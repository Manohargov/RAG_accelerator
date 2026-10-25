"""Embedding wrapper for Jina CLIP v2 and Jina's v5 omni family.

The heavy dependencies (``torch``, ``transformers``) are imported lazily so
that the rest of the application - CLI help, settings, store management and
the test-suite - keeps working on a machine where the ML stack is not
installed.

Both checkpoints are *dual-encoders*: text and images live in the same vector
space, which is what allows a text query to retrieve a figure and an image
query to retrieve the paragraph that explains it.  They differ in how they are
loaded and in how asymmetric retrieval is expressed, so this wrapper works out
which of the two shapes it is holding and adapts.

API notes (verified against ``jinaai/jina-clip-implementation``)
----------------------------------------------------------------
The model class ``JinaCLIPModel`` exposes three generations of embedding API.
This wrapper prefers the newest and falls back to the older ones:

``encode_text`` / ``encode_image``
    The current API.  Handles tokenisation, image preprocessing, batching and
    L2 normalisation internally.  ``encode_text`` additionally accepts a
    ``task`` name (e.g. ``"retrieval.query"``) which prepends a task
    instruction to the text - the correct way to run *asymmetric* retrieval,
    where documents are indexed without an instruction and queries carry one.

``get_text_features`` / ``get_image_features``
    Accepts pre-tokenised / pre-processed tensors.

``get_text_embeddings`` / ``get_image_embeddings``
    The oldest form.  Kept only as a fallback for older checkpoints.

``jina-embeddings-v5`` (``embed``)
----------------------------------
One ``embed()`` for every modality, driven by the processor rather than by a
method per tower.  The instruction table is replaced by two fixed role
prefixes - ``Query:`` for what is being searched for, ``Document:`` for what
is indexed - which is the same asymmetry expressed differently, and maps onto
the existing ``query_task`` / ``index_task`` contract without changing what
callers pass in.  The checkpoint also carries an audio tower BookRAG never
uses, dropped at load time through ``modality="vision"``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import paths
from ..errors import DependencyError, ModelUnavailableError
from ..logs.logging_setup import get_logger
from ..logs.monitoring import cuda_peak_mb, get_monitor, parameter_count, process_peak_rss_mb, stage
from ..Rag_app.settings import AppSettings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

log = get_logger("embedder")

#: Dimensionality of the Jina CLIP v2 embedding space.
JINA_CLIP_V2_DIM = 1024

_INSTALL_HINT = "pip install -r requirements.txt"


@contextlib.contextmanager
def _quiet_third_party():
    """Silence the noise Jina's remote code produces while loading.

    The remote code warns about missing CUDA extras (FlashAttention, xFormers)
    and uses the deprecated ``torch_dtype`` argument.  None of that is
    actionable for the user, and it buries the app's own log lines.
    """
    patterns = (
        r".*torch_dtype.*",
        r".*Flash attention requires CUDA.*",
        r".*xFormers requires CUDA.*",
        r".*apex.*",
    )
    for pattern in patterns:
        warnings.filterwarnings("ignore", message=pattern)

    loggers = ("transformers", "huggingface_hub", "transformers_modules")
    previous: dict[str, int] = {}
    for name in loggers:
        logger = logging.getLogger(name)
        previous[name] = logger.level
        logger.setLevel(logging.ERROR)

    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def resolve_device(preference: str = "auto") -> str:
    """Pick the torch device to run on.

    ``auto`` prefers CUDA, then Apple Silicon (MPS), then CPU.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised via doctor
        raise DependencyError("PyTorch is not installed.", _INSTALL_HINT) from exc

    preference = (preference or "auto").lower()
    if preference != "auto":
        if preference.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        if preference.startswith("xpu") and not _xpu_available(torch):
            log.warning("XPU requested but unavailable; falling back to CPU.")
            return "cpu"
        return preference

    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available(torch):
        return "xpu"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def _xpu_available(torch: Any) -> bool:
    """True when an Intel XPU device is present and usable.

    ``torch.xpu`` only exists on the xpu build of PyTorch, so ``getattr``
    is the guard - probing a CPU-only install must not raise.
    """
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except Exception:  # noqa: BLE001 - a probe must never raise
        return False


def resolve_dtype(preference: str, device: str) -> str:
    """Choose the floating point precision for the model weights.

    The published checkpoint is ``bfloat16``.  That is a good fit for modern
    GPUs but slow on CPU, so ``auto`` downgrades to ``float32`` everywhere
    except CUDA and XPU - and both are asked first whether they handle bf16.
    """
    preference = (preference or "auto").lower()
    if preference != "auto":
        return preference

    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_bf16_supported():
                return "bfloat16"
        except Exception:  # noqa: BLE001 - fall through to the safe default
            pass
        return "float16"
    if device.startswith("xpu"):
        try:
            import torch

            if torch.xpu.is_bf16_supported():
                return "bfloat16"
        except Exception:  # noqa: BLE001 - fall through to the safe default
            pass
        return "float16"
    return "float32"


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving zero rows untouched.

    Jina CLIP v2 already returns unit vectors, but normalising defensively
    guarantees that the cosine space configured in ChromaDB behaves as
    expected even if a different checkpoint is swapped in.
    """
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


#: Role prefixes Jina's v5 family uses in place of CLIP's instruction table.
_OMNI_QUERY_PREFIX = "Query: "
_OMNI_DOCUMENT_PREFIX = "Document: "

#: Substrings that identify a ``jina-embeddings-v5`` omni checkpoint.  Matched
#: against both the HuggingFace repository id and the directory name a local
#: copy is fetched into, so ``.../models/v5-omni-small`` is recognised too.
_OMNI_MARKERS = ("jina-embeddings-v5", "jina_embeddings_v5", "v5-omni")


def _is_omni_model(model_id: str) -> bool:
    """Is ``model_id`` one of Jina's v5 omni checkpoints?

    That family needs a different load call and a different embedding
    contract from Jina CLIP v2, and before the weights are in memory there is
    no attribute shared by exactly those models - so the id is what says
    which one we are about to open.
    """
    lowered = (model_id or "").lower()
    return any(marker in lowered for marker in _OMNI_MARKERS)


def _local_copy(model_id: str) -> Path | None:
    """A fully materialised copy of ``model_id``, if one has been fetched."""
    if "/" not in model_id or "\\" in model_id:
        return None  # not a repository id - nothing to look up
    candidate = paths.model_cache_dir() / model_id.rsplit("/", 1)[-1]
    return candidate if (candidate / "config.json").is_file() else None


def _resolve_source(model_id: str) -> str:
    """Turn a model id into something ``from_pretrained`` can open directly.

    HuggingFace's cache is built from symlinks, which Windows only permits
    under Developer Mode or with elevation.  Where they are unavailable the
    cache cannot be populated at all, so we keep a plain copy of each
    checkpoint in the application's own model directory and read from there.
    """
    local = _local_copy(model_id)
    return str(local) if local is not None else model_id


def _materialise(model_id: str) -> Path | None:
    """Fetch ``model_id`` into the model directory as ordinary files.

    Returns ``None`` when there is nothing to fetch or the download fails -
    the signal for the caller to surface its own error rather than this one.
    """
    if _local_copy(model_id) is not None or "/" not in model_id:
        return None
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    destination = paths.model_cache_dir() / model_id.rsplit("/", 1)[-1]
    try:
        log.info("Fetching a plain copy of %s ...", model_id)
        with _quiet_third_party():
            snapshot_download(model_id, local_dir=str(destination))
    except Exception as exc:  # noqa: BLE001 - a download failure is the caller's to report
        log.warning("Could not fetch a plain copy of %s: %s", model_id, exc)
        return None
    return destination


def _load_kwargs(model_id: str) -> dict[str, Any]:
    """Extra ``from_pretrained`` arguments a particular checkpoint needs.

    The omni family carries an audio tower BookRAG never uses, and ``modality``
    drops it at load time rather than paying for it in memory.  Jina CLIP v2
    takes no such argument, so it gets an empty dict.
    """
    return {"modality": "vision"} if _is_omni_model(model_id) else {}


class ClipEmbedder:
    """Loads a Jina checkpoint and turns text / images into normalised vectors."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._torch: Any = None
        self._api: str = "encode"

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        """The device the model currently lives on."""
        return self._device

    @property
    def api(self) -> str:
        """Which embedding API the loaded checkpoint exposes."""
        return self._api

    def load(self) -> ClipEmbedder:
        """Load the model and processor. Safe to call repeatedly."""
        if self._model is not None:
            return self

        # The timer wraps the imports as well as the checkpoint load, because
        # a cold `import torch` is a real part of what the user waits for.
        started = time.perf_counter()
        with stage("model.load") as handle:
            try:
                import torch
                from transformers import AutoModel
            except ImportError as exc:
                raise DependencyError(
                    "Jina CLIP v2 needs PyTorch and Transformers.", _INSTALL_HINT
                ) from exc

            self._torch = torch
            self._device = resolve_device(self.settings.device)
            dtype = resolve_dtype(self.settings.dtype, self._device)

            cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
            if cache:
                Path(cache).mkdir(parents=True, exist_ok=True)

            log.info("Loading %s on %s (%s) ...", self.settings.model_id, self._device, dtype)
            with _quiet_third_party():
                try:
                    model = self._load_checkpoint(AutoModel)
                except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
                    raise ModelUnavailableError(
                        f"Could not load {self.settings.model_id!r}: {exc}",
                        "Check your network connection and that the model id is correct.",
                    ) from exc

            model = self._cast(model, dtype)
            self._model = model.to(self._device).eval()
            self._api = self._detect_api()
            handle.detail.update(
                id=self.settings.model_id,
                device=self._device,
                dtype=dtype,
                api=self._api,
            )

        load_s = time.perf_counter() - started
        log.info("Model ready (device=%s, dtype=%s, api=%s).", self._device, dtype, self._api)

        # Everything about *how* the model is running, captured once, on load.
        get_monitor().set_model(
            id=self.settings.model_id,
            device=self._device,
            dtype=dtype,
            api=self._api,
            load_s=load_s,
            parameters=parameter_count(self._model),
            peak_rss_mb=process_peak_rss_mb(),
            cuda_peak_mb=cuda_peak_mb(),
            threads=getattr(torch, "get_num_threads", lambda: None)(),
            cpus=os.cpu_count(),
            torch=getattr(torch, "__version__", None),
            cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
            python=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
        )
        return self

    def _load_checkpoint(self, auto_model: Any) -> Any:
        """Load the weights, working around Windows' symlink restriction.

        The first attempt reads whatever is already available.  If that fails
        and the checkpoint has not been materialised locally, fetch it as
        ordinary files and try again: HuggingFace's own cache cannot be
        written on a Windows box without Developer Mode, and a failed cache
        write must not become "this model does not exist".
        """
        model_id = self.settings.model_id
        source = _resolve_source(model_id)
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
            **_load_kwargs(model_id),
        }
        try:
            return auto_model.from_pretrained(source, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retried, then reported as a domain error
            destination = _materialise(model_id)
            if destination is None:
                raise
            log.warning(
                "Could not load %s directly (%s); retrying from %s.",
                model_id,
                exc,
                destination,
            )
            return auto_model.from_pretrained(str(destination), **kwargs)

    def _cast(self, model: Any, dtype: str) -> Any:
        """Cast the model weights to the requested precision."""
        if dtype == "float32":
            return model.float()
        if dtype == "float16":
            return model.half()
        if dtype == "bfloat16":
            return model.bfloat16()
        return model

    def _detect_api(self) -> str:
        """Work out which embedding API this checkpoint exposes."""
        model = self._model
        if hasattr(model, "encode_text") and hasattr(model, "encode_image"):
            return "encode"
        if hasattr(model, "get_text_features") and hasattr(model, "get_image_features"):
            return "features"
        if hasattr(model, "get_text_embeddings") and hasattr(model, "get_image_embeddings"):
            return "embeddings"
        if hasattr(model, "embed") and callable(model.embed):
            # jina-embeddings-v5: one embed() per modality, driven by the
            # processor rather than by a method per tower.  Checked last, so
            # the detection order every existing checkpoint relies on is
            # left exactly as it was.
            return "embed"
        raise ModelUnavailableError(
            f"{self.settings.model_id!r} exposes no known embedding API.",
            "Expected one of: encode_text, get_text_features, get_text_embeddings or embed.",
        )

    def unload(self) -> None:
        """Release GPU/CPU memory held by the model."""
        self._model = None
        self._processor = None
        if self._torch is not None:
            try:
                import gc

                gc.collect()
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _processor_or_fail(self) -> Any:
        """Load the processor on demand (only the fallback APIs need it)."""
        if self._processor is None:
            if _is_omni_model(self.settings.model_id):
                self._processor = self._build_omni_processor()
            else:
                try:
                    from transformers import AutoProcessor
                except ImportError as exc:
                    raise DependencyError(
                        "Jina CLIP v2 needs PyTorch and Transformers.", _INSTALL_HINT
                    ) from exc
                self._processor = AutoProcessor.from_pretrained(
                    _resolve_source(self.settings.model_id),
                    trust_remote_code=self.settings.trust_remote_code,
                )
        return self._processor

    def _build_omni_processor(self) -> Any:
        """Assemble the omni processor explicitly.

        Its ``processor_class`` is ``Qwen3VLProcessor``, which always resolves
        a video component.  The auto registry only knows how to name that
        component for model types it recognises, and ``jina_embeddings_v5_omni``
        is not one of them under transformers 4.57 - so the plain
        ``AutoProcessor.from_pretrained`` fails before it ever looks at our
        files.  Handing the class its three parts directly skips that lookup.
        The video processor is inert here: BookRAG embeds text and page images.
        """
        source = _resolve_source(self.settings.model_id)
        try:
            from transformers import (
                AutoImageProcessor,
                AutoTokenizer,
                Qwen3VLProcessor,
                Qwen3VLVideoProcessor,
            )
        except ImportError as exc:
            raise DependencyError(
                "jina-embeddings-v5 needs PyTorch and Transformers.", _INSTALL_HINT
            ) from exc

        trust = self.settings.trust_remote_code
        with _quiet_third_party():
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust)
            image_processor = AutoImageProcessor.from_pretrained(source, trust_remote_code=trust)
        return Qwen3VLProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=Qwen3VLVideoProcessor(),
        )

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch is not None and isinstance(value, torch.Tensor):
                moved[key] = value.to(self._device)
            else:
                moved[key] = value
        return moved

    def _resolve_task(self, task: str | None) -> str | None:
        """Validate a task name against the model's instruction table.

        Returns the task name to use, or ``None`` when the task is unknown
        (in which case the model is called without an instruction).
        """
        if not task:
            return None
        text_tower = getattr(self._model, "text_model", None)
        resolver = getattr(text_tower, "get_instruction_from_task", None)
        if resolver is None:
            return task
        try:
            resolver(task)
        except Exception as exc:  # noqa: BLE001 - unknown task names raise ValueError
            log.warning("Task %r is not supported by this model (%s); ignoring it.", task, exc)
            return None
        return task

    def _encode_text_omni(self, texts: list[str], task: str | None) -> np.ndarray:
        """Embed text through the omni ``embed()`` API.

        Jina's v5 family dropped the instruction table for two fixed role
        prefixes, and the existing contract already says which one applies: a
        non-empty ``task`` means this string is what is being searched for, so
        it takes ``Query:``; no task means it is being indexed, so it takes
        ``Document:``.  Callers keep passing exactly what they pass today.
        """
        proc = self._processor_or_fail()
        prefix = _OMNI_QUERY_PREFIX if task else _OMNI_DOCUMENT_PREFIX
        size = self.settings.batch_size
        batches: list[np.ndarray] = []

        # The tokenizer is loaded lazily on the first call, which is noisy.
        with _quiet_third_party():
            for start in range(0, len(texts), size):
                chunk = [prefix + text for text in texts[start : start + size]]
                inputs = self._to_device(
                    proc(
                        text=chunk,
                        padding=True,
                        truncation=True,
                        max_length=self.settings.max_text_length,
                        return_tensors="pt",
                    )
                )
                with self._torch.no_grad():
                    vectors = self._model.embed(**inputs)
                batches.append(vectors.float().cpu().numpy())
        return np.vstack(batches)

    def _omni_image_token(self, processor: Any) -> str:
        """The token marking where an image goes, as this processor defines it.

        It cannot be invented: the processor and the model config have to agree
        on which id means "an image starts here", and only the processor knows
        it.  A token the tokenizer does not recognise is worse than absent -
        it encodes to ``<unk>``, the sequence reaching the model then carries
        no image marker at all, and the vision output is never merged into the
        text.  Every image embeds as the same empty placeholder, every vector
        comes back identical, and nothing raises; so the one cheap place to
        notice is here.
        """
        token = getattr(processor, "image_token", None)
        if not token:
            raise ModelUnavailableError(
                f"{type(processor).__name__} exposes no image_token, so images cannot be embedded.",
                "Check that settings.model_id matches the checkpoint being "
                "loaded; this is not a jina-embeddings-v5 processor.",
            )
        convert = getattr(getattr(processor, "tokenizer", None), "convert_tokens_to_ids", None)
        if convert is not None and convert(token) is None:
            raise ModelUnavailableError(
                f"The processor's image_token {token!r} is not in its vocabulary.",
                "The tokenizer and the model disagree about the image marker; "
                "make sure both come from the same checkpoint.",
            )
        return token

    def _encode_images_omni(self, images: list[Image]) -> np.ndarray:
        """Embed images through the omni ``embed()`` API.

        The processor wants one text slot per image to hang that image's patches
        off, and the marker for that slot comes from the processor itself - see
        :meth:`_omni_image_token` for why guessing it fails silently.
        """
        proc = self._processor_or_fail()
        placeholder = self._omni_image_token(proc)
        size = self.settings.batch_size
        batches: list[np.ndarray] = []

        with _quiet_third_party():
            for start in range(0, len(images), size):
                chunk = images[start : start + size]
                # Padding is not optional here.  Every image becomes as many
                # patches as its aspect ratio divides into, so two figures of
                # different sizes yield different-length inputs that cannot
                # stack into one tensor.  Real book figures are never uniform,
                # so this is the common case rather than the exception.
                inputs = self._to_device(
                    proc(
                        images=chunk,
                        text=[placeholder] * len(chunk),
                        padding=True,
                        return_tensors="pt",
                    )
                )
                with self._torch.no_grad():
                    vectors = self._model.embed(**inputs)
                batches.append(vectors.float().cpu().numpy())
        return np.vstack(batches)

    def _encode_text(self, texts: list[str], task: str | None) -> np.ndarray:
        """Embed text with the modern ``encode_text`` API."""
        kwargs: dict[str, Any] = {
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "batch_size": self.settings.batch_size,
        }
        if self.settings.max_text_length:
            kwargs["max_length"] = self.settings.max_text_length

        resolved = self._resolve_task(task)
        if resolved:
            kwargs["task"] = resolved

        # The tokenizer is loaded lazily on the first call, which is noisy.
        with _quiet_third_party():
            try:
                result = self._model.encode_text(list(texts), **kwargs)
            except ValueError:
                # Unknown task name raised deep inside the model: retry without it.
                if not resolved:
                    raise
                log.warning("Retrying text encoding without the task instruction.")
                kwargs.pop("task", None)
                result = self._model.encode_text(list(texts), **kwargs)

        return np.asarray(result, dtype=np.float32).reshape(len(texts), -1)

    def _encode_images(self, images: list[Any]) -> np.ndarray:
        """Embed images with the modern ``encode_image`` API."""
        # The image processor is loaded lazily on the first call.
        with _quiet_third_party():
            result = self._model.encode_image(
                images,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.settings.batch_size,
            )
        return np.asarray(result, dtype=np.float32).reshape(len(images), -1)

    def _features_text(self, texts: list[str]) -> np.ndarray:
        """Embed text via the processor + ``get_text_features`` fallback."""
        processor = self._processor_or_fail()
        inputs = processor(
            text=list(texts),
            padding=True,
            truncation=True,
            max_length=self.settings.max_text_length or 512,
            return_tensors="pt",
        )
        inputs = self._to_device(dict(inputs))
        with self._torch.no_grad():
            embeddings = self._model.get_text_features(**inputs)
        return embeddings.detach().cpu().float().numpy()

    def _features_images(self, images: list[Any]) -> np.ndarray:
        """Embed images via the processor + ``get_image_features`` fallback."""
        processor = self._processor_or_fail()
        inputs = processor(images=list(images), return_tensors="pt")
        inputs = self._to_device(dict(inputs))
        with self._torch.no_grad():
            embeddings = self._model.get_image_features(**inputs)
        return embeddings.detach().cpu().float().numpy()

    def _legacy_text(self, texts: list[str]) -> np.ndarray:
        """Embed text via the ``get_text_embeddings`` fallback."""
        processor = self._processor_or_fail()
        inputs = processor(
            text=list(texts),
            padding=True,
            truncation=True,
            max_length=self.settings.max_text_length or 512,
            return_tensors="pt",
        )
        inputs = self._to_device(dict(inputs))
        with self._torch.no_grad():
            embeddings = self._model.get_text_embeddings(**inputs)
        return embeddings.detach().cpu().float().numpy()

    def _legacy_images(self, images: list[Any]) -> np.ndarray:
        """Embed images via the ``get_image_embeddings`` fallback."""
        processor = self._processor_or_fail()
        inputs = processor(images=list(images), return_tensors="pt")
        inputs = self._to_device(dict(inputs))
        with self._torch.no_grad():
            embeddings = self._model.get_image_embeddings(**inputs)
        return embeddings.detach().cpu().float().numpy()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed_texts(self, texts: Sequence[str], task: str | None = None) -> np.ndarray:
        """Embed a list of strings. Returns a ``(n, dim)`` float32 array.

        ``task`` selects a Jina task instruction (e.g. ``"retrieval.query"``)
        which is prepended to every string.
        """
        if not texts:
            return np.zeros((0, JINA_CLIP_V2_DIM), dtype=np.float32)
        # Loaded outside the timer: `model.load` and `embed.text` are sibling
        # leaves, so the report never counts the same wall-clock twice.
        self.load()
        items = list(texts)

        with stage("embed.text", items=len(items), api=self._api) as handle:
            if task:
                handle.detail["task"] = task
            if self._api == "encode":
                vectors = self._encode_text(items, task)
            elif self._api == "features":
                vectors = self._features_text(items)
            elif self._api == "embed":
                vectors = self._encode_text_omni(items, task)
            else:
                vectors = self._legacy_text(items)
            vectors = l2_normalise(vectors)
        return vectors

    def embed_text(self, text: str, task: str | None = None) -> np.ndarray:
        """Embed a single string. Returns a ``(dim,)`` float32 array."""
        return self.embed_texts([text], task=task)[0]

    def embed_images(self, images: Sequence[Image]) -> np.ndarray:
        """Embed PIL images. Returns a ``(n, dim)`` float32 array."""
        if not images:
            return np.zeros((0, JINA_CLIP_V2_DIM), dtype=np.float32)
        self.load()
        items = list(images)

        with stage("embed.image", items=len(items), api=self._api):
            if self._api == "encode":
                vectors = self._encode_images(items)
            elif self._api == "features":
                vectors = self._features_images(items)
            elif self._api == "embed":
                vectors = self._encode_images_omni(items)
            else:
                vectors = self._legacy_images(items)
            vectors = l2_normalise(vectors)
        return vectors

    def embed_image_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Open and embed images from disk."""
        from PIL import Image, ImageOps

        opened: list[Image.Image] = []
        try:
            # Decoding is timed separately from `embed.image`: opening files and
            # running the vision tower have very different costs, and keeping
            # them as sibling leaves stops the report double-counting them.
            with stage("embed.image_open") as open_stats:
                for path in paths:
                    with Image.open(path) as image_file:
                        image = ImageOps.exif_transpose(image_file).convert("RGB")
                        opened.append(self._downscale(image))
                    open_stats.items += 1
            return self.embed_images(opened)
        finally:
            for image in opened:
                image.close()

    def embed_image(self, path: str | Path) -> np.ndarray:
        """Embed a single image file. Returns a ``(dim,)`` float32 array."""
        return self.embed_image_paths([path])[0]

    def _downscale(self, image: Image) -> Image:
        """Shrink oversized images before they reach the processor."""
        from PIL import Image

        limit = self.settings.max_image_side
        if not limit:
            return image
        width, height = image.size
        longest = max(width, height)
        if longest <= limit:
            return image
        scale = limit / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.LANCZOS)

    @property
    def dimension(self) -> int:
        """Dimensionality of the vectors produced by the loaded model."""
        return JINA_CLIP_V2_DIM
