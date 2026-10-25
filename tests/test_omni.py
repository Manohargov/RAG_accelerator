"""The v5 omni checkpoint: how it is recognised, fetched and driven.

Nothing here touches torch, transformers or the network.  What is under test
is the adapter - which shapes get detected, which prefix a string gets, and
how a model id becomes something loadable - which are precisely the parts a
real checkpoint cannot tell you about.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest

from bookrag.accelerator.embedder import (
    ClipEmbedder,
    _is_omni_model,
    _load_kwargs,
    _local_copy,
    _resolve_source,
)
from bookrag.data import paths
from bookrag.errors import ModelUnavailableError
from bookrag.Rag_app.settings import AppSettings

OMNI_ID = "jinaai/jina-embeddings-v5-omni-small"
CLIP_ID = "jinaai/jina-clip-v2"


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
def test_omni_is_recognised_by_repository_id() -> None:
    assert _is_omni_model(OMNI_ID)


def test_omni_is_recognised_by_local_directory_name() -> None:
    # The plain folder the snapshot lands in on a Windows box without symlinks
    # - which is what a machine that already fetched it will load from.
    assert _is_omni_model(r"C:\models\v5-omni-small")


def test_clip_is_not_mistaken_for_omni() -> None:
    assert not _is_omni_model(CLIP_ID)


@pytest.mark.parametrize("model_id", ["", None])
def test_an_empty_id_is_not_omni(model_id: str | None) -> None:
    assert not _is_omni_model(model_id)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_omni_drops_the_audio_tower_it_cannot_use() -> None:
    assert _load_kwargs(OMNI_ID) == {"modality": "vision"}


def test_clip_is_not_passed_an_argument_its_class_does_not_take() -> None:
    assert _load_kwargs(CLIP_ID) == {}


def test_a_materialised_copy_wins_over_the_repository() -> None:
    local = paths.model_cache_dir() / "jina-embeddings-v5-omni-small"
    local.mkdir(parents=True, exist_ok=True)
    (local / "config.json").write_text("{}", encoding="utf-8")

    assert Path(_resolve_source(OMNI_ID)) == local


def test_without_a_copy_the_repository_id_is_left_alone() -> None:
    # Nothing fetched yet: the caller must try HuggingFace first, and only
    # materialise a copy if that fails.
    assert _resolve_source(CLIP_ID) == CLIP_ID


def test_a_plain_path_passes_through_untouched(tmp_path: Path) -> None:
    path = tmp_path / "some-checkpoint"
    assert _resolve_source(str(path)) == str(path)


def test_a_plain_path_is_never_treated_as_a_repository(tmp_path: Path) -> None:
    assert _local_copy(str(tmp_path / "elsewhere")) is None


def test_a_directory_without_a_config_is_not_a_usable_copy() -> None:
    # A half-finished download must not pass for a checkpoint, or the
    # fallback would load rubbish instead of fetching the real thing.
    partial = paths.model_cache_dir() / "jina-clip-v2"
    partial.mkdir(parents=True, exist_ok=True)

    assert _local_copy(CLIP_ID) is None


# --------------------------------------------------------------------------- #
# Driving it
# --------------------------------------------------------------------------- #
class _FakeTensor:
    """Enough of a torch tensor to survive the embedder's post-processing."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)

    def float(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeTorch:
    """The two members the omni encode paths reach for."""

    class Tensor:  # noqa: D106 - only used as an isinstance() target
        pass

    def no_grad(self):  # noqa: D102
        return contextlib.nullcontext()


class _FakeTokenizer:
    """Only the lookup the image-token guard needs."""

    def __init__(self, known: str) -> None:
        self._known = known

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return 7 if token == self._known else None


class _FakeOmniProcessor:
    """Stands in for Qwen3VLProcessor, which needs real weights to build."""

    #: The real one reads this off the processor; mirroring that keeps the
    #: tests honest about where the marker is supposed to come from.
    IMAGE_TOKEN = "<image>"

    def __init__(self) -> None:
        self.image_token = self.IMAGE_TOKEN
        self.tokenizer = _FakeTokenizer(self.IMAGE_TOKEN)
        self.text_batches: list[list[str]] = []
        self.image_batches: list[int] = []
        self.image_texts: list[list[str]] = []
        self.max_lengths: list[int | None] = []

    def __call__(self, *, text=None, images=None, **kwargs):  # noqa: ANN001, ANN003
        self.max_lengths.append(kwargs.get("max_length"))
        if text is not None and images is None:
            self.text_batches.append(list(text))
            return {"input_ids": [[0] for _ in text]}
        self.image_batches.append(len(images))
        self.image_texts.append(list(text or []))
        return {"pixel_values": [[0.0] for _ in images]}


class _FakeOmniModel:
    """The v5 shape: one embed() for every modality."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def embed(self, **kwargs):  # noqa: ANN003
        self.seen.append(kwargs)
        size = len(kwargs.get("input_ids") or kwargs.get("pixel_values"))
        return _FakeTensor(np.arange(size * 4, dtype=np.float32).reshape(size, 4))


def _omni_embedder(**overrides) -> tuple[ClipEmbedder, _FakeOmniProcessor]:
    processor = _FakeOmniProcessor()
    embedder = ClipEmbedder(AppSettings(model_id=OMNI_ID, **overrides))
    embedder._model = _FakeOmniModel()
    embedder._processor = processor
    embedder._torch = _FakeTorch()
    embedder._device = "cpu"
    embedder._api = embedder._detect_api()
    return embedder, processor


def test_detect_api_finds_the_omni_embed_method() -> None:
    embedder, _ = _omni_embedder()
    assert embedder.api == "embed"


def test_a_query_takes_the_query_prefix() -> None:
    embedder, processor = _omni_embedder()
    embedder._encode_text_omni(["what is energy"], task="retrieval.query")

    assert processor.text_batches == [["Query: what is energy"]]


def test_an_unprompted_string_is_indexed_as_a_document() -> None:
    embedder, processor = _omni_embedder()
    embedder._encode_text_omni(["Ek = 1/2 m v^2"], task=None)

    assert processor.text_batches == [["Document: Ek = 1/2 m v^2"]]


def test_text_is_batched_at_the_configured_size() -> None:
    embedder, processor = _omni_embedder(batch_size=2)
    embedder._encode_text_omni([f"chunk {i}" for i in range(5)], task=None)

    assert [len(batch) for batch in processor.text_batches] == [2, 2, 1]


def test_text_is_truncated_to_the_configured_length() -> None:
    embedder, processor = _omni_embedder(max_text_length=256)
    embedder._encode_text_omni(["x"], task=None)

    assert processor.max_lengths == [256]


def test_images_are_paired_with_one_placeholder_each() -> None:
    embedder, processor = _omni_embedder()
    embedder._encode_images_omni([object(), object()])

    assert processor.image_batches == [2]


def test_images_are_marked_with_the_processors_own_image_token() -> None:
    # The marker has to come from the processor.  A plausible character of our
    # own choosing is not in the vocabulary, encodes to <unk>, and leaves the
    # sequence with no image in it at all - so every image comes back with a
    # byte-identical vector and nothing raises.
    embedder, processor = _omni_embedder()
    embedder._encode_images_omni([object(), object()])

    assert processor.image_texts == [[processor.image_token, processor.image_token]]


def test_a_processor_without_an_image_token_is_refused() -> None:
    embedder, processor = _omni_embedder()
    del processor.image_token

    with pytest.raises(ModelUnavailableError, match="no image_token"):
        embedder._encode_images_omni([object()])


def test_an_image_token_missing_from_the_vocabulary_is_refused() -> None:
    embedder, processor = _omni_embedder()
    processor.tokenizer = _FakeTokenizer("<not-in-the-vocab>")

    with pytest.raises(ModelUnavailableError, match="not in its vocabulary"):
        embedder._encode_images_omni([object()])


def test_text_runs_through_the_public_entry_point() -> None:
    embedder, _ = _omni_embedder()

    vectors = embedder.embed_texts(["a", "b"], task="retrieval.query")

    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_images_run_through_the_public_entry_point() -> None:
    embedder, _ = _omni_embedder()

    vectors = embedder.embed_images([object(), object(), object()])

    assert vectors.shape == (3, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
