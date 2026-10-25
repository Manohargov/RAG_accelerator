"""User settings with JSON persistence.

Settings are deliberately plain data so they can be inspected, hand-edited or
version-controlled.  Every field can also be overridden for a single run via
an environment variable (``BOOKRAG_<FIELD>``), which keeps CI and one-off
experiments reproducible without touching the saved file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ..data import paths
from ..errors import ConfigError

#: Environment variable prefix used for per-run overrides.
ENV_PREFIX = "BOOKRAG_"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return fallback


@dataclass
class AppSettings:
    """Everything the user can tune for an indexing / retrieval run."""

    # --- Model ------------------------------------------------------------
    model_id: str = "jinaai/jina-embeddings-v5-omni-small"
    device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    trust_remote_code: bool = True
    max_text_length: int = 512  # tokens fed to the text tower
    #: Task instruction used for *queries* (Jina's asymmetric retrieval setup).
    query_task: str = "retrieval.query"
    #: Task instruction used for *documents* while indexing. Empty = none.
    index_task: str = ""

    # --- Vector store -----------------------------------------------------
    collection_name: str = "book_chunks"
    hnsw_space: str = "cosine"
    top_k: int = 5

    # --- Ingestion --------------------------------------------------------
    chunk_size: int = 1200  # characters per text chunk
    chunk_overlap: int = 150  # characters of overlap between chunks
    batch_size: int = 16  # items per model forward pass
    extract_images: bool = True
    keep_extracted_images: bool = True
    min_image_pixels: int = 4096  # drop decorative sprites below this area
    max_image_side: int = 1024  # downscale images before embedding

    # --- Runtime ----------------------------------------------------------
    verbose: bool = False
    seed: int = 42

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppSettings:
        """Build settings from a mapping, ignoring unknown or malformed keys."""
        known = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {}
        for key, value in (data or {}).items():
            if key not in known or value is None:
                continue
            clean[key] = value
        instance = cls(**clean)
        instance.sanitise()
        return instance

    # ------------------------------------------------------------------ #
    def sanitise(self) -> None:
        """Clamp values into sane ranges so a hand-edited file cannot break the app."""
        self.model_id = (self.model_id or "").strip() or AppSettings.model_id
        self.device = (self.device or "auto").strip().lower() or "auto"
        self.dtype = (self.dtype or "auto").strip().lower() or "auto"
        self.collection_name = (self.collection_name or "").strip() or "book_chunks"
        self.hnsw_space = (self.hnsw_space or "cosine").strip().lower()
        self.query_task = str(self.query_task or "").strip()
        self.index_task = str(self.index_task or "").strip()
        self.max_text_length = max(32, min(8192, int(self.max_text_length or 512)))

        self.top_k = max(1, min(200, int(self.top_k or 5)))
        self.chunk_size = max(200, int(self.chunk_size or 1200))
        self.chunk_overlap = max(0, min(self.chunk_size - 1, int(self.chunk_overlap or 0)))
        self.batch_size = max(1, min(256, int(self.batch_size or 16)))
        self.min_image_pixels = max(0, int(self.min_image_pixels or 0))
        self.max_image_side = max(64, int(self.max_image_side or 1024))
        self.seed = int(self.seed)

        self.extract_images = bool(self.extract_images)
        self.keep_extracted_images = bool(self.keep_extracted_images)
        self.trust_remote_code = bool(self.trust_remote_code)
        self.verbose = bool(self.verbose)

        if self.hnsw_space not in {"cosine", "l2", "ip"}:
            raise ConfigError(
                f"Unsupported hnsw_space {self.hnsw_space!r}; expected cosine, l2 or ip."
            )
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigError(f"Unsupported dtype {self.dtype!r}.")
        if self.device != "auto":
            allowed = self.device.startswith("cuda") or self.device in {"cpu", "mps"}
            if not allowed:
                raise ConfigError(f"Unsupported device {self.device!r}.")

    # ------------------------------------------------------------------ #
    def update(self, **kwargs: Any) -> None:
        """Apply a partial update and re-validate the result."""
        known = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key in known and value is not None:
                setattr(self, key, value)
        self.sanitise()

    def apply_environment(self, environ: dict[str, str] | None = None) -> AppSettings:
        """Overlay ``BOOKRAG_*`` environment variables on top of this instance."""
        env = environ if environ is not None else dict(os.environ)
        known = {f.name for f in fields(self)}
        overrides: dict[str, Any] = {}
        for raw_key, raw_value in env.items():
            if not raw_key.startswith(ENV_PREFIX):
                continue
            name = raw_key[len(ENV_PREFIX) :].lower()
            if name not in known:
                continue
            current = getattr(self, name)
            if isinstance(current, bool):
                overrides[name] = _as_bool(raw_value, current)
            elif isinstance(current, int):
                try:
                    overrides[name] = int(raw_value)
                except ValueError:
                    continue
            else:
                overrides[name] = raw_value
        if overrides:
            self.update(**overrides)
        return self

    # ------------------------------------------------------------------ #
    def load(self, path: Path | None = None) -> AppSettings:
        """Read settings from disk. Missing or corrupt files yield defaults."""
        target = Path(path) if path else paths.settings_file()
        try:
            raw = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return self
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self
        if isinstance(data, dict):
            self.update(**data)
        return self

    def save(self, path: Path | None = None) -> Path:
        """Persist settings atomically and return the file path."""
        target = Path(path) if path else paths.settings_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return target
