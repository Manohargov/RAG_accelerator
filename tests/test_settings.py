"""Tests for settings loading, validation and environment overrides."""

from __future__ import annotations

import json

import pytest

from bookrag.errors import ConfigError
from bookrag.Rag_app.settings import AppSettings


def test_defaults_are_sane():
    settings = AppSettings()
    settings.sanitise()
    assert settings.model_id == "jinaai/jina-embeddings-v5-omni-small"
    assert settings.hnsw_space == "cosine"
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.top_k >= 1


def test_overlap_is_clamped_below_chunk_size():
    settings = AppSettings(chunk_size=300, chunk_overlap=5000)
    settings.sanitise()
    assert settings.chunk_overlap == 299


def test_unknown_keys_are_ignored():
    settings = AppSettings.from_dict({"top_k": 7, "not_a_setting": 1})
    assert settings.top_k == 7
    assert not hasattr(settings, "not_a_setting")


def test_invalid_hnsw_space_raises():
    with pytest.raises(ConfigError):
        AppSettings(hnsw_space="manhattan").sanitise()


def test_invalid_device_raises():
    with pytest.raises(ConfigError):
        AppSettings(device="quantum").sanitise()


def test_environment_overrides_are_applied():
    settings = AppSettings()
    settings.apply_environment(
        {
            "BOOKRAG_TOP_K": "11",
            "BOOKRAG_VERBOSE": "true",
            "BOOKRAG_MODEL_ID": "other/model",
            "BOOKRAG_UNRELATED": "x",
        }
    )
    assert settings.top_k == 11
    assert settings.verbose is True
    assert settings.model_id == "other/model"


def test_environment_override_ignores_bad_int():
    settings = AppSettings(top_k=5)
    settings.apply_environment({"BOOKRAG_TOP_K": "not-a-number"})
    assert settings.top_k == 5


def test_save_and_load_roundtrip(tmp_path):
    settings = AppSettings(top_k=9, chunk_size=800)
    target = tmp_path / "settings.json"
    settings.save(target)

    restored = AppSettings().load(target)
    assert restored.top_k == 9
    assert restored.chunk_size == 800


def test_load_tolerates_corrupt_file(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text("{not json", encoding="utf-8")
    restored = AppSettings(top_k=4).load(target)
    assert restored.top_k == 4


def test_load_tolerates_missing_file(tmp_path):
    restored = AppSettings(top_k=4).load(tmp_path / "nope.json")
    assert restored.top_k == 4


def test_save_writes_valid_json(tmp_path):
    target = tmp_path / "settings.json"
    AppSettings().save(target)
    assert json.loads(target.read_text(encoding="utf-8"))["model_id"] == (
        "jinaai/jina-embeddings-v5-omni-small"
    )
