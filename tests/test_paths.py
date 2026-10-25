"""Where logs are allowed to go.

The precedence is what keeps two promises at once: a developer running from
the repo sees every artefact in one place, and the hermetic test-suite never
leaves files behind in that checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bookrag.data import paths


def test_explicit_log_dir_override_beats_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "elsewhere"
    monkeypatch.setenv("BOOKRAG_LOG_DIR", str(target))
    monkeypatch.setenv("BOOKRAG_HOME", str(tmp_path / "home"))
    assert paths.logs_dir() == target.resolve()


def test_hermetic_mode_keeps_logs_with_the_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BOOKRAG_LOG_DIR", raising=False)
    monkeypatch.setenv("BOOKRAG_HOME", str(tmp_path))
    assert paths.logs_dir() == paths.data_dir() / "logs"


def test_logs_land_in_the_project_Log_folder_when_running_from_a_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BOOKRAG_LOG_DIR", raising=False)
    monkeypatch.delenv("BOOKRAG_HOME", raising=False)
    if paths._source_checkout_root() is None:
        # Installed as a package: there is no project folder to write into.
        assert paths.logs_dir() == paths.data_dir() / "logs"
    else:
        assert paths.logs_dir() == paths.PROJECT_ROOT / "Log"


def test_checkout_root_is_none_outside_a_source_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No pyproject.toml and no src/bookrag at that depth - a bare directory
    # must not be mistaken for a checkout.
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    assert paths._source_checkout_root() is None


def test_log_dir_relative_path_is_not_left_ambiguous(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A relative override would resolve against whatever directory the CLI
    # happened to be started in, which is not a place a log belongs.
    monkeypatch.setenv("BOOKRAG_LOG_DIR", "logs")
    monkeypatch.chdir(tmp_path)
    assert paths.logs_dir().is_absolute()
