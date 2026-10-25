"""Tests for the command line interface."""

from __future__ import annotations

import json

import pytest

from bookrag.Rag_app.cli import build_parser, main


def test_parser_has_all_commands():
    parser = build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    commands = set(actions[0].choices)
    assert {
        "doctor",
        "ingest",
        "query",
        "find-image",
        "demo",
        "stats",
        "reset",
        "settings",
    } <= commands


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_doctor_runs_without_model(capsys):
    assert main(["doctor"]) in (0, 1)
    out = capsys.readouterr().out
    assert "BookRAG doctor" in out
    assert "Python packages" in out


def test_settings_command_outputs_json(capsys):
    assert main(["settings"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_id"] == "jinaai/jina-embeddings-v5-omni-small"


def test_settings_command_can_override(capsys):
    assert main(["settings", "--set", "top_k=13"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["top_k"] == 13


def test_settings_command_rejects_unknown_key(capsys):
    assert main(["settings", "--set", "nonsense=1"]) == 0
    assert "Unknown setting" in capsys.readouterr().err


def test_stats_command_reports_totals(capsys):
    assert main(["stats", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "total" in payload


def test_query_without_model_reports_error(capsys, monkeypatch):
    """A missing model must surface as a clean error, not a traceback."""
    from bookrag.errors import ModelUnavailableError

    def boom(self, *args, **kwargs):
        raise ModelUnavailableError("model missing", "download it")

    monkeypatch.setattr("bookrag.accelerator.embedder.ClipEmbedder.embed_text", boom)
    assert main(["query", "anything"]) == 1
    assert "error:" in capsys.readouterr().err


def test_reset_requires_confirmation(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    assert main(["reset"]) == 1
    assert "Aborted" in capsys.readouterr().out


@pytest.mark.parametrize("command", [["--help"], ["ingest", "--help"], ["query", "--help"]])
def test_help_exits_cleanly(command):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(command)
    assert excinfo.value.code == 0
