"""Tests for stage-timing and model-runtime monitoring.

The monitoring must be invisible: it never changes control flow, never raises
into the code it instruments, and never leaks its own lines into stdout, which
some commands reserve for JSON.
"""

from __future__ import annotations

import json
import logging

import pytest

from bookrag.logs.monitoring import (
    MARKER,
    Monitor,
    StageStats,
    get_monitor,
    parameter_count,
    process_peak_rss_mb,
)


@pytest.fixture(autouse=True)
def monitor() -> Monitor:
    """Hand each test a fresh collector so aggregates never bleed across."""
    shared = get_monitor()
    shared.reset()
    yield shared
    shared.reset()


def _stage_by_name(monitor: Monitor, name: str) -> StageStats:
    return {stat.stage: stat for stat in monitor.stages()}[name]


class _Capture(logging.Handler):
    """Minimal handler that keeps the records it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.records.append(record)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_repeated_calls_accumulate(monitor: Monitor) -> None:
    monitor.record("store.upsert", 0.5, items=3)
    monitor.record("store.upsert", 2.0, items=5)

    stat = _stage_by_name(monitor, "store.upsert")
    assert stat.calls == 2
    assert stat.total_s == pytest.approx(2.5)
    assert stat.min_s == pytest.approx(0.5)
    assert stat.max_s == pytest.approx(2.0)
    assert stat.last_s == pytest.approx(2.0)
    assert stat.items == 8
    assert stat.mean_s == pytest.approx(1.25)


def test_rate_is_items_over_total_time() -> None:
    stat = StageStats(stage="embed.text", calls=2, total_s=4.0, items=8)
    assert stat.rate_per_s == pytest.approx(2.0)


def test_rate_is_none_without_items() -> None:
    stat = StageStats(stage="parse.pdf", calls=1, total_s=4.0)
    assert stat.rate_per_s is None


def test_stages_are_sorted_slowest_first(monitor: Monitor) -> None:
    monitor.record("store.query", 0.1)
    monitor.record("embed.text", 5.0)
    monitor.record("parse.pdf", 0.5)

    assert [stat.stage for stat in monitor.stages()] == ["embed.text", "parse.pdf", "store.query"]


# --------------------------------------------------------------------------- #
# The container/leaf split - the whole point of the coverage figure
# --------------------------------------------------------------------------- #
def test_containers_are_reported_but_not_summed(monitor: Monitor) -> None:
    monitor.record("run.total", 10.0, container=True)
    monitor.record("ingest.total", 9.0, container=True)
    monitor.record("embed.text", 4.0, items=8)
    monitor.record("store.upsert", 1.0, items=8)

    # Only the leaves count as timed work, or the same wall-clock would be
    # billed twice and coverage would exceed the run length.
    assert monitor.work_total_s == pytest.approx(5.0)
    assert {stat.stage for stat in monitor.stages()} == {
        "run.total",
        "ingest.total",
        "embed.text",
        "store.upsert",
    }


def test_container_flag_survives_on_the_stats() -> None:
    assert StageStats(stage="run.total", total_s=9.0, container=True).work_total_s == 0.0
    assert StageStats(stage="embed.text", total_s=9.0).work_total_s == pytest.approx(9.0)


def test_coverage_is_reported_unclamped(monitor: Monitor) -> None:
    """Coverage is work/wall with no ceiling - a cap would hide overlap."""
    # Five seconds of claimed work against a few microseconds of real elapsed
    # time; a clamp at 1.0 would round this down to 100%.
    monitor.record("embed.text", 5.0)

    assert monitor.coverage > 1.0
    assert monitor.work_total_s == pytest.approx(5.0)


def test_coverage_above_one_means_stages_overlapped(monitor: Monitor) -> None:
    # Ingestion encodes batch k while batch k-1 is still being written, so the
    # timed work legitimately exceeds the wall clock. A container is shown but
    # never added to the sum.
    monitor.record("embed.text", 6.0)
    monitor.record("store.upsert", 6.0)
    monitor.record("run.total", 7.0, container=True)

    assert monitor.work_total_s == pytest.approx(12.0)
    assert monitor.coverage > 1.0
    assert {stat.stage for stat in monitor.stages()} >= {"embed.text", "run.total"}


# --------------------------------------------------------------------------- #
# The context manager
# --------------------------------------------------------------------------- #
def test_stage_times_the_block(monitor: Monitor) -> None:
    with monitor.stage("embed.text", items=6, api="encode"):
        pass

    stat = _stage_by_name(monitor, "embed.text")
    assert stat.calls == 1
    assert stat.items == 6
    assert stat.total_s >= 0.0
    assert stat.detail == {"api": "encode"}


def test_handle_is_read_back_when_the_block_ends(monitor: Monitor) -> None:
    with monitor.stage("parse.pdf") as handle:
        # Known only once parsing has finished.
        handle.items = 4
        handle.detail.update(chunks=6, images=2)

    stat = _stage_by_name(monitor, "parse.pdf")
    assert stat.items == 4
    assert stat.detail == {"chunks": 6, "images": 2}


def test_failures_inside_a_stage_are_still_recorded(monitor: Monitor) -> None:
    with pytest.raises(ValueError, match="boom"), monitor.stage("embed.text"):
        raise ValueError("boom")

    assert _stage_by_name(monitor, "embed.text").calls == 1


def test_the_stage_failure_is_not_swallowed(monitor: Monitor) -> None:
    error = RuntimeError("disk full")
    with pytest.raises(RuntimeError) as excinfo, monitor.stage("store.upsert"):
        raise error
    assert excinfo.value is error


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_report_renders_model_and_stages(monitor: Monitor) -> None:
    monitor.set_model(
        id="jinaai/jina-clip-v2",
        device="cpu",
        dtype="float32",
        api="encode",
        parameters=865_278_477,
        load_s=12.7,
        cuda_peak_mb=None,  # dropped rather than rendered as None
    )
    monitor.record("model.load", 12.7)
    monitor.record("embed.text", 1.0, items=4)
    monitor.record("run.total", 30.0, container=True)

    report = monitor.report()
    assert "checkpoint     jinaai/jina-clip-v2" in report
    assert "865.3M" in report
    assert "12.700s" in report
    assert "embed.text" in report
    assert "run.total" in report
    assert "coverage" in report


def test_report_says_so_when_nothing_was_recorded(monitor: Monitor) -> None:
    assert "No stages recorded." in monitor.report()


def test_set_model_ignores_missing_values(monitor: Monitor) -> None:
    monitor.set_model(id="jinaai/jina-clip-v2", device="cpu", cuda_peak_mb=None)
    assert monitor.model_info == {"id": "jinaai/jina-clip-v2", "device": "cpu"}


def test_model_info_is_a_copy(monitor: Monitor) -> None:
    monitor.set_model(device="cpu")
    monitor.model_info["device"] = "cuda"
    assert monitor.model_info["device"] == "cpu"


def test_summary_is_json_serialisable(monitor: Monitor) -> None:
    monitor.set_model(id="jina", device="cpu")
    monitor.record("embed.text", 0.5, items=2, api="encode")

    payload = json.loads(json.dumps(monitor.summary()))
    assert payload["stages"][0]["stage"] == "embed.text"
    assert payload["stages"][0]["rate_per_s"] == pytest.approx(4.0)
    assert payload["model"]["device"] == "cpu"
    assert isinstance(payload["coverage"], float)


def test_reset_forgets_everything(monitor: Monitor) -> None:
    monitor.record("embed.text", 1.0, items=4)
    monitor.set_model(device="cpu")

    monitor.reset()

    assert monitor.stages() == []
    assert monitor.model_info == {}


# --------------------------------------------------------------------------- #
# Log output
# --------------------------------------------------------------------------- #
def test_records_reach_the_monitor_logger_as_key_value(monitor: Monitor) -> None:
    from bookrag.logs.logging_setup import setup_logging

    setup_logging()  # raises the effective level above WARNING
    sink = _Capture()
    logger = logging.getLogger("bookrag.monitor")
    logger.addHandler(sink)
    try:
        monitor.record(
            "embed.text",
            1.5,
            items=6,
            api="encode",
            task="retrieval.query",
            note="batch of six",
        )
    finally:
        logger.removeHandler(sink)

    assert len(sink.records) == 1
    message = sink.records[0].getMessage()
    assert message.startswith(MARKER)
    assert "stage=embed.text" in message
    assert "took=1.500s" in message
    assert "items=6" in message
    assert "rate=4.00/s" in message
    # Plain values stay bare; ones that would break a naive split are quoted.
    assert "task=retrieval.query" in message
    assert 'note="batch of six"' in message


def test_monitor_logger_does_not_propagate_into_the_console() -> None:
    from bookrag.logs.logging_setup import setup_logging

    setup_logging()  # idempotent: the first call configures, later ones no-op
    logger = logging.getLogger("bookrag.monitor")
    assert logger.propagate is False
    assert logger.handlers, "a dedicated handler must keep the file alive on its own"


# --------------------------------------------------------------------------- #
# Best-effort behaviour: monitoring never breaks the app
# --------------------------------------------------------------------------- #
def test_broken_detail_values_do_not_escape(monitor: Monitor) -> None:
    class Awkward:
        def __str__(self) -> str:
            raise RuntimeError("no string for you")

    monitor.record("store.query", 0.1, source=Awkward())  # must not raise
    assert _stage_by_name(monitor, "store.query").calls == 1


def test_a_failing_logger_does_not_stop_aggregation(monitor: Monitor, monkeypatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("log file vanished")

    monkeypatch.setattr(logging.Logger, "info", boom)

    monitor.record("embed.text", 1.0, items=4)  # must not raise
    monitor.set_model(device="cpu")  # must not raise

    assert _stage_by_name(monitor, "embed.text").total_s == pytest.approx(1.0)
    assert monitor.model_info == {"device": "cpu"}


def test_memory_probes_return_a_float_or_none() -> None:
    value = process_peak_rss_mb()
    assert value is None or (isinstance(value, float) and value > 0)


def test_parameter_count_rejects_things_without_parameters() -> None:
    assert parameter_count(object()) is None
    assert parameter_count(None) is None


def test_parameter_count_totals_a_fake_module() -> None:
    class FakeParameter:
        def __init__(self, count: int) -> None:
            self._count = count

        def numel(self) -> int:
            return self._count

    class FakeModule:
        def parameters(self):
            return [FakeParameter(3), FakeParameter(5)]

    assert parameter_count(FakeModule()) == 8


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_profile_flag_prints_the_report(capsys) -> None:
    from bookrag.Rag_app.cli import main

    assert main(["--profile", "stats"]) == 0
    out = capsys.readouterr().out
    assert "Stage timings" in out
    assert "run.total" in out


def test_profile_flag_is_accepted_after_the_subcommand(capsys) -> None:
    from bookrag.Rag_app.cli import main

    assert main(["stats", "--profile"]) == 0
    assert "Stage timings" in capsys.readouterr().out


def test_profile_moves_to_stderr_so_json_stays_parseable(capsys) -> None:
    from bookrag.Rag_app.cli import main

    assert main(["stats", "--json", "--profile"]) == 0
    captured = capsys.readouterr()

    assert "Stage timings" not in captured.out
    assert json.loads(captured.out)["collection"] == "book_chunks"
    assert "Stage timings" in captured.err


def test_commands_stay_quiet_without_profile(capsys) -> None:
    from bookrag.Rag_app.cli import main

    assert main(["stats"]) == 0
    assert "Stage timings" not in capsys.readouterr().out


def test_monitoring_survives_a_failing_command(capsys) -> None:
    from bookrag.Rag_app.cli import main

    # Ingesting a file that does not exist is a domain error, not a crash,
    # and the profile still renders.
    assert main(["ingest", "nope-does-not-exist.pdf", "--profile"]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err and "nope-does-not-exist" in captured.err
    assert "Stage timings" in captured.out  # the report still renders
