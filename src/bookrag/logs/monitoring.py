"""Stage timings and model runtime capture.

BookRAG spends nearly all of its wall-clock time in a handful of places:
loading the checkpoint, encoding text, encoding images, writing to ChromaDB
and running the nearest-neighbour search.  This module records how long each
of those stages took and how the model was actually running - device, dtype,
detected API, parameter count, peak memory - so a slow run can be diagnosed
from the log instead of guessed at.

Two things are produced:

* one machine-readable line per completed stage, written to a dedicated
  ``monitoring.log`` next to the application log::

      MONITOR stage=embed.text calls=1 total=1.234s items=6 rate=4.86/s

* a summary table printed by ``bookrag --profile``.

Stage types
-----------
A stage is either a *container* or a *leaf*:

``container=True``
    Wraps a whole operation (``run.total``, ``ingest.total``, ``query.total``).
    Shown in the report but excluded from the timed sum.

``container=False``
    Times one indivisible piece of real work.  These are summed, because a
    leaf must never contain another stage - otherwise the sum would count the
    same wall-clock twice and the coverage figure would be a lie.

Everything here is best-effort: monitoring must never change what the
application does, and failures to report are swallowed.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .logging_setup import get_logger

log = get_logger("monitor")

#: Prefix on every machine-readable monitoring line.
MARKER = "MONITOR"

#: Models are reported in millions of parameters.
_PARAM_DIVISOR = 1_000_000

#: Ordering used when rendering the model block of the report.
_MODEL_KEYS = (
    ("id", "checkpoint"),
    ("device", "device"),
    ("dtype", "dtype"),
    ("api", "api"),
    ("parameters", "parameters"),
    ("load_s", "load time"),
    ("peak_rss_mb", "peak RSS"),
    ("cuda_peak_mb", "cuda peak"),
    ("threads", "torch threads"),
    ("cpus", "cpu count"),
    ("torch", "torch"),
    ("cuda_version", "cuda"),
    ("python", "python"),
    ("platform", "platform"),
)


# --------------------------------------------------------------------------- #
# Memory probes (optional, never fatal)
# --------------------------------------------------------------------------- #
def _windows_peak_working_set_mb() -> float | None:
    """Peak working set on Windows, via ``psapi``. Returns None on failure."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):  # noqa: N801 - mirrors the Win32 name
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        # Without declared argtypes ctypes passes the handle as a 32-bit int,
        # which is mangled on 64-bit Windows and makes the call fail with no
        # error - it just returns False.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return float(counters.PeakWorkingSetSize) / (1024 * 1024)
    except Exception:  # noqa: BLE001 - monitoring must never raise
        return None


def process_peak_rss_mb() -> float | None:
    """Peak resident memory for this process, in megabytes.

    Uses ``resource`` on POSIX and ``psapi`` on Windows; returns None when
    neither is available rather than raising.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return float(peak) / (1024 * 1024) if sys.platform == "darwin" else float(peak) / 1024
    except Exception:  # noqa: BLE001 - `resource` does not exist on Windows
        pass

    if sys.platform == "win32":
        return _windows_peak_working_set_mb()
    return None


def cuda_peak_mb() -> float | None:
    """Peak CUDA allocation in megabytes, or None when not on CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.max_memory_allocated()
        return float(allocated) / (1024 * 1024) if allocated else None
    except Exception:  # noqa: BLE001
        return None


def parameter_count(model: Any) -> int | None:
    """Total parameter count of a torch module, or None if it cannot be read."""
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _fmt_value(value: Any) -> str:
    """Render a detail value so a line can be split on spaces safely."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - a hostile __str__ must not break logging
        return "<unprintable>"
    if not text:
        return '""'
    if " " in text or "=" in text or '"' in text:
        return '"' + text.replace('"', "'") + '"'
    return text


def _format_bytes_mb(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} MB"


def _format_params(value: Any) -> str:
    if not isinstance(value, int) or value <= 0:
        return "-"
    return f"{value / _PARAM_DIVISOR:.1f}M"


@dataclass
class StageStats:
    """Aggregated wall-clock time for one named stage."""

    stage: str
    calls: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0
    last_s: float = 0.0
    items: int = 0
    container: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0

    @property
    def rate_per_s(self) -> float | None:
        """Items processed per second across every call, if items were given."""
        if not self.total_s or not self.items:
            return None
        return self.items / self.total_s

    @property
    def work_total_s(self) -> float:
        """Time counted towards the coverage figure (containers are excluded)."""
        return 0.0 if self.container else self.total_s

    def as_dict(self) -> dict[str, Any]:
        rate = self.rate_per_s
        return {
            "stage": self.stage,
            "container": self.container,
            "calls": self.calls,
            "total_s": round(self.total_s, 4),
            "mean_s": round(self.mean_s, 4),
            "min_s": round(self.min_s, 4) if self.calls else 0.0,
            "max_s": round(self.max_s, 4),
            "last_s": round(self.last_s, 4),
            "items": self.items,
            "rate_per_s": round(rate, 4) if rate else None,
            "detail": dict(self.detail),
        }


class _StageHandle:
    """Yielded by :meth:`Monitor.stage` so the caller can add detail later.

    ``items`` feeds the throughput figure; ``detail`` becomes key=value pairs
    on the log line and the report.
    """

    def __init__(self, items: int, detail: dict[str, Any]) -> None:
        self.items = items
        self.detail: dict[str, Any] = dict(detail)


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
class Monitor:
    """Process-wide collector for stage timings and model runtime info."""

    def __init__(self) -> None:
        self._stages: dict[str, StageStats] = {}
        self._model: dict[str, Any] = {}
        self._origin = time.perf_counter()
        self._wall_start = time.time()
        # Ingestion records from two threads: the embedder on the calling
        # thread and ChromaDB on the writer.  `count += n` is not atomic, so
        # the mutations - not the log emission - are what need guarding.
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------- #
    def reset(self) -> None:
        """Forget everything recorded so far (used by tests)."""
        with self._lock:
            self._stages.clear()
            self._model.clear()
            self._origin = time.perf_counter()
            self._wall_start = time.time()

    @contextlib.contextmanager
    def stage(
        self, name: str, items: int = 0, *, container: bool = False, **detail: Any
    ) -> Iterator[_StageHandle]:
        """Time a block of work.

        Yields a handle whose ``items`` and ``detail`` are read back when the
        block finishes, so values only known afterwards can still be reported.
        Failures inside the block still record the time - monitoring must not
        swallow or alter the application's own control flow.
        """
        handle = _StageHandle(items, detail)
        started = time.perf_counter()
        try:
            yield handle
        finally:
            self.record(
                name,
                time.perf_counter() - started,
                handle.items,
                container=container,
                **handle.detail,
            )

    def record(
        self,
        name: str,
        seconds: float,
        items: int = 0,
        *,
        container: bool = False,
        **detail: Any,
    ) -> None:
        """Fold one measurement into the running totals and log it."""
        with self._lock:
            stat = self._stages.get(name)
            if stat is None:
                stat = self._stages[name] = StageStats(stage=name, container=container)
            stat.calls += 1
            stat.total_s += seconds
            stat.min_s = min(stat.min_s, seconds)
            stat.max_s = max(stat.max_s, seconds)
            stat.last_s = seconds
            stat.items += items
            stat.detail.update(detail)

        # Emission is best-effort and deliberately happens *outside* the lock:
        # `stage()` calls record() from a `finally` block, so a failure here
        # would replace whatever exception the instrumented code was raising,
        # and a handler must never be able to re-enter and deadlock.
        with contextlib.suppress(Exception):
            self._emit_stage(stat, seconds, detail, items)

    def _emit_stage(
        self, stat: StageStats, seconds: float, detail: dict[str, Any], items: int
    ) -> None:
        extras = "".join(f" {key}={_fmt_value(value)}" for key, value in sorted(detail.items()))
        if items:
            extras += f" items={items}"
        rate = stat.rate_per_s
        if rate:
            extras += f" rate={rate:.2f}/s"
        log.info("%s stage=%s took=%.3fs%s", MARKER, stat.stage, seconds, extras)

    # -- model runtime ---------------------------------------------------- #
    def set_model(self, **info: Any) -> None:
        """Record how the model was actually loaded."""
        with self._lock:
            self._model.update({key: value for key, value in info.items() if value is not None})
            rendered = dict(self._model)
        with contextlib.suppress(Exception):
            summary = " ".join(
                f"{key}={_fmt_value(rendered[key])}"
                for key in ("id", "device", "dtype", "api", "parameters", "load_s")
                if key in rendered
            )
            log.info("%s model %s", MARKER, summary)

    @property
    def model_info(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._model)

    # -- reporting --------------------------------------------------------- #
    def stages(self) -> list[StageStats]:
        """Every recorded stage, slowest first."""
        with self._lock:
            return sorted(self._stages.values(), key=lambda stat: stat.total_s, reverse=True)

    @property
    def wall_clock_s(self) -> float:
        """Seconds since this collector was created (or last reset)."""
        return time.perf_counter() - self._origin

    @property
    def work_total_s(self) -> float:
        """Sum of the leaf stages - the part of the run we actually timed."""
        return sum(stat.work_total_s for stat in self.stages())

    @property
    def coverage(self) -> float:
        """Timed leaf work as a fraction of the wall clock.

        At or below 1.0 when stages run one after another.  Above 1.0 is not
        an error: it means stages genuinely overlapped, which is exactly what
        the ingestion pipeline is built to do (encode batch *k* while batch
        *k-1* is still being written).  Capping it would hide the parallelism
        it exists to measure, so it is reported unclamped.
        """
        wall = self.wall_clock_s
        return self.work_total_s / wall if wall > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable view of everything recorded."""
        return {
            "wall_clock_s": round(self.wall_clock_s, 4),
            "work_total_s": round(self.work_total_s, 4),
            "coverage": round(self.coverage, 4),
            "model": self.model_info,
            "stages": [stat.as_dict() for stat in self.stages()],
        }

    def report(self) -> str:
        """Human-readable summary of the model block and every stage."""
        lines: list[str] = []
        model = self.model_info

        if model:
            lines.append("Model")
            lines.append("-" * 74)
            rendered: dict[str, Any] = dict(model)
            rendered["parameters"] = _format_params(rendered.get("parameters"))
            rendered["peak_rss_mb"] = _format_bytes_mb(rendered.get("peak_rss_mb"))
            rendered["cuda_peak_mb"] = _format_bytes_mb(rendered.get("cuda_peak_mb"))
            for key, label in _MODEL_KEYS:
                if key in rendered and rendered[key] not in (None, ""):
                    value = rendered[key]
                    text = f"{value:.3f}s" if key == "load_s" else str(value)
                    lines.append(f"  {label:<14} {text}")
            lines.append("")

        stats = self.stages()
        if not stats:
            lines.append("No stages recorded.")
            return "\n".join(lines)

        lines.append("Stage timings")
        lines.append("-" * 74)
        lines.append(f"  {'stage':<20}{'calls':>7}{'total':>11}{'mean':>11}{'max':>11}{'items':>8}")
        for stat in stats:
            count = "-" if stat.container else (str(stat.items) if stat.items else "-")
            lines.append(
                f"  {stat.stage:<20}{stat.calls:>7}"
                f"{stat.total_s:>10.3f}s{stat.mean_s:>10.3f}s{stat.max_s:>10.3f}s{count:>8}"
            )

        lines.append("-" * 74)
        lines.append(f"  wall clock     {self.wall_clock_s:>9.3f}s")
        lines.append(f"  timed work     {self.work_total_s:>9.3f}s")
        lines.append(f"  coverage       {self.coverage * 100:>8.1f}%")
        return "\n".join(lines)

    def write_report(self) -> None:
        """Write the whole report to the monitoring log as a single record.

        The monitor logger has no console handler, so this reaches only
        ``monitoring.log`` and leaves stdout free for command output.
        """
        log.info("%s summary\n%s", MARKER, self.report())


#: Process-wide collector. Tests call :func:`get_monitor` and ``reset()``.
_MONITOR = Monitor()


def get_monitor() -> Monitor:
    """Return the shared collector."""
    return _MONITOR


def stage(
    name: str, items: int = 0, *, container: bool = False, **detail: Any
) -> contextlib.AbstractContextManager[_StageHandle]:
    """Convenience wrapper around ``get_monitor().stage(...)``."""
    return _MONITOR.stage(name, items, container=container, **detail)
