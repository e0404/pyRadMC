"""Progress reporting for long-running ``run``/``run_dij`` calls.

Dependency-free by design (AGENTS.md section 6: no dependency on pyRadPlan inside
the core) so a caller — the pyRadPlan adapter or anyone else — can observe a
transport in flight without pyRadMC taking on any consumer-side machinery.

Two ways to observe progress, usable independently or together:

* Pass a :data:`ProgressCallback` to ``run``/``run_dij``; it receives one
  :class:`ProgressEvent` per tick.
* Set ``logging.getLogger("pyRadMC.progress")`` to ``DEBUG``; every tick is also
  logged there, callback or not (AGENTS.md section 5: library output goes through
  module loggers, no handlers installed here).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["ProgressCallback", "ProgressEmitter", "ProgressEvent"]


@dataclass(frozen=True)
class ProgressEvent:
    """A snapshot of transport progress, emitted at a tick.

    Tick granularity is backend- and method-specific (a batch, or a beamlet
    group) — see the ``progress`` parameter docs on ``run``/``run_dij`` for what
    a tick represents. Treat ``histories_done / histories_total`` as the
    portable signal; do not assume a fixed tick count or uniform spacing.
    """

    histories_done: int
    histories_total: int
    elapsed_s: float
    """Wall time since the run started."""
    rate_hz: float
    """Histories per second since the *previous* tick (instantaneous, not averaged)."""


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressEmitter:
    """Thread-safe fan-in for progress ticks from one or more worker threads.

    ``tick()`` may be called freely from any thread (``run_dij`` shards beamlet
    groups across device threads). The update-and-dispatch is serialized under one
    lock, so the callback is never invoked concurrently with itself and always
    observes strictly increasing ``histories_done`` — regardless of which thread's
    tick arrives first.

    A callback that raises is logged and otherwise ignored: an observer must never
    be able to break a Monte Carlo run.
    """

    def __init__(self, callback: ProgressCallback | None, histories_total: int) -> None:
        self._callback = callback
        self._total = histories_total
        self._done = 0
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._last_t = self._start

    def tick(self, histories: int) -> None:
        """Record that ``histories`` more histories completed; dispatch one event."""
        with self._lock:
            self._done += histories
            now = time.monotonic()
            dt = now - self._last_t
            rate = histories / dt if dt > 0 else 0.0
            self._last_t = now
            event = ProgressEvent(
                histories_done=self._done,
                histories_total=self._total,
                elapsed_s=now - self._start,
                rate_hz=rate,
            )
            logger.debug(
                "progress: %d/%d histories (%.1f Hz)",
                event.histories_done,
                event.histories_total,
                event.rate_hz,
            )
            if self._callback is None:
                return
            try:
                self._callback(event)
            except Exception:  # an observer must never break a run
                logger.exception("Progress callback raised; ignoring.")
