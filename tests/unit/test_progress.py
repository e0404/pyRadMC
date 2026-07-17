"""``ProgressEmitter``: thread-safe fan-in of progress ticks to one callback.

Pinned here in isolation (no engine, no transport) because the property under
test — the callback never runs concurrently with itself and always sees strictly
increasing ``histories_done``, however many threads call ``tick()`` — is a
concurrency guarantee, not a physics one; ``tests/dij`` exercises it end to end
through the actual device-shard threads.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from pyRadMC.progress import ProgressEmitter, ProgressEvent


def test_ticks_accumulate_histories_done() -> None:
    emitter = ProgressEmitter(lambda _: None, histories_total=100)
    emitter.tick(30)
    emitter.tick(20)
    emitter.tick(50)
    assert emitter._done == 100  # inspecting internal state is the point of this test


def test_callback_receives_the_final_total() -> None:
    events: list[ProgressEvent] = []
    emitter = ProgressEmitter(events.append, histories_total=100)
    emitter.tick(40)
    emitter.tick(60)
    assert events[-1].histories_done == 100
    assert events[-1].histories_total == 100


def test_none_callback_is_a_safe_noop() -> None:
    emitter = ProgressEmitter(None, histories_total=10)
    emitter.tick(10)  # must not raise


def test_concurrent_ticks_lose_no_updates() -> None:
    """100 threads each tick once; the lock must serialize every increment."""
    n_threads = 100
    emitter = ProgressEmitter(lambda _: None, histories_total=n_threads)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(emitter.tick, [1] * n_threads))
    assert emitter._done == n_threads


def test_concurrent_ticks_deliver_strictly_increasing_histories_done() -> None:
    """The callback is only ever invoked while the emitter holds its lock.

    So however many threads race into ``tick()``, the sequence of
    ``histories_done`` values the callback observes must be strictly
    increasing — no two calls interleaved, no value skipped or repeated.
    """
    n_threads = 64
    events: list[ProgressEvent] = []
    lock = threading.Lock()

    def record(event: ProgressEvent) -> None:
        # The emitter's own lock should already make this single-threaded;
        # guard the test's bookkeeping too so a regression shows up as a
        # length/ordering failure rather than a corrupted list.
        with lock:
            events.append(event)

    emitter = ProgressEmitter(record, histories_total=n_threads)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(emitter.tick, [1] * n_threads))

    assert len(events) == n_threads
    done_values = [e.histories_done for e in events]
    assert done_values == sorted(done_values)
    assert len(set(done_values)) == n_threads
    assert done_values[-1] == n_threads


def test_callback_exception_is_swallowed_and_logged(caplog) -> None:
    def broken(_: ProgressEvent) -> None:
        raise RuntimeError("observer bug")

    emitter = ProgressEmitter(broken, histories_total=10)
    with caplog.at_level(logging.ERROR, logger="pyRadMC.progress"):
        emitter.tick(10)  # must not raise
    assert "Progress callback raised" in caplog.text


def test_elapsed_and_rate_are_nonnegative() -> None:
    events: list[ProgressEvent] = []
    emitter = ProgressEmitter(events.append, histories_total=10)
    emitter.tick(10)
    assert events[0].elapsed_s >= 0.0
    assert events[0].rate_hz >= 0.0
