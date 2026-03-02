import queue
import threading

from aimemory.io_workers import IOWorkers, SpillHostJob, PrefetchJob


def _dummy_workers():
    w = IOWorkers.__new__(IOWorkers)
    w._q = queue.Queue(maxsize=1)
    w._stop = threading.Event()
    return w


def main():
    w = _dummy_workers()
    w._q.put(("X", object()))

    ok_spill = w.submit_spill_host(SpillHostJob("", 0, None, 0, "", (), None, threading.Event()), timeout_s=0.0)  # type: ignore[arg-type]
    ok_prefetch = w.submit_prefetch(PrefetchJob("", None))  # type: ignore[arg-type]
    assert ok_spill is False, "submit_spill_host should fail fast on full queue"
    assert ok_prefetch is False, "submit_prefetch should fail fast on full queue"

    w2 = _dummy_workers()
    ok_spill2 = w2.submit_spill_host(SpillHostJob("", 0, None, 0, "", (), None, threading.Event()), timeout_s=0.0)  # type: ignore[arg-type]
    ok_prefetch2 = w2.submit_prefetch(PrefetchJob("", None))  # type: ignore[arg-type]
    assert ok_spill2 is True
    assert ok_prefetch2 is False, "second enqueue should fail since maxsize=1"

    print("BACKPRESSURE_QUEUE_OK")


if __name__ == "__main__":
    main()
