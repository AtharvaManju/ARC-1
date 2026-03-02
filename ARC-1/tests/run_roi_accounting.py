import tempfile

from aimemory.roi import ROITracker, WorkloadIdentity


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_roi_") as td:
        tr = ROITracker(td)
        wid = WorkloadIdentity(model="m", batch_size=8, seq_len=2048, precision="fp16", world_size=1, job="j")
        tr.capture_baseline(
            wid,
            {
                "memory_headroom_pct": 5.0,
                "oom_events": 3,
                "reruns": 2,
                "throughput": 90.0,
                "step_samples_ms": [12.0, 13.0, 12.5],
            },
        )
        rep = tr.attribution(
            wid,
            {
                "memory_headroom_pct": 25.0,
                "oom_events": 0,
                "reruns": 0,
                "throughput": 120.0,
                "step_samples_ms": [10.0, 10.5, 9.8],
            },
        )
        assert rep["ooms_prevented"] == 3
        assert rep["headroom_gain_pct"] > 0
        assert rep["anti_gaming"]["ok"] is True
    print("ROI_ACCOUNTING_OK")


if __name__ == "__main__":
    main()
