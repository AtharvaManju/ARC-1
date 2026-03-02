from aimemory.admission import AdmissionController, JobRequest, NodeProfile


def main():
    req = JobRequest(
        job_id="j1",
        model_fingerprint="mfp",
        requested_hbm_bytes=90_000,
        batch_size=16,
        seq_len=4096,
        world_size=1,
        policy="max_headroom",
        never_oom=True,
    )
    nodes = [
        NodeProfile(node_id="n1", hbm_bytes=80_000, nvme_write_mb_s=3000, nvme_read_mb_s=3000, numa_node=0),
        NodeProfile(node_id="n2", hbm_bytes=80_000, nvme_write_mb_s=200, nvme_read_mb_s=200, numa_node=1),
    ]
    ac = AdmissionController()
    rep = ac.admit_many(req, nodes)
    assert "best" in rep and "candidates" in rep
    best_dec = rep["best"]["decision"]["decision"]
    assert best_dec in ("admit", "reshape", "reject")
    d = ac.admit(req, nodes[0])
    assert d.effective_hbm_bytes > 0
    print("ADMISSION_CONTROL_OK")


if __name__ == "__main__":
    main()
