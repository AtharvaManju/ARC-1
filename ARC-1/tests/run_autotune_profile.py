import os
import tempfile

from aimemory.config import AIMemoryConfig
from aimemory.metrics import Metrics
from aimemory.autotune import RuntimeAutoTuner


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_tune_") as td:
        cfg = AIMemoryConfig(
            pool_dir=td,
            autotune_enabled=True,
            autotune_profile_dir=os.path.join(td, "profiles"),
            autotune_adjust_interval_steps=1,
            spill_min_bytes=8 * 1024 * 1024,
            pcc_lookahead=4,
        )
        m = Metrics()
        m.backend = "NVME_FILE"
        tuner = RuntimeAutoTuner(cfg, m)
        m.prefetch_hits = 90
        m.prefetch_misses = 10
        m.io_queue_depth_ema.value = 0.5
        m.io_queue_depth_ema.inited = True
        tuner.observe_step(1)
        assert m.autotune_updates >= 1
        assert cfg.spill_min_bytes <= 8 * 1024 * 1024
        assert os.path.exists(tuner.state.profile_path)
    print("AUTOTUNE_PROFILE_OK")


if __name__ == "__main__":
    main()
