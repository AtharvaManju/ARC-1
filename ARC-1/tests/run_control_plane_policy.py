import json
import os
import tempfile

from aimemory.config import AIMemoryConfig
from aimemory.control_plane import PolicyStore


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_cp_") as td:
        os.environ["AIMEMORY_TEST_KEY_HEX"] = "11" * 32
        store = PolicyStore(td)
        p1 = {"spill_min_bytes": 1024, "pcc_lookahead": 2}
        p2 = {"spill_min_bytes": 2048, "pcc_lookahead": 3}
        store.push("pilot", p1, reason="seed", stage="canary", canary_ratio=0.5, key_uri="env://AIMEMORY_TEST_KEY_HEX")
        store.push("pilot", p2, reason="tune", stage="stable", canary_ratio=1.0, key_uri="env://AIMEMORY_TEST_KEY_HEX")
        cur = store.pull("pilot")
        assert cur == p2
        cur_sig = store.pull("pilot", require_signature=True, key_uri="env://AIMEMORY_TEST_KEY_HEX")
        assert cur_sig == p2
        canary = store.select_for_rank("pilot", rank=0, require_signature=True, key_uri="env://AIMEMORY_TEST_KEY_HEX", stage="stable")
        assert canary == p2
        old = store.rollback("pilot")
        assert old == p1
        cfg = AIMemoryConfig(pool_dir="/tmp/aimemory_pool")
        applied = store.apply_to_config(cfg, old)
        assert "spill_min_bytes" in applied and cfg.spill_min_bytes == 1024
        env = store.pull_envelope("pilot")
        assert int(env.get("version", 0)) >= 1
        rep = {"policy": cur, "rollback": old}
        print(json.dumps(rep))
    print("CONTROL_PLANE_POLICY_OK")


if __name__ == "__main__":
    main()
