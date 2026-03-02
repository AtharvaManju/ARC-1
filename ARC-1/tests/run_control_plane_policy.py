import json
import tempfile

from aimemory.config import AIMemoryConfig
from aimemory.control_plane import PolicyStore


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_cp_") as td:
        store = PolicyStore(td)
        p1 = {"spill_min_bytes": 1024, "pcc_lookahead": 2}
        p2 = {"spill_min_bytes": 2048, "pcc_lookahead": 3}
        store.push("pilot", p1, reason="seed")
        store.push("pilot", p2, reason="tune")
        cur = store.pull("pilot")
        assert cur == p2
        old = store.rollback("pilot")
        assert old == p1
        cfg = AIMemoryConfig(pool_dir="/tmp/aimemory_pool")
        applied = store.apply_to_config(cfg, old)
        assert "spill_min_bytes" in applied and cfg.spill_min_bytes == 1024
        rep = {"policy": cur, "rollback": old}
        print(json.dumps(rep))
    print("CONTROL_PLANE_POLICY_OK")


if __name__ == "__main__":
    main()
