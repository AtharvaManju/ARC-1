import tempfile

from aimemory.control_plane import PolicyStore


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_roll_") as td:
        store = PolicyStore(td)
        p1 = {"spill_min_bytes": 1024}
        p2 = {"spill_min_bytes": 2048}
        store.push("pilot", p1, reason="seed")
        store.push("pilot", p2, reason="risky")
        fleet = {"fleet_step_p99_ms_max": 999.0, "safe_mode_ranks": 2}
        rolled = store.auto_rollback_on_slo("pilot", fleet, p99_ms_max=100.0, safe_mode_max=0)
        assert rolled == p1
        cur = store.pull("pilot")
        assert cur == p1
    print("POLICY_AUTO_ROLLBACK_OK")


if __name__ == "__main__":
    main()
