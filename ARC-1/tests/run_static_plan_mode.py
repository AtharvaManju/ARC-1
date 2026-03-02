import tempfile

from aimemory.static_plan import StaticPlanCompiler, model_fingerprint


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_plan_") as td:
        comp = StaticPlanCompiler(td)
        mfp = model_fingerprint("m1", (3,), "float16")
        plan = comp.compile_from_restore_order(mfp, [3, 2, 1], lookahead=4)
        p = comp.save(plan)
        assert p.endswith(".json")
        loaded = comp.load(plan.plan_key)
        assert loaded is not None
        assert loaded.plan_key == plan.plan_key
        assert [e.pack_idx for e in loaded.entries] == [3, 2, 1]
        parity = comp.compile_capture_parity({"loss": 1.0, "grad": 2.0}, {"loss": 1.02, "grad": 2.08}, tol_ratio=0.05)
        assert parity["ok"] is True
        parity2 = comp.compile_capture_parity({"loss": 1.0}, {"loss": 1.2}, tol_ratio=0.05)
        assert parity2["ok"] is False
    print("STATIC_PLAN_MODE_OK")


if __name__ == "__main__":
    main()
