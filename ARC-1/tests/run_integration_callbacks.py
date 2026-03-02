from types import SimpleNamespace

from aimemory.hf import AIMemoryTrainerCallback
from aimemory.lightning import AIMemoryLightningCallback


class _DummyCtx:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class _DummyCtrl:
    def __init__(self):
        self.finalize_calls = 0
        self.step_calls = 0

    def step(self, profiling_warmup=False):
        self.step_calls += 1
        return _DummyCtx()

    def finalize_pcc_profile(self):
        self.finalize_calls += 1


def test_hf_callback():
    ctrl = _DummyCtrl()
    cb = AIMemoryTrainerCallback(ctrl)
    state0 = SimpleNamespace(global_step=0)
    state1 = SimpleNamespace(global_step=1)

    cb.on_step_begin(None, state0, None)
    cb.on_step_end(None, state0, None)
    cb.on_step_begin(None, state1, None)
    cb.on_step_end(None, state1, None)

    assert ctrl.step_calls == 2
    assert ctrl.finalize_calls == 1, "PCC finalize should run exactly once"


def test_lightning_callback():
    ctrl = _DummyCtrl()
    cb = AIMemoryLightningCallback(ctrl)
    trainer0 = SimpleNamespace(global_step=0)
    trainer1 = SimpleNamespace(global_step=1)

    cb.on_train_batch_start(trainer0, None, None, 0)
    cb.on_train_batch_end(trainer0, None, None, None, 0)
    cb.on_train_batch_start(trainer1, None, None, 1)
    cb.on_train_batch_end(trainer1, None, None, None, 1)

    assert ctrl.step_calls == 2
    assert ctrl.finalize_calls == 1, "PCC finalize should run exactly once"


def main():
    test_hf_callback()
    test_lightning_callback()
    print("INTEGRATION_CALLBACKS_OK")


if __name__ == "__main__":
    main()
