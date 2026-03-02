class AIMemoryTrainerCallback:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self._pcc_finalized = False

    def on_step_begin(self, args, state, control, **kwargs):
        self._ctx = self.ctrl.step(profiling_warmup=(state.global_step == 0))
        self._ctx.__enter__()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, "_ctx") and self._ctx is not None:
            self._ctx.__exit__(None, None, None)
        if not self._pcc_finalized:
            try:
                self.ctrl.finalize_pcc_profile()
                self._pcc_finalized = True
            except Exception:
                pass
        return control
