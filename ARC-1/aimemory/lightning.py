class AIMemoryLightningCallback:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self._ctx = None
        self._pcc_finalized = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._ctx = self.ctrl.step(profiling_warmup=(trainer.global_step == 0))
        self._ctx.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        if not self._pcc_finalized:
            try:
                self.ctrl.finalize_pcc_profile()
                self._pcc_finalized = True
            except Exception:
                pass
