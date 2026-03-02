from contextlib import contextmanager

@contextmanager
def wrap_accelerate_backward(ctrl, profiling_warmup: bool = False):
    ctx = ctrl.step(profiling_warmup=profiling_warmup)
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)
