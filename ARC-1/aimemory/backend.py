import os
import torch

def detect_distributed():
    rank = 0
    world = 1
    is_dist = False
    is_multinode = False
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            is_dist = True
            rank = dist.get_rank()
            world = dist.get_world_size()
            nnodes = int(os.environ.get("WORLD_SIZE", str(world))) // max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
            is_multinode = nnodes > 1
    except Exception:
        pass
    return is_dist, rank, world, is_multinode

def choose_backend(cfg_backend: str, pool_dir: str) -> str:
    if cfg_backend and cfg_backend != "AUTO":
        return cfg_backend

    if not torch.cuda.is_available():
        return "NOOP"

    try:
        os.makedirs(pool_dir, exist_ok=True)
        test = os.path.join(pool_dir, ".aimemory_write_test")
        with open(test, "wb") as f:
            f.write(b"ok")
        os.remove(test)
    except Exception:
        return "RAM"

    return "NVME_FILE"
