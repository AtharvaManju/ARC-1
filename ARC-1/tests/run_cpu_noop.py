from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController

def main():
    cfg = AIMemoryConfig(backend="AUTO", pool_dir="/tmp/aimemory_pool")
    ctrl = AIMemoryController(cfg)
    with ctrl.step():
        pass
    print("CPU/NOOP OK:", ctrl.quick_summary())

if __name__ == "__main__":
    main()
