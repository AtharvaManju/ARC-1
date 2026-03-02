import os
import platform
import socket
from typing import Dict, Any

import torch


def detect_topology(rank: int = 0) -> Dict[str, Any]:
    host = socket.gethostname()
    info: Dict[str, Any] = {
        "hostname": host,
        "platform": platform.system(),
        "machine": platform.machine(),
        "rank": int(rank),
        "numa_node": -1,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": "",
        "local_rank_env": int(os.environ.get("LOCAL_RANK", "-1")),
        "node_rank_env": int(os.environ.get("NODE_RANK", "-1")),
    }
    if torch.cuda.is_available():
        try:
            dev = int(os.environ.get("LOCAL_RANK", "0"))
            dev = max(0, min(dev, max(0, torch.cuda.device_count() - 1)))
            info["cuda_device_name"] = str(torch.cuda.get_device_name(dev))
        except Exception:
            info["cuda_device_name"] = "unknown"

    # Best-effort Linux NUMA lookup.
    try:
        sys_nodes = "/sys/devices/system/node"
        if os.path.isdir(sys_nodes):
            nodes = [x for x in os.listdir(sys_nodes) if x.startswith("node")]
            if nodes:
                info["numa_node"] = int(nodes[0].replace("node", ""))
    except Exception:
        pass
    return info
