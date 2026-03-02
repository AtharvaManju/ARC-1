import os
import json
import hashlib
from typing import Tuple

ALIGN = 4096

def round_up(x: int, a: int = ALIGN) -> int:
    return ((x + a - 1) // a) * a

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dtype_to_str(dtype) -> str:
    return str(dtype).replace("torch.", "")

def str_to_dtype(s: str):
    import torch
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported dtype: {s}")
    return mapping[s]

def shape_to_json(shape: Tuple[int, ...]) -> str:
    return json.dumps(list(shape))

def shape_from_json(s: str) -> Tuple[int, ...]:
    arr = json.loads(s)
    return tuple(int(x) for x in arr)

def sha_meta(dtype: str, shape: Tuple[int, ...], nbytes: int, padded_bytes: int, offset: int, encrypted: int, step_id: int, pool_id: int) -> str:
    h = hashlib.sha256()
    h.update(dtype.encode())
    h.update(str(shape).encode())
    h.update(str(nbytes).encode())
    h.update(str(padded_bytes).encode())
    h.update(str(offset).encode())
    h.update(str(int(encrypted)).encode())
    h.update(str(int(step_id)).encode())
    h.update(str(int(pool_id)).encode())
    return h.hexdigest()
