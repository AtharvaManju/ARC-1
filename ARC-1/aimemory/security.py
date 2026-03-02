import os
import json
import time
import base64
import hashlib
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

ENC_NONCE = 12
ENC_TAG = 16
ENC_OVERHEAD = ENC_NONCE + ENC_TAG


def _read_key_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read().strip()


def resolve_key_from_uri(uri: str) -> bytes:
    """
    Supported:
      - env://VARNAME_HEX      (hex encoded 32-byte key)
      - env://VARNAME_B64      (base64 encoded 32-byte key)
      - file:///abs/path       (raw 32-byte key file)
      - raw:/abs/path          (raw 32-byte key file)
    """
    u = str(uri or "").strip()
    if not u:
        raise ValueError("empty key uri")

    if u.startswith("env://"):
        name = u[len("env://"):].strip()
        val = os.environ.get(name, "").strip()
        if not val:
            raise ValueError(f"env key not found: {name}")
        # Try hex first, then b64.
        try:
            k = bytes.fromhex(val)
        except Exception:
            k = base64.b64decode(val)
        if len(k) != 32:
            raise ValueError("resolved env key must be 32 bytes")
        return k

    if u.startswith("file://"):
        p = u[len("file://"):]
        k = _read_key_bytes(p)
        if len(k) != 32:
            raise ValueError("file key must be 32 bytes")
        return k

    if u.startswith("raw:"):
        p = u[len("raw:"):]
        k = _read_key_bytes(p)
        if len(k) != 32:
            raise ValueError("raw file key must be 32 bytes")
        return k

    if u.startswith("kms://"):
        # KMS/HSM abstraction layer (pluggable by URI convention):
        #   kms://env/VARNAME
        #   kms://file/abs/path
        body = u[len("kms://"):]
        if body.startswith("env/"):
            name = body[len("env/"):].strip()
            val = os.environ.get(name, "").strip()
            if not val:
                raise ValueError(f"kms env key not found: {name}")
            try:
                k = bytes.fromhex(val)
            except Exception:
                k = base64.b64decode(val)
            if len(k) != 32:
                raise ValueError("kms env key must decode to 32 bytes")
            return k
        if body.startswith("file/"):
            p = "/" + body[len("file/"):].lstrip("/")
            k = _read_key_bytes(p)
            if len(k) != 32:
                raise ValueError("kms file key must be 32 bytes")
            return k
        raise ValueError(f"unsupported kms uri format: {uri}")

    raise ValueError(f"unsupported key uri: {uri}")

def load_or_create_key(key_path: str) -> bytes:
    os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            k = f.read().strip()
        if len(k) != 32:
            raise ValueError("Encryption key must be 32 bytes")
        return k
    k = os.urandom(32)
    with open(key_path, "wb") as f:
        f.write(k)
    os.chmod(key_path, 0o600)
    return k


def load_key(key_path: str = "", key_uri: str = "") -> bytes:
    if key_uri:
        return resolve_key_from_uri(key_uri)
    if not key_path:
        raise ValueError("either key_path or key_uri is required")
    return load_or_create_key(key_path)


def rotate_key(key_path: str, new_key_path: str = "") -> bytes:
    """
    Generates a new key and writes to new_key_path (or key_path if omitted).
    NOTE: does not re-encrypt existing payloads by itself.
    """
    p = new_key_path or key_path
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    k = os.urandom(32)
    with open(p, "wb") as f:
        f.write(k)
    os.chmod(p, 0o600)
    audit_log(os.path.join(os.path.dirname(p) or ".", "aimemory.audit.jsonl"), "key_rotated", key_path=p)
    return k

def _as_contig_u8_view(t_u8: torch.Tensor, nbytes: int) -> memoryview:
    if t_u8.device.type != "cpu":
        t_u8 = t_u8.cpu()
    t_u8 = t_u8[:nbytes]
    if not t_u8.is_contiguous():
        t_u8 = t_u8.contiguous()
    arr = t_u8.numpy()
    return memoryview(arr)

def _copy_bytes_into(dst_u8: torch.Tensor, blob: bytes):
    n = len(blob)
    if dst_u8.numel() < n:
        raise ValueError("dst_u8 too small")
    if not dst_u8.is_contiguous():
        dst_u8 = dst_u8.contiguous()
    dst_np = dst_u8[:n].numpy()
    src_np = np.frombuffer(blob, dtype=np.uint8)
    np.copyto(dst_np, src_np)


def audit_log(path: str, event: str, **kwargs):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"ts": time.time(), "event": str(event)}
    row.update(kwargs)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def encrypt_into(dst_u8: torch.Tensor, plain_u8: torch.Tensor, key: bytes, nbytes: int) -> int:
    aead = ChaCha20Poly1305(key)
    nonce = os.urandom(ENC_NONCE)
    pt_mv = _as_contig_u8_view(plain_u8, nbytes)
    ct = aead.encrypt(nonce, pt_mv, None)
    blob = nonce + ct
    _copy_bytes_into(dst_u8, blob)
    return len(blob)

def decrypt_to(dst_u8: torch.Tensor, blob_u8: torch.Tensor, key: bytes, nbytes: int):
    need = ENC_NONCE + nbytes + ENC_TAG
    if blob_u8.numel() < need:
        raise ValueError("blob_u8 too small")
    aead = ChaCha20Poly1305(key)

    blob_mv = _as_contig_u8_view(blob_u8, need)
    nonce = bytes(blob_mv[:ENC_NONCE])
    ct_mv = blob_mv[ENC_NONCE:]
    pt = aead.decrypt(nonce, ct_mv, None)
    if len(pt) != nbytes:
        raise ValueError("decrypt wrong length")
    _copy_bytes_into(dst_u8, pt)


def secure_wipe(path: str, passes: int = 1, verify: bool = False):
    if not os.path.exists(path):
        return
    sz = os.path.getsize(path)
    if sz <= 0:
        try:
            os.remove(path)
        except Exception:
            pass
        return
    for _ in range(max(1, int(passes))):
        with open(path, "r+b") as f:
            f.seek(0)
            left = sz
            chunk = 1024 * 1024
            while left > 0:
                n = min(chunk, left)
                f.write(os.urandom(n))
                left -= n
            f.flush()
            os.fsync(f.fileno())
    if verify:
        with open(path, "rb") as f:
            sample = f.read(min(4096, sz))
        if not sample:
            raise RuntimeError("secure wipe verification failed: empty sample")
        if len(set(sample)) <= 1:
            raise RuntimeError("secure wipe verification failed: low entropy sample")
        digest = hashlib.sha256(sample).hexdigest()
        if digest == hashlib.sha256(b"\x00" * len(sample)).hexdigest():
            raise RuntimeError("secure wipe verification failed: zeroed sample")
    os.remove(path)


def enforce_retention(path: str, keep_days: int, wipe: bool = False, wipe_passes: int = 1) -> dict:
    now = time.time()
    keep_s = max(0, int(keep_days)) * 86400
    removed = []
    scanned = 0
    base = os.path.abspath(path or ".")
    if not os.path.exists(base):
        return {"ok": True, "scanned": 0, "removed": 0, "files": []}
    for root, _, files in os.walk(base):
        for fn in files:
            fp = os.path.join(root, fn)
            scanned += 1
            try:
                mt = os.path.getmtime(fp)
            except Exception:
                continue
            if keep_s > 0 and (now - mt) <= keep_s:
                continue
            try:
                if wipe:
                    secure_wipe(fp, passes=wipe_passes, verify=True)
                else:
                    os.remove(fp)
                removed.append(fp)
            except Exception:
                continue
    return {"ok": True, "scanned": int(scanned), "removed": int(len(removed)), "files": removed[:128]}
