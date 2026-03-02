import os
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

ENC_NONCE = 12
ENC_TAG = 16
ENC_OVERHEAD = ENC_NONCE + ENC_TAG

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
