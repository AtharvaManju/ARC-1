import os
import sqlite3
import threading
import zlib
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import numpy as np

from .util import ensure_dir, round_up, sha_meta, shape_to_json, shape_from_json
from .security import load_or_create_key, encrypt_into, decrypt_to, ENC_OVERHEAD
from .pinned import PinnedBlockPool, PinnedBlock

try:
    from aimemory_engine import SARCCore  # type: ignore
except Exception:
    SARCCore = None  # type: ignore

@dataclass(frozen=True)
class TensorMeta:
    key: str
    step_id: int
    pool_id: int
    offset: int
    nbytes: int
    padded_bytes: int
    dtype_s: str
    shape: Tuple[int, ...]
    sha256: str
    encrypted: int
    crc32: int

def crc32_u8(u8: torch.Tensor, nbytes: int) -> int:
    arr = u8[:nbytes].contiguous().numpy()
    return int(zlib.crc32(memoryview(arr)) & 0xFFFFFFFF)

class _EngineShim:
    def __init__(self, staging_mb: int, strict_direct: bool):
        self.strict_direct = bool(strict_direct)
        self._native = None
        if SARCCore is not None:
            try:
                self._native = SARCCore(staging_mb)
            except Exception:
                self._native = None

    def alloc_pinned_u8(self, bytes_: int) -> torch.Tensor:
        n = int(round_up(bytes_, 4096))
        if self._native is not None:
            try:
                return self._native.alloc_pinned_u8(n)
            except Exception:
                pass
        try:
            return torch.empty((n,), dtype=torch.uint8, pin_memory=True)
        except Exception:
            return torch.empty((n,), dtype=torch.uint8)

    def last_used_direct(self) -> bool:
        if self._native is not None:
            try:
                return bool(self._native.last_used_direct())
            except Exception:
                return False
        return False

class SARCStorage:
    def __init__(
        self,
        pool_dir: str,
        rank: int,
        staging_mb: int = 512,
        durable: bool = False,
        encrypt_at_rest: bool = False,
        key_path: str = "",
        strict_direct: bool = False,
        backend: str = "NVME_FILE",
        pinned_pool: Optional[PinnedBlockPool] = None,
        pool_window_steps: int = 50,
        ram_max_bytes: int = 64 * 1024**3,
    ):
        self.rank_dir = os.path.join(pool_dir, f"rank_{rank}")
        ensure_dir(self.rank_dir)

        self.db_path = os.path.join(self.rank_dir, "metadata.db")
        self.backend = backend  # NVME_FILE or RAM

        self.durable = bool(durable)
        self.strict_direct = bool(strict_direct)

        self.pool_window_steps = max(1, int(pool_window_steps))
        self.ram_max_bytes = int(ram_max_bytes)

        self.encrypt_at_rest = bool(encrypt_at_rest)
        self._key: Optional[bytes] = None
        if self.encrypt_at_rest:
            if not key_path:
                key_path = os.path.join(self.rank_dir, "enc.key")
            self._key = load_or_create_key(key_path)

        self.engine = _EngineShim(staging_mb=staging_mb, strict_direct=self.strict_direct)
        self.engine_lock = threading.Lock()
        self._alloc_lock = threading.Lock()
        self._local = threading.local()

        # Track created SQLite connections so we can close them on shutdown
        self._conns_lock = threading.Lock()
        self._conns: List[sqlite3.Connection] = []

        self._ram: Dict[str, tuple[PinnedBlock, TensorMeta]] = {}
        self._ram_lock = threading.Lock()
        self._ram_bytes = 0

        self.pinned_pool = pinned_pool

        self._init_db()
        self._migrate_if_needed()
        self.recover()

    def close(self):
        # Close all tracked connections
        with self._conns_lock:
            for c in self._conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()
        # Best-effort close threadlocal
        try:
            c = getattr(self._local, "conn", None)
            if c is not None:
                c.close()
                self._local.conn = None
        except Exception:
            pass

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=%s;" % ("FULL" if self.durable else "NORMAL"))
            c.execute("PRAGMA temp_store=MEMORY;")
            c.execute("PRAGMA busy_timeout=30000;")
            c.execute("PRAGMA wal_autocheckpoint=1000;")
            self._local.conn = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=%s;" % ("FULL" if self.durable else "NORMAL"))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tensors (
                key TEXT PRIMARY KEY,
                step_id INTEGER NOT NULL DEFAULT 0,
                pool_id INTEGER NOT NULL DEFAULT 0,
                offset INTEGER NOT NULL,
                nbytes INTEGER NOT NULL,
                padded_bytes INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                shape TEXT NOT NULL,
                checksum TEXT NOT NULL,
                committed INTEGER NOT NULL,
                encrypted INTEGER NOT NULL,
                crc32 INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _migrate_if_needed(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tensors)").fetchall()]
        if "step_id" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN step_id INTEGER NOT NULL DEFAULT 0")
        if "pool_id" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN pool_id INTEGER NOT NULL DEFAULT 0")
        conn.commit()
        conn.close()

    def _pool_id_for_step(self, step_id: int) -> int:
        return int(step_id) // self.pool_window_steps

    def _pool_path(self, pool_id: int) -> str:
        return os.path.join(self.rank_dir, f"pool_{pool_id}.bin")

    def _meta_key_next_offset(self, pool_id: int) -> str:
        return f"next_offset_{pool_id}"

    def _load_next_offset(self, pool_id: int) -> int:
        conn = self._conn()
        row = conn.execute("SELECT v FROM meta WHERE k=?", (self._meta_key_next_offset(pool_id),)).fetchone()
        return int(row[0]) if row else 0

    def _store_next_offset(self, pool_id: int, v: int):
        conn = self._conn()
        conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (self._meta_key_next_offset(pool_id), str(int(v))))
        conn.commit()

    def alloc(self, pool_id: int, padded_bytes: int) -> int:
        need = round_up(padded_bytes)
        with self._alloc_lock:
            off = self._load_next_offset(pool_id)
            nxt = off + need
            self._store_next_offset(pool_id, nxt)
            return off

    def recover(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=%s;" % ("FULL" if self.durable else "NORMAL"))
        conn.execute("DELETE FROM tensors WHERE committed=0")

        rows = conn.execute("SELECT pool_id, MAX(offset + padded_bytes) FROM tensors WHERE committed=1 GROUP BY pool_id").fetchall()
        for pool_id, max_end in rows:
            max_end = int(max_end) if max_end is not None else 0
            conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (f"next_offset_{int(pool_id)}", str(max_end)))

        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def mark_failed(self, key: str):
        # Remove any uncommitted row to avoid weird mid-run states
        conn = self._conn()
        conn.execute("DELETE FROM tensors WHERE key=? AND committed=0", (key,))
        conn.commit()

    # --- pooled pinned helpers ---
    def acquire_pinned(self, nbytes: int) -> PinnedBlock:
        if self.pinned_pool is None:
            t = self.engine.alloc_pinned_u8(nbytes)
            return PinnedBlock(u8=t, size=int(t.numel()))
        blk = self.pinned_pool.acquire(nbytes)
        if blk is None:
            t = self.engine.alloc_pinned_u8(nbytes)
            return PinnedBlock(u8=t, size=int(t.numel()))
        return blk

    def release_pinned(self, blk: Optional[PinnedBlock]):
        if blk is None:
            return
        if self.pinned_pool is None:
            return
        self.pinned_pool.release(blk)
    # ----------------------------

    def _python_write_tensor(self, path: str, offset: int, u8: torch.Tensor, nbytes: int):
        if not u8.is_contiguous():
            u8 = u8.contiguous()
        mv = memoryview(u8[:nbytes].numpy())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            f = open(path, "r+b")
        except FileNotFoundError:
            f = open(path, "w+b")
        with f:
            f.seek(int(offset))
            f.write(mv)

    def _python_read_into_tensor(self, path: str, offset: int, dst_u8: torch.Tensor, nbytes: int):
        if not dst_u8.is_contiguous():
            dst_u8 = dst_u8.contiguous()

        buf = bytearray(int(nbytes))
        mv = memoryview(buf)
        with open(path, "rb") as f:
            f.seek(int(offset))
            got = f.readinto(mv)
            if got != int(nbytes):
                raise RuntimeError("short read")

        # copy into dst tensor via numpy views (no intermediate torch tensor)
        dst_np = dst_u8[:nbytes].numpy()
        src_np = np.frombuffer(mv, dtype=np.uint8)
        np.copyto(dst_np, src_np)

    def put_host_bytes(self, key: str, plain_blk: PinnedBlock, nbytes: int,
                       dtype_s: str, shape: Tuple[int, ...], step_id: int,
                       spill_done_evt=None) -> TensorMeta:
        try:
            encrypted = 1 if self.encrypt_at_rest else 0
            payload_bytes = (nbytes + ENC_OVERHEAD) if encrypted else nbytes
            padded_bytes = round_up(payload_bytes)
            pool_id = self._pool_id_for_step(step_id)

            offset = self.alloc(pool_id, padded_bytes) if self.backend == "NVME_FILE" else 0

            crc = 0 if encrypted else crc32_u8(plain_blk.u8, nbytes)
            checksum = sha_meta(dtype_s, shape, nbytes, padded_bytes, offset, encrypted, step_id, pool_id)

            conn = self._conn()
            # Single transaction: insert uncommitted
            conn.execute("BEGIN;")
            conn.execute(
                "INSERT OR REPLACE INTO tensors(key, step_id, pool_id, offset, nbytes, padded_bytes, dtype, shape, checksum, committed, encrypted, crc32) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (key, int(step_id), int(pool_id), int(offset), int(nbytes), int(padded_bytes), dtype_s, shape_to_json(shape), checksum, encrypted, int(crc)),
            )
            conn.execute("COMMIT;")

            pool_path = None
            if self.backend == "RAM":
                if encrypted:
                    assert self._key is not None
                    enc_blk = self.acquire_pinned(padded_bytes)
                    blob_len = encrypt_into(enc_blk.u8, plain_blk.u8, self._key, nbytes)
                    if padded_bytes > blob_len:
                        enc_blk.u8[blob_len:padded_bytes].zero_()
                    meta = TensorMeta(key, step_id, pool_id, 0, nbytes, padded_bytes, dtype_s, shape, checksum, encrypted, crc)
                    with self._ram_lock:
                        self._ram[key] = (enc_blk, meta)
                        self._ram_bytes += int(enc_blk.size)
                else:
                    if padded_bytes > nbytes:
                        plain_blk.u8[nbytes:padded_bytes].zero_()
                    meta = TensorMeta(key, step_id, pool_id, 0, nbytes, padded_bytes, dtype_s, shape, checksum, encrypted, crc)
                    with self._ram_lock:
                        ram_blk = self.acquire_pinned(padded_bytes)
                        ram_blk.u8[:padded_bytes].copy_(plain_blk.u8[:padded_bytes], non_blocking=False)
                        self._ram[key] = (ram_blk, meta)
                        self._ram_bytes += int(ram_blk.size)
            else:
                pool_path = self._pool_path(pool_id)
                if not os.path.exists(pool_path):
                    open(pool_path, "ab").close()

                if encrypted:
                    assert self._key is not None
                    enc_blk = self.acquire_pinned(padded_bytes)
                    blob_len = encrypt_into(enc_blk.u8, plain_blk.u8, self._key, nbytes)
                    if padded_bytes > blob_len:
                        enc_blk.u8[blob_len:padded_bytes].zero_()

                    with self.engine_lock:
                        if self.engine._native is not None:
                            self.engine._native.write_host_ptr_to_file(pool_path, enc_blk.u8.data_ptr(), padded_bytes, offset, self.strict_direct)
                        else:
                            self._python_write_tensor(pool_path, offset, enc_blk.u8, padded_bytes)
                    self.release_pinned(enc_blk)
                else:
                    if padded_bytes > nbytes:
                        plain_blk.u8[nbytes:padded_bytes].zero_()
                    with self.engine_lock:
                        if self.engine._native is not None:
                            self.engine._native.write_host_ptr_to_file(pool_path, plain_blk.u8.data_ptr(), padded_bytes, offset, self.strict_direct)
                        else:
                            self._python_write_tensor(pool_path, offset, plain_blk.u8, padded_bytes)

                if self.durable:
                    try:
                        fd = os.open(pool_path, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    except Exception:
                        pass
                    try:
                        dfd = os.open(os.path.dirname(pool_path) or ".", os.O_RDONLY)
                        try:
                            os.fsync(dfd)
                        finally:
                            os.close(dfd)
                    except Exception:
                        pass

            # Commit metadata only after payload is written
            conn.execute("UPDATE tensors SET committed=1 WHERE key=?", (key,))
            conn.commit()
            if self.durable:
                try:
                    dfd = os.open(os.path.dirname(self.db_path) or ".", os.O_RDONLY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
                except Exception:
                    pass

            return TensorMeta(key, step_id, pool_id, offset, nbytes, padded_bytes, dtype_s, shape, checksum, encrypted, crc)
        finally:
            if spill_done_evt is not None:
                spill_done_evt.set()

    def get_meta(self, key: str) -> TensorMeta:
        if self.backend == "RAM":
            with self._ram_lock:
                if key in self._ram:
                    return self._ram[key][1]

        conn = self._conn()
        row = conn.execute(
            "SELECT step_id, pool_id, offset, nbytes, padded_bytes, dtype, shape, checksum, committed, encrypted, crc32 "
            "FROM tensors WHERE key=?",
            (key,),
        ).fetchone()
        if not row:
            raise KeyError(key)
        step_id, pool_id, offset, nbytes, padded_bytes, dtype_s, shape_s, checksum, committed, encrypted, crc32v = row
        if int(committed) != 1:
            raise KeyError(f"{key} not committed")

        shape = shape_from_json(shape_s)
        expected = sha_meta(dtype_s, shape, int(nbytes), int(padded_bytes), int(offset), int(encrypted), int(step_id), int(pool_id))
        if expected != checksum:
            raise RuntimeError("metadata integrity check failed")

        return TensorMeta(
            key=key, step_id=int(step_id), pool_id=int(pool_id), offset=int(offset),
            nbytes=int(nbytes), padded_bytes=int(padded_bytes),
            dtype_s=dtype_s, shape=shape, sha256=checksum, encrypted=int(encrypted), crc32=int(crc32v)
        )

    def read_block_to_pinned(self, meta: TensorMeta, dst_blk: PinnedBlock):
        if self.backend == "RAM":
            with self._ram_lock:
                blk, _ = self._ram[meta.key]
                dst_blk.u8[:meta.padded_bytes].copy_(blk.u8[:meta.padded_bytes], non_blocking=False)
            return

        pool_path = self._pool_path(meta.pool_id)
        with self.engine_lock:
            if self.engine._native is not None:
                self.engine._native.read_file_to_host_ptr(pool_path, dst_blk.u8.data_ptr(), meta.padded_bytes, meta.offset, self.strict_direct)
            else:
                self._python_read_into_tensor(pool_path, meta.offset, dst_blk.u8, meta.padded_bytes)

    def restore_to_cuda_from_pinned(self, meta: TensorMeta, enc_blk: PinnedBlock, out: torch.Tensor, scratch_plain: Optional[PinnedBlock] = None):
        if meta.encrypted:
            assert self._key is not None
            plain_blk = scratch_plain if scratch_plain is not None else self.acquire_pinned(meta.nbytes)
            decrypt_to(plain_blk.u8, enc_blk.u8, self._key, meta.nbytes)
            src_u8 = plain_blk.u8[:meta.nbytes].view(torch.uint8).reshape(-1)
        else:
            got = crc32_u8(enc_blk.u8, meta.nbytes)
            if got != meta.crc32:
                raise RuntimeError("CRC32 mismatch: data corruption detected")
            src_u8 = enc_blk.u8[:meta.nbytes].view(torch.uint8).reshape(-1)

        out_u8 = out.view(torch.uint8).reshape(-1)
        out_u8.copy_(src_u8, non_blocking=True)

        if meta.encrypted and scratch_plain is None:
            self.release_pinned(plain_blk)

    def ram_gc_keep_last_windows(self, keep_last_windows: int, current_pool_id: int):
        if self.backend != "RAM":
            return
        cutoff_pool_id = max(0, int(current_pool_id) - int(keep_last_windows) + 1)
        to_del = []
        with self._ram_lock:
            for k, (_, meta) in self._ram.items():
                if int(meta.pool_id) < cutoff_pool_id:
                    to_del.append(k)
            for k in to_del:
                blk, _ = self._ram.pop(k)
                self._ram_bytes -= int(blk.size)
                self.release_pinned(blk)

            if self._ram_bytes > self.ram_max_bytes:
                items = sorted(self._ram.items(), key=lambda kv: int(kv[1][1].pool_id))
                for k, (blk, _) in items:
                    if self._ram_bytes <= self.ram_max_bytes:
                        break
                    self._ram.pop(k, None)
                    self._ram_bytes -= int(blk.size)
                    self.release_pinned(blk)
