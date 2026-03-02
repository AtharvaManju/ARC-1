import os
import sqlite3
import threading
import zlib
import time
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import numpy as np

from .util import ensure_dir, round_up, sha_meta, shape_to_json, shape_from_json
from .security import load_key, encrypt_into, decrypt_to, ENC_OVERHEAD, audit_log
from .compression import compress_blob, decompress_blob, normalize_codec
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
    status_s: str = "COMMITTED"
    compressed: int = 0
    stored_nbytes: int = 0
    codec: str = "none"
    restored_count: int = 0
    chunk_manifest_json: str = ""


STATUS_RESERVED = "RESERVED"
STATUS_WRITING = "WRITING"
STATUS_COMMITTED = "COMMITTED"
STATUS_EVICTABLE = "EVICTABLE"
STATUS_PREFETCHING = "PREFETCHING"
STATUS_RESIDENT = "RESIDENT"
STATUS_RELEASED = "RELEASED"
STATUS_RESTORED = "RESTORED"
STATUS_FAILED = "FAILED"
VALID_STATUSES = {
    STATUS_RESERVED,
    STATUS_WRITING,
    STATUS_COMMITTED,
    STATUS_EVICTABLE,
    STATUS_PREFETCHING,
    STATUS_RESIDENT,
    STATUS_RELEASED,
    STATUS_RESTORED,
    STATUS_FAILED,
}
READABLE_STATUSES = {
    STATUS_COMMITTED,
    STATUS_EVICTABLE,
    STATUS_PREFETCHING,
    STATUS_RESIDENT,
    STATUS_RESTORED,
}
ALLOWED_TRANSITIONS = {
    STATUS_RESERVED: {STATUS_WRITING, STATUS_FAILED},
    STATUS_WRITING: {STATUS_COMMITTED, STATUS_FAILED},
    STATUS_COMMITTED: {STATUS_EVICTABLE, STATUS_PREFETCHING, STATUS_RESIDENT, STATUS_RESTORED, STATUS_RELEASED, STATUS_FAILED},
    STATUS_EVICTABLE: {STATUS_PREFETCHING, STATUS_RESIDENT, STATUS_RESTORED, STATUS_RELEASED, STATUS_FAILED},
    STATUS_PREFETCHING: {STATUS_RESIDENT, STATUS_RESTORED, STATUS_EVICTABLE, STATUS_FAILED},
    STATUS_RESIDENT: {STATUS_RESTORED, STATUS_RELEASED, STATUS_EVICTABLE, STATUS_FAILED},
    STATUS_RESTORED: {STATUS_RESIDENT, STATUS_RELEASED, STATUS_EVICTABLE, STATUS_FAILED},
    STATUS_RELEASED: {STATUS_EVICTABLE, STATUS_PREFETCHING, STATUS_RESIDENT, STATUS_RESTORED, STATUS_FAILED},
    STATUS_FAILED: set(),
}


def _try_import_xxhash():
    try:
        import xxhash  # type: ignore
        return xxhash
    except Exception:
        return None


def build_chunk_manifest(payload: bytes, chunk_size: int = 1 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        chunk_size = 1 * 1024 * 1024
    xxh = _try_import_xxhash()
    chunks = []
    off = 0
    total = len(payload)
    while off < total:
        part = payload[off:off + chunk_size]
        row = {
            "off": off,
            "n": len(part),
            "crc32": int(zlib.crc32(part) & 0xFFFFFFFF),
        }
        if xxh is not None:
            row["xxh64"] = xxh.xxh64(part).hexdigest()
        chunks.append(row)
        off += chunk_size
    manifest = {"chunk_size": int(chunk_size), "chunks": chunks}
    return json.dumps(manifest, separators=(",", ":"))


def verify_chunk_manifest(payload: bytes, manifest_json: str) -> bool:
    if not manifest_json:
        return True
    xxh = _try_import_xxhash()
    try:
        m = json.loads(manifest_json)
        chunks = list(m.get("chunks", []))
    except Exception:
        return False
    for c in chunks:
        off = int(c["off"])
        n = int(c["n"])
        part = payload[off:off + n]
        if len(part) != n:
            return False
        if int(zlib.crc32(part) & 0xFFFFFFFF) != int(c["crc32"]):
            return False
        if "xxh64" in c and xxh is not None:
            if xxh.xxh64(part).hexdigest() != str(c["xxh64"]):
                return False
    return True


def is_valid_transition(prev_state: str, next_state: str) -> bool:
    if prev_state == next_state:
        return True
    return next_state in ALLOWED_TRANSITIONS.get(prev_state, set())

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
            if torch.cuda.is_available():
                return torch.empty((n,), dtype=torch.uint8, pin_memory=True)
            return torch.empty((n,), dtype=torch.uint8)
        except Exception:
            return torch.empty((n,), dtype=torch.uint8)

    def last_used_direct(self) -> bool:
        if self._native is not None:
            try:
                return bool(self._native.last_used_direct())
            except Exception:
                return False
        return False

    def gds_enabled(self) -> bool:
        if self._native is not None:
            try:
                return bool(self._native.gds_enabled())
            except Exception:
                return False
        return False

    def uring_enabled(self) -> bool:
        if self._native is not None:
            try:
                return bool(self._native.uring_enabled())
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
        key_uri: str = "",
        compression_codec: str = "none",
        compression_min_bytes: int = 4 * 1024 * 1024,
        compression_min_gain_ratio: float = 0.05,
        audit_log_path: str = "",
        enable_audit_log: bool = False,
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
            self._key = load_key(key_path=key_path, key_uri=key_uri)

        self.compression_codec = normalize_codec(compression_codec)
        self.compression_min_bytes = int(compression_min_bytes)
        self.compression_min_gain_ratio = float(compression_min_gain_ratio)
        self.enable_audit_log = bool(enable_audit_log)
        self.audit_log_path = audit_log_path or os.path.join(self.rank_dir, "aimemory.audit.jsonl")

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
                status TEXT NOT NULL DEFAULT 'COMMITTED',
                encrypted INTEGER NOT NULL,
                compressed INTEGER NOT NULL DEFAULT 0,
                stored_nbytes INTEGER NOT NULL DEFAULT 0,
                codec TEXT NOT NULL DEFAULT 'none',
                chunk_manifest TEXT NOT NULL DEFAULT '',
                restored_count INTEGER NOT NULL DEFAULT 0,
                crc32 INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                state TEXT NOT NULL,
                ts REAL NOT NULL,
                details TEXT NOT NULL DEFAULT ''
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
        if "status" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN status TEXT NOT NULL DEFAULT 'COMMITTED'")
        if "compressed" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN compressed INTEGER NOT NULL DEFAULT 0")
        if "stored_nbytes" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN stored_nbytes INTEGER NOT NULL DEFAULT 0")
            conn.execute("UPDATE tensors SET stored_nbytes=nbytes WHERE stored_nbytes=0")
        if "codec" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN codec TEXT NOT NULL DEFAULT 'none'")
        if "chunk_manifest" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN chunk_manifest TEXT NOT NULL DEFAULT ''")
        if "restored_count" not in cols:
            conn.execute("ALTER TABLE tensors ADD COLUMN restored_count INTEGER NOT NULL DEFAULT 0")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                state TEXT NOT NULL,
                ts REAL NOT NULL,
                details TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("PRAGMA user_version=3;")
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
        conn.execute(
            "DELETE FROM tensors WHERE committed=0 OR status IN (?, ?, ?, ?)",
            (STATUS_RESERVED, STATUS_WRITING, STATUS_FAILED, STATUS_PREFETCHING),
        )
        conn.execute(
            "UPDATE tensors SET status=? WHERE committed=1 AND status NOT IN (?, ?, ?, ?)",
            (STATUS_EVICTABLE, STATUS_EVICTABLE, STATUS_RESIDENT, STATUS_RESTORED, STATUS_RELEASED),
        )

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
        # Best-effort drift report; recover should never fail because of manifest parse.
        try:
            self.replay_manifest(repair=True)
        except Exception:
            pass

    def mark_failed(self, key: str):
        conn = self._conn()
        conn.execute("UPDATE tensors SET status=? WHERE key=?", (STATUS_FAILED, key))
        conn.execute("DELETE FROM tensors WHERE key=? AND committed=0", (key,))
        conn.commit()
        self._append_manifest_event(key, STATUS_FAILED, "")

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

    def _u8_to_bytes(self, u8: torch.Tensor, nbytes: int) -> bytes:
        arr = u8[:nbytes]
        if not arr.is_contiguous():
            arr = arr.contiguous()
        return bytes(memoryview(arr.numpy()))

    def _bytes_to_block(self, data: bytes) -> PinnedBlock:
        blk = self.acquire_pinned(len(data))
        if data:
            np.copyto(blk.u8[:len(data)].numpy(), np.frombuffer(data, dtype=np.uint8))
        return blk

    def _set_status(self, key: str, status_s: str):
        if status_s not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status_s}")
        conn = self._conn()
        row = conn.execute("SELECT status, committed FROM tensors WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        prev_status, committed = str(row[0]), int(row[1])
        if not is_valid_transition(prev_status, status_s):
            raise RuntimeError(f"invalid state transition key={key}: {prev_status}->{status_s}")
        if committed != 1 and status_s in READABLE_STATUSES:
            raise RuntimeError(f"illegal readable status before commit for key={key}: {status_s}")
        conn.execute("UPDATE tensors SET status=? WHERE key=?", (status_s, key))
        conn.commit()
        self._append_manifest_event(key, status_s, "")

    def _append_manifest_event(self, key: str, state: str, details: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO manifest_events(key, state, ts, details) VALUES(?,?,?,?)",
            (str(key), str(state), float(time.time()), str(details or "")),
        )
        conn.commit()

    def mark_restored(self, key: str):
        conn = self._conn()
        row = conn.execute("SELECT status, committed FROM tensors WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        prev_status, committed = str(row[0]), int(row[1])
        if committed != 1:
            raise RuntimeError(f"restore attempted before commit for key={key}")
        if not is_valid_transition(prev_status, STATUS_RESTORED):
            raise RuntimeError(f"invalid state transition key={key}: {prev_status}->{STATUS_RESTORED}")
        conn.execute("UPDATE tensors SET status=?, restored_count=restored_count+1 WHERE key=?", (STATUS_RESTORED, key))
        conn.commit()
        self._append_manifest_event(key, STATUS_RESTORED, "")

    def mark_prefetching(self, key: str):
        self._set_status(key, STATUS_PREFETCHING)

    def mark_resident(self, key: str):
        self._set_status(key, STATUS_RESIDENT)

    def mark_released(self, key: str):
        self._set_status(key, STATUS_RELEASED)

    def mark_evictable(self, key: str):
        self._set_status(key, STATUS_EVICTABLE)

    def replay_manifest(self, repair: bool = False) -> dict:
        conn = self._conn()
        rows = conn.execute("SELECT key, state, seq FROM manifest_events ORDER BY seq ASC").fetchall()
        last: Dict[str, str] = {}
        prev: Dict[str, str] = {}
        mismatches = []
        for k, st, seq in rows:
            key = str(k)
            state = str(st)
            if state not in VALID_STATUSES:
                mismatches.append({"key": key, "seq": int(seq), "issue": f"invalid_manifest_state:{state}"})
                continue
            p = prev.get(key)
            if p is not None and (not is_valid_transition(p, state)):
                mismatches.append({"key": key, "seq": int(seq), "issue": "illegal_manifest_transition", "from": p, "to": state})
                if repair:
                    conn.execute("UPDATE tensors SET status=? WHERE key=?", (STATUS_FAILED, key))
            prev[key] = state
            last[key] = state
        for key, st in last.items():
            row = conn.execute("SELECT committed,status FROM tensors WHERE key=?", (key,)).fetchone()
            if row is None:
                continue
            committed, db_state = int(row[0]), str(row[1])
            if committed != 1 and st in READABLE_STATUSES:
                mismatches.append({"key": key, "manifest": st, "db": db_state, "issue": "manifest_readable_uncommitted"})
                if repair:
                    conn.execute("UPDATE tensors SET status=? WHERE key=?", (STATUS_FAILED, key))
            elif db_state != st and st in VALID_STATUSES:
                mismatches.append({"key": key, "manifest": st, "db": db_state, "issue": "state_drift"})
                if repair:
                    conn.execute("UPDATE tensors SET status=? WHERE key=?", (st, key))
        if repair:
            conn.commit()
        return {"ok": len(mismatches) == 0, "count": len(mismatches), "mismatches": mismatches}

    def wait_until_readable(self, key: str, timeout_s: float = 2.0, poll_s: float = 0.002) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        conn = self._conn()
        while True:
            row = conn.execute("SELECT committed,status FROM tensors WHERE key=?", (key,)).fetchone()
            if row is not None:
                committed, status = int(row[0]), str(row[1])
                if committed == 1 and status in READABLE_STATUSES:
                    return True
                if status == STATUS_FAILED:
                    return False
            if timeout_s <= 0:
                return False
            if time.time() >= deadline:
                return False
            time.sleep(max(0.0005, float(poll_s)))

    def consistency_report(self, repair: bool = False) -> dict:
        conn = self._conn()
        rows = conn.execute(
            "SELECT key, pool_id, offset, padded_bytes, committed, status FROM tensors"
        ).fetchall()
        issues: list[dict] = []
        for key, pool_id, offset, padded_bytes, committed, status in rows:
            if status not in VALID_STATUSES:
                issues.append({"key": key, "issue": f"invalid_status:{status}"})
                if repair:
                    conn.execute("UPDATE tensors SET status=? WHERE key=?", (STATUS_FAILED, key))
                continue
            if int(committed) != 1 and status in READABLE_STATUSES:
                issues.append({"key": key, "issue": f"status_without_commit:{status}"})
                if repair:
                    conn.execute("UPDATE tensors SET status=? WHERE key=?", (STATUS_FAILED, key))
            if self.backend == "NVME_FILE":
                p = self._pool_path(int(pool_id))
                end = int(offset) + int(padded_bytes)
                if (not os.path.exists(p)) or (os.path.getsize(p) < end):
                    issues.append({"key": key, "issue": "pool_payload_missing_or_short"})
                    if repair:
                        conn.execute("UPDATE tensors SET status=? WHERE key=?", (STATUS_FAILED, key))
        if repair:
            conn.commit()
        replay = self.replay_manifest(repair=repair)
        if not replay.get("ok", True):
            issues.append({"issue": "manifest_replay_mismatch", "count": int(replay.get("count", 0))})
        return {"ok": len(issues) == 0, "issues": issues, "count": len(issues), "manifest": replay}

    def put_host_bytes(self, key: str, plain_blk: PinnedBlock, nbytes: int,
                       dtype_s: str, shape: Tuple[int, ...], step_id: int,
                       spill_done_evt=None) -> TensorMeta:
        try:
            encrypted = 1 if self.encrypt_at_rest else 0
            pool_id = self._pool_id_for_step(step_id)

            plain_crc = int(zlib.crc32(memoryview(plain_blk.u8[:nbytes].contiguous().numpy())) & 0xFFFFFFFF)
            plain_bytes = self._u8_to_bytes(plain_blk.u8, nbytes)
            comp_payload, codec_used, compressed = compress_blob(
                plain_bytes,
                codec=self.compression_codec,
                min_bytes=self.compression_min_bytes,
                min_gain_ratio=self.compression_min_gain_ratio,
            )
            stored_nbytes = len(comp_payload)
            chunk_manifest_json = build_chunk_manifest(comp_payload)

            payload_bytes = (stored_nbytes + ENC_OVERHEAD) if encrypted else stored_nbytes
            padded_bytes = round_up(payload_bytes)
            offset = self.alloc(pool_id, padded_bytes) if self.backend == "NVME_FILE" else 0
            checksum = sha_meta(dtype_s, shape, nbytes, padded_bytes, offset, encrypted, step_id, pool_id)

            conn = self._conn()
            conn.execute("BEGIN;")
            conn.execute(
                "INSERT OR REPLACE INTO tensors(key, step_id, pool_id, offset, nbytes, padded_bytes, dtype, shape, checksum, committed, status, encrypted, compressed, stored_nbytes, codec, chunk_manifest, restored_count, crc32) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    key, int(step_id), int(pool_id), int(offset), int(nbytes), int(padded_bytes),
                    dtype_s, shape_to_json(shape), checksum, STATUS_RESERVED, encrypted,
                    int(compressed), int(stored_nbytes), codec_used, chunk_manifest_json, int(plain_crc),
                ),
            )
            conn.execute("COMMIT;")
            self._set_status(key, STATUS_WRITING)

            pool_path = None
            if self.backend == "RAM":
                src_blk = self._bytes_to_block(comp_payload) if compressed else plain_blk
                if encrypted:
                    assert self._key is not None
                    enc_blk = self.acquire_pinned(padded_bytes)
                    blob_len = encrypt_into(enc_blk.u8, src_blk.u8, self._key, stored_nbytes)
                    if padded_bytes > blob_len:
                        enc_blk.u8[blob_len:padded_bytes].zero_()
                    meta = TensorMeta(
                        key=key, step_id=step_id, pool_id=pool_id, offset=0, nbytes=nbytes,
                        padded_bytes=padded_bytes, dtype_s=dtype_s, shape=shape, sha256=checksum,
                        encrypted=encrypted, crc32=plain_crc, status_s=STATUS_EVICTABLE,
                        compressed=int(compressed), stored_nbytes=int(stored_nbytes), codec=codec_used, restored_count=0,
                        chunk_manifest_json=chunk_manifest_json,
                    )
                    with self._ram_lock:
                        self._ram[key] = (enc_blk, meta)
                        self._ram_bytes += int(enc_blk.size)
                    if compressed:
                        self.release_pinned(src_blk)
                else:
                    ram_blk = self.acquire_pinned(padded_bytes)
                    ram_blk.u8[:stored_nbytes].copy_(src_blk.u8[:stored_nbytes], non_blocking=False)
                    if padded_bytes > stored_nbytes:
                        ram_blk.u8[stored_nbytes:padded_bytes].zero_()
                    meta = TensorMeta(
                        key=key, step_id=step_id, pool_id=pool_id, offset=0, nbytes=nbytes,
                        padded_bytes=padded_bytes, dtype_s=dtype_s, shape=shape, sha256=checksum,
                        encrypted=encrypted, crc32=plain_crc, status_s=STATUS_EVICTABLE,
                        compressed=int(compressed), stored_nbytes=int(stored_nbytes), codec=codec_used, restored_count=0,
                        chunk_manifest_json=chunk_manifest_json,
                    )
                    with self._ram_lock:
                        self._ram[key] = (ram_blk, meta)
                        self._ram_bytes += int(ram_blk.size)
                    if compressed:
                        self.release_pinned(src_blk)
            else:
                pool_path = self._pool_path(pool_id)
                if not os.path.exists(pool_path):
                    open(pool_path, "ab").close()

                src_blk = self._bytes_to_block(comp_payload) if compressed else plain_blk
                try:
                    if encrypted:
                        assert self._key is not None
                        enc_blk = self.acquire_pinned(padded_bytes)
                        blob_len = encrypt_into(enc_blk.u8, src_blk.u8, self._key, stored_nbytes)
                        if padded_bytes > blob_len:
                            enc_blk.u8[blob_len:padded_bytes].zero_()
                        with self.engine_lock:
                            if self.engine._native is not None:
                                self.engine._native.write_host_ptr_to_file(pool_path, enc_blk.u8.data_ptr(), padded_bytes, offset, self.strict_direct)
                            else:
                                self._python_write_tensor(pool_path, offset, enc_blk.u8, padded_bytes)
                        self.release_pinned(enc_blk)
                    else:
                        if padded_bytes > stored_nbytes:
                            src_blk.u8[stored_nbytes:padded_bytes].zero_()
                        with self.engine_lock:
                            if self.engine._native is not None:
                                self.engine._native.write_host_ptr_to_file(pool_path, src_blk.u8.data_ptr(), padded_bytes, offset, self.strict_direct)
                            else:
                                self._python_write_tensor(pool_path, offset, src_blk.u8, padded_bytes)
                finally:
                    if compressed:
                        self.release_pinned(src_blk)

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
            conn.execute("UPDATE tensors SET committed=1, status=? WHERE key=?", (STATUS_COMMITTED, key))
            conn.commit()
            self._append_manifest_event(key, STATUS_COMMITTED, f"stored_nbytes={stored_nbytes}")
            self._set_status(key, STATUS_EVICTABLE)
            if self.durable:
                try:
                    dfd = os.open(os.path.dirname(self.db_path) or ".", os.O_RDONLY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
                except Exception:
                    pass

            if self.enable_audit_log:
                audit_log(
                    self.audit_log_path,
                    "spill_commit",
                    key=key,
                    step_id=int(step_id),
                    nbytes=int(nbytes),
                    stored_nbytes=int(stored_nbytes),
                    codec=str(codec_used),
                    encrypted=bool(encrypted),
                    backend=self.backend,
                    chunk_manifest=chunk_manifest_json,
                )

            return TensorMeta(
                key=key, step_id=step_id, pool_id=pool_id, offset=offset, nbytes=nbytes,
                padded_bytes=padded_bytes, dtype_s=dtype_s, shape=shape, sha256=checksum,
                encrypted=encrypted, crc32=plain_crc, status_s=STATUS_EVICTABLE,
                compressed=int(compressed), stored_nbytes=int(stored_nbytes), codec=codec_used, restored_count=0,
                chunk_manifest_json=chunk_manifest_json,
            )
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
            "SELECT step_id, pool_id, offset, nbytes, padded_bytes, dtype, shape, checksum, committed, status, encrypted, compressed, stored_nbytes, codec, chunk_manifest, restored_count, crc32 "
            "FROM tensors WHERE key=?",
            (key,),
        ).fetchone()
        if not row:
            raise KeyError(key)
        step_id, pool_id, offset, nbytes, padded_bytes, dtype_s, shape_s, checksum, committed, status_s, encrypted, compressed, stored_nbytes, codec, chunk_manifest_json, restored_count, crc32v = row
        if int(committed) != 1:
            raise KeyError(f"{key} not committed")
        if str(status_s) not in READABLE_STATUSES:
            raise KeyError(f"{key} status={status_s}")

        shape = shape_from_json(shape_s)
        expected = sha_meta(dtype_s, shape, int(nbytes), int(padded_bytes), int(offset), int(encrypted), int(step_id), int(pool_id))
        if expected != checksum:
            raise RuntimeError("metadata integrity check failed")

        return TensorMeta(
            key=key, step_id=int(step_id), pool_id=int(pool_id), offset=int(offset),
            nbytes=int(nbytes), padded_bytes=int(padded_bytes),
            dtype_s=dtype_s, shape=shape, sha256=checksum, encrypted=int(encrypted), crc32=int(crc32v),
            status_s=str(status_s), compressed=int(compressed), stored_nbytes=int(stored_nbytes), codec=str(codec), restored_count=int(restored_count),
            chunk_manifest_json=str(chunk_manifest_json or ""),
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

    def decode_to_plain_block(
        self,
        meta: TensorMeta,
        payload_blk: PinnedBlock,
        scratch_plain: Optional[PinnedBlock] = None,
        scratch_work: Optional[PinnedBlock] = None,
    ) -> Tuple[PinnedBlock, Optional[PinnedBlock], bool, bool]:
        """
        Returns: (plain_blk, owns_plain, owns_work)
        """
        owns_plain = False
        owns_work = False

        if int(meta.encrypted):
            assert self._key is not None
            if scratch_work is None:
                scratch_work = self.acquire_pinned(meta.stored_nbytes or meta.nbytes)
                owns_work = True
            decrypt_to(scratch_work.u8, payload_blk.u8, self._key, meta.stored_nbytes or meta.nbytes)
            src_u8 = scratch_work.u8[:(meta.stored_nbytes or meta.nbytes)]
        else:
            src_u8 = payload_blk.u8[:(meta.stored_nbytes or meta.nbytes)]

        stored_blob = self._u8_to_bytes(src_u8, meta.stored_nbytes or meta.nbytes)
        if not verify_chunk_manifest(stored_blob, meta.chunk_manifest_json):
            raise RuntimeError("chunk manifest verification failed")

        if int(meta.compressed):
            if scratch_plain is None:
                scratch_plain = self.acquire_pinned(meta.nbytes)
                owns_plain = True
            plain = decompress_blob(stored_blob, codec=meta.codec, expected_nbytes=meta.nbytes)
            np.copyto(scratch_plain.u8[:meta.nbytes].numpy(), np.frombuffer(plain, dtype=np.uint8))
            plain_blk = scratch_plain
        else:
            if scratch_plain is None:
                scratch_plain = self.acquire_pinned(meta.nbytes)
                owns_plain = True
            scratch_plain.u8[:meta.nbytes].copy_(src_u8[:meta.nbytes], non_blocking=False)
            plain_blk = scratch_plain

        got = crc32_u8(plain_blk.u8, meta.nbytes)
        if got != meta.crc32:
            raise RuntimeError("CRC32 mismatch: data corruption detected")
        return plain_blk, scratch_work, owns_plain, owns_work

    def restore_to_cuda_from_pinned(self, meta: TensorMeta, enc_blk: PinnedBlock, out: torch.Tensor, scratch_plain: Optional[PinnedBlock] = None):
        plain_blk, work_blk, owns_plain, owns_work = self.decode_to_plain_block(meta, enc_blk, scratch_plain=scratch_plain, scratch_work=None)
        src_u8 = plain_blk.u8[:meta.nbytes].view(torch.uint8).reshape(-1)
        out_u8 = out.view(torch.uint8).reshape(-1)
        out_u8.copy_(src_u8, non_blocking=True)
        self.mark_restored(meta.key)
        if owns_plain:
            self.release_pinned(plain_blk)
        if owns_work and work_blk is not None:
            self.release_pinned(work_blk)

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
