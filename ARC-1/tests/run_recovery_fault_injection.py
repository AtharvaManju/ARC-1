import os
import sqlite3
import tempfile

from aimemory.storage import SARCStorage
from aimemory.util import sha_meta, shape_to_json


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_recovery_") as td:
        st = SARCStorage(
            pool_dir=td,
            rank=0,
            backend="NVME_FILE",
            durable=False,
            encrypt_at_rest=False,
        )
        db = os.path.join(td, "rank_0", "metadata.db")

        conn = sqlite3.connect(db, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")

        shape = (1024,)
        committed_ck = sha_meta("uint8", shape, 1024, 4096, 8192, 0, 3, 0)
        uncommitted_ck = sha_meta("uint8", shape, 1024, 4096, 12288, 0, 3, 0)

        conn.execute(
            "INSERT OR REPLACE INTO tensors(key, step_id, pool_id, offset, nbytes, padded_bytes, dtype, shape, checksum, committed, encrypted, crc32) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k_committed", 3, 0, 8192, 1024, 4096, "uint8", shape_to_json(shape), committed_ck, 1, 0, 0),
        )
        conn.execute(
            "INSERT OR REPLACE INTO tensors(key, step_id, pool_id, offset, nbytes, padded_bytes, dtype, shape, checksum, committed, encrypted, crc32) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k_uncommitted", 3, 0, 12288, 1024, 4096, "uint8", shape_to_json(shape), uncommitted_ck, 0, 0, 0),
        )
        conn.commit()
        conn.close()

        st.recover()

        conn = sqlite3.connect(db, timeout=30.0, check_same_thread=False)
        row0 = conn.execute("SELECT COUNT(*) FROM tensors WHERE key='k_uncommitted'").fetchone()
        row1 = conn.execute("SELECT v FROM meta WHERE k='next_offset_0'").fetchone()
        conn.close()
        st.close()

        assert row0 is not None and int(row0[0]) == 0, "recover() must delete uncommitted rows"
        assert row1 is not None and int(row1[0]) == 12288, "recover() must set next offset from committed max end"

    print("RECOVERY_FAULT_INJECTION_OK")


if __name__ == "__main__":
    main()
