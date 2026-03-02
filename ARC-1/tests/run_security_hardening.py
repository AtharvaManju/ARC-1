import os
import base64
import tempfile

from aimemory.security import resolve_key_from_uri, rotate_key, secure_wipe


def main():
    raw = bytes(range(32))
    os.environ["AIMEMORY_KEY_HEX"] = raw.hex()
    os.environ["AIMEMORY_KEY_B64"] = base64.b64encode(raw).decode()
    k1 = resolve_key_from_uri("env://AIMEMORY_KEY_HEX")
    k2 = resolve_key_from_uri("env://AIMEMORY_KEY_B64")
    assert k1 == raw and k2 == raw

    with tempfile.TemporaryDirectory(prefix="aimemory_sec_") as td:
        p = os.path.join(td, "k.bin")
        p2 = os.path.join(td, "k2.bin")
        rk = rotate_key(p2)
        assert len(rk) == 32 and os.path.exists(p2)
        data = os.path.join(td, "data.bin")
        with open(data, "wb") as f:
            f.write(os.urandom(8192))
        secure_wipe(data, passes=1, verify=True)
        assert not os.path.exists(data)
    print("SECURITY_HARDENING_OK")


if __name__ == "__main__":
    main()
