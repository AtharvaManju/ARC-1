import zlib
from typing import Optional, Tuple


def _try_import_lz4():
    try:
        import lz4.frame as lz4f  # type: ignore
        return lz4f
    except Exception:
        return None


def _try_import_zstd():
    try:
        import zstandard as zstd  # type: ignore
        return zstd
    except Exception:
        return None


def available_codecs() -> list[str]:
    out = ["none", "zlib"]
    if _try_import_lz4() is not None:
        out.append("lz4")
    if _try_import_zstd() is not None:
        out.append("zstd")
    return out


def normalize_codec(codec: str) -> str:
    c = str(codec or "none").strip().lower()
    if c == "auto":
        if _try_import_zstd() is not None:
            return "zstd"
        if _try_import_lz4() is not None:
            return "lz4"
        return "zlib"
    if c not in ("none", "zlib", "lz4", "zstd"):
        return "none"
    return c


def compress_blob(
    src: bytes,
    codec: str,
    min_bytes: int,
    min_gain_ratio: float = 0.05,
) -> Tuple[bytes, str, int]:
    """
    Returns (payload, codec_used, compressed_flag).
    Uses codec only if payload is large enough and compression is beneficial.
    """
    if len(src) < int(min_bytes):
        return src, "none", 0

    c = normalize_codec(codec)
    if c == "none":
        return src, "none", 0

    try:
        if c == "zlib":
            comp = zlib.compress(src, level=1)
        elif c == "lz4":
            lz4f = _try_import_lz4()
            if lz4f is None:
                return src, "none", 0
            comp = lz4f.compress(src, compression_level=0)
        elif c == "zstd":
            zstd = _try_import_zstd()
            if zstd is None:
                return src, "none", 0
            comp = zstd.ZstdCompressor(level=1).compress(src)
        else:
            return src, "none", 0
    except Exception:
        return src, "none", 0

    src_n = max(1, len(src))
    gain = 1.0 - (len(comp) / float(src_n))
    if gain < float(min_gain_ratio):
        return src, "none", 0
    return comp, c, 1


def decompress_blob(blob: bytes, codec: str, expected_nbytes: Optional[int] = None) -> bytes:
    c = normalize_codec(codec)
    if c == "none":
        out = blob
    elif c == "zlib":
        out = zlib.decompress(blob)
    elif c == "lz4":
        lz4f = _try_import_lz4()
        if lz4f is None:
            raise RuntimeError("lz4 codec requested but lz4 is not installed")
        out = lz4f.decompress(blob)
    elif c == "zstd":
        zstd = _try_import_zstd()
        if zstd is None:
            raise RuntimeError("zstd codec requested but zstandard is not installed")
        out = zstd.ZstdDecompressor().decompress(blob)
    else:
        raise RuntimeError(f"Unsupported codec: {codec}")

    if expected_nbytes is not None and len(out) != int(expected_nbytes):
        raise RuntimeError(f"decompress length mismatch: got={len(out)} expected={int(expected_nbytes)}")
    return out
