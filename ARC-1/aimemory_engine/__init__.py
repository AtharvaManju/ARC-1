"""
Stable engine import path.

Users (and aimemory core) should import:
  from aimemory_engine import SARCCore
"""

try:
    from sarc_engine import SARCCore  # type: ignore
except Exception as e:
    SARCCore = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

def available() -> bool:
    return SARCCore is not None

def import_error():
    return _IMPORT_ERROR
