from .version import __version__

try:
    from aimemory_engine import SARCCore as _SARCCore  # type: ignore
except Exception:
    _SARCCore = None

from .controller import AIMemoryController
from .doctor import run_doctor
from .installer import run_installer
from .auto import enable, disable, status

# ARC-1 branded aliases (backward compatible with AIMemory names).
ARC1Controller = AIMemoryController

def engine_available() -> bool:
    return _SARCCore is not None
