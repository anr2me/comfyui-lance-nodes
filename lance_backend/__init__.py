# lance_backend/__init__.py
# Ensure Lance submodule is importable from within the backend package.

import sys
from pathlib import Path

_LANCE_ROOT = Path(__file__).parent.parent / "Lance"
_s = str(_LANCE_ROOT)
if _s not in sys.path:
    sys.path.insert(0, _s)
