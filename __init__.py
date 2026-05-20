"""
ComfyUI-Lance-Nodes
===================
ComfyUI custom nodes for the Lance multimodal model (bytedance/Lance).
The Lance source is included as a git submodule at Lance/.

Supported tasks
---------------
  Text → Image  |  Image editing
  Text → Video  |  Video editing
  Image VQA / captioning
  Video VQA / captioning
"""

import sys
from pathlib import Path

# Guarantee Lance submodule is importable before any node code runs.
_LANCE_ROOT = Path(__file__).parent / "Lance"
_lance_str  = str(_LANCE_ROOT)
if _lance_str not in sys.path:
    sys.path.insert(0, _lance_str)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
