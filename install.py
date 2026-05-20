"""
install.py
==========
Executed automatically by ComfyUI's custom-node loader when the node pack is
first installed (or when the user clicks "Install" in ComfyUI Manager).

Responsibilities
----------------
1. Ensure the Lance git submodule is initialised and up-to-date.
2. Add the submodule root to sys.path so that Lance's internal packages
   (modeling, data, common, config) are importable without any extra manual
   steps.
3. Install Python dependencies listed in Lance/requirements.txt.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

# Root of *this* custom-node package
NODE_ROOT = Path(__file__).parent.resolve()
# Submodule location (matches .gitmodules path = Lance)
LANCE_ROOT = NODE_ROOT / "Lance"


# ---------------------------------------------------------------------------
# 1. Initialise / update the submodule
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"[LanceNodes/install] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd or NODE_ROOT), capture_output=False)
    return result.returncode


def ensure_submodule():
    # If the submodule has never been initialised the Lance directory will be
    # empty (just a placeholder created by git).
    sentinel = LANCE_ROOT / "modeling" / "lance.py"
    if not sentinel.exists():
        print("[LanceNodes/install] Initialising Lance submodule …")
        # Works whether the user cloned with --recurse-submodules or not.
        rc = _run(["git", "submodule", "update", "--init", "--recursive"], cwd=NODE_ROOT)
        if rc != 0:
            print(
                "[LanceNodes/install] WARNING: git submodule update failed (exit code "
                f"{rc}).\n"
                "  If you downloaded this as a zip (not git clone), please run:\n"
                "    git submodule update --init --recursive\n"
                "  inside the comfyui-lance-nodes directory, or manually clone\n"
                "  https://github.com/bytedance/Lance into comfyui-lance-nodes/Lance/"
            )
    else:
        print("[LanceNodes/install] Lance submodule already present.")


# ---------------------------------------------------------------------------
# 2. Inject Lance root into sys.path
# ---------------------------------------------------------------------------

def ensure_sys_path():
    lance_str = str(LANCE_ROOT)
    if lance_str not in sys.path:
        sys.path.insert(0, lance_str)
        print(f"[LanceNodes/install] Added to sys.path: {lance_str}")


# ---------------------------------------------------------------------------
# 3. Install Lance's Python requirements
# ---------------------------------------------------------------------------

def install_requirements():
    req_file = LANCE_ROOT / "requirements.txt"
    if not req_file.exists():
        print("[LanceNodes/install] requirements.txt not found – skipping pip install.")
        return

    # Check if the heavy deps are already satisfied to avoid a slow re-install
    # on every ComfyUI restart.
    try:
        import safetensors  # noqa: F401
        import transformers  # noqa: F401
        print("[LanceNodes/install] Core dependencies already installed.")
        return
    except ImportError:
        pass

    print("[LanceNodes/install] Installing Lance requirements …")
    rc = _run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        cwd=NODE_ROOT,
    )
    if rc != 0:
        print(
            f"[LanceNodes/install] WARNING: pip install returned exit code {rc}. "
            "Some packages may be missing."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def install():
    ensure_submodule()
    ensure_sys_path()
    install_requirements()


if __name__ == "__main__":
    install()
else:
    # Called by ComfyUI Manager automatically
    install()
