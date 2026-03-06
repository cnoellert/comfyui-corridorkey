"""ComfyUI auto-install script.

ComfyUI-Manager calls this automatically when installing the custom node.
It can also be run manually: python install.py
"""

import subprocess
import sys
from pathlib import Path


def install():
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        )


if __name__ == "__main__":
    install()
