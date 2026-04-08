"""ComfyUI auto-install script.

ComfyUI-Manager calls this automatically when installing the custom node.
It can also be run manually: python install.py

CorridorKey's upstream package currently pins its own torch/torchvision/triton
versions. Installing it with normal dependency resolution can downgrade or
replace ComfyUI's working PyTorch stack, so we install the package itself
without dependencies and rely on ComfyUI's existing torch environment.
"""

import subprocess
import sys
from pathlib import Path


CORRIDORKEY_GIT = "corridorkey @ git+https://github.com/nikopueringer/CorridorKey.git"


def _pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def install():
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        _pip_install("-r", str(req_file))

    _pip_install("--no-deps", CORRIDORKEY_GIT)


if __name__ == "__main__":
    install()
