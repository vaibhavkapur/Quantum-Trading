"""Let the local XGBoost wheel find the OpenMP runtime bundled with PyTorch.

No system files are changed. On Linux/Windows this command is a no-op.
The standard alternative on macOS is `brew install libomp`.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != "darwin":
        return
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run inside the project's virtual environment")
    xgb = Path(importlib.util.find_spec("xgboost").origin).parent / "lib/libxgboost.dylib"
    omp = Path(importlib.util.find_spec("torch").origin).parent / "lib/libomp.dylib"
    if not xgb.resolve().is_relative_to(Path(sys.prefix).resolve()) or not omp.exists():
        raise SystemExit("Expected local XGBoost and PyTorch wheels; otherwise install libomp with Homebrew")
    rpath = "@loader_path/../../torch/lib"
    info = subprocess.check_output(["otool", "-l", str(xgb)], text=True)
    if rpath not in info:
        subprocess.run(["install_name_tool", "-add_rpath", rpath, str(xgb)], check=True)
        subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(xgb)], check=True)
    print("Local XGBoost OpenMP path configured")


if __name__ == "__main__":
    main()
