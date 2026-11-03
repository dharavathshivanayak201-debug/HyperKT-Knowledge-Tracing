"""Capture package versions and artifact container hashes, never credentials."""
from importlib import metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]


def main():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "torch", "pyarrow", "scikit-learn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": platform.python_version(), "platform": platform.system(),
              "machine": platform.machine(), "packages": packages,
              "scope": "CPU release audit, not original training or the original CUDA evaluation environment."}
    (ROOT / "analysis/environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
