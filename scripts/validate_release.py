"""Run local public tests, Markdown-target checks and optional asset-byte checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_public import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/public-validation.json")
    args = parser.parse_args()
    report = {"public_structure": verify()}
    missing = []
    for path in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            resolved = path.parent / urllib.parse.unquote(target.split("#")[0])
            if not resolved.exists():
                missing.append({"file": str(path.relative_to(ROOT)), "target": target})
    assert not missing, missing
    report["unresolved_local_markdown_targets"] = missing
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    report["tests"] = {"run": tests.testsRun, "skipped": len(tests.skipped),
                       "failures": len(tests.failures), "errors": len(tests.errors),
                       "successful": tests.wasSuccessful()}
    assert tests.wasSuccessful()
    if args.assets:
        manifest = json.loads((ROOT / "provenance/checkpoint-release-manifest.json").read_text())
        asset_checks = []
        for row in manifest["assets"]:
            archive_path = args.assets / row["asset"]
            assert archive_path.stat().st_size == row["bytes"]
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == row["sha256"]
            entries = {r["entry"]: r for r in manifest["checkpoints"] if r["asset"] == row["asset"]}
            with zipfile.ZipFile(archive_path) as archive:
                assert set(archive.namelist()) == set(entries)
                assert archive.testzip() is None
                for entry, checkpoint in entries.items():
                    data = archive.read(entry)
                    assert len(data) == checkpoint["bytes"]
                    assert hashlib.sha256(data).hexdigest() == checkpoint["sha256"]
            asset_checks.append({"asset": row["asset"], "checkpoints": len(entries),
                                 "all_entry_hashes_verified": True, "crc_pass": True})
        report["release_assets"] = asset_checks
    report["publication"] = {
        "github_content_pushed": False,
        "github_release_published": False,
        "github_ci_executed": False,
        "reason": "Owner access re-verification and disconnected Comet blocked the approved temporary-collaborator invitation. Local tests and asset checks are not remote CI verification."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
