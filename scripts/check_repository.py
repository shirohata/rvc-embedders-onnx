"""Check local source packaging and pinned license receipts without model files."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vc_embedders.release import ROOT, catalog, check_licenses, source_manifest


def main():
    check_licenses(catalog())
    snapshot = source_manifest(ROOT)
    print(f"License receipts and {len(snapshot['files'])} allowlisted source files checked")
    print("Model files and numerical release evidence were not checked")


if __name__ == "__main__":
    main()
