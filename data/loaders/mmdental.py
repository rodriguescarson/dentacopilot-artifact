"""MMDental (Sci Data 2025) — multimodal CBCT + medical-record loader.

Source: Nature Scientific Data 2025 — A multimodal dataset of tooth CBCT
images with expert medical records.

660 patients with paired 3D CBCT and structured chart records. Licensed
CC-BY but still ~5 GB; we require manual download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE.parent / "cache" / "mmdental"

SOURCE_URL = "https://www.nature.com/articles/s41597-025-05398-7"
EXPECTED_SHA256: str | None = None


@dataclass
class MMDentalSample:
    cbct_path: pathlib.Path
    record_path: pathlib.Path
    record: dict


def _expected_layout(root: pathlib.Path) -> list[pathlib.Path]:
    return [root / "cbct", root / "records"]


def fetch(force: bool = False) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    needed = _expected_layout(CACHE)
    if all(p.exists() for p in needed) and not force:
        print(f"MMDental cache present at {CACHE}")
        return CACHE

    print("=" * 70)
    print("MMDental — manual download required (CC-BY, ~5 GB).")
    print(f"  1. Open:  {SOURCE_URL}")
    print(f"  2. Follow the data availability section to fetch the archive.")
    print(f"  3. Extract under:  {CACHE}")
    print("     Expected layout:")
    for p in needed:
        print(f"       {p}")
    print("  4. Re-run this loader to verify.")
    print("=" * 70)
    raise SystemExit(1)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(archive_path: pathlib.Path, expected: str | None = None) -> bool:
    expected = expected or EXPECTED_SHA256
    actual = _sha256(archive_path)
    print(f"  SHA-256 actual:   {actual}")
    if expected:
        print(f"  SHA-256 expected: {expected}")
        return actual == expected
    print("  (no expected SHA on file; record actual for next users)")
    return True


def load_split(split: str = "all") -> Iterable[MMDentalSample]:
    if not CACHE.exists():
        fetch()
    rec_dir = CACHE / "records"
    img_dir = CACHE / "cbct"
    for rec_path in sorted(rec_dir.glob("*.json")):
        record = json.loads(rec_path.read_text())
        cbct_name = record.get("cbct_file", rec_path.stem + ".nii.gz")
        yield MMDentalSample(
            cbct_path=img_dir / cbct_name,
            record_path=rec_path,
            record=record,
        )


def to_chart_context(sample: MMDentalSample, *, on_tooth: int | None = None) -> str:
    findings = sample.record.get("imaging_findings", [])
    if on_tooth is not None:
        findings = [f for f in findings if f.get("tooth_id") == on_tooth]
    if not findings:
        return "CBCT: no findings on annotated teeth."
    parts = [f"tooth {f.get('tooth_id', '?')}: {f.get('summary', 'finding')}"
             for f in findings]
    return "CBCT findings: " + "; ".join(parts) + "."


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--archive", type=pathlib.Path)
    args = p.parse_args()
    if args.archive:
        verify_archive(args.archive)
        return 0
    if args.check:
        try:
            fetch()
            print("OK — MMDental cache layout looks correct.")
            return 0
        except SystemExit as exc:
            return int(exc.code or 1)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = ["MMDentalSample", "fetch", "load_split", "to_chart_context", "SOURCE_URL"]
