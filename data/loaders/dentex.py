"""DENTEX (MICCAI 2023) panoramic dental X-ray loader.

Source: https://dentex.grand-challenge.org/
Reference: Hamamci et al., 2023.

Hierarchy:
  - Quadrant detection task     (693 X-rays)
  - Tooth detection task        (634 X-rays)
  - Abnormal-tooth detection    (1,005 fully labelled)
Diagnosis classes: caries, deep caries, periapical lesions, impacted teeth.

We do NOT auto-download — the dataset requires Grand Challenge registration
and acceptance of the data-use terms. This loader documents the manual
fetch step, verifies the archive's SHA-256, and parses annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE.parent / "cache" / "dentex"

REGISTRATION_URL = "https://dentex.grand-challenge.org/data/"
EXPECTED_SHA256: str | None = None  # set after first verified download


@dataclass
class DentexSample:
    image_path: pathlib.Path
    annotations: dict  # raw COCO-style dict
    quadrants: list[dict]
    teeth: list[dict]
    diagnoses: list[dict]


def _expected_layout(root: pathlib.Path) -> list[pathlib.Path]:
    return [
        root / "training_data" / "images",
        root / "training_data" / "annotations",
    ]


def fetch(force: bool = False) -> pathlib.Path:
    """Either confirm an existing cache or print the manual-download steps.

    Returns the cache root; raises if the user has not yet downloaded.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    needed = _expected_layout(CACHE)
    if all(p.exists() for p in needed) and not force:
        print(f"DENTEX cache present at {CACHE}")
        return CACHE

    print("=" * 70)
    print("DENTEX is not auto-downloadable.")
    print(f"  1. Register and accept terms at:  {REGISTRATION_URL}")
    print(f"  2. Download the archive (~2 GB).")
    print(f"  3. Extract into:  {CACHE}")
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
    print("  (no expected SHA on file; record actual for the next user)")
    return True


def load_split(split: str = "training") -> Iterable[DentexSample]:
    """Yield DentexSample for each annotated image."""
    if not CACHE.exists():
        fetch()
    ann_dir = CACHE / f"{split}_data" / "annotations"
    img_dir = CACHE / f"{split}_data" / "images"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Missing {ann_dir} or {img_dir}")

    for ann_path in sorted(ann_dir.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        img_path = img_dir / ann.get("image_filename", ann_path.stem + ".png")
        yield DentexSample(
            image_path=img_path,
            annotations=ann,
            quadrants=ann.get("quadrants", []),
            teeth=ann.get("teeth", []),
            diagnoses=ann.get("diagnoses", []),
        )


_DX_LABEL_MAP = {
    0: "caries",
    1: "deep caries",
    2: "periapical lesion",
    3: "impacted tooth",
}


def to_chart_context(sample: DentexSample, *, on_tooth: int | None = None) -> str:
    """Render a one-paragraph prose summary suitable for prompting."""
    findings: list[str] = []
    for dx in sample.diagnoses:
        tooth = dx.get("tooth_id", "?")
        if on_tooth is not None and tooth != on_tooth:
            continue
        label = _DX_LABEL_MAP.get(dx.get("category_id"), "finding")
        findings.append(f"tooth {tooth}: {label}")
    if not findings:
        return "Panoramic radiograph: no abnormal findings on annotated teeth."
    return "Panoramic radiograph findings: " + "; ".join(findings) + "."


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify cache layout")
    p.add_argument("--archive", type=pathlib.Path,
                   help="path to a downloaded archive to SHA-verify")
    args = p.parse_args()
    if args.archive:
        verify_archive(args.archive)
        return 0
    if args.check:
        try:
            fetch()
            print("OK — DENTEX cache layout looks correct.")
            return 0
        except SystemExit as exc:
            return int(exc.code or 1)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "DentexSample",
    "fetch",
    "load_split",
    "to_chart_context",
    "verify_archive",
    "REGISTRATION_URL",
]
