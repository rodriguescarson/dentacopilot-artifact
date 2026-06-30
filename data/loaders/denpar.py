"""DenPAR (Sci Data 2025) periapical X-ray loader.

Source: Nature Scientific Data 2025 — DenPAR: Annotated intra-oral periapical
radiographs dataset for machine learning.

This dataset is openly licensed; we still avoid auto-download to keep the
repo bytecount low. The loader prints the source URL and verifies the
archive after the user fetches it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE.parent / "cache" / "denpar"

SOURCE_URL = "https://www.nature.com/articles/s41597-025-05906-9"
EXPECTED_SHA256: str | None = None


@dataclass
class DenParSample:
    image_path: pathlib.Path
    tooth_id: int | None
    annotations: dict


def _expected_layout(root: pathlib.Path) -> list[pathlib.Path]:
    return [root / "images", root / "annotations"]


def fetch(force: bool = False) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    needed = _expected_layout(CACHE)
    if all(p.exists() for p in needed) and not force:
        print(f"DenPAR cache present at {CACHE}")
        return CACHE

    print("=" * 70)
    print("DenPAR — manual download required (one-time).")
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


def load_split(split: str = "all") -> Iterable[DenParSample]:
    if not CACHE.exists():
        fetch()
    ann_dir = CACHE / "annotations"
    img_dir = CACHE / "images"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Missing {ann_dir} or {img_dir}")

    for ann_path in sorted(ann_dir.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        img_name = ann.get("image_filename", ann_path.stem + ".png")
        yield DenParSample(
            image_path=img_dir / img_name,
            tooth_id=ann.get("tooth_id"),
            annotations=ann,
        )


def to_chart_context(sample: DenParSample, *, on_tooth: int | None = None) -> str:
    if on_tooth is not None and sample.tooth_id != on_tooth:
        return ""
    finding = sample.annotations.get("primary_finding") or "no abnormality"
    tooth = sample.tooth_id if sample.tooth_id is not None else "?"
    return f"Periapical radiograph (tooth {tooth}): {finding}."


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
            print("OK — DenPAR cache layout looks correct.")
            return 0
        except SystemExit as exc:
            return int(exc.code or 1)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = ["DenParSample", "fetch", "load_split", "to_chart_context", "SOURCE_URL"]
