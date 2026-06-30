"""ADA Current Dental Terminology (CDT) vocabulary and category map.

This module is the single source of truth for the procedure-code label set
used throughout Paper 7. It encodes the nine CDT 2024 categories and a
representative-but-non-exhaustive set of high-frequency codes per category.

Why representative rather than exhaustive: the full CDT 2024 codeset is ADA
copyright. We include the ~80 most frequent codes that appear in published
procedure-distribution analyses (e.g. published dental EHR distributions, Kalenderian-group
papers). The full code list is loaded at runtime from
``data/cache/cdt_2024_full.csv`` if present (user must download separately
from ADA), and merged with the representative set below.

References:
    American Dental Association. CDT 2024: Current Dental Terminology.
    Walji M.F. et al. JAMIA 2022 (dental EHR code-frequency analysis).
"""

from __future__ import annotations

import csv
import pathlib
from typing import NamedTuple

CACHE = pathlib.Path(__file__).resolve().parent / "cache" / "cdt_2024_full.csv"

CATEGORIES: dict[str, tuple[str, str]] = {
    "diagnostic":      ("D0100", "D0999"),
    "preventive":      ("D1000", "D1999"),
    "restorative":     ("D2000", "D2999"),
    "endodontic":      ("D3000", "D3999"),
    "periodontic":     ("D4000", "D4999"),
    "prosthodontic_removable": ("D5000", "D5899"),
    "maxillofacial":   ("D5900", "D5999"),
    "implant":         ("D6000", "D6199"),
    "prosthodontic_fixed":     ("D6200", "D6999"),
    "oral_surgery":    ("D7000", "D7999"),
    "orthodontic":     ("D8000", "D8999"),
    "adjunctive":      ("D9000", "D9999"),
}


class CDTCode(NamedTuple):
    code: str
    short: str
    category: str


REPRESENTATIVE_CODES: list[CDTCode] = [
    # Diagnostic
    CDTCode("D0120", "periodic oral evaluation", "diagnostic"),
    CDTCode("D0140", "limited oral evaluation, problem-focused", "diagnostic"),
    CDTCode("D0150", "comprehensive oral evaluation", "diagnostic"),
    CDTCode("D0210", "intraoral, complete series", "diagnostic"),
    CDTCode("D0220", "intraoral, periapical first", "diagnostic"),
    CDTCode("D0230", "intraoral, periapical each additional", "diagnostic"),
    CDTCode("D0274", "bitewings, four films", "diagnostic"),
    CDTCode("D0330", "panoramic radiographic image", "diagnostic"),
    CDTCode("D0367", "cone beam CT, both jaws", "diagnostic"),
    # Preventive
    CDTCode("D1110", "prophylaxis, adult", "preventive"),
    CDTCode("D1120", "prophylaxis, child", "preventive"),
    CDTCode("D1206", "topical fluoride varnish", "preventive"),
    CDTCode("D1351", "sealant, per tooth", "preventive"),
    CDTCode("D1354", "interim caries arresting medicament (SDF)", "preventive"),
    # Restorative — amalgam
    CDTCode("D2140", "amalgam, one surface", "restorative"),
    CDTCode("D2150", "amalgam, two surfaces", "restorative"),
    CDTCode("D2160", "amalgam, three surfaces", "restorative"),
    # Restorative — composite
    CDTCode("D2330", "resin-based composite, anterior, one surface", "restorative"),
    CDTCode("D2331", "resin-based composite, anterior, two surfaces", "restorative"),
    CDTCode("D2332", "resin-based composite, anterior, three surfaces", "restorative"),
    CDTCode("D2391", "resin-based composite, posterior, one surface", "restorative"),
    CDTCode("D2392", "resin-based composite, posterior, two surfaces", "restorative"),
    CDTCode("D2393", "resin-based composite, posterior, three surfaces", "restorative"),
    CDTCode("D2394", "resin-based composite, posterior, four+ surfaces", "restorative"),
    # Restorative — crowns
    CDTCode("D2740", "crown, porcelain/ceramic", "restorative"),
    CDTCode("D2750", "crown, porcelain fused to high-noble", "restorative"),
    CDTCode("D2790", "crown, full cast high-noble", "restorative"),
    CDTCode("D2950", "core buildup, including pins", "restorative"),
    CDTCode("D2954", "prefabricated post and core", "restorative"),
    # Endodontic
    CDTCode("D3220", "therapeutic pulpotomy", "endodontic"),
    CDTCode("D3310", "endodontic therapy, anterior", "endodontic"),
    CDTCode("D3320", "endodontic therapy, premolar", "endodontic"),
    CDTCode("D3330", "endodontic therapy, molar", "endodontic"),
    CDTCode("D3346", "retreatment, anterior", "endodontic"),
    CDTCode("D3347", "retreatment, premolar", "endodontic"),
    CDTCode("D3348", "retreatment, molar", "endodontic"),
    CDTCode("D3410", "apicoectomy, anterior", "endodontic"),
    # Periodontic
    CDTCode("D4341", "scaling and root planing, four+ teeth/quadrant", "periodontic"),
    CDTCode("D4342", "scaling and root planing, one to three teeth/quadrant", "periodontic"),
    CDTCode("D4346", "scaling, gingival inflammation, full mouth", "periodontic"),
    CDTCode("D4910", "periodontal maintenance", "periodontic"),
    CDTCode("D4263", "bone replacement graft, first site/quadrant", "periodontic"),
    # Prostho — removable
    CDTCode("D5110", "complete denture, maxillary", "prosthodontic_removable"),
    CDTCode("D5120", "complete denture, mandibular", "prosthodontic_removable"),
    CDTCode("D5213", "RPD with metal framework, maxillary", "prosthodontic_removable"),
    CDTCode("D5214", "RPD with metal framework, mandibular", "prosthodontic_removable"),
    # Implant
    CDTCode("D6010", "surgical placement, endosteal implant", "implant"),
    CDTCode("D6056", "prefabricated implant abutment", "implant"),
    CDTCode("D6057", "custom implant abutment", "implant"),
    CDTCode("D6058", "abutment-supported porcelain/ceramic crown", "implant"),
    CDTCode("D6065", "implant-supported porcelain/ceramic crown", "implant"),
    # Prostho — fixed
    CDTCode("D6240", "pontic, porcelain fused to high-noble", "prosthodontic_fixed"),
    CDTCode("D6750", "retainer crown, porcelain fused to high-noble", "prosthodontic_fixed"),
    # Oral surgery
    CDTCode("D7140", "extraction, erupted tooth or exposed root", "oral_surgery"),
    CDTCode("D7210", "extraction, erupted tooth requiring elevation/section", "oral_surgery"),
    CDTCode("D7220", "removal of impacted tooth, soft tissue", "oral_surgery"),
    CDTCode("D7230", "removal of impacted tooth, partial bony", "oral_surgery"),
    CDTCode("D7240", "removal of impacted tooth, complete bony", "oral_surgery"),
    CDTCode("D7241", "removal of impacted tooth, complete bony, w/ complications", "oral_surgery"),
    CDTCode("D7250", "surgical removal of residual roots", "oral_surgery"),
    CDTCode("D7286", "incisional biopsy of soft tissue", "oral_surgery"),
    # Orthodontic
    CDTCode("D8080", "comprehensive orthodontic treatment, adolescent", "orthodontic"),
    CDTCode("D8090", "comprehensive orthodontic treatment, adult", "orthodontic"),
    CDTCode("D8210", "removable appliance therapy", "orthodontic"),
    # Adjunctive
    CDTCode("D9110", "palliative treatment of dental pain", "adjunctive"),
    CDTCode("D9215", "local anesthesia, in conjunction with operative", "adjunctive"),
    CDTCode("D9230", "inhalation of nitrous oxide / analgesia", "adjunctive"),
    CDTCode("D9310", "consultation by another dentist", "adjunctive"),
    CDTCode("D9944", "occlusal guard, hard appliance, full arch", "adjunctive"),
]


def _load_cached() -> list[CDTCode]:
    if not CACHE.exists():
        return []
    out: list[CDTCode] = []
    with CACHE.open() as f:
        for row in csv.DictReader(f):
            cat = _category_for_code(row["code"])
            out.append(CDTCode(row["code"], row.get("short", ""), cat))
    return out


def _category_for_code(code: str) -> str:
    if not code.startswith("D") or len(code) != 5 or not code[1:].isdigit():
        return "unknown"
    n = int(code[1:])
    for name, (lo, hi) in CATEGORIES.items():
        if int(lo[1:]) <= n <= int(hi[1:]):
            return name
    return "unknown"


def _build() -> dict[str, CDTCode]:
    merged: dict[str, CDTCode] = {c.code: c for c in REPRESENTATIVE_CODES}
    for c in _load_cached():
        merged.setdefault(c.code, c)
    return merged


CDT_CODES: dict[str, CDTCode] = _build()


def codes_in_category(category: str) -> list[CDTCode]:
    return [c for c in CDT_CODES.values() if c.category == category]


def category_for(code: str) -> str:
    if code in CDT_CODES:
        return CDT_CODES[code].category
    return _category_for_code(code)


__all__ = [
    "CDTCode",
    "CDT_CODES",
    "CATEGORIES",
    "REPRESENTATIVE_CODES",
    "codes_in_category",
    "category_for",
]
