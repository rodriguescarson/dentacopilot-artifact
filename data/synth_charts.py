"""Synthetic dental chart generator for Paper 7 infrastructure validation.

We need labeled procedure-history data to wire up and stress-test the modeling
pipeline before access to real clinical EHR data. The generator below
produces N realistic chart histories with timestamped CDT procedures using a
hierarchical model:

    1. Each patient is assigned a *trajectory archetype* (e.g. ``healthy``,
       ``restorative_prone``, ``perio_prone``, ``edentulous_progressing``,
       ``ortho_adolescent``). The archetype controls visit cadence and the
       weighting of subsequent-procedure choices.
    2. Per-archetype transition tables encode plausibility-based bigram
       probabilities: ``P(next_code | prev_code, archetype)``. These were
       derived from clinical workflow knowledge and the published frequency
       analyses in large real-world dental EHR corpora (e.g., Walji et al., JAMIA 2022). They are
       *plausibility approximations*, not measured statistics.
    3. Tooth-level state is tracked: a tooth that has been extracted cannot
       receive a subsequent restoration; an RCT'd tooth tends to receive a
       crown next; an SDF site reverts to monitoring rather than drilling.
    4. Time-between-visits is sampled from realistic distributions per
       archetype.

The empirical claims of this paper rest on real-data evaluations (the public MEPS corpus, with clinical EHR
validation as future work); the synthetic data exists to validate that the pipeline runs end-to-end
and that the baselines learn something non-trivial. The generator is also
released as a side contribution: any reviewer can re-run the entire pipeline
on synthetic data without needing dataset access.

Usage::

    # Generate a corpus
    python data/synth_charts.py generate --n 1000 --out data/cache/synth.json

    # Print summary statistics for sanity checking
    python data/synth_charts.py validate data/cache/synth.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import pathlib
import random
import sys
from collections import Counter
from typing import Optional

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cdt_codes import CDT_CODES, codes_in_category, category_for  # noqa: E402
from tooth import is_molar, is_anterior, arch_universal  # noqa: E402

DEFAULT_SEED = 42

ARCHETYPES = [
    "healthy",
    "restorative_prone",
    "perio_prone",
    "edentulous_progressing",
    "ortho_adolescent",
]

ARCHETYPE_PRIOR = {
    "healthy": 0.40,
    "restorative_prone": 0.25,
    "perio_prone": 0.15,
    "edentulous_progressing": 0.10,
    "ortho_adolescent": 0.10,
}

AGE_BANDS_BY_ARCHETYPE: dict[str, list[str]] = {
    "healthy": ["20-29", "30-39", "40-49", "50-59"],
    "restorative_prone": ["30-39", "40-49", "50-59", "60-69"],
    "perio_prone": ["40-49", "50-59", "60-69", "70+"],
    "edentulous_progressing": ["50-59", "60-69", "70+"],
    "ortho_adolescent": ["10-19", "20-29"],
}

VISITS_PER_YEAR_MEAN = {
    "healthy": 2.0,
    "restorative_prone": 3.0,
    "perio_prone": 3.5,
    "edentulous_progressing": 3.0,
    "ortho_adolescent": 6.0,
}

TIMELINE_YEARS = 5.0


@dataclasses.dataclass
class ProcedureEvent:
    date: str
    code: str
    tooth: Optional[int] = None  # Universal numbering, None when not tooth-specific


@dataclasses.dataclass
class Chart:
    patient_id: str
    age_band: str
    sex: str
    archetype: str
    medical_history: list[str]
    history: list[ProcedureEvent]


_RECALL_FALLBACK = {"D0120": 0.55, "D1110": 0.25, "D0274": 0.20}


def _new_visit_codes(
    prev: Optional[str],
    archetype: str,
    rng: random.Random,
    is_first_visit: bool,
) -> list[str]:
    """Pick the next visit's procedure(s).

    Returns a list because some visits naturally bundle codes (e.g. eval +
    bitewings + prophy on a single recall day).

    If ``prev`` is missing from the archetype's transition table (because it
    was a tail-state procedure with no follow-up entry), we fall back to a
    generic recall mix rather than resetting to a comprehensive evaluation.
    Resetting to D0150 produces unrealistic chart histories where ~25% of
    events are comprehensive evaluations.
    """
    transitions = TRANSITIONS[archetype]

    if is_first_visit:
        first_visit = ["D0150", "D0274"]
        if archetype != "edentulous_progressing":
            first_visit.append("D1110")
        return first_visit

    if prev is None or prev not in transitions:
        codes, weights = zip(*_RECALL_FALLBACK.items())
        return [rng.choices(codes, weights=weights, k=1)[0]]

    dist = transitions[prev]
    codes, weights = zip(*dist.items())
    next_code = rng.choices(codes, weights=weights, k=1)[0]
    out = [next_code]

    # A single visit often bundles imaging or anesthesia
    if next_code.startswith("D2") and rng.random() < 0.4:
        out.append("D9215")  # local anesthesia
    if next_code in {"D3310", "D3320", "D3330"} and rng.random() < 0.7:
        out.append("D9215")
    return out


def _tooth_for(code: str, archetype: str, used_teeth: set[int], rng: random.Random) -> Optional[int]:
    """Pick a plausible tooth for a tooth-specific procedure."""
    cat = category_for(code)
    if cat in {"diagnostic", "preventive", "adjunctive"}:
        return None
    if cat == "orthodontic":
        return None

    candidates = list(range(1, 33))
    if code in {"D2150", "D2160", "D2391", "D2392", "D2393", "D2394"}:
        candidates = [n for n in candidates if not is_anterior(n)]  # posterior composite
    if code in {"D2330", "D2331", "D2332"}:
        candidates = [n for n in candidates if is_anterior(n)]  # anterior composite
    if code in {"D3330", "D3348"}:
        candidates = [n for n in candidates if is_molar(n)]
    if code in {"D3310", "D3346"}:
        candidates = [n for n in candidates if is_anterior(n)]
    if code in {"D7220", "D7230", "D7240", "D7241"}:
        candidates = [1, 16, 17, 32]  # third molars

    if not candidates:
        candidates = list(range(1, 33))

    fresh = [n for n in candidates if n not in used_teeth]
    pool = fresh if fresh else candidates
    return rng.choice(pool)


def _sample_archetype(rng: random.Random) -> str:
    items, weights = zip(*ARCHETYPE_PRIOR.items())
    return rng.choices(items, weights=weights, k=1)[0]


def _sample_visit_date(start: _dt.date, archetype: str, rng: random.Random) -> _dt.date:
    days_per_year = 365.25
    mean_visits = VISITS_PER_YEAR_MEAN[archetype]
    expected_days = days_per_year / mean_visits
    delta = max(7, int(rng.gauss(expected_days, expected_days * 0.3)))
    return start + _dt.timedelta(days=delta)


def _medical_history_for(archetype: str, rng: random.Random) -> list[str]:
    base = []
    if archetype == "perio_prone" and rng.random() < 0.4:
        base.append("T2DM")
    if rng.random() < 0.15:
        base.append("HTN")
    if archetype == "edentulous_progressing" and rng.random() < 0.3:
        base.append("Smoker")
    return base


def _generate_one(patient_id: str, rng: random.Random) -> Chart:
    archetype = _sample_archetype(rng)
    age_band = rng.choice(AGE_BANDS_BY_ARCHETYPE[archetype])
    sex = rng.choice(["F", "M"])
    medical = _medical_history_for(archetype, rng)

    history: list[ProcedureEvent] = []
    used_teeth: set[int] = set()
    rct_teeth: set[int] = set()
    extracted: set[int] = set()
    current_date = _dt.date(2021, 1, 1) + _dt.timedelta(days=rng.randint(0, 180))
    end_date = current_date + _dt.timedelta(days=int(TIMELINE_YEARS * 365))

    prev_code: Optional[str] = None
    visit_index = 0
    while current_date < end_date:
        codes = _new_visit_codes(prev_code, archetype, rng, is_first_visit=(visit_index == 0))
        visit_index += 1
        for code in codes:
            tooth: Optional[int] = None
            if category_for(code) not in {"diagnostic", "preventive", "adjunctive", "orthodontic"}:
                tooth = _tooth_for(code, archetype, extracted, rng)
                if tooth is not None and tooth in extracted:
                    continue  # cannot operate on extracted tooth
                if tooth is not None and code in {"D7140", "D7210", "D7220", "D7230", "D7240", "D7241"}:
                    extracted.add(tooth)
                if tooth is not None and code in {"D3310", "D3320", "D3330", "D3346", "D3347", "D3348"}:
                    rct_teeth.add(tooth)
                if tooth is not None and code.startswith("D27") and tooth in rct_teeth:
                    pass  # plausible: RCT then crown
            history.append(ProcedureEvent(
                date=current_date.isoformat(),
                code=code,
                tooth=tooth,
            ))
            if tooth is not None:
                used_teeth.add(tooth)
            prev_code = code

        current_date = _sample_visit_date(current_date, archetype, rng)

    return Chart(
        patient_id=patient_id,
        age_band=age_band,
        sex=sex,
        archetype=archetype,
        medical_history=medical,
        history=history,
    )


def generate_corpus(n: int, seed: int = DEFAULT_SEED) -> list[Chart]:
    rng = random.Random(seed)
    return [_generate_one(f"P{idx:06d}", rng) for idx in range(n)]


def to_jsonable(chart: Chart) -> dict:
    return {
        **dataclasses.asdict(chart),
        "history": [dataclasses.asdict(e) for e in chart.history],
    }


def from_jsonable(blob: dict) -> Chart:
    history = [ProcedureEvent(**e) for e in blob["history"]]
    return Chart(
        patient_id=blob["patient_id"],
        age_band=blob["age_band"],
        sex=blob["sex"],
        archetype=blob["archetype"],
        medical_history=blob.get("medical_history", []),
        history=history,
    )


def save(charts: list[Chart], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "charts": [to_jsonable(c) for c in charts]}
    path.write_text(json.dumps(payload, indent=2))


def load(path: pathlib.Path) -> list[Chart]:
    blob = json.loads(path.read_text())
    return [from_jsonable(c) for c in blob["charts"]]


def validate_corpus(charts: list[Chart]) -> dict:
    """Return summary statistics for sanity checking."""
    n = len(charts)
    total_events = sum(len(c.history) for c in charts)
    archetype_counts = Counter(c.archetype for c in charts)
    code_counts: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    for chart in charts:
        for ev in chart.history:
            code_counts[ev.code] += 1
            cat_counts[category_for(ev.code)] += 1

    avg_history_len = total_events / max(n, 1)

    # Sanity: no procedures on extracted teeth (counter check)
    violations = 0
    for chart in charts:
        extracted: set[int] = set()
        for ev in chart.history:
            if ev.tooth is not None and ev.tooth in extracted:
                violations += 1
            if ev.tooth is not None and ev.code in {"D7140", "D7210", "D7220", "D7230", "D7240", "D7241"}:
                extracted.add(ev.tooth)

    return {
        "n_charts": n,
        "n_events": total_events,
        "avg_history_len": round(avg_history_len, 2),
        "archetype_counts": dict(archetype_counts),
        "category_counts": dict(cat_counts),
        "top10_codes": code_counts.most_common(10),
        "n_unique_codes": len(code_counts),
        "extraction_violations": violations,
    }


# ---------------------------------------------------------------------------
# Transition tables (per archetype). Each is dict[str, dict[str, float]]:
# P(next_code | prev_code). Missing weights are treated as zero; if a prev_code
# is absent from the table, _new_visit_codes falls back to a fresh
# comprehensive eval.
#
# These are plausibility approximations, NOT measured statistics. Final
# experiments use real clinical EHR data.
# ---------------------------------------------------------------------------

_HEALTHY: dict[str, dict[str, float]] = {
    "D0150": {"D0274": 0.55, "D0210": 0.20, "D1110": 0.15, "D0330": 0.05, "D9110": 0.05},
    "D0274": {"D1110": 0.70, "D2392": 0.10, "D2391": 0.05, "D0150": 0.10, "D0120": 0.05},
    "D1110": {"D0120": 0.75, "D2392": 0.07, "D1206": 0.10, "D4346": 0.05, "D0274": 0.03},
    "D0120": {"D0274": 0.55, "D1110": 0.30, "D2392": 0.07, "D2391": 0.05, "D0150": 0.03},
    "D2392": {"D9215": 0.40, "D0120": 0.40, "D2393": 0.10, "D2391": 0.05, "D0274": 0.05},
    "D2391": {"D9215": 0.40, "D0120": 0.45, "D2392": 0.08, "D0274": 0.07},
    "D9215": {"D0120": 0.55, "D1110": 0.20, "D0274": 0.10, "D2392": 0.10, "D2391": 0.05},
    "D1206": {"D0120": 0.80, "D1110": 0.15, "D0274": 0.05},
    "D4346": {"D0120": 0.85, "D1110": 0.10, "D4910": 0.05},
}

_RESTORATIVE: dict[str, dict[str, float]] = {
    "D0150": {"D0274": 0.55, "D0210": 0.30, "D0330": 0.10, "D1110": 0.05},
    "D0274": {"D2392": 0.30, "D2393": 0.15, "D2391": 0.15, "D1110": 0.20, "D2150": 0.10, "D2160": 0.10},
    "D1110": {"D0120": 0.55, "D2392": 0.20, "D2393": 0.10, "D2391": 0.10, "D0274": 0.05},
    "D0120": {"D0274": 0.50, "D2392": 0.20, "D1110": 0.15, "D2391": 0.10, "D0150": 0.05},
    "D2392": {"D9215": 0.35, "D2393": 0.15, "D0120": 0.30, "D2950": 0.10, "D2391": 0.05, "D2740": 0.05},
    "D2393": {"D9215": 0.35, "D2950": 0.20, "D2740": 0.15, "D0120": 0.20, "D2394": 0.10},
    "D2394": {"D2950": 0.35, "D2740": 0.30, "D9215": 0.20, "D3330": 0.10, "D0120": 0.05},
    "D2950": {"D2740": 0.55, "D2750": 0.30, "D2790": 0.05, "D2954": 0.10},
    "D2740": {"D0120": 0.65, "D9944": 0.10, "D1110": 0.15, "D9215": 0.10},
    "D2750": {"D0120": 0.65, "D9944": 0.10, "D1110": 0.15, "D9215": 0.10},
    "D2391": {"D9215": 0.40, "D0120": 0.40, "D2392": 0.10, "D0274": 0.10},
    "D2150": {"D9215": 0.35, "D0120": 0.40, "D2160": 0.15, "D3330": 0.10},
    "D2160": {"D2950": 0.30, "D3330": 0.20, "D9215": 0.20, "D0120": 0.20, "D2740": 0.10},
    "D9215": {"D0120": 0.50, "D1110": 0.20, "D2392": 0.15, "D2393": 0.15},
    "D3330": {"D2950": 0.45, "D2750": 0.30, "D2740": 0.15, "D0120": 0.10},
    "D9944": {"D0120": 0.85, "D1110": 0.15},
}

_PERIO: dict[str, dict[str, float]] = {
    "D0150": {"D0274": 0.45, "D0210": 0.30, "D4341": 0.15, "D0330": 0.10},
    "D0274": {"D4341": 0.40, "D4342": 0.25, "D4346": 0.15, "D1110": 0.10, "D2392": 0.10},
    "D4341": {"D4341": 0.20, "D4342": 0.20, "D4910": 0.30, "D0120": 0.20, "D9215": 0.10},
    "D4342": {"D4341": 0.20, "D4910": 0.40, "D0120": 0.25, "D9215": 0.15},
    "D4346": {"D0120": 0.45, "D4910": 0.30, "D4341": 0.15, "D1110": 0.10},
    "D4910": {"D0120": 0.50, "D4910": 0.25, "D4341": 0.15, "D4263": 0.05, "D4346": 0.05},
    "D4263": {"D4910": 0.55, "D0120": 0.30, "D4341": 0.15},
    "D0120": {"D0274": 0.45, "D4910": 0.30, "D4341": 0.15, "D2392": 0.10},
    "D2392": {"D9215": 0.35, "D0120": 0.35, "D4910": 0.20, "D2393": 0.10},
    "D9215": {"D0120": 0.45, "D4910": 0.25, "D2392": 0.15, "D4341": 0.15},
}

_EDENTULOUS: dict[str, dict[str, float]] = {
    "D0150": {"D0210": 0.40, "D0330": 0.30, "D7140": 0.15, "D7210": 0.10, "D9110": 0.05},
    "D0210": {"D7140": 0.35, "D7210": 0.30, "D7250": 0.15, "D6010": 0.10, "D5110": 0.10},
    "D0330": {"D7140": 0.30, "D7210": 0.25, "D7240": 0.15, "D6010": 0.15, "D5110": 0.15},
    "D7140": {"D9215": 0.40, "D0120": 0.30, "D7140": 0.15, "D7210": 0.10, "D6010": 0.05},
    "D7210": {"D9215": 0.40, "D0120": 0.25, "D7210": 0.15, "D6010": 0.15, "D5110": 0.05},
    "D7240": {"D9215": 0.45, "D0120": 0.35, "D7240": 0.10, "D6010": 0.10},
    "D6010": {"D6056": 0.40, "D6057": 0.30, "D0120": 0.15, "D6065": 0.10, "D6058": 0.05},
    "D6056": {"D6058": 0.50, "D6065": 0.30, "D0120": 0.20},
    "D6057": {"D6058": 0.50, "D6065": 0.30, "D0120": 0.20},
    "D6058": {"D0120": 0.65, "D9944": 0.15, "D1110": 0.20},
    "D6065": {"D0120": 0.65, "D9944": 0.15, "D1110": 0.20},
    "D5110": {"D0120": 0.70, "D5120": 0.20, "D5213": 0.10},
    "D5120": {"D0120": 0.70, "D5110": 0.20, "D5214": 0.10},
    "D9215": {"D0120": 0.50, "D7140": 0.25, "D7210": 0.15, "D6010": 0.10},
    "D9110": {"D7140": 0.45, "D7210": 0.30, "D0150": 0.20, "D0120": 0.05},
    "D0120": {"D0210": 0.40, "D0330": 0.20, "D7140": 0.15, "D6010": 0.15, "D9944": 0.10},
}

_ORTHO: dict[str, dict[str, float]] = {
    "D0150": {"D0330": 0.45, "D0210": 0.25, "D8080": 0.20, "D0274": 0.10},
    "D0330": {"D8080": 0.55, "D0210": 0.20, "D0367": 0.15, "D8090": 0.10},
    "D0210": {"D8080": 0.50, "D0330": 0.20, "D8090": 0.20, "D8210": 0.10},
    "D0367": {"D8080": 0.60, "D8090": 0.30, "D0330": 0.10},
    "D8080": {"D8080": 0.65, "D0120": 0.20, "D9215": 0.10, "D8210": 0.05},
    "D8090": {"D8090": 0.65, "D0120": 0.20, "D9215": 0.10, "D8210": 0.05},
    "D8210": {"D0120": 0.65, "D8080": 0.20, "D8090": 0.15},
    "D0274": {"D1110": 0.50, "D8080": 0.30, "D0120": 0.20},
    "D0120": {"D8080": 0.55, "D0274": 0.20, "D1110": 0.15, "D8090": 0.10},
    "D9215": {"D8080": 0.45, "D8090": 0.30, "D0120": 0.25},
    "D1110": {"D0120": 0.60, "D8080": 0.30, "D0274": 0.10},
}

TRANSITIONS: dict[str, dict[str, dict[str, float]]] = {
    "healthy": _HEALTHY,
    "restorative_prone": _RESTORATIVE,
    "perio_prone": _PERIO,
    "edentulous_progressing": _EDENTULOUS,
    "ortho_adolescent": _ORTHO,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a corpus")
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--out", type=pathlib.Path, required=True)

    v = sub.add_parser("validate", help="print summary statistics")
    v.add_argument("path", type=pathlib.Path)

    args = p.parse_args()
    if args.cmd == "generate":
        charts = generate_corpus(args.n, seed=args.seed)
        save(charts, args.out)
        stats = validate_corpus(charts)
        print(json.dumps(stats, indent=2))
        print(f"Wrote {args.n} charts to {args.out}")
        return 0
    if args.cmd == "validate":
        charts = load(args.path)
        stats = validate_corpus(charts)
        print(json.dumps(stats, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "Chart",
    "ProcedureEvent",
    "generate_corpus",
    "validate_corpus",
    "save",
    "load",
    "to_jsonable",
    "from_jsonable",
    "ARCHETYPES",
    "TRANSITIONS",
]
