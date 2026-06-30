# DentaCoPilot: reproducibility code and runs

This directory holds the code, the deterministic synthetic corpora, and the
per-run records for the paper *DentaCoPilot: An LLM-Augmented Next-Procedure
Recommender for General Dentistry, Designed for Dentist Augmentation*.

## Layout

```
data/
├── requirements.txt        pinned Python dependencies
├── cc_compute.py           Claude Code compute wrapper (every LLM call goes through here)
├── cc_compute_test.py      wrapper smoke test
├── cdt_codes.py            ADA CDT 2024 vocabulary + category map
├── tooth.py                Universal/FDI tooth-numbering helpers
├── synth_charts.py         synthetic chart generator (deterministic, seed 42)
├── retrieval.py            BM25 retriever over the evidence corpus
├── loaders/                public dental dataset loaders (DENTEX, MMDental, DENPAR)
├── models/                 classical baselines (B0-B3) + LLM variants (M1-M6)
├── evidence_corpus/        BM25 evidence cards
├── calibration.py          temperature scaling + verbalized confidence
├── abstention.py           abstention threshold rule + coverage-risk curves
├── abstention_ood.py       out-of-distribution abstention probes
├── evaluate.py             synthetic-corpus evaluation entry point
├── meps_benchmark.py       downloads + parses the public MEPS 2023 file; real-data baselines
├── meps_conformal.py       split-conformal selective prediction on MEPS
├── meps_ci_fairness.py     patient-clustered bootstrap CIs + subgroup fairness
├── meps_llm_eval.py        zero-shot LLM probe on coarse MEPS history
├── meps_synth_fidelity.py  synthetic-vs-MEPS generator face-validity
├── hybrid_analysis.py      classical + LLM hybrid (gating, M5/M6 priming)
├── plot_results.py         regenerates the paper figures from results_*/
├── results_<descriptor>/   one directory per run
│   ├── config.json         git SHA, seed, model versions, CLI args
│   ├── results.json        per-trial predictions and gold labels
│   ├── summary.json        aggregated metrics
│   └── llm_calls.jsonl     every LLM call (prompt, response, model, latency, tokens)
└── cache/
    └── synth_n{100,500}_seed42.json, synth_smoke.json   deterministic synthetic corpora
```

The MEPS 2023 Dental Visits file (AHRQ, HC-248B) is **not redistributed**: it is
public data that `meps_benchmark.py` downloads and parses reproducibly into a
local cache. No patient-identifiable data is included anywhere in this release.

## Compute substrate: Claude Code, not the metered API

Every LLM call is routed through `cc_compute.py`, which shells out to the local
`claude` CLI (Claude Code) rather than importing the Anthropic SDK. Each call is
logged to `results_<run>/llm_calls.jsonl` with prompt, response, model, latency,
and token counts, so any decision can be audited without re-running the
experiment. If `claude` is not on `PATH`, scripts fail loudly rather than fall
back to a paid API.

## Conventions

- **Seeds:** `random.seed(42); np.random.seed(42)` at the top of every script (override with `--seed`).
- **Smoke vs full:** every script supports `--smoke` (tiny input) and a full run.
- **Run directories:** `results_<condition>_<UTC-ISO>/`; existing directories are never overwritten.
- **Determinism:** any non-LLM model run twice with the same seed produces a byte-identical `summary.json`; LLM models reproduce within the stochastic tolerance reported in the paper.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r data/requirements.txt

# Real-data benchmark: downloads the public MEPS file (no access agreement needed)
python data/meps_benchmark.py
python data/meps_conformal.py

# Synthetic-corpus pipeline (LLM steps require the `claude` CLI on PATH)
python data/evaluate.py --smoke
```

## What is not here

- The Anthropic Python SDK; compute is routed through Claude Code (see above).
- Raw patient records. The loaders fetch public datasets into `cache/`.
