# DentaCoPilot

Reproducibility artifact for the paper **"DentaCoPilot: An LLM-Augmented Next-Procedure Recommender for General Dentistry, Designed for Dentist Augmentation."**

DentaCoPilot is a next-procedure recommender for general dentistry. Given a structured patient chart, it returns a calibrated top-K distribution over Current Dental Terminology (CDT) codes, a verbalized confidence label, an explicit abstain flag, and a chart-grounded rationale. It is designed to augment a clinician, not to act autonomously.

This repository contains everything needed to reproduce the paper's results:

- the synthetic chart generator and the deterministic synthetic corpora;
- four classical baselines (frequency bigram, TF-IDF with logistic regression, gradient boosting, and a MultiTP-style CNN-RNN) and six LLM variants;
- the calibration, abstention, and split-conformal selective-prediction code;
- the public-data benchmark on the MEPS 2023 Dental Visits file (AHRQ); and
- the per-run records behind every reported number.

## Key results

- On an apples-to-apples synthetic evaluation, classical baselines reach 0.567 top-1 while pure LLM variants trail at 0.267 to 0.467. Prompt-conditioning a small LLM on the classical model's candidate list (M5) recovers most of the gap while keeping calibration, abstention, and the chart-grounded rationale.
- On the public MEPS 2023 corpus (11,016 next-visit transitions across 5,088 patients), procedure history predicts the next treatment category, the classical model is well calibrated (expected calibration error 0.031), and split-conformal selective prediction supplies a distribution-free coverage guarantee.

A note on scope: because the synthetic generator shares structure with one baseline, synthetic accuracy indexes pipeline behavior rather than clinical performance. Externally valid accuracy claims are reserved for the real data.

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r data/requirements.txt

# Real-data benchmark: downloads the public MEPS file (no access agreement needed)
python data/meps_benchmark.py
python data/meps_conformal.py

# Synthetic-corpus pipeline (LLM steps require the `claude` CLI on PATH)
python data/evaluate.py --smoke
```

See `data/README.md` for the full directory layout and run conventions.

## Data

- **MEPS 2023 Dental Visits (AHRQ, HC-248B):** public data, not redistributed here; downloaded and parsed by `data/meps_benchmark.py`.
- **Synthetic corpora:** deterministic (seed 42), included under `data/cache/`.
- No patient-identifiable data is included in this release.

## Citation

If you use this artifact, please cite the paper:

> Rodrigues CC, Rebello SD. DentaCoPilot: An LLM-Augmented Next-Procedure Recommender for General Dentistry, Designed for Dentist Augmentation. Preprint, 2026. doi:10.64898/2026.05.07.26352635

Machine-readable metadata is in `CITATION.cff`.

## License

- **Code** (`*.py`): MIT License (see `LICENSE`).
- **Data and per-run records** (synthetic corpora and `results_*/`): Creative Commons Attribution 4.0 International (see `LICENSE-DATA`).

## 2026-09-05 update

The manuscript was retitled after an audit: "Calibrated Next-Procedure Recommendation with Split-Conformal Abstention: A MEPS-Anchored Evaluation of Classical and Large Language Model Designs for General Dentistry". New in this artifact: `data/results_stratified_n200/` (the 200-example stratified re-run promised in the earlier draft: 400 Sonnet calls, 0 failures; analysed by `analyze_stratified_n200.py`), `audit_recompute_2026-09-05.py` (held-out calibration, calibrated abstention table, paired McNemar tests), and `check_headline_numbers.py` (diffs every number restated in the manuscript against these result files).
