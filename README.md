# DentaCoPilot

Reproducibility artifact for the paper **"A Zero-Shot Decision Model (Jev) Matches a Trained Classifier but Ranks Its Own Errors Worse: Calibrated, Split-Conformal Next-Procedure Recommendation in General Dentistry"** (earlier titles: "Calibrated Next-Procedure Recommendation with Split-Conformal Abstention: A MEPS-Anchored Evaluation of Classical and Large Language Model Designs for General Dentistry"; "DentaCoPilot: An LLM-Augmented Next-Procedure Recommender for General Dentistry, Designed for Dentist Augmentation").

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

## 2026-10-10 update: Jev and Jev Router

The manuscript is retitled "A Zero-Shot Decision Model (Jev) Matches a Trained Classifier but Ranks Its Own Errors Worse: Calibrated, Split-Conformal Next-Procedure Recommendation in General Dentistry". It adds TypeSafe's Jev (a decision model that returns a probability for every option) and Jev Router (which routes each chat request to an underlying LLM), both called through the OpenRouter API on 10 October 2026.

- `data/meps_jev_decisions.py` asks Jev (`typesafe/jev-1.13`, snapshot 20260917) the MEPS next-category question from the same two features the logistic baseline uses, zero-shot (J0) and with the training-split next-category frequencies (J1); 44 distinct states x 5 repeats x 2 designs = 440 calls, every request and probability vector in `data/results_meps_jev_decisions/calls.jsonl`, analysis (top-1/top-3, ECE raw and after isotonic recalibration, AURC, split-conformal selective risk, paired patient-clustered bootstrap against the refit baseline, repeat stability) in `analysis.json`. Rerun the analysis without any API call: `python3 data/meps_jev_decisions.py --analyse-only`.
- `data/run_jev_openrouter.py` sends the M2 and M5 prompts, byte-identical (checked against the logged Sonnet prompts before any call), to Jev Router on the stratified 200-example sample (`data/results_jev_n200/`, designs M7 and M8), and the Haiku prompt on the 150-transition MEPS probe (`data/results_meps_llm_jev/`). `data/openrouter_compute.py` logs the model that served each request, the provider and the billed cost; the API key is read from the environment and never logged.
- `data/analyze_jev.py` writes `data/analyze_jev.json` (paired McNemar tests against B1 and the Sonnet design with the same prompt, routed-model distribution, cost, latency). `check_headline_numbers.py` now checks the Jev numbers too (215 checks).

Headline: on MEPS, zero-shot Jev matches the trained model's top-1 (0.485 vs 0.477; paired difference +0.008, 95% CI -0.001 to +0.017) but is overconfident (ECE 0.098; 0.014 after isotonic recalibration) and ranks its own errors worse (AURC 0.474 vs 0.397; difference +0.077, CI +0.050 to +0.102). On the synthetic n=200 sample Jev Router reaches 0.300 (CoT) and 0.425 (classical prior), matching Sonnet with the prior (p=0.66). Jev Router's routing policy is unpublished and may change, so those runs can be audited from the logs but not re-executed exactly. All Jev calls cost US$1.87.
