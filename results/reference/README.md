# Reference results

These are the immutable computational artifacts used by the PNAS draft.
They were copied from the manuscript workspace without numerical modification.

- `central_three_block/` contains theorem-level and full-grid certificates,
  deterministic and stratified limiting curves, and both finite-sample Monte
  Carlo studies.
- `finite_85_tests/` contains the archived finite-dimensional certificate
  transcript.
- `larger_violation_six_block/` contains deterministic, Monte Carlo, and Arb
  certificate outputs for the supplementary six-block model.

Use `python scripts/verify_results.py` from the repository root to check every
manuscript-level value and the repository-wide SHA-256 manifest. Reproduction
commands write to `reproduced/`, not here. JSON files containing
`elapsed_seconds` are not expected to be byte-identical across reruns.
