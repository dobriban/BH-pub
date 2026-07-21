# Larger-violation six-block example

This directory contains the exact six-block model and the three associated
programs: a deterministic limiting evaluation, a probability-stratified Monte
Carlo check, and an outward-rounded certificate at `alpha=0.05`.

From the repository root, use:

```bash
python scripts/reproduce.py six-numerical
python scripts/reproduce.py six-mc
python scripts/reproduce.py six-certificate
```

Reference outputs are in
`../../results/reference/larger_violation_six_block/`; regenerated files go to
`../../reproduced/larger_violation_six_block/`.

The exact common block denominator is `5e43`. Several signal proportions are
extremely small, so the Monte Carlo program estimates the limiting conditional
functional rather than a moderate-dimensional finite realization. The
certificate proves `liminf FDR_N(0.05) > 0.0582252019310`.
