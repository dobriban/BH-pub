# Central `163 N + N + 3 N` model

This directory contains source code and exact rational witness inputs for the
main PNAS model. Immutable outputs used by the draft are in
`../../results/reference/central_three_block/`; new runs go to
`../../reproduced/central_three_block/`.

From the repository root, use:

```bash
python scripts/reproduce.py theorem-certificates
python scripts/reproduce.py certificate-grid
python scripts/reproduce.py limiting-curve
python scripts/reproduce.py limiting-mc
python scripts/reproduce.py finite-mc
python scripts/reproduce.py selected-mc
python scripts/reproduce.py figure
```

The theorem certificates and full-grid certificate use outward-rounded Arb
arithmetic. The limiting curve and stratified calculation are numerical
diagnostics. The finite-sample scripts use exact conditional binomial thinning
to simulate ordinary BH without generating and sorting all `167 N` p-values.

The standalone PNAS panel uses the rigorous lower-bound curve and the
independent 100,000-replication finite-sample run at `N=1000`. The abstract's
selected values come from the separate 500,000-replication run.
