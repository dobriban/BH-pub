# Finite-dimensional 85-test certificate

This directory contains the outward-rounded certificate for the separate
example with 83 true nulls, two singleton nonnulls, and nominal level
`alpha=1/10000`.

Run it safely from the repository root with:

```bash
python scripts/reproduce.py finite-85
```

The runner creates its own temporary directory and writes the transcript to
`reproduced/finite_85_tests/`. The certificate proves
`FDR - alpha > 9.27e-9`. The archived submission transcript is under
`results/reference/finite_85_tests/`.
