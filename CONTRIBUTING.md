# Contributing

Bug reports and reproducibility questions are welcome through GitHub issues.
Please include the Python version, operating system, package versions, command,
and complete error output.

When changing a numerical method or model parameter:

1. keep archived draft results under `results/reference/` unchanged;
2. write new outputs under `reproduced/`;
3. run `python scripts/reproduce.py check`; and
4. describe whether the change affects a rigorous certificate, a numerical
   diagnostic, a Monte Carlo result, or only presentation.

Update `SHA256SUMS` with `python scripts/checksums.py --write` only after the
intended repository contents have been reviewed.
