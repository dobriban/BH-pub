#!/usr/bin/env python3
"""Probability-stratified Monte Carlo for the limiting six-block FDP functional."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtri
from scipy.stats import t as student_t

ALPHAS = (0.01, 0.05, 0.10)
WEIGHTS = np.array([
    0.99716298822048339213308976913422506279207606,
    1.035724637633261e-5,
    9.78796812256908e-10,
    3.0492299816422394e-28,
    2.826653554343463e-3,
    2.2305608519390437e-24,
])
MEANS = np.array([
    0.0,
    0.6770732230955452,
    1.586522735054591,
    3.2037098102462105,
    4.469876769506591,
    5.053055030678688,
])
LOADINGS = np.array([
    0.05609416921615385,
    -0.0830487863256059,
    -0.1770420820546788,
    -0.23038325840329232,
    -0.8002685127003462,
    -0.414555784395906,
])
SDS = np.sqrt(1.0 - LOADINGS * LOADINGS)

# Extra resolution in the Gaussian tails; each segment is integrated by an
# independent randomized stratified estimate and then weighted by its exact
# probability mass.
SEGMENTS = (
    (0.0, 1.0e-6, 100),
    (1.0e-6, 1.0e-4, 100),
    (1.0e-4, 1.0e-2, 200),
    (1.0e-2, 9.9e-1, 400),
    (9.9e-1, 9.999e-1, 200),
    (9.999e-1, 9.99999e-1, 100),
    (9.99999e-1, 1.0, 100),
)


def c_grid(alpha: float) -> np.ndarray:
    ca = float(ndtri(1.0 - alpha / 2.0))
    return np.unique(
        np.r_[
            ca,
            np.arange(ca, 10.0, 0.10),
            np.arange(max(ca, 10.0), 30.0, 0.30),
            np.arange(max(ca, 30.0), 100.0, 1.0),
            np.arange(max(ca, 100.0), 300.0, 4.0),
            np.arange(max(ca, 300.0), 1500.0, 20.0),
            np.arange(max(ca, 1500.0), 10000.0, 200.0),
            10000.0,
        ]
    )


def log_qtail(c: np.ndarray, a: np.ndarray, sd: float) -> np.ndarray:
    return np.logaddexp(log_ndtr((a - c) / sd), log_ndtr((-a - c) / sd))


def log_mixture_tail(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    out: np.ndarray | None = None
    for w, mu, rho, sd in zip(WEIGHTS, MEANS, LOADINGS, SDS):
        a = np.abs(mu + rho * z)
        term = math.log(float(w)) + log_qtail(c, a, float(sd))
        out = term if out is None else np.logaddexp(out, term)
    assert out is not None
    return out


def crossing_functional_batch(alpha: float, z: np.ndarray, bisections: int = 22) -> np.ndarray:
    grid = c_grid(alpha)
    C = grid[None, :]
    Z = z[:, None]
    logu = math.log(2.0) + log_ndtr(-grid)
    logg = log_mixture_tail(C, Z)
    score = math.log(alpha) + logg - logu[None, :]
    feasible = score >= 0.0
    has = feasible.any(axis=1)
    first = np.argmax(feasible, axis=1)
    cstar = np.full(z.shape, np.nan)
    cstar[has & (first == 0)] = grid[0]

    rows = np.flatnonzero(has & (first > 0))
    if rows.size:
        cols = first[rows]
        lo = grid[cols - 1].copy()
        hi = grid[cols].copy()
        zz = z[rows]
        for _ in range(bisections):
            mid = (lo + hi) / 2.0
            lg = log_mixture_tail(mid, zz)
            smid = math.log(alpha) + lg - (math.log(2.0) + log_ndtr(-mid))
            positive = smid >= 0.0
            hi = np.where(positive, mid, hi)
            lo = np.where(positive, lo, mid)
        cstar[rows] = (lo + hi) / 2.0

    fdp = np.zeros_like(z)
    rows = np.flatnonzero(has)
    if rows.size:
        cc = cstar[rows]
        zz = z[rows]
        lg = log_mixture_tail(cc, zz)
        a0 = np.abs(LOADINGS[0] * zz)
        l0 = math.log(float(WEIGHTS[0])) + log_qtail(cc, a0, float(SDS[0]))
        fdp[rows] = np.exp(l0 - lg)
        np.clip(fdp, 0.0, 1.0, out=fdp)
    return fdp


def crossing_functional(alpha: float, z: np.ndarray, batch_size: int = 400) -> np.ndarray:
    out = np.empty_like(z)
    for start in range(0, z.size, batch_size):
        stop = min(start + batch_size, z.size)
        out[start:stop] = crossing_functional_batch(alpha, z[start:stop])
    return out


def sample_macro(rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    pieces: list[np.ndarray] = []
    metadata: list[tuple[int, int, float]] = []
    offset = 0
    for lo, hi, n in SEGMENTS:
        j = np.arange(n, dtype=float)
        u = lo + (hi - lo) * (j + rng.random(n)) / n
        # Avoid an endpoint under any future change to the RNG implementation.
        u = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        pieces.append(ndtri(u))
        metadata.append((offset, offset + n, hi - lo))
        offset += n
    return np.concatenate(pieces), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--output-prefix", type=Path, default=Path("larger_violation_mc"))
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least 2")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int]] = []
    for rep in range(args.replications):
        z, meta = sample_macro(rng)
        row: dict[str, float | int] = {"replication": rep + 1, "seed": args.seed}
        for alpha in ALPHAS:
            vals = crossing_functional(alpha, z)
            estimate = 0.0
            for start, stop, mass in meta:
                estimate += mass * float(vals[start:stop].mean())
            row[f"alpha_{alpha:.2f}"] = estimate
        rows.append(row)
        print(f"completed macro-replication {rep + 1}/{args.replications}", flush=True)

    summary: dict[str, object] = {
        "seed": args.seed,
        "replications": args.replications,
        "stratification_segments": [list(x) for x in SEGMENTS],
        "total_strata_per_replication": sum(x[2] for x in SEGMENTS),
        "alphas": {},
    }
    for alpha in ALPHAS:
        x = np.array([float(r[f"alpha_{alpha:.2f}"]) for r in rows])
        estimate = float(x.mean())
        mcse = float(x.std(ddof=1) / math.sqrt(len(x)))
        critical = float(student_t.ppf(0.975, df=len(x) - 1))
        summary["alphas"][f"{alpha:.2f}"] = {
            "estimate": estimate,
            "mcse": mcse,
            "student_t_ci95": [
                estimate - critical * mcse,
                estimate + critical * mcse,
            ],
        }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["replication", "seed"] + [f"alpha_{a:.2f}" for a in ALPHAS]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    prefix.with_name(prefix.name + "_summary.txt").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
