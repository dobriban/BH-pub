#!/usr/bin/env python3
"""Numerical limiting-FDR evaluation for the six-block larger-violation model.

This is a floating-point diagnostic, not the rigorous certificate.  Gaussian
probabilities are evaluated in the log domain; the conditional first BH
crossing is found on a piecewise cutoff grid and refined by bisection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtri

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE.parents[1]
    / "reproduced"
    / "larger_violation_six_block"
    / "larger_violation_numerical_results.json"
)
ALPHAS = (0.01, 0.05, 0.10)

WEIGHT_STRINGS = (
    "0.99716298822048339213308976913422506279207606",
    "1.035724637633261e-5",
    "9.78796812256908e-10",
    "3.0492299816422394e-28",
    "2.826653554343463e-3",
    "2.2305608519390437e-24",
)
MEAN_STRINGS = (
    "0",
    "0.6770732230955452",
    "1.586522735054591",
    "3.2037098102462105",
    "4.469876769506591",
    "5.053055030678688",
)
LOADING_STRINGS = (
    "0.05609416921615385",
    "-0.0830487863256059",
    "-0.1770420820546788",
    "-0.23038325840329232",
    "-0.8002685127003462",
    "-0.414555784395906",
)

WEIGHTS = np.array([float(x) for x in WEIGHT_STRINGS])
MEANS = np.array([float(x) for x in MEAN_STRINGS])
LOADINGS = np.array([float(x) for x in LOADING_STRINGS])
SDS = np.sqrt(1.0 - LOADINGS * LOADINGS)

ZMAX = 8.5
DZ = 0.00125
BISECTION_STEPS = 24
BATCH_SIZE = 350


def make_validation_c_grid(alpha: float) -> np.ndarray:
    c_alpha = float(ndtri(1.0 - alpha / 2.0))
    return np.unique(
        np.r_[
            c_alpha,
            np.arange(c_alpha, 10.0, 0.02),
            np.arange(max(c_alpha, 10.0), 30.0, 0.06),
            np.arange(max(c_alpha, 30.0), 100.0, 0.25),
            np.arange(max(c_alpha, 100.0), 300.0, 1.0),
            np.arange(max(c_alpha, 300.0), 1500.0, 5.0),
            np.arange(max(c_alpha, 1500.0), 10000.0, 50.0),
            10000.0,
        ]
    )


def log_two_sided_tail(c: np.ndarray, abs_mean: np.ndarray, sd: float) -> np.ndarray:
    return np.logaddexp(
        log_ndtr((abs_mean - c) / sd),
        log_ndtr((-abs_mean - c) / sd),
    )


def log_mixture_tail(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    out: np.ndarray | None = None
    for weight, mean, loading, sd in zip(WEIGHTS, MEANS, LOADINGS, SDS):
        abs_mean = np.abs(mean + loading * z)
        term = math.log(float(weight)) + log_two_sided_tail(c, abs_mean, float(sd))
        out = term if out is None else np.logaddexp(out, term)
    assert out is not None
    return out


def evaluate_batch(alpha: float, c_grid: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = c_grid[None, :]
    Z = z[:, None]
    log_u = math.log(2.0) + log_ndtr(-c_grid)
    log_g = log_mixture_tail(C, Z)
    score = math.log(alpha) + log_g - log_u[None, :]
    feasible = score >= 0.0
    has_crossing = feasible.any(axis=1)
    first = np.argmax(feasible, axis=1)

    cstar = np.full(z.shape, np.nan)
    at_left = has_crossing & (first == 0)
    cstar[at_left] = c_grid[0]

    rows = np.flatnonzero(has_crossing & (first > 0))
    if rows.size:
        cols = first[rows]
        lo = c_grid[cols - 1].copy()
        hi = c_grid[cols].copy()
        zz = z[rows]
        for _ in range(BISECTION_STEPS):
            mid = (lo + hi) / 2.0
            log_g_mid = log_mixture_tail(mid, zz)
            score_mid = math.log(alpha) + log_g_mid - (
                math.log(2.0) + log_ndtr(-mid)
            )
            positive = score_mid >= 0.0
            hi = np.where(positive, mid, hi)
            lo = np.where(positive, lo, mid)
        cstar[rows] = (lo + hi) / 2.0

    fdp = np.zeros_like(z)
    rows = np.flatnonzero(has_crossing)
    if rows.size:
        cc = cstar[rows]
        zz = z[rows]
        log_g_star = log_mixture_tail(cc, zz)
        null_abs_mean = np.abs(LOADINGS[0] * zz)
        log_null = math.log(float(WEIGHTS[0])) + log_two_sided_tail(
            cc, null_abs_mean, float(SDS[0])
        )
        fdp[rows] = np.exp(log_null - log_g_star)
        np.clip(fdp, 0.0, 1.0, out=fdp)
    return fdp, cstar


def evaluate_alpha(alpha: float) -> dict[str, float | int | None]:
    z = np.arange(-ZMAX, ZMAX + DZ / 2.0, DZ)
    z_weights = np.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi) * DZ
    z_weights[0] *= 0.5
    z_weights[-1] *= 0.5
    c_grid = make_validation_c_grid(alpha)

    integral = 0.0
    crossing_count = 0
    maximum_c: float | None = None
    maximum_fdp = 0.0
    for start in range(0, z.size, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, z.size)
        fdp, cstar = evaluate_batch(alpha, c_grid, z[start:stop])
        integral += float(z_weights[start:stop] @ fdp)
        finite = np.isfinite(cstar)
        crossing_count += int(finite.sum())
        if finite.any():
            batch_max = float(np.nanmax(cstar))
            maximum_c = batch_max if maximum_c is None else max(maximum_c, batch_max)
        maximum_fdp = max(maximum_fdp, float(fdp.max()))

    return {
        "limiting_fdr": integral,
        "z_grid_points": int(z.size),
        "c_grid_points": int(c_grid.size),
        "crossing_grid_fraction": crossing_count / z.size,
        "maximum_located_c": maximum_c,
        "maximum_conditional_fdp": maximum_fdp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output: dict[str, object] = {
        "model": {
            "K": 6,
            "weights_exact_decimal": list(WEIGHT_STRINGS),
            "means_exact_decimal": list(MEAN_STRINGS),
            "loadings_exact_decimal": list(LOADING_STRINGS),
            "residual_standard_deviations": "sqrt(1-rho_g^2)",
            "common_block_denominator": "50000000000000000000000000000000000000000000",
        },
        "z_range": [-ZMAX, ZMAX],
        "z_mesh": DZ,
        "bisection_steps": BISECTION_STEPS,
        "results": {},
    }
    for alpha in ALPHAS:
        result = evaluate_alpha(alpha)
        output["results"][f"{alpha:.2f}"] = result
        print(f"alpha={alpha:.2f} limiting FDR={result['limiting_fdr']:.17g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
