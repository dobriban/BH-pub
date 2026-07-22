#!/usr/bin/env python3
"""Evaluate the exploratory first-crossing bound from SI Section S4.

For the three-block Gaussian factor model, this script evaluates

    L_tilde(alpha) = int ell_alpha(z) phi(z) dz,
    ell_alpha(z) = alpha*w0*q(c_*(alpha,z); |M0(z)|, s0),

where c_*(alpha,z) is the first regular upward crossing of
R_{alpha,z}(c)=1 for c >= c_alpha.  Calculations are performed in the
log domain.  The z-integral is truncated to [-8,8]; because the
pointwise contribution is at most one at a crossing, the omitted mass is
at most 2*Phi(-8) < 1.25e-15.

The first-crossing branch has one jump in z for this model.  The code
locates the associated tangency by maximizing log R over the early-c
branch, splits the quadrature at that point, and applies adaptive
Gauss--Kronrod quadrature on the two smooth pieces.

This is a reproducible floating-point evaluation, not an outward-rounded
certificate.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from scipy.special import log_ndtr, ndtri


# Model constants from SI Sections S1 and S3.
WEIGHTS = np.array([163.0 / 167.0, 1.0 / 167.0, 3.0 / 167.0])
SDS = np.array([
    math.sqrt(91.0) / 10.0,
    3.0 * math.sqrt(13.0) / 11.0,
    math.sqrt(41.0) / 21.0,
])
W0 = float(WEIGHTS[0])
SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Settings:
    z_limit: float = 8.0
    c_max: float = 20.0
    peak_mesh: float = 0.04
    tangency_mesh: float = 0.025
    jump_z_left: float = -4.0
    jump_z_right: float = 0.5
    jump_z_mesh: float = 0.02
    epsabs: float = 2.0e-10
    epsrel: float = 2.0e-9


def phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / SQRT_2PI


def abs_block_means(z: float) -> np.ndarray:
    return np.abs(np.array([
        0.3 * z,
        37.0 / 20.0 + 2.0 * z / 11.0,
        59.0 / 12.0 + 20.0 * z / 21.0,
    ]))


def log_q(c: np.ndarray | float, a: float, s: float) -> np.ndarray | float:
    """Log of q(c;a,s)=Q(c;a,s)/(2*Phi_bar(c)), stably."""
    c_arr = np.asarray(c)
    log_q_num = np.logaddexp(
        log_ndtr(-(c_arr - a) / s),
        log_ndtr(-(c_arr + a) / s),
    )
    log_u = math.log(2.0) + log_ndtr(-c_arr)
    out = log_q_num - log_u
    return float(out) if np.ndim(out) == 0 else out


def log_s(c: np.ndarray | float, z: float) -> np.ndarray | float:
    """Log of S_z(c)=sum_g w_g q(c;|M_g(z)|,s_g)."""
    c_arr = np.asarray(c)
    means = abs_block_means(z)
    out = np.full(c_arr.shape, -np.inf, dtype=float)
    for weight, mean, sd in zip(WEIGHTS, means, SDS):
        out = np.logaddexp(out, math.log(float(weight)) + log_q(c_arr, mean, sd))
    return float(out) if np.ndim(out) == 0 else out


def c_alpha(alpha: float) -> float:
    return float(ndtri(1.0 - alpha / 2.0))


def first_crossing(alpha: float, z: float, settings: Settings) -> float:
    """Return the first upward root of log(alpha*S_z(c))=0.

    Local maxima of log S_z are detected on a coarse grid and refined.
    The first refined maximum above zero identifies the first positive
    excursion; Brent's method then finds its left endpoint.  Detecting
    the maximum rather than scanning only for positive grid values makes
    the calculation stable near a tangency, where the positive excursion
    can be arbitrarily narrow.
    """
    ca = c_alpha(alpha)
    log_alpha = math.log(alpha)
    n_grid = max(3, int(math.ceil((settings.c_max - ca) / settings.peak_mesh)) + 1)
    grid = np.linspace(ca, settings.c_max, n_grid)
    values = log_s(grid, z) + log_alpha

    candidates = list(find_peaks(values)[0])
    if values[-1] > values[-2]:
        candidates.append(len(grid) - 1)

    for idx in sorted(set(candidates)):
        if idx == len(grid) - 1:
            peak_c = float(grid[-1])
            peak_value = float(values[-1])
        else:
            result = minimize_scalar(
                lambda c: -(log_s(c, z) + log_alpha),
                bounds=(float(grid[idx - 1]), float(grid[idx + 1])),
                method="bounded",
                options={"xatol": 1.0e-13, "maxiter": 200},
            )
            peak_c = float(result.x)
            peak_value = float(log_s(peak_c, z) + log_alpha)

        if peak_value > 0.0:
            left_value = float(log_s(ca, z) + log_alpha)
            if not left_value < 0.0:
                raise RuntimeError(
                    f"Expected strict infeasibility at c_alpha; alpha={alpha}, z={z}, value={left_value}"
                )
            return float(
                brentq(
                    lambda c: log_s(c, z) + log_alpha,
                    ca,
                    peak_c,
                    xtol=1.0e-13,
                    rtol=1.0e-14,
                    maxiter=200,
                )
            )

    return math.inf


def interval_max_log_r(
    alpha: float,
    z: float,
    c_left: float,
    c_right: float,
    settings: Settings,
) -> tuple[float, float]:
    """Maximize log(alpha*S_z(c)) on a compact c interval."""
    log_alpha = math.log(alpha)
    n_grid = max(
        3,
        int(math.ceil((c_right - c_left) / settings.tangency_mesh)) + 1,
    )
    grid = np.linspace(c_left, c_right, n_grid)
    values = log_s(grid, z) + log_alpha
    candidates = list(find_peaks(values)[0]) + [0, len(grid) - 1]

    best_value = -math.inf
    best_c = math.nan
    for idx in sorted(set(candidates)):
        if idx == 0 or idx == len(grid) - 1:
            candidate_c = float(grid[idx])
            candidate_value = float(values[idx])
        else:
            result = minimize_scalar(
                lambda c: -(log_s(c, z) + log_alpha),
                bounds=(float(grid[idx - 1]), float(grid[idx + 1])),
                method="bounded",
                options={"xatol": 1.0e-13, "maxiter": 200},
            )
            candidate_c = float(result.x)
            candidate_value = float(log_s(candidate_c, z) + log_alpha)

        if candidate_value > best_value:
            best_value = candidate_value
            best_c = candidate_c

    return best_value, best_c


def locate_branch_jump(alpha: float, settings: Settings) -> tuple[float, float]:
    """Locate the unique first-crossing branch jump for this model.

    A coarse z scan brackets the drop from the later root to the earlier
    root.  Within that bracket, the jump occurs when the maximum of the
    earlier c-lobe is exactly zero, i.e. at a tangency.
    """
    z_grid = np.arange(
        settings.jump_z_left,
        settings.jump_z_right + 0.5 * settings.jump_z_mesh,
        settings.jump_z_mesh,
    )
    roots = np.array([first_crossing(alpha, float(z), settings) for z in z_grid])
    if not np.all(np.isfinite(roots)):
        raise RuntimeError(f"A nonfinite first crossing occurred while locating the jump for alpha={alpha}")

    root_differences = np.diff(roots)
    jump_index = int(np.argmin(root_differences))
    if root_differences[jump_index] >= -0.5:
        raise RuntimeError(
            f"Could not identify the expected branch jump for alpha={alpha}; "
            f"largest downward change={root_differences[jump_index]}"
        )

    z_left = float(z_grid[jump_index])
    z_right = float(z_grid[jump_index + 1])
    c_cut = 0.5 * float(roots[jump_index] + roots[jump_index + 1])
    ca = c_alpha(alpha)

    def early_lobe_height(z: float) -> float:
        return interval_max_log_r(alpha, z, ca, c_cut, settings)[0]

    left_height = early_lobe_height(z_left)
    right_height = early_lobe_height(z_right)
    if not (left_height < 0.0 < right_height):
        raise RuntimeError(
            "The coarse jump bracket did not bracket a tangency: "
            f"alpha={alpha}, heights=({left_height}, {right_height})"
        )

    jump_z = float(
        brentq(
            early_lobe_height,
            z_left,
            z_right,
            xtol=2.0e-12,
            rtol=2.0e-13,
            maxiter=100,
        )
    )
    _, tangency_c = interval_max_log_r(alpha, jump_z, ca, c_cut, settings)
    return jump_z, float(tangency_c)


def ell(alpha: float, z: float, settings: Settings) -> float:
    crossing = first_crossing(alpha, z, settings)
    if not math.isfinite(crossing):
        return 0.0
    value = alpha * W0 * math.exp(log_q(crossing, abs(0.3 * z), float(SDS[0])))
    if value < -1.0e-13 or value > 1.0 + 1.0e-8:
        raise RuntimeError(f"Pointwise contribution outside [0,1]: alpha={alpha}, z={z}, value={value}")
    return min(max(value, 0.0), 1.0)


def evaluate_alpha(alpha: float, settings: Settings) -> dict[str, float | int]:
    jump_z, tangency_c = locate_branch_jump(alpha, settings)
    endpoints = (-settings.z_limit, jump_z, settings.z_limit)

    integral = 0.0
    quadrature_error = 0.0
    function_evaluations = 0
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        value, error, info = quad(
            lambda z: ell(alpha, z, settings) * phi(z),
            left,
            right,
            epsabs=settings.epsabs / 2.0,
            epsrel=settings.epsrel,
            limit=250,
            full_output=1,
        )[:3]
        integral += float(value)
        quadrature_error += float(error)
        function_evaluations += int(info["neval"])

    return {
        "alpha": alpha,
        "first_crossing_bound": integral,
        "excess": integral - alpha,
        "quadrature_error_estimate": quadrature_error,
        "jump_z": jump_z,
        "tangency_c": tangency_c,
        "function_evaluations": function_evaluations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("s4_first_crossing_grid.csv"),
        help="CSV output path (default: %(default)s)",
    )
    parser.add_argument("--alpha-min-milli", type=int, default=1)
    parser.add_argument("--alpha-max-milli", type=int, default=200)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of worker processes (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.alpha_min_milli <= args.alpha_max_milli <= 999):
        raise ValueError("Require 1 <= alpha-min-milli <= alpha-max-milli <= 999")

    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")

    settings = Settings()
    alphas = [milli / 1000.0 for milli in range(args.alpha_min_milli, args.alpha_max_milli + 1)]
    if args.jobs == 1:
        rows = [evaluate_alpha(alpha, settings) for alpha in alphas]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            rows = list(executor.map(evaluate_alpha, alphas, [settings] * len(alphas)))

    for row in rows:
        alpha = float(row["alpha"])
        print(
            f"alpha={alpha:0.3f}  "
            f"L_tilde={row['first_crossing_bound']:.12f}  "
            f"excess={row['excess']:+.12f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "alpha",
        "first_crossing_bound",
        "excess",
        "quadrature_error_estimate",
        "jump_z",
        "tangency_c",
        "function_evaluations",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "alpha": f"{float(row['alpha']):.3f}",
                "first_crossing_bound": f"{float(row['first_crossing_bound']):.15f}",
                "excess": f"{float(row['excess']):+.15f}",
                "quadrature_error_estimate": f"{float(row['quadrature_error_estimate']):.3e}",
                "jump_z": f"{float(row['jump_z']):.15f}",
                "tangency_c": f"{float(row['tangency_c']):.15f}",
                "function_evaluations": int(row["function_evaluations"]),
            })

    violating = [row for row in rows if float(row["excess"]) > 0.0]
    if violating:
        last = violating[-1]
        print(
            "Largest violating grid point: "
            f"alpha={float(last['alpha']):.3f}, "
            f"L_tilde={float(last['first_crossing_bound']):.12f}, "
            f"excess={float(last['excess']):+.12f}"
        )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
