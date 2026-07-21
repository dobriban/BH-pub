#!/usr/bin/env python3
"""Outward-rounded certificate for a finite-dimensional multi-block BH violation.

The exact model has m=85 tests: 83 exchangeable true nulls and two singleton
nonnull blocks. Every model input, quadrature panel, and Gauss--Legendre node
is evaluated with Arb ball arithmetic. A global analytic derivative bound
certifies the quadrature error. The worker mode limits memory use by evaluating
small groups of panels in fresh Python processes.

Dependency: python-flint 0.8.0
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from math import comb, factorial

from flint import arb, arb_series, ctx

ctx.dps = 50
ctx.cap = 128

M = 85
N0 = 83
ALPHA = arb(1) / 10000
RHO0 = arb(1107) / 2000
MU = (arb(3673) / 500, arb(3723) / 500)
RHO = (-arb(999) / 1000, -arb(999) / 1000)

Z_LEFT = arb(-8)
Z_RIGHT = arb(8)
PANEL_WIDTH = arb(1) / 4
NUMBER_OF_PANELS = 64
GL_ORDER = 18
PANELS_PER_WORKER = 8
TARGET = arb("0.000100005")
MANUSCRIPT_TARGET = arb("0.00010000927")
WORKDIR_MARKER = ".bh_m85_temporary_workdir"

SQRT2 = arb(2).sqrt()
SQRT2PI = (2 * arb.pi()).sqrt()


def normal_cdf(x: arb) -> arb:
    return (-x / SQRT2).erfc() / 2


def conditional_p_cdf(t: arb, mu: arb, rho: arb, z: arb) -> arb:
    """P(2*Phi-bar(|X|)<=t | Z=z) for X=mu+rho*z+s*epsilon."""
    c = SQRT2 * t.erfcinv()
    s = (1 - rho * rho).sqrt()
    mean = mu + rho * z
    return normal_cdf((-c - mean) / s) + normal_cdf((mean - c) / s)


def compose_with_binomial_thinning(coeffs: list[arb], p: arb, prec: int) -> arb_series:
    """For P(x)=sum_a coeffs[a]x^a, return P(1-p+p*x)."""
    linear = arb_series([1 - p, p], prec)
    out = arb_series([], prec)
    for coefficient in reversed(coeffs):
        out = out * linear + coefficient
    return out


def conditional_integrand(z: arb) -> arb:
    """Return phi(z) E[FDP|Z=z] by an exact finite-state BH recursion."""
    f0 = [arb(0)] * (M + 1)
    fs = [[arb(0)] * (M + 1) for _ in MU]

    for j in range(1, M + 1):
        t = ALPHA * j / M
        f0[j] = conditional_p_cdf(t, arb(0), RHO0, z)
        for g in range(len(MU)):
            fs[g][j] = conditional_p_cdf(t, MU[g], RHO[g], z)

    # Force one chosen true-null p-value to zero. At t_M, the other null count
    # is Binomial(N0-1,f0[M]), and each singleton signal gives one mask bit.
    n = N0 - 1
    p = f0[M]
    q = 1 - p
    null_pmf = [arb(0)] * (n + 1)
    null_pmf[0] = q**n
    for a in range(n):
        null_pmf[a + 1] = null_pmf[a] * (n - a) * p / ((a + 1) * q)

    state: list[arb_series] = []
    for mask in range(1 << len(MU)):
        signal_mass = arb(1)
        for g in range(len(MU)):
            signal_mass *= fs[g][M] if (mask >> g) & 1 else 1 - fs[g][M]
        state.append(arb_series([x * signal_mass for x in null_pmf], n + 1))

    leave_one_out_sum = arb(0)

    for j in range(M, 0, -1):
        q_j = arb(0)
        survivors: list[arb_series] = []

        for mask, polynomial in enumerate(state):
            coefficients = polynomial.coeffs()
            # The leading one is the forced-zero null. Absorb exactly when
            # 1+A_{0,j}+number of included singleton signals >= j.
            cutoff = j - 1 - mask.bit_count()
            if cutoff <= 0:
                for coefficient in coefficients:
                    q_j += coefficient
                survivors.append(arb_series([], 1))
            else:
                for coefficient in coefficients[cutoff:]:
                    q_j += coefficient
                survivors.append(arb_series(coefficients[:cutoff], cutoff))

        leave_one_out_sum += f0[j] * q_j / j
        if j == 1:
            break

        # Independently thin the surviving coordinates from t_j to t_{j-1}.
        p0 = f0[j - 1] / f0[j]
        precision = j
        current = [
            compose_with_binomial_thinning(poly.coeffs(), p0, precision)
            for poly in survivors
        ]

        for g in range(len(MU)):
            pg = fs[g][j - 1] / fs[g][j]
            updated = [arb_series([], precision) for _ in current]
            bit = 1 << g
            for mask, polynomial in enumerate(current):
                if mask & bit:
                    updated[mask] = updated[mask] + polynomial * pg
                    updated[mask ^ bit] = updated[mask ^ bit] + polynomial * (1 - pg)
                else:
                    updated[mask] = updated[mask] + polynomial
            current = updated
        state = current

    return N0 * leave_one_out_sum * (-z * z / 2).exp() / SQRT2PI


def panel_sum(start_panel: int, stop_panel: int) -> arb:
    roots_weights = [
        arb.legendre_p_root(GL_ORDER, k, weight=True) for k in range(GL_ORDER)
    ]
    half_width = PANEL_WIDTH / 2
    total = arb(0)
    for panel in range(start_panel, stop_panel):
        left = Z_LEFT + panel * PANEL_WIDTH
        center = left + half_width
        for root, weight in roots_weights:
            z = center + half_width * root
            total += half_width * weight * conditional_integrand(z)
    return total


def aggregate_worker_files(paths: list[str]) -> arb:
    total = arb(0)
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            total += arb(handle.read().strip())
    return total

def normal_density_derivative_bound(order: int) -> arb:
    """Fourier bound on sup_z |phi^(order)(z)|."""
    return (
        arb(2) ** (arb(order - 1) / 2)
        * (arb(order + 1) / 2).gamma()
        / arb.pi()
    )


def conditional_derivative_bound(order: int) -> arb:
    """Bound sup_z |D^(order)(z)| for D(z)=E[FDP|Z=z]."""
    a_squared = N0 * RHO0 * RHO0 / (1 - RHO0 * RHO0)
    for rho in RHO:
        a_squared += rho * rho / (1 - rho * rho)
    return a_squared ** (arb(order) / 2) * arb(factorial(order)).sqrt()


def integrand_derivative_bound(order: int) -> arb:
    total = arb(0)
    for j in range(order + 1):
        total += (
            comb(order, j)
            * normal_density_derivative_bound(order - j)
            * conditional_derivative_bound(j)
        )
    return total


def quadrature_error_bound() -> arb:
    derivative_order = 2 * GL_ORDER
    length = Z_RIGHT - Z_LEFT
    gauss_constant = arb(factorial(GL_ORDER)) ** 4 / (
        (2 * GL_ORDER + 1) * arb(factorial(2 * GL_ORDER)) ** 3
    )
    return (
        length
        * PANEL_WIDTH**derivative_order
        * gauss_constant
        * integrand_derivative_bound(derivative_order)
    )


def print_and_check_certificate(quadrature: arb) -> None:
    error = quadrature_error_bound()
    certified_lower = quadrature - error

    print(f"composite Gauss-Legendre enclosure: {quadrature}")
    print(f"global quadrature-error bound:       {error}")
    print(f"certified truncated-integral lower:  {certified_lower}")
    print(f"nominal level alpha:                 {ALPHA}")
    print(f"claimed strict lower target:         {TARGET}")
    print(f"manuscript lower target:             {MANUSCRIPT_TARGET}")

    # The omitted factor tails are nonnegative, so the full FDR is at least
    # the integral over [-8,8]. Every strict comparison is an Arb test.
    assert certified_lower > TARGET
    assert certified_lower > MANUSCRIPT_TARGET
    assert TARGET > ALPHA
    print("CERTIFIED: FDR - alpha > 9.27e-9 > 0")


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1:] == ["--run"]:
        with tempfile.TemporaryDirectory(prefix="bh-m85-") as temporary:
            workdir = Path(temporary)
            (workdir / WORKDIR_MARKER).write_text(
                "managed by bh_finite_multiblock_m85_certificate.py\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--chain",
                    "0",
                    str(workdir),
                ],
                check=True,
            )
        return

    if sys.argv[1:] in (["--help"], ["-h"]):
        print(
            "usage: certificate.py [--run] | --worker START STOP | "
            "--aggregate WORKER_OUTPUT...\n"
            "The default --run mode creates and removes its own temporary directory."
        )
        return

    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(panel_sum(int(sys.argv[2]), int(sys.argv[3])))
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--aggregate":
        print_and_check_certificate(aggregate_worker_files(sys.argv[2:]))
        return

    if len(sys.argv) == 4 and sys.argv[1] == "--chain":
        start = int(sys.argv[2])
        workdir = Path(sys.argv[3]).resolve()
        if not (workdir / WORKDIR_MARKER).is_file():
            raise SystemExit(
                "refusing unmanaged --chain directory; use the default --run mode"
            )
        stop = min(start + PANELS_PER_WORKER, NUMBER_OF_PANELS)
        output = workdir / f"panels_{start:02d}_{stop:02d}.txt"
        output.write_text(str(panel_sum(start, stop)) + "\n", encoding="utf-8")

        if stop < NUMBER_OF_PANELS:
            os.execv(
                sys.executable,
                [sys.executable, str(Path(__file__).resolve()), "--chain", str(stop), str(workdir)],
            )

        paths = [
            str(workdir / f"panels_{s:02d}_{min(s + PANELS_PER_WORKER, NUMBER_OF_PANELS):02d}.txt")
            for s in range(0, NUMBER_OF_PANELS, PANELS_PER_WORKER)
        ]
        quadrature = aggregate_worker_files(paths)
        print_and_check_certificate(quadrature)
        return

    raise SystemExit(
        "usage: certificate.py [--run] | certificate.py --worker START STOP | "
        "certificate.py --aggregate WORKER_OUTPUT... | "
        "certificate.py --chain START MANAGED_WORKDIR"
    )


if __name__ == "__main__":
    main()
