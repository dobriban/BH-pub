#!/usr/bin/env python3
"""Outward-rounded Arb certificate for the six-block alpha=0.05 model.

Dependency
----------
    python-flint == 0.8.0

Mathematical inputs
-------------------
Every displayed model parameter, z-bin endpoint, and c-grid endpoint is an
exact rational number.  No binary floating-point literal is used in a
mathematical comparison.  Transcendental functions are evaluated as Arb balls.
A sign is accepted only when the entire resulting ball has the required sign.

For a block with absolute conditional mean a and residual standard deviation s,
write

    Q(c;a,s) = P(|N(a,s^2)| >= c),
    u(c)     = 2 * Phi_bar(c),
    q(c;a,s) = Q(c;a,s) / u(c).

BH feasibility is alpha * sum_g w_g q_g(c,z) >= 1.  On every z-bin the code
uses exact extrema of |mu_g + rho_g z|.  It certifies a prefix on which the
feasibility ratio is strictly below one, and then a point at which it is
strictly above one.  Interval branch-and-bound controls each component q_g on
a c-interval using the sign of d/dc log q_g whenever Arb can certify it.

The binwise FDP bound preserves the common threshold:

    alpha*w_0*inf_{c in [a_k,b_k]} q(c;m_{0,k}^-,s_0).

This is sharper than bounding Q at b_k and u at a_k separately.  The final
Gaussian integration discards all factor values outside the stated certified
range, which is valid because the FDP is nonnegative.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from flint import arb, ctx, fmpq

ctx.dps = 80

SQRT2 = arb(2).sqrt()
SQRT2PI = (2 * arb.pi()).sqrt()


def decimal_rational(text: str) -> fmpq:
    """Convert a finite decimal/scientific-notation string to an exact fmpq."""
    value = Decimal(text)
    sign, digits, exponent = value.as_tuple()
    numerator = 0
    for digit in digits:
        numerator = 10 * numerator + digit
    if sign:
        numerator = -numerator
    if exponent >= 0:
        return fmpq(numerator * (10**exponent), 1)
    return fmpq(numerator, 10 ** (-exponent))


def arb_interval(lo: fmpq, hi: fmpq) -> arb:
    """Closed interval [lo,hi] as an Arb midpoint-radius ball."""
    return arb((lo + hi) / 2, (hi - lo) / 2)


def normal_upper_tail(x: arb) -> arb:
    return (x / SQRT2).erfc() / 2


def normal_density(x: arb) -> arb:
    return (-(x * x) / 2).exp() / SQRT2PI


def two_sided_tail(c: arb, abs_mean: arb, sd: arb) -> arb:
    return (
        normal_upper_tail((c - abs_mean) / sd)
        + normal_upper_tail((c + abs_mean) / sd)
    )


def p_threshold(c: arb) -> arb:
    return 2 * normal_upper_tail(c)


def tail_ratio(c: arb, abs_mean: arb, sd: arb) -> arb:
    return two_sided_tail(c, abs_mean, sd) / p_threshold(c)


def tail_ratio_log_derivative(c: arb, abs_mean: arb, sd: arb) -> arb:
    """d/dc log(Q(c;a,s)/u(c)), evaluated as an Arb enclosure."""
    left = (c - abs_mean) / sd
    right = (c + abs_mean) / sd
    q_tail = normal_upper_tail(left) + normal_upper_tail(right)
    conditional_hazard = (
        normal_density(left) + normal_density(right)
    ) / (sd * q_tail)
    null_hazard = normal_density(c) / normal_upper_tail(c)
    return null_hazard - conditional_hazard


def abs_range_of_affine(
    mean: fmpq, loading: fmpq, z_lo: fmpq, z_hi: fmpq
) -> tuple[fmpq, fmpq]:
    """Exact range of |mean + loading*z| on [z_lo,z_hi]."""
    left = mean + loading * z_lo
    right = mean + loading * z_hi
    abs_left = abs(left)
    abs_right = abs(right)
    if (left <= 0 <= right) or (right <= 0 <= left):
        minimum = fmpq(0)
    else:
        minimum = min(abs_left, abs_right)
    maximum = max(abs_left, abs_right)
    return minimum, maximum


@dataclass(frozen=True)
class ModelSpec:
    key: str
    alpha: str
    weights_nonnull: tuple[str, ...]
    means: tuple[str, ...]
    loadings: tuple[str, ...]
    numerical_fdr: str
    certified_claim: str


@dataclass(frozen=True)
class CertificateSettings:
    z_denominator: int
    z_lo_numerator: int
    z_hi_numerator: int
    c_denominator: int
    c_start_numerator: int
    c_stop_numerator: int
    c_tolerance_units: int


MODEL = ModelSpec(
    key="alpha_0.05",
    alpha="0.05",
    weights_nonnull=(
        "1.035724637633261e-5",
        "9.78796812256908e-10",
        "3.0492299816422394e-28",
        "2.826653554343463e-3",
        "2.2305608519390437e-24",
    ),
    means=(
        "0",
        "0.6770732230955452",
        "1.586522735054591",
        "3.2037098102462105",
        "4.469876769506591",
        "5.053055030678688",
    ),
    loadings=(
        "0.05609416921615385",
        "-0.0830487863256059",
        "-0.1770420820546788",
        "-0.23038325840329232",
        "-0.8002685127003462",
        "-0.414555784395906",
    ),
    numerical_fdr="0.06204595215960602",
    certified_claim="0.0582252019310",
)

SETTINGS = CertificateSettings(
    z_denominator=20,
    z_lo_numerator=-100,
    z_hi_numerator=60,
    c_denominator=1000,
    c_start_numerator=1959,
    c_stop_numerator=60000,
    c_tolerance_units=2,
)

class Certificate:
    def __init__(self, model: ModelSpec, settings: CertificateSettings) -> None:
        self.model = model
        self.settings = settings
        self.alpha_q = decimal_rational(model.alpha)
        self.alpha = arb(self.alpha_q)
        self.means_q = tuple(decimal_rational(x) for x in model.means)
        self.loadings_q = tuple(decimal_rational(x) for x in model.loadings)
        nonnull_q = tuple(decimal_rational(x) for x in model.weights_nonnull)
        self.weights_q = (fmpq(1) - sum(nonnull_q, fmpq(0)),) + nonnull_q
        self.sds = tuple(
            (arb(1) - arb(rho) * arb(rho)).sqrt() for rho in self.loadings_q
        )
        self.prefix_nodes = 0
        self.interval_evaluations = 0
        self.qmin_nodes = 0
        self.maximum_b_index = settings.c_start_numerator

        if not (
            len(self.weights_q) == len(self.means_q) == len(self.loadings_q)
        ):
            raise ValueError("Inconsistent model dimensions")
        if not all(w > 0 for w in self.weights_q):
            raise ValueError("Every exact block weight must be positive")
        if sum(self.weights_q, fmpq(0)) != 1:
            raise ValueError("Exact block weights do not sum to one")

        c_start = arb(
            fmpq(settings.c_start_numerator, settings.c_denominator)
        )
        if not (p_threshold(c_start) > self.alpha):
            raise AssertionError("The rational c-grid must start below c_alpha")

    def _component_upper_on_interval(
        self, c_lo_q: fmpq, c_hi_q: fmpq, abs_mean_q: fmpq, sd: arb
    ) -> arb:
        c_ball = arb_interval(c_lo_q, c_hi_q)
        abs_mean = arb(abs_mean_q)
        derivative = tail_ratio_log_derivative(c_ball, abs_mean, sd)
        if derivative > 0:
            value = tail_ratio(arb(c_hi_q), abs_mean, sd)
        elif derivative < 0:
            value = tail_ratio(arb(c_lo_q), abs_mean, sd)
        else:
            value = tail_ratio(c_ball, abs_mean, sd)
        return value.upper()

    def _uniformly_infeasible(
        self, c_lo_index: int, c_hi_index: int, abs_means_hi: tuple[fmpq, ...]
    ) -> bool:
        self.interval_evaluations += 1
        denominator = self.settings.c_denominator
        c_lo_q = fmpq(c_lo_index, denominator)
        c_hi_q = fmpq(c_hi_index, denominator)
        upper = arb(0)
        for weight, abs_mean, sd in zip(
            self.weights_q, abs_means_hi, self.sds
        ):
            upper += arb(weight) * self._component_upper_on_interval(
                c_lo_q, c_hi_q, abs_mean, sd
            )
        return self.alpha * upper < 1

    def _first_unverified_index(
        self,
        c_lo_index: int,
        c_hi_index: int,
        abs_means_hi: tuple[fmpq, ...],
    ) -> int:
        """Left-to-right branch-and-bound for the certified infeasible prefix."""
        self.prefix_nodes += 1
        if self._uniformly_infeasible(c_lo_index, c_hi_index, abs_means_hi):
            return c_hi_index
        if c_hi_index - c_lo_index <= self.settings.c_tolerance_units:
            return c_lo_index
        midpoint = (c_lo_index + c_hi_index) // 2
        left_result = self._first_unverified_index(
            c_lo_index, midpoint, abs_means_hi
        )
        if left_result < midpoint:
            return left_result
        return self._first_unverified_index(midpoint, c_hi_index, abs_means_hi)

    def _uniformly_feasible_at_point(
        self, c_index: int, abs_means_lo: tuple[fmpq, ...]
    ) -> bool:
        self.interval_evaluations += 1
        c = arb(fmpq(c_index, self.settings.c_denominator))
        lower = arb(0)
        for weight, abs_mean, sd in zip(
            self.weights_q, abs_means_lo, self.sds
        ):
            lower += arb(weight) * tail_ratio(c, arb(abs_mean), sd)
        return self.alpha * lower > 1

    def _find_feasible_index(
        self, a_index: int, abs_means_lo: tuple[fmpq, ...]
    ) -> int:
        """Find and refine any common feasible point to the right of a_index."""
        if self._uniformly_feasible_at_point(a_index, abs_means_lo):
            return a_index

        step = self.settings.c_tolerance_units
        b_index = a_index + step
        found = False
        for _ in range(40):
            if b_index > self.settings.c_stop_numerator:
                break
            if self._uniformly_feasible_at_point(b_index, abs_means_lo):
                found = True
                break
            step *= 2
            b_index = a_index + step

        if not found:
            # A deterministic exact-grid fallback.  It is only a search for a
            # witness; every accepted point still passes the Arb sign test.
            step = max(
                self.settings.c_tolerance_units,
                self.settings.c_denominator // 100,
            )
            b_index = a_index + step
            while b_index <= self.settings.c_stop_numerator:
                if self._uniformly_feasible_at_point(b_index, abs_means_lo):
                    found = True
                    break
                b_index += step

        if not found:
            raise RuntimeError("No common feasible point found in this z-bin")

        lo = a_index
        hi = b_index
        while hi - lo > self.settings.c_tolerance_units:
            midpoint = (lo + hi) // 2
            if self._uniformly_feasible_at_point(midpoint, abs_means_lo):
                hi = midpoint
            else:
                lo = midpoint
        return hi

    def _tail_ratio_infimum_lower_bound(
        self, c_lo_index: int, c_hi_index: int, null_abs_mean_lo: fmpq
    ) -> arb:
        """Outward-rounded lower bound for inf q_0 on a c-interval."""
        self.qmin_nodes += 1
        denominator = self.settings.c_denominator
        c_lo_q = fmpq(c_lo_index, denominator)
        c_hi_q = fmpq(c_hi_index, denominator)
        c_ball = arb_interval(c_lo_q, c_hi_q)
        abs_mean = arb(null_abs_mean_lo)
        sd = self.sds[0]
        derivative = tail_ratio_log_derivative(c_ball, abs_mean, sd)

        if derivative > 0:
            return tail_ratio(arb(c_lo_q), abs_mean, sd).lower()
        if derivative < 0:
            return tail_ratio(arb(c_hi_q), abs_mean, sd).lower()
        if c_hi_index - c_lo_index <= self.settings.c_tolerance_units:
            return tail_ratio(c_ball, abs_mean, sd).lower()

        midpoint = (c_lo_index + c_hi_index) // 2
        left = self._tail_ratio_infimum_lower_bound(
            c_lo_index, midpoint, null_abs_mean_lo
        )
        right = self._tail_ratio_infimum_lower_bound(
            midpoint, c_hi_index, null_abs_mean_lo
        )
        return left if left < right else right

    def certify(self) -> dict[str, object]:
        start_time = time.time()
        settings = self.settings
        total_lower = arb(0)
        grouped_lower: dict[str, arb] = {}
        bin_records: list[dict[str, str]] = []

        for z_index in range(
            settings.z_lo_numerator, settings.z_hi_numerator
        ):
            z_lo_q = fmpq(z_index, settings.z_denominator)
            z_hi_q = fmpq(z_index + 1, settings.z_denominator)
            ranges = tuple(
                abs_range_of_affine(mean, loading, z_lo_q, z_hi_q)
                for mean, loading in zip(self.means_q, self.loadings_q)
            )
            abs_means_lo = tuple(x[0] for x in ranges)
            abs_means_hi = tuple(x[1] for x in ranges)

            a_index = self._first_unverified_index(
                settings.c_start_numerator,
                settings.c_stop_numerator,
                abs_means_hi,
            )
            a = arb(fmpq(a_index, settings.c_denominator))
            if not (p_threshold(a) < self.alpha):
                raise AssertionError("Certified prefix did not reach c_alpha")

            b_index = self._find_feasible_index(a_index, abs_means_lo)
            self.maximum_b_index = max(self.maximum_b_index, b_index)

            q0_inf_lower = self._tail_ratio_infimum_lower_bound(
                a_index, b_index, abs_means_lo[0]
            )
            d_lower = (
                self.alpha * arb(self.weights_q[0]) * q0_inf_lower
            ).lower()
            gaussian_mass = (
                normal_upper_tail(arb(z_lo_q))
                - normal_upper_tail(arb(z_hi_q))
            ).lower()
            contribution_lower = (d_lower * gaussian_mass).lower()
            total_lower = (total_lower + contribution_lower).lower()

            group_left = z_index // settings.z_denominator
            group_right = group_left + 1
            label = f"[{group_left},{group_right}]"
            grouped_lower[label] = (
                grouped_lower.get(label, arb(0)) + contribution_lower
            ).lower()

            bin_records.append(
                {
                    "z_lo": str(z_lo_q),
                    "z_hi": str(z_hi_q),
                    "a_index": str(a_index),
                    "b_index": str(b_index),
                    "a": str(fmpq(a_index, settings.c_denominator)),
                    "b": str(fmpq(b_index, settings.c_denominator)),
                    "fdp_lower_ball": str(d_lower),
                    "contribution_lower_ball": str(contribution_lower),
                }
            )

        claim = arb(decimal_rational(self.model.certified_claim))
        if not (total_lower > claim):
            raise AssertionError(
                f"Certificate did not prove the advertised bound {claim}"
            )
        if not (total_lower > self.alpha):
            raise AssertionError("Certificate did not prove FDR > alpha")

        elapsed = time.time() - start_time
        result: dict[str, object] = {
            "model_key": self.model.key,
            "alpha": self.model.alpha,
            "number_of_blocks": len(self.weights_q),
            "exact_null_weight": str(self.weights_q[0]),
            "exact_nonnull_weights": list(self.model.weights_nonnull),
            "exact_means": list(self.model.means),
            "exact_loadings": list(self.model.loadings),
            "numerical_fdr_estimate": self.model.numerical_fdr,
            "advertised_certified_strict_lower_bound": self.model.certified_claim,
            "computed_total_lower_ball": str(total_lower),
            "certified_factor_range": [
                str(fmpq(settings.z_lo_numerator, settings.z_denominator)),
                str(fmpq(settings.z_hi_numerator, settings.z_denominator)),
            ],
            "z_mesh": str(fmpq(1, settings.z_denominator)),
            "c_grid_unit": str(fmpq(1, settings.c_denominator)),
            "c_terminal_cell_width": str(
                fmpq(settings.c_tolerance_units, settings.c_denominator)
            ),
            "c_search_range": [
                str(
                    fmpq(
                        settings.c_start_numerator, settings.c_denominator
                    )
                ),
                str(
                    fmpq(settings.c_stop_numerator, settings.c_denominator)
                ),
            ],
            "maximum_feasible_witness_b": str(
                fmpq(self.maximum_b_index, settings.c_denominator)
            ),
            "grouped_contribution_lower_balls": {
                key: str(value) for key, value in sorted(grouped_lower.items())
            },
            "prefix_branch_nodes": self.prefix_nodes,
            "qmin_branch_nodes": self.qmin_nodes,
            "interval_sign_evaluations": self.interval_evaluations,
            "elapsed_seconds": elapsed,
            "bins": bin_records,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        choices=("0.05",),
        default="0.05",
        help="fixed at 0.05 for this certificate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arb_certificate_results.json"),
    )
    args = parser.parse_args()

    certificate = Certificate(MODEL, SETTINGS)
    result = certificate.certify()
    print(
        "CERTIFIED:",
        f"alpha={result['alpha']}",
        f"liminf FDR > {result['advertised_certified_strict_lower_bound']}",
        f"> alpha; computed lower ball {result['computed_total_lower_ball']}",
        flush=True,
    )
    print(
        "  factor range",
        result["certified_factor_range"],
        "z mesh",
        result["z_mesh"],
        "c terminal width",
        result["c_terminal_cell_width"],
        "maximum b",
        result["maximum_feasible_witness_b"],
        flush=True,
    )
    for interval, value in result[
        "grouped_contribution_lower_balls"
    ].items():
        print(f"  z in {interval}: {value}")

    payload = {
        "software": {
            "python_flint": "0.8.0",
            "arb_decimal_digits": ctx.dps,
        },
        "results": [result],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
