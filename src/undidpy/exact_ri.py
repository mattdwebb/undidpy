#!/usr/bin/env python3
"""Exact and smoothed randomization-inference helpers.

These helpers are intentionally separate from the published Monte Carlo runner.
They support focused diagnostics without changing the current 399-draw RI path.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np


Assignment = Dict[Any, str]


def assignment_support_size(
    r: int,
    j: int,
    l: int,
    *,
    focal_cohort: Optional[str] = None,
) -> int:
    """Return the number of staggered treatment assignments.

    If ``focal_cohort`` is ``"early"`` or ``"late"``, one focal state is held in
    that observed cohort and only the remaining labels are assigned.
    """
    if r < 1 or j < 0 or l < 0 or j + l > r:
        raise ValueError("Require r >= 1, j >= 0, l >= 0, and j + l <= r.")
    if focal_cohort is None:
        return math.factorial(r) // (
            math.factorial(j) * math.factorial(l) * math.factorial(r - j - l)
        )
    if focal_cohort == "early":
        if j < 1:
            raise ValueError("An early focal state requires j >= 1.")
        return assignment_support_size(r - 1, j - 1, l)
    if focal_cohort == "late":
        if l < 1:
            raise ValueError("A late focal state requires l >= 1.")
        return assignment_support_size(r - 1, j, l - 1)
    raise ValueError("focal_cohort must be None, 'early', or 'late'.")


def assignment_key(states: Sequence[Any], assignment: Mapping[Any, str]) -> tuple[str, ...]:
    return tuple(str(assignment.get(s, "0")) for s in states)


def enumerate_assignments(
    states: Sequence[Any],
    *,
    early_count: int,
    late_count: int,
    focal_state: Optional[Any] = None,
    focal_cohort: Optional[str] = None,
    exclude_assignment: Optional[Mapping[Any, str]] = None,
) -> Iterator[Assignment]:
    """Yield each admissible early/late/untreated assignment exactly once.

    Labels are strings: ``"early"``, ``"late"``, and ``"0"``.  When a focal
    state/cohort is supplied, that focal state is fixed in its observed cohort.
    """
    state_list = list(states)
    if len(set(state_list)) != len(state_list):
        raise ValueError("states must be unique.")
    if early_count < 0 or late_count < 0 or early_count + late_count > len(state_list):
        raise ValueError("Invalid early/late counts for state list.")
    if (focal_state is None) != (focal_cohort is None):
        raise ValueError("focal_state and focal_cohort must be supplied together.")
    if focal_state is not None and focal_state not in state_list:
        raise ValueError("focal_state is not in states.")

    exclude_key = (
        assignment_key(state_list, exclude_assignment)
        if exclude_assignment is not None
        else None
    )

    base: Assignment = {s: "0" for s in state_list}
    remaining = list(state_list)
    choose_early = early_count
    choose_late = late_count

    if focal_cohort == "early":
        if early_count < 1:
            raise ValueError("An early focal state requires early_count >= 1.")
        base[focal_state] = "early"
        remaining.remove(focal_state)
        choose_early -= 1
    elif focal_cohort == "late":
        if late_count < 1:
            raise ValueError("A late focal state requires late_count >= 1.")
        base[focal_state] = "late"
        remaining.remove(focal_state)
        choose_late -= 1
    elif focal_cohort is not None:
        raise ValueError("focal_cohort must be None, 'early', or 'late'.")

    for early_extra in combinations(remaining, choose_early):
        early_set = set(early_extra)
        late_pool = [s for s in remaining if s not in early_set]
        for late_extra in combinations(late_pool, choose_late):
            assignment = dict(base)
            for s in early_extra:
                assignment[s] = "early"
            for s in late_extra:
                assignment[s] = "late"
            if exclude_key is not None and assignment_key(state_list, assignment) == exclude_key:
                continue
            yield assignment


def target_defined_for_state(
    assignment: Mapping[Any, str],
    focal_state: Any,
    *,
    target: str,
    target_period: Optional[float] = None,
    early_time: float = 5.0,
    late_time: float = 7.0,
) -> bool:
    """Return whether a fixed focal-state target is defined.

    ``target`` is ``"att_s"`` or ``"att_st"``.  For ``att_s``, the focal state
    must be treated in either cohort.  For ``att_st``, the focal state must be
    treated by the retained target period.
    """
    label = str(assignment.get(focal_state, "0"))
    if target == "att_s":
        return label in {"early", "late"}
    if target != "att_st":
        raise ValueError("target must be 'att_s' or 'att_st'.")
    if target_period is None:
        raise ValueError("target_period is required for att_st.")
    if label == "early":
        return float(target_period) >= float(early_time)
    if label == "late":
        return float(target_period) >= float(late_time)
    return False


def enumerate_target_defined_assignments(
    states: Sequence[Any],
    *,
    early_count: int,
    late_count: int,
    focal_state: Any,
    target: str,
    target_period: Optional[float] = None,
    early_time: float = 5.0,
    late_time: float = 7.0,
) -> Iterator[Assignment]:
    """Yield full-support assignments where the fixed target is defined."""
    for assignment in enumerate_assignments(
        states,
        early_count=early_count,
        late_count=late_count,
    ):
        if target_defined_for_state(
            assignment,
            focal_state,
            target=target,
            target_period=target_period,
            early_time=early_time,
            late_time=late_time,
        ):
            yield assignment


def enumerated_target_defined_support_size(
    states: Sequence[Any],
    *,
    early_count: int,
    late_count: int,
    focal_state: Any,
    target: str,
    target_period: Optional[float] = None,
    early_time: float = 5.0,
    late_time: float = 7.0,
) -> int:
    """Count the target-defined conditional support by enumeration."""
    return sum(
        1
        for _ in enumerate_target_defined_assignments(
            states,
            early_count=early_count,
            late_count=late_count,
            focal_state=focal_state,
            target=target,
            target_period=target_period,
            early_time=early_time,
            late_time=late_time,
        )
    )


def target_defined_support_size(
    states: Sequence[Any],
    *,
    early_count: int,
    late_count: int,
    focal_state: Any,
    target: str,
    target_period: Optional[float] = None,
    early_time: float = 5.0,
    late_time: float = 7.0,
) -> int:
    """Count target-defined support without enumerating every assignment."""
    n = len(states)
    if focal_state not in states:
        raise ValueError("focal_state is not in states.")
    support = 0
    if target == "att_s" or (target == "att_st" and target_period is not None and float(target_period) >= float(early_time)):
        support += assignment_support_size(n - 1, early_count - 1, late_count) if early_count > 0 else 0
    if target == "att_s" or (target == "att_st" and target_period is not None and float(target_period) >= float(late_time)):
        support += assignment_support_size(n - 1, early_count, late_count - 1) if late_count > 0 else 0
    if target == "att_st" and target_period is None:
        raise ValueError("target_period is required for att_st.")
    if target not in {"att_s", "att_st"}:
        raise ValueError("target must be 'att_s' or 'att_st'.")
    return support


def exact_ri_pvalue(
    observed_stat: float,
    placebo_stats: Iterable[float],
    *,
    two_sided: bool = True,
) -> float:
    """Inclusive RI p-value with the observed assignment added once.

    ``placebo_stats`` should exclude the observed assignment.  The formula is
    ``(1 + hits) / (M + 1)``.
    """
    if not np.isfinite(observed_stat):
        return float("nan")
    vals = np.asarray([v for v in placebo_stats if np.isfinite(v)], dtype=float)
    if vals.size < 1:
        return float("nan")
    obs = abs(float(observed_stat)) if two_sided else float(observed_stat)
    comp = np.abs(vals) if two_sided else vals
    hits = int(np.sum(comp >= obs))
    return float((1 + hits) / (vals.size + 1))


def exact_pvalue_grid(k: int) -> List[float]:
    if k < 1:
        raise ValueError("k must be positive.")
    return [i / k for i in range(1, k + 1)]


def kernel_bandwidth(
    values: Iterable[float],
    *,
    coefficient: float = 1.575,
    exponent: float = -4.0 / 9.0,
    scale: float = 1.0,
) -> Dict[str, Any]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    m = int(vals.size)
    if m < 2:
        return {"bandwidth": np.nan, "sd": np.nan, "m": m, "flag": "too_few_values"}
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return {"bandwidth": np.nan, "sd": sd, "m": m, "flag": "degenerate_sd"}
    bandwidth = float(scale * coefficient * (m ** exponent) * sd)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return {"bandwidth": np.nan, "sd": sd, "m": m, "flag": "invalid_bandwidth"}
    return {"bandwidth": bandwidth, "sd": sd, "m": m, "flag": "ok"}


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def kernel_smoothed_cdf(
    value: float,
    support_values: Iterable[float],
    *,
    coefficient: float = 1.575,
    exponent: float = -4.0 / 9.0,
    bandwidth_scale: float = 1.0,
) -> Dict[str, Any]:
    vals = np.asarray([v for v in support_values if np.isfinite(v)], dtype=float)
    bw = kernel_bandwidth(
        vals,
        coefficient=coefficient,
        exponent=exponent,
        scale=bandwidth_scale,
    )
    h = bw["bandwidth"]
    if bw["flag"] != "ok":
        return {"cdf": np.nan, **bw}
    cdf = float(np.mean([normal_cdf((float(value) - v) / h) for v in vals]))
    return {"cdf": cdf, **bw}


def kernel_smoothed_ri_pvalue(
    observed_stat: float,
    placebo_stats: Iterable[float],
    *,
    coefficient: float = 1.575,
    exponent: float = -4.0 / 9.0,
    bandwidth_scale: float = 1.0,
) -> Dict[str, Any]:
    """Racine-MacKinnon smoothed two-sided RI p-value.

    ``placebo_stats`` are the S alternative re-randomized statistics and
    should exclude the observed assignment.  For two-sided tests, the
    bandwidth scale ``s`` is estimated from raw, unsigned ``T_j^*`` values, but
    the kernel comparison is applied to ``|T_obs|`` and ``|T_j^*|``:

        p_h = 1 - S^{-1} sum_j Phi((|T_obs|-|T_j^*|)/h),

    with h = 1.575 * sd(T_j^*) * S^{-4/9} for five percent tests.
    """
    if not np.isfinite(observed_stat):
        return {
            "p_value": np.nan,
            "cdf": np.nan,
            "bandwidth": np.nan,
            "sd": np.nan,
            "m": 0,
            "flag": "invalid_observed",
        }
    vals = np.asarray([v for v in placebo_stats if np.isfinite(v)], dtype=float)
    bw = kernel_bandwidth(
        vals,
        coefficient=coefficient,
        exponent=exponent,
        scale=bandwidth_scale,
    )
    h = bw["bandwidth"]
    if bw["flag"] != "ok":
        return {"p_value": np.nan, "cdf": np.nan, **bw}
    obs_abs = abs(float(observed_stat))
    comp = np.abs(vals)
    cdf = float(np.mean([normal_cdf((obs_abs - v) / h) for v in comp]))
    p_val = float(1.0 - cdf)
    return {"p_value": p_val, "cdf": cdf, **bw}
