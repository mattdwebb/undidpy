"""Canonical ATT(s,t)-first aggregation for DID-INT/UN-DID outputs.

The primitive input is a supported state-time treatment-effect table. Higher
targets are explicit weighted averages of that table; no aggregate-level
pooled-treated-state calculation is performed here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


LEVEL_COLUMNS = [
    "estimate",
    "se",
    "p_value",
    "weight",
    "n_treated",
    "n_control",
]
POLICY_EXTRA_COLUMNS = ["support_states", "support_state_time_cells"]


def _two_sided_normal_pvalue(z_score: float) -> float:
    if not np.isfinite(z_score):
        return np.nan
    return float(math.erfc(abs(float(z_score)) / math.sqrt(2.0)))


def weighted_mean_and_se(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.nan, np.nan
    total = float(weights.sum())
    if total <= 0:
        return np.nan, np.nan
    estimate = float(np.sum(values * weights) / total)
    if values.size < 2:
        return estimate, np.nan
    centered = values - estimate
    var_mu = float(np.sum((weights**2) * (centered**2)) / (total**2))
    var_mu *= values.size / (values.size - 1.0)
    return estimate, float(math.sqrt(max(var_mu, 0.0)))


def _empty(group_cols: Iterable[str], policy: bool = False) -> pd.DataFrame:
    cols = list(group_cols) + LEVEL_COLUMNS
    if policy:
        cols += POLICY_EXTRA_COLUMNS
    return pd.DataFrame(columns=cols)


def _weight_vector(block: pd.DataFrame, column: str) -> np.ndarray:
    if column in block.columns:
        weights = pd.to_numeric(block[column], errors="coerce").to_numpy(dtype=float)
    else:
        weights = np.ones(len(block), dtype=float)
    return weights


def _aggregate_block(
    block: pd.DataFrame,
    keys: Dict[str, Any],
    *,
    estimate_col: str = "estimate",
    average_weight_col: str,
    output_weight_col: str,
    policy: bool = False,
) -> Dict[str, Any]:
    values = pd.to_numeric(block[estimate_col], errors="coerce").to_numpy(dtype=float)
    avg_weights = _weight_vector(block, average_weight_col)
    estimate, se = weighted_mean_and_se(values, avg_weights)
    p_value = _two_sided_normal_pvalue(estimate / se) if np.isfinite(se) and se > 0 else np.nan
    output_weights = _weight_vector(block, output_weight_col)
    active = np.isfinite(values) & np.isfinite(avg_weights) & (avg_weights > 0)
    output_active = np.isfinite(output_weights) & (output_weights > 0) & active

    rec: Dict[str, Any] = dict(keys)
    rec.update(
        {
            "estimate": float(estimate),
            "se": float(se),
            "p_value": float(p_value),
            "weight": float(np.sum(output_weights[output_active])) if output_active.any() else 0.0,
            "n_treated": float(active.sum()),
            "n_control": np.nan,
        }
    )
    if policy:
        rec["support_states"] = float(block.loc[active, "state"].nunique()) if "state" in block.columns else np.nan
        rec["support_state_time_cells"] = float(active.sum())
    return rec


def _aggregate_frame(
    frame: pd.DataFrame,
    group_cols: List[str],
    *,
    average_weight_col: str,
    output_weight_col: str = "target_weight",
    estimate_col: str = "estimate",
    policy: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return _empty(group_cols, policy=policy)
    records: List[Dict[str, Any]] = []
    grouped = frame.groupby(group_cols, sort=True, dropna=False) if group_cols else [((), frame)]
    for key, block in grouped:
        keys: Dict[str, Any] = {}
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            keys.update(dict(zip(group_cols, key_tuple)))
        records.append(
            _aggregate_block(
                block,
                keys,
                estimate_col=estimate_col,
                average_weight_col=average_weight_col,
                output_weight_col=output_weight_col,
                policy=policy,
            )
        )
    return pd.DataFrame.from_records(records) if records else _empty(group_cols, policy=policy)


def prepare_att_st_table(att_st: pd.DataFrame, *, weighting: str = "both") -> pd.DataFrame:
    """Normalize metadata columns on a state-time primitive table."""
    out = att_st.copy()
    if out.empty:
        return out
    if "ATT_st" not in out.columns and "estimate" in out.columns:
        out["ATT_st"] = out["estimate"]
    if "target_weight" not in out.columns:
        out["target_weight"] = pd.to_numeric(out.get("weight", 1.0), errors="coerce")
    if "contrast_weight" not in out.columns:
        if "w_diff_target" in out.columns:
            out["contrast_weight"] = pd.to_numeric(out["w_diff_target"], errors="coerce")
        elif weighting in {"both", "diff"} and {"treated_target_N", "base_target_N"}.issubset(out.columns):
            out["contrast_weight"] = (
                pd.to_numeric(out["treated_target_N"], errors="coerce")
                + pd.to_numeric(out["base_target_N"], errors="coerce")
            )
        else:
            out["contrast_weight"] = 1.0
    if "weight" not in out.columns:
        out["weight"] = out["target_weight"]
    if "support" not in out.columns:
        out["support"] = True
    return out


def aggregate_from_att_st(
    att_st: pd.DataFrame,
    *,
    weighting: str = "both",
    include_policy: bool = True,
    cross_state_weight: str = "target",
    include_aggregation_weights: bool = True,
) -> Dict[str, Any]:
    """Aggregate all DID-INT targets from the supported ATT(s,t) table."""
    if cross_state_weight not in {"target", "contrast"}:
        raise ValueError("cross_state_weight must be 'target' or 'contrast'.")
    st = prepare_att_st_table(att_st, weighting=weighting)
    if st.empty:
        empty = _empty([])
        return {
            "att": {
                "estimate": np.nan,
                "se": np.nan,
                "p_value": np.nan,
                "weight": np.nan,
                "n_treated": np.nan,
                "n_control": np.nan,
            },
            "att_g": _empty(["g"]),
            "att_gt": _empty(["g", "t"]),
            "att_s": _empty(["state", "g"]),
            "att_st": st,
            "att_p": _empty(["pvar"], policy=True),
            "att_pt": _empty(["pvar", "t"], policy=True),
            "aggregation_weights": pd.DataFrame(),
        }

    avg_target_weight = "target_weight" if weighting in {"both", "att"} else "__equal_weight"
    avg_contrast_weight = "contrast_weight" if weighting in {"both", "diff"} else "__equal_weight"
    avg_cross_state_weight = (
        avg_target_weight if cross_state_weight == "target" else avg_contrast_weight
    )
    st["__equal_weight"] = 1.0

    att_s = _aggregate_frame(
        st,
        ["state", "g"],
        average_weight_col=avg_target_weight,
        output_weight_col="target_weight",
    )
    att_gt = _aggregate_frame(
        st,
        ["g", "t"],
        average_weight_col=avg_cross_state_weight,
        output_weight_col="target_weight",
    )
    att_g = _aggregate_frame(
        att_gt,
        ["g"],
        average_weight_col="weight" if weighting in {"both", "att"} else "__equal_weight",
        output_weight_col="weight",
    )
    if not att_g.empty:
        att_g["__equal_weight"] = 1.0
    att_frame = _aggregate_frame(
        att_g,
        [],
        average_weight_col="weight" if weighting in {"both", "att"} else "__equal_weight",
        output_weight_col="weight",
    )
    att = att_frame.iloc[0].to_dict() if not att_frame.empty else {
        "estimate": np.nan,
        "se": np.nan,
        "p_value": np.nan,
        "weight": np.nan,
        "n_treated": np.nan,
        "n_control": np.nan,
    }

    if include_policy and "pvar" in st.columns:
        policy_st = st.loc[st["pvar"].fillna("0").astype(str) != "0"].copy()
    else:
        policy_st = pd.DataFrame()
    if policy_st.empty:
        att_p = _empty(["pvar"], policy=True)
        att_pt = _empty(["pvar", "t"], policy=True)
    else:
        # Preserve the Which Policy implementation choice: policy targets average
        # active state-time effects using the target/post-period mass.
        att_pt = _aggregate_frame(
            policy_st,
            ["pvar", "t"],
            average_weight_col=avg_target_weight,
            output_weight_col="target_weight",
            policy=True,
        )
        att_p = _aggregate_frame(
            policy_st,
            ["pvar"],
            average_weight_col=avg_target_weight,
            output_weight_col="target_weight",
            policy=True,
        )

    weights = (
        aggregation_identity_weights(
            st, weighting=weighting, cross_state_weight=cross_state_weight
        )
        if include_aggregation_weights
        else pd.DataFrame()
    )
    return {
        "att": att,
        "att_g": att_g.drop(columns=["__equal_weight"], errors="ignore"),
        "att_gt": att_gt,
        "att_s": att_s,
        "att_st": st.drop(columns=["__equal_weight"], errors="ignore"),
        "att_p": att_p,
        "att_pt": att_pt,
        "aggregation_weights": weights,
    }


def aggregation_identity_weights(
    att_st: pd.DataFrame,
    *,
    weighting: str = "both",
    cross_state_weight: str = "target",
) -> pd.DataFrame:
    """Return normalized ATT(s,t)-row weights used by each supported target."""
    if cross_state_weight not in {"target", "contrast"}:
        raise ValueError("cross_state_weight must be 'target' or 'contrast'.")
    st = prepare_att_st_table(att_st, weighting=weighting)
    if st.empty:
        return pd.DataFrame(columns=["target", "key", "state", "g", "t", "normalized_weight"])
    avg_target = "target_weight" if weighting in {"both", "att"} else None
    avg_contrast = "contrast_weight" if weighting in {"both", "diff"} else None
    avg_cross_state = avg_target if cross_state_weight == "target" else avg_contrast
    records: List[Dict[str, Any]] = []

    def add(block: pd.DataFrame, target: str, key: str, weight_col: Optional[str]) -> None:
        raw = np.ones(len(block), dtype=float) if weight_col is None else pd.to_numeric(block[weight_col], errors="coerce").to_numpy(float)
        mask = np.isfinite(raw) & (raw > 0)
        denom = float(raw[mask].sum())
        if denom <= 0:
            return
        for idx, row in enumerate(block.itertuples(index=False)):
            if not mask[idx]:
                continue
            records.append(
                {
                    "target": target,
                    "key": key,
                    "state": getattr(row, "state"),
                    "g": getattr(row, "g"),
                    "t": getattr(row, "t"),
                    "normalized_weight": float(raw[idx] / denom),
                }
            )

    for (state, g), block in st.groupby(["state", "g"], sort=True):
        add(block, "att_s", f"{state}|{g}", avg_target)
    for (g, t), block in st.groupby(["g", "t"], sort=True):
        add(block, "att_gt", f"{g}|{t}", avg_cross_state)
    if "pvar" in st.columns:
        policy = st.loc[st["pvar"].fillna("0").astype(str) != "0"]
        for (pvar, t), block in policy.groupby(["pvar", "t"], sort=True):
            add(block, "att_pt", f"{pvar}|{t}", avg_target)
        for pvar, block in policy.groupby("pvar", sort=True):
            add(block, "att_p", str(pvar), avg_target)

    # The overall ATT implied by the hierarchy: cross-state weights within (g,t),
    # target weights across (g,t), and target weights across cohorts.
    gt = []
    for (g, t), block in st.groupby(["g", "t"], sort=True):
        raw_inner = (
            np.ones(len(block), dtype=float)
            if avg_cross_state is None
            else pd.to_numeric(block[avg_cross_state], errors="coerce").to_numpy(float)
        )
        inner_mask = np.isfinite(raw_inner) & (raw_inner > 0)
        inner_sum = float(raw_inner[inner_mask].sum())
        if inner_sum <= 0:
            continue
        gt_weight = float(pd.to_numeric(block["target_weight"], errors="coerce").sum()) if weighting in {"both", "att"} else 1.0
        for idx, row in enumerate(block.itertuples(index=False)):
            if inner_mask[idx]:
                gt.append((g, t, row.state, float(raw_inner[idx] / inner_sum), gt_weight))
    if weighting in {"both", "att"}:
        gt_cell_weights = {}
        for g, t, _state, _inner_w, gt_weight in gt:
            gt_cell_weights[(g, t)] = gt_weight
        group_weight = {}
        for (g, _t), gt_weight in gt_cell_weights.items():
            group_weight[g] = group_weight.get(g, 0.0) + gt_weight
        total_group_weight = sum(group_weight.values())
    else:
        groups = sorted({x[0] for x in gt})
        group_weight = {g: 1.0 for g in groups}
        total_group_weight = float(len(groups))
        gt_counts = {g: len({x[1] for x in gt if x[0] == g}) for g in groups}
    if total_group_weight > 0:
        for g, t, state, inner_w, gt_weight in gt:
            if group_weight[g] <= 0:
                continue
            within_gt_weight = (
                gt_weight / group_weight[g]
                if weighting in {"both", "att"}
                else 1.0 / max(gt_counts[g], 1)
            )
            records.append(
                {
                    "target": "att",
                    "key": "overall",
                    "state": state,
                    "g": g,
                    "t": t,
                    "normalized_weight": float(
                        inner_w
                        * within_gt_weight
                        * (group_weight[g] / total_group_weight)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def identity_diagnostics(result: Dict[str, Any], *, tolerance: float = 1e-10) -> pd.DataFrame:
    """Check that aggregate estimates equal weighted sums of ATT(s,t) rows."""
    st = result.get("att_st", pd.DataFrame())
    weights = result.get("aggregation_weights", pd.DataFrame())
    if st.empty or weights.empty:
        return pd.DataFrame(columns=["target", "key", "estimate", "reconstructed", "abs_diff", "weight_sum", "passes"])
    st_values = st.set_index(["state", "g", "t"])["estimate"].to_dict()
    rows: List[Dict[str, Any]] = []
    target_frames = {
        "att_s": result.get("att_s", pd.DataFrame()),
        "att_gt": result.get("att_gt", pd.DataFrame()),
        "att_p": result.get("att_p", pd.DataFrame()),
        "att_pt": result.get("att_pt", pd.DataFrame()),
    }

    def target_estimate(target: str, key: str) -> float:
        if target == "att":
            return float(result.get("att", {}).get("estimate", np.nan))
        frame = target_frames.get(target, pd.DataFrame())
        if frame.empty:
            return np.nan
        parts = key.split("|")
        if target == "att_s":
            match = frame[(frame["state"].astype(str) == parts[0]) & (frame["g"].astype(str) == parts[1])]
        elif target == "att_gt":
            match = frame[(frame["g"].astype(str) == parts[0]) & (frame["t"].astype(str) == parts[1])]
        elif target == "att_pt":
            match = frame[(frame["pvar"].astype(str) == parts[0]) & (frame["t"].astype(str) == parts[1])]
        elif target == "att_p":
            match = frame[frame["pvar"].astype(str) == key]
        else:
            match = pd.DataFrame()
        return float(match["estimate"].iloc[0]) if not match.empty else np.nan

    for (target, key), block in weights.groupby(["target", "key"], sort=True):
        reconstructed = 0.0
        weight_sum = float(block["normalized_weight"].sum())
        for row in block.itertuples(index=False):
            reconstructed += float(row.normalized_weight) * float(st_values[(row.state, row.g, row.t)])
        estimate = target_estimate(target, key)
        abs_diff = abs(estimate - reconstructed) if np.isfinite(estimate) else np.nan
        rows.append(
            {
                "target": target,
                "key": key,
                "estimate": estimate,
                "reconstructed": float(reconstructed),
                "abs_diff": float(abs_diff),
                "weight_sum": weight_sum,
                "passes": bool(np.isfinite(abs_diff) and abs_diff < tolerance and abs(weight_sum - 1.0) < tolerance),
            }
        )
    return pd.DataFrame.from_records(rows)
