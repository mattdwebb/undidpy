"""
Pooled UNDiD estimator with ATT / ATTg / ATTgt outputs and three inference modes:
- Conventional (analytic weighted contrast SEs)
- Cluster jackknife (leave-one-state-out)
- Randomization inference (state-level cohort-label permutation)
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .canonical_aggregation import aggregate_from_att_st
    from .exact_ri import kernel_smoothed_ri_pvalue
except ImportError:
    from canonical_aggregation import aggregate_from_att_st
    from exact_ri import kernel_smoothed_ri_pvalue

from ._probability import student_t
from ._inference import InferenceError, strict_lstsq


VALID_WEIGHTING = {"none", "diff", "att", "both"}


def _parse_pg_label(value: Any) -> Tuple[float, str]:
    if pd.isna(value):
        return 0.0, "0"
    text = str(value).strip()
    if text in {"", "0", "0.0", "nan", "None"}:
        return 0.0, "0"
    match = re.fullmatch(r"([A-Za-z]+)([-+]?\d+(?:\.\d+)?)", text)
    if match is None:
        raise ValueError(
            f"Invalid pg label '{value}'. Expected labels like A2, B4, or 0."
        )
    return float(match.group(2)), match.group(1)


def _pg_gvar(value: Any) -> float:
    return _parse_pg_label(value)[0]


def _pg_pvar(value: Any) -> str:
    return _parse_pg_label(value)[1]


def _as_generator(random_state: Optional[Any]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _to_numeric_or_raise(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        bad_count = int(out.isna().sum())
        raise ValueError(
            f"Column '{name}' must be numeric. Found {bad_count} non-numeric values."
        )
    return out


def _two_sided_pvalue(z_score: float, dof: Optional[int] = None) -> float:
    if not np.isfinite(z_score):
        return np.nan
    z_abs = abs(float(z_score))

    if dof is not None:
        if dof <= 0: raise ValueError("insufficient_clusters")
        from ._probability import student_t as t
        return float(2.0 * t.sf(z_abs, dof))

    return float(math.erfc(z_abs / math.sqrt(2.0)))


def _weighted_mean_and_se(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.nan, np.nan

    sw = float(np.sum(weights))
    if sw <= 0:
        return np.nan, np.nan

    mu = float(np.sum(weights * values) / sw)
    if values.size < 2:
        return mu, np.nan

    centered = values - mu
    var_mu = float(np.sum((weights**2) * (centered**2)) / (sw**2))
    var_mu *= values.size / (values.size - 1.0)  # HC1-style small-sample inflation
    return mu, float(math.sqrt(max(var_mu, 0.0)))


def _weighted_diff_stats(block: pd.DataFrame) -> Optional[Dict[str, float]]:
    treat = block.loc[block["treat"] == 1]
    control = block.loc[block["treat"] == 0]
    if treat.empty or control.empty:
        return None

    y = block["delta"].to_numpy(dtype=float)
    d = block["treat"].to_numpy(dtype=float)
    w = block["w_diff"].to_numpy(dtype=float)
    return _weighted_diff_stats_from_arrays(
        y=y,
        d=d,
        w=w,
        treat_w_att=treat["w_att"].to_numpy(dtype=float),
        n_treated=float(len(treat)),
        n_control=float(len(control)),
    )


def _weighted_diff_stats_from_arrays(
    *,
    y: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    treat_w_att: np.ndarray,
    n_treated: float,
    n_control: float,
) -> Optional[Dict[str, float]]:

    mask = np.isfinite(y) & np.isfinite(d) & np.isfinite(w) & (w > 0)
    y = y[mask]
    d = d[mask]
    w = w[mask]

    if y.size < 2 or d.min() == d.max():
        return None

    x = np.column_stack([np.ones(y.size), d])
    xtwx = x.T @ (x * w[:, None])

    try:
        xtwx_inv = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        return None

    beta = xtwx_inv @ (x.T @ (w * y))
    resid = y - (x @ beta)
    est = float(beta[1])

    if y.size <= x.shape[1]:
        return {
            "estimate": est,
            "se": np.nan,
            "p_value": np.nan,
            "weight": float(np.nansum(treat_w_att)),
            "n_treated": float(n_treated),
            "n_control": float(n_control),
        }

    # HC1-robust covariance for weighted least squares via transformed residuals.
    meat = x.T @ (((w * resid) ** 2)[:, None] * x)
    n_obs, k = x.shape

    cov = xtwx_inv @ meat @ xtwx_inv
    cov *= n_obs / (n_obs - k)

    se = float(math.sqrt(max(cov[1, 1], 0.0))) if np.isfinite(cov[1, 1]) else np.nan
    p_val = (
        _two_sided_pvalue(est / se, dof=n_obs - k)
        if np.isfinite(se) and se > 0
        else np.nan
    )

    return {
        "estimate": est,
        "se": se,
        "p_value": float(p_val),
        "weight": float(np.nansum(treat_w_att)),
        "n_treated": float(n_treated),
        "n_control": float(n_control),
    }


def _empty_level_frame(group_cols: Iterable[str]) -> pd.DataFrame:
    cols = list(group_cols) + [
        "estimate",
        "se",
        "p_value",
        "weight",
        "n_treated",
        "n_control",
    ]
    return pd.DataFrame(columns=cols)


def _level_stats(rows: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if rows.empty:
        return _empty_level_frame(group_cols)

    records: List[Dict[str, Any]] = []
    if group_cols:
        grouped = rows.groupby(group_cols, sort=True, dropna=False)
    else:
        grouped = [((), rows)]

    for key, block in grouped:
        stats = _weighted_diff_stats(block)
        if stats is None:
            continue

        rec: Dict[str, Any] = {}
        if group_cols:
            key_t = key if isinstance(key, tuple) else (key,)
            rec.update({col: val for col, val in zip(group_cols, key_t)})
        rec.update(stats)
        records.append(rec)

    if not records:
        return _empty_level_frame(group_cols)
    return pd.DataFrame.from_records(records)


def prepare_panel_data(
    df: pd.DataFrame,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    slope_sample: str = "full",
) -> pd.DataFrame:
    """
    Prepare pooled data for UNDiD.

    - No covariates: collapse to state-time means/counts.
    - With covariates: keep micro rows so pair-specific residualization can be
      computed for each (state, g-1, t) contrast.
    """
    required = {outcome_col, state_col, time_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    covariates = covariates or []
    missing_cov = [c for c in covariates if c not in df.columns]
    if missing_cov:
        raise ValueError(f"Missing covariate columns: {missing_cov}")

    if slope_sample not in {"full", "untreated_pre"}:
        raise ValueError("slope_sample must be one of ['full', 'untreated_pre'].")

    work_cols = [state_col, time_col, outcome_col]
    if gvar_col in df.columns:
        work_cols.append(gvar_col)
    if pg_col and pg_col in df.columns:
        work_cols.append(pg_col)
    work_cols.extend(covariates)
    work = df[work_cols].copy()

    work = work.dropna(subset=[state_col, time_col, outcome_col])
    work[time_col] = _to_numeric_or_raise(work[time_col], time_col)

    if gvar_col in work.columns:
        work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)
    else:
        work[gvar_col] = 0.0
    if pg_col:
        if pg_col in work.columns:
            work[pg_col] = work[pg_col].fillna("0").astype(str)
        else:
            work[pg_col] = "0"

    for cov in covariates:
        work[cov] = pd.to_numeric(work[cov], errors="coerce")
    if covariates:
        work = work.dropna(subset=covariates)

    if covariates:
        # Keep individual rows for pair-specific covariate adjustment in stage 2.
        out_cols = [state_col, time_col, gvar_col]
        if pg_col:
            out_cols.append(pg_col)
        out_cols = out_cols + [outcome_col] + covariates
        out = work[out_cols].copy()
        out = out.rename(columns={outcome_col: "_y_raw"})
        return out.reset_index(drop=True)

    panel = (
        work.groupby([state_col, time_col], as_index=False)
        .agg(lambda_st=(outcome_col, "mean"), n_t=(outcome_col, "size"))
        .reset_index(drop=True)
    )

    g_map = (
        work.groupby(state_col, as_index=False)[gvar_col]
        .agg(lambda x: float(x.iloc[0]) if len(x) else 0.0)
        .reset_index(drop=True)
    )
    panel = panel.merge(g_map, on=state_col, how="left")
    panel[gvar_col] = panel[gvar_col].fillna(0.0)
    out_cols = [state_col, time_col, gvar_col, "lambda_st", "n_t"]

    if pg_col:
        pg_map = (
            work.groupby(state_col, as_index=False)[pg_col]
            .agg(lambda x: str(x.iloc[0]) if len(x) else "0")
            .reset_index(drop=True)
        )
        panel = panel.merge(pg_map, on=state_col, how="left")
        panel[pg_col] = panel[pg_col].fillna("0").astype(str)
        out_cols.insert(3, pg_col)

    return panel[out_cols]


def _pair_adjusted_delta_from_arrays(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    x_pre: Optional[np.ndarray],
    x_post: Optional[np.ndarray],
    has_covariates: bool,
) -> float:
    y_pre = np.asarray(y_pre, dtype=float)
    y_post = np.asarray(y_post, dtype=float)
    if y_post.size == 0 or y_pre.size == 0:
        return np.nan
    base_diff = float(np.mean(y_post) - np.mean(y_pre))

    if not has_covariates:
        return base_diff

    if x_pre is None or x_post is None:
        raise InferenceError('invalid_input', 'requested covariates are missing')
    x_cov = np.vstack([np.asarray(x_pre, dtype=float), np.asarray(x_post, dtype=float)])
    y = np.concatenate([y_pre, y_post])
    post = np.concatenate([
        np.zeros(y_pre.size, dtype=float),
        np.ones(y_post.size, dtype=float),
    ])
    x = np.column_stack([np.ones(y.size, dtype=float), post, x_cov])
    beta = strict_lstsq(x, y)
    return float(beta[1])


def _pair_adjusted_delta(
    state_block: pd.DataFrame,
    time_col: str,
    pre_t: float,
    post_t: float,
    covariate_cols: List[str],
) -> float:
    pair = state_block.loc[state_block[time_col].isin([pre_t, post_t])].copy()
    if pair.empty:
        return np.nan

    pre_mask = pair[time_col].to_numpy(dtype=float) == float(pre_t)
    post_mask = pair[time_col].to_numpy(dtype=float) == float(post_t)
    if not pre_mask.any() or not post_mask.any():
        return np.nan
    y = pair["_y_raw"].to_numpy(dtype=float)
    x_cov = pair[covariate_cols].to_numpy(dtype=float) if covariate_cols else None
    return _pair_adjusted_delta_from_arrays(
        y_pre=y[pre_mask],
        y_post=y[post_mask],
        x_pre=x_cov[pre_mask] if x_cov is not None else None,
        x_post=x_cov[post_mask] if x_cov is not None else None,
        has_covariates=bool(covariate_cols),
    )


def _micro_contrast_cache_key(
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str],
) -> str:
    return f"_undid_micro_contrast_cache::{state_col}::{time_col}::{gvar_col}::{pg_col or ''}"


def _micro_contrast_time_lookup_key(time_col: str) -> str:
    return f"_undid_micro_contrast_time_lookup::{time_col}"


class _SharedMicroContrastCache:
    def __init__(self, cache: Dict[str, Any]) -> None:
        self.cache = cache

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_SharedMicroContrastCache":
        return self


def _state_time_arrays(
    work: pd.DataFrame,
    state_col: str,
    time_col: str,
    covariate_cols: List[str],
) -> Dict[Any, Dict[float, Tuple[np.ndarray, Optional[np.ndarray]]]]:
    out: Dict[Any, Dict[float, Tuple[np.ndarray, Optional[np.ndarray]]]] = {}
    for (state, t_val), block in work.groupby([state_col, time_col], sort=False, dropna=False):
        y = block["_y_raw"].to_numpy(dtype=float)
        x = block[covariate_cols].to_numpy(dtype=float) if covariate_cols else None
        out.setdefault(state, {})[float(t_val)] = (y, x)
    return out


def _required_pairs_from_g_values(
    times: List[float],
    g_values: Iterable[float],
) -> List[Tuple[float, float]]:
    required_pairs: List[Tuple[float, float]] = []
    for g_val_raw in g_values:
        g_val = float(g_val_raw)
        if not np.isfinite(g_val) or g_val <= 0:
            continue
        pre = [float(tt) for tt in times if float(tt) < g_val]
        if not pre:
            continue
        r1 = float(pre[-1])
        for post_t in times:
            post_f = float(post_t)
            if post_f >= g_val:
                required_pairs.append((r1, post_f))
    required_pairs = sorted(set(required_pairs))
    if required_pairs:
        return required_pairs
    return [
        (float(pre_t), float(post_t))
        for pre_t in times
        for post_t in times
        if float(post_t) > float(pre_t)
    ]


def _compile_micro_contrast_cache(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str],
    required_pairs: Optional[Iterable[Tuple[float, float]]] = None,
    global_cache: bool = False,
) -> Dict[str, Any]:
    required = {state_col, time_col, gvar_col, "_y_raw"}
    missing = [c for c in required if c not in panel_df.columns]
    if missing:
        raise ValueError(f"panel_df is missing required columns: {missing}")

    work = panel_df.copy()
    work = work.dropna(subset=[state_col, time_col, gvar_col, "_y_raw"])
    work[time_col] = _to_numeric_or_raise(work[time_col], time_col)
    work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)

    non_covariate_cols = {state_col, time_col, gvar_col, "_y_raw"}
    if pg_col:
        non_covariate_cols.add(pg_col)
    covariate_cols = [c for c in work.columns if c not in non_covariate_cols]
    for c in covariate_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if covariate_cols:
        work = work.dropna(subset=covariate_cols)

    counts = (
        work.groupby([state_col, time_col], as_index=False)["_y_raw"]
        .size()
        .rename(columns={"size": "n_t"})
    )
    n_wide = counts.pivot(index=state_col, columns=time_col, values="n_t")
    times = sorted(work[time_col].unique().tolist())
    state_arrays = _state_time_arrays(work, state_col, time_col, covariate_cols)
    state_g = work[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
    g_values = sorted(
        [float(g) for g in np.unique(state_g[gvar_col].to_numpy()) if np.isfinite(g) and g > 0]
    )
    if not required_pairs:
        required_pair_list = _required_pairs_from_g_values(times, g_values)
    else:
        required_pair_list = sorted(
            set((float(pre_t), float(post_t)) for pre_t, post_t in required_pairs)
        )

    delta_cache: Dict[Tuple[Any, float, float], float] = {}
    for s, time_arrays in state_arrays.items():
        for pre_t, post_t in required_pair_list:
            pre_arrays = time_arrays.get(float(pre_t))
            post_arrays = time_arrays.get(float(post_t))
            if pre_arrays is None or post_arrays is None:
                delta_cache[(s, float(pre_t), float(post_t))] = np.nan
                continue
            delta_cache[(s, float(pre_t), float(post_t))] = _pair_adjusted_delta_from_arrays(
                y_pre=pre_arrays[0],
                y_post=post_arrays[0],
                x_pre=pre_arrays[1],
                x_post=post_arrays[1],
                has_covariates=bool(covariate_cols),
            )

    return {
        "times": times,
        "states_index": list(state_arrays.keys()),
        "n_wide": n_wide,
        "delta_cache": delta_cache,
        "required_pairs": required_pair_list,
        "global_cache": bool(global_cache),
        "covariate_cols": list(covariate_cols),
        "fingerprint": {
            "n_rows": int(len(work)),
            "n_states": int(len(state_arrays)),
            "n_times": int(len(times)),
            "y_sum": float(np.nansum(work["_y_raw"].to_numpy(dtype=float))),
            "y_sumsq": float(np.nansum(work["_y_raw"].to_numpy(dtype=float) ** 2)),
        },
    }


def clear_micro_contrast_caches(panel_df: pd.DataFrame) -> None:
    for key in list(panel_df.attrs.keys()):
        if str(key).startswith("_undid_micro_contrast_cache::") or str(key).startswith(
            "_undid_micro_contrast_time_lookup::"
        ):
            panel_df.attrs.pop(key, None)


def set_micro_contrast_time_lookup(
    panel_df: pd.DataFrame,
    time_col: str,
    relative_to_source: Dict[float, float],
) -> None:
    panel_df.attrs[_micro_contrast_time_lookup_key(time_col)] = {
        float(k): float(v) for k, v in relative_to_source.items()
    }


def attach_micro_contrast_cache(
    panel_df: pd.DataFrame,
    *,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str] = None,
    required_pairs: Optional[Iterable[Tuple[float, float]]] = None,
    global_cache: bool = True,
) -> Dict[str, Any]:
    cache = _compile_micro_contrast_cache(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        required_pairs=required_pairs,
        global_cache=global_cache,
    )
    panel_df.attrs[_micro_contrast_cache_key(state_col, time_col, gvar_col, pg_col)] = (
        _SharedMicroContrastCache(cache)
    )
    return cache


def _materialize_micro_contrast_cache_view(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    source_cache: Dict[str, Any],
) -> Dict[str, Any]:
    work = panel_df[[state_col, time_col, gvar_col]].copy()
    work = work.dropna(subset=[state_col, time_col, gvar_col])
    work[time_col] = _to_numeric_or_raise(work[time_col], time_col)
    work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)
    times = sorted(float(x) for x in work[time_col].unique().tolist())
    states_present = work[state_col].drop_duplicates().tolist()
    state_g = work[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
    g_values = sorted(
        [float(g) for g in np.unique(state_g[gvar_col].to_numpy()) if np.isfinite(g) and g > 0]
    )
    required_pairs = _required_pairs_from_g_values(times, g_values)

    rel_lookup = panel_df.attrs.get(_micro_contrast_time_lookup_key(time_col), {})
    rel_lookup = {float(k): float(v) for k, v in dict(rel_lookup).items()}

    def source_time(t_val: float) -> float:
        return float(rel_lookup.get(float(t_val), float(t_val)))

    source_n = source_cache["n_wide"]
    if rel_lookup:
        n_cols: Dict[float, pd.Series] = {}
        for t_val in times:
            src_t = source_time(t_val)
            if src_t in source_n.columns:
                n_cols[t_val] = source_n[src_t]
            else:
                n_cols[t_val] = pd.Series(np.nan, index=source_n.index)
        n_wide = pd.DataFrame(n_cols).reindex(index=states_present)
    else:
        n_wide = source_n.reindex(index=states_present, columns=times)

    source_delta = source_cache["delta_cache"]
    delta_cache: Dict[Tuple[Any, float, float], float] = {}
    for s in states_present:
        for pre_t, post_t in required_pairs:
            delta_cache[(s, float(pre_t), float(post_t))] = source_delta.get(
                (s, source_time(pre_t), source_time(post_t)),
                np.nan,
            )

    return {
        "times": times,
        "states_index": states_present,
        "n_wide": n_wide,
        "delta_cache": delta_cache,
        "required_pairs": required_pairs,
        "global_cache_view": True,
        "source_fingerprint": source_cache.get("fingerprint", {}),
    }


def _get_micro_contrast_cache(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str],
) -> Dict[str, Any]:
    key = _micro_contrast_cache_key(state_col, time_col, gvar_col, pg_col)
    cache = panel_df.attrs.get(key)
    if isinstance(cache, _SharedMicroContrastCache):
        if cache.cache.get("global_cache"):
            view_key = f"{key}::view"
            view = panel_df.attrs.get(view_key)
            if isinstance(view, _SharedMicroContrastCache):
                return view.cache
            if isinstance(view, dict):
                return view
            materialized = _materialize_micro_contrast_cache_view(
                panel_df=panel_df,
                state_col=state_col,
                time_col=time_col,
                gvar_col=gvar_col,
                source_cache=cache.cache,
            )
            panel_df.attrs[view_key] = _SharedMicroContrastCache(materialized)
            return materialized
        return cache.cache
    if isinstance(cache, dict):
        if cache.get("global_cache"):
            view_key = f"{key}::view"
            view = panel_df.attrs.get(view_key)
            if isinstance(view, _SharedMicroContrastCache):
                return view.cache
            if isinstance(view, dict):
                return view
            materialized = _materialize_micro_contrast_cache_view(
                panel_df=panel_df,
                state_col=state_col,
                time_col=time_col,
                gvar_col=gvar_col,
                source_cache=cache,
            )
            panel_df.attrs[view_key] = _SharedMicroContrastCache(materialized)
            return materialized
        return cache

    cache = _compile_micro_contrast_cache(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
    )
    panel_df.attrs[key] = _SharedMicroContrastCache(cache)
    return cache


def _build_contrast_rows(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
) -> pd.DataFrame:
    if weighting not in VALID_WEIGHTING:
        raise ValueError(f"weighting must be one of {sorted(VALID_WEIGHTING)}.")

    using_micro = "_y_raw" in panel_df.columns
    if using_micro:
        required = {state_col, time_col, gvar_col}
        missing = [c for c in required if c not in panel_df.columns]
        if missing:
            raise ValueError(f"panel_df is missing required columns: {missing}")

        cache = _get_micro_contrast_cache(panel_df, state_col, time_col, gvar_col, pg_col)
        work = panel_df[[state_col, time_col, gvar_col]].copy()
        work = work.dropna(subset=[state_col, time_col, gvar_col])
        work[time_col] = _to_numeric_or_raise(work[time_col], time_col)
        work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)
        n_wide = cache["n_wide"]
        states_present = set(work[state_col].drop_duplicates().tolist())
        states_index = [s for s in cache["states_index"] if s in states_present]
        delta_cache = cache["delta_cache"]
    else:
        required = {state_col, time_col, gvar_col, "lambda_st", "n_t"}
        missing = [c for c in required if c not in panel_df.columns]
        if missing:
            raise ValueError(f"panel_df is missing required columns: {missing}")

        work_cols = [state_col, time_col, gvar_col, "lambda_st", "n_t"]
        if pg_col and pg_col in panel_df.columns:
            work_cols.append(pg_col)
        work = panel_df[work_cols].copy()
        work = work.dropna(subset=[state_col, time_col, gvar_col, "lambda_st", "n_t"])
        work[time_col] = _to_numeric_or_raise(work[time_col], time_col)
        work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)

        lambda_wide = work.pivot(index=state_col, columns=time_col, values="lambda_st")
        n_wide = work.pivot(index=state_col, columns=time_col, values="n_t")
        states_index = list(lambda_wide.index)

    times = sorted(work[time_col].unique().tolist())
    if len(times) < 2:
        return pd.DataFrame(
            columns=["state", "g", "t", "delta", "w_diff", "w_att", "treat", "gt_key"]
        )

    state_g = work[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
    g_map = state_g.set_index(state_col)[gvar_col]
    states_index = list(states_index)
    g_by_state = g_map.reindex(states_index).to_numpy(dtype=float)
    n_wide = n_wide.reindex(index=states_index)
    n_arrays = {
        float(col): n_wide[col].to_numpy(dtype=float)
        for col in n_wide.columns
    }
    lambda_arrays: Dict[float, np.ndarray] = {}
    if not using_micro:
        lambda_wide = lambda_wide.reindex(index=states_index)
        lambda_arrays = {
            float(col): lambda_wide[col].to_numpy(dtype=float)
            for col in lambda_wide.columns
        }

    cohorts = sorted([g for g in np.unique(g_by_state) if np.isfinite(g) and g > 0])
    records: List[Dict[str, Any]] = []

    for g in cohorts:
        pre = [tt for tt in times if tt < g]
        if not pre:
            continue
        r1 = pre[-1]

        for t in times:
            if t < g:
                continue
            n_t_vec = n_arrays.get(float(t))
            n_r1_vec = n_arrays.get(float(r1))
            if n_t_vec is None or n_r1_vec is None:
                continue
            if using_micro:
                delta_vec_arr = np.asarray(
                    [delta_cache.get((s, float(r1), float(t)), np.nan) for s in states_index],
                    dtype=float,
                )
            else:
                lambda_t = lambda_arrays.get(float(t))
                lambda_r1 = lambda_arrays.get(float(r1))
                if lambda_t is None or lambda_r1 is None:
                    continue
                delta_vec_arr = lambda_t - lambda_r1

            for idx, s in enumerate(states_index):
                gv = g_by_state[idx]
                treat = np.nan
                if gv == g:
                    treat = 1
                elif gv == 0 or (not_yet_treated and gv > t):
                    treat = 0
                else:
                    continue

                n_t_val = n_t_vec[idx]
                n_r1_val = n_r1_vec[idx]
                if weighting in {"both", "diff"}:
                    wd_val = n_t_val + n_r1_val
                else:
                    wd_val = 1.0
                if weighting in {"both", "att"}:
                    wa_val = n_t_val
                else:
                    wa_val = 1.0

                d_val = delta_vec_arr[idx]

                if not (
                    np.isfinite(d_val)
                    and np.isfinite(wd_val)
                    and np.isfinite(wa_val)
                    and wd_val > 0
                    and wa_val > 0
                ):
                    continue

                records.append(
                    {
                        "state": s,
                        "g": g,
                        "t": t,
                        "base_period": r1,
                        "delta": float(d_val),
                        "w_diff": float(wd_val),
                        "w_att": float(wa_val),
                        "treated_target_N": float(n_t_val),
                        "base_target_N": float(n_r1_val),
                        "contrast_weight": float(wd_val),
                        "target_weight": float(wa_val),
                        "treat": int(treat),
                        "gt_key": (g, t),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "state",
                "g",
                "t",
                "base_period",
                "delta",
                "w_diff",
                "w_att",
                "treated_target_N",
                "base_target_N",
                "contrast_weight",
                "target_weight",
                "treat",
                "gt_key",
            ]
        )
    return pd.DataFrame.from_records(records)
def _state_target_stats(
    rows: pd.DataFrame,
    panel_df: pd.DataFrame,
    state_col: str,
    gvar_col: str,
    weighting: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st_records: List[Dict[str, Any]] = []

    if not rows.empty:
        for (g_val, t_val), block in rows.groupby(["g", "t"], sort=True, dropna=False):
            control = block.loc[block["treat"] == 0]
            treated = block.loc[block["treat"] == 1]
            if control.empty or treated.empty:
                continue

            c_delta = control["delta"].to_numpy(dtype=float)
            c_treat = np.zeros(len(control), dtype=float)
            c_w = control["w_diff"].to_numpy(dtype=float)
            support_controls = float(control["state"].nunique())

            for treated_row in treated.itertuples(index=False):
                y = np.concatenate(
                    [
                        np.asarray([float(getattr(treated_row, "delta"))], dtype=float),
                        c_delta,
                    ]
                )
                d = np.concatenate([np.ones(1, dtype=float), c_treat])
                w = np.concatenate(
                    [
                        np.asarray([float(getattr(treated_row, "w_diff"))], dtype=float),
                        c_w,
                    ]
                )
                stats = _weighted_diff_stats_from_arrays(
                    y=y,
                    d=d,
                    w=w,
                    treat_w_att=np.asarray([float(getattr(treated_row, "w_att"))], dtype=float),
                    n_treated=1.0,
                    n_control=float(len(control)),
                )
                if stats is None:
                    continue

                rec: Dict[str, Any] = {
                    "state": getattr(treated_row, "state"),
                    "g": float(g_val),
                    "t": float(t_val),
                }
                rec.update(stats)
                rec.update(
                    {
                        "base_period": float(getattr(treated_row, "base_period")),
                        "ATT_st": float(stats["estimate"]),
                        "treated_target_N": float(getattr(treated_row, "treated_target_N")),
                        "base_target_N": float(getattr(treated_row, "base_target_N")),
                        "contrast_weight": float(getattr(treated_row, "contrast_weight")),
                        "target_weight": float(getattr(treated_row, "target_weight")),
                        "support": True,
                        "support_controls": support_controls,
                    }
                )
                st_records.append(rec)

    att_st_cols = [
        "state",
        "g",
        "t",
        "base_period",
        "estimate",
        "ATT_st",
        "se",
        "p_value",
        "weight",
        "target_weight",
        "contrast_weight",
        "treated_target_N",
        "base_target_N",
        "n_treated",
        "n_control",
        "support",
        "support_controls",
    ]

    if not st_records:
        att_st = pd.DataFrame(columns=att_st_cols)
        att_s = pd.DataFrame(columns=[
            "state",
            "g",
            "estimate",
            "se",
            "p_value",
            "weight",
            "n_treated",
            "n_control",
        ])
        return att_s, att_st

    att_st = pd.DataFrame.from_records(st_records)

    s_records: List[Dict[str, Any]] = []
    for st, block in att_st.groupby("state", sort=True):
        g_val = float(block["g"].iloc[0])
        est_vals = block["estimate"].to_numpy(dtype=float)
        if weighting in {"both", "att"}:
            w_vals = block["weight"].to_numpy(dtype=float)
        else:
            w_vals = np.ones(len(block), dtype=float)

        s_est, s_se = _weighted_mean_and_se(est_vals, w_vals)
        dof = len(block) - 1
        s_p = (
            _two_sided_pvalue(s_est / s_se, dof=dof)
            if np.isfinite(s_est) and np.isfinite(s_se) and s_se > 0
            else np.nan
        )

        s_records.append(
            {
                "state": st,
                "g": g_val,
                "estimate": float(s_est),
                "se": float(s_se),
                "p_value": float(s_p),
                "weight": float(np.sum(w_vals)),
                "n_treated": float(len(block)),
                "n_control": np.nan,
            }
        )

    att_s = pd.DataFrame.from_records(s_records)
    return att_s, att_st


def _legacy_contrast_row_aggregates(
    rows: pd.DataFrame,
    weighting: str,
) -> Dict[str, Any]:
    """Deprecated direct aggregate path retained only for validation reports."""
    att_gt = _level_stats(rows, ["g", "t"])
    att_g = _level_stats(rows, ["g"])
    if att_g.empty:
        att = {
            "estimate": np.nan,
            "se": np.nan,
            "p_value": np.nan,
            "weight": np.nan,
            "n_treated": np.nan,
            "n_control": np.nan,
        }
    else:
        agg_w = att_g["weight"].to_numpy(dtype=float) if weighting in {"both", "att"} else np.ones(len(att_g), dtype=float)
        agg_est = att_g["estimate"].to_numpy(dtype=float)
        att_est, att_se = _weighted_mean_and_se(agg_est, agg_w)
        att = {
            "estimate": float(att_est),
            "se": float(att_se),
            "p_value": float(
                _two_sided_pvalue(att_est / att_se, dof=len(att_g) - 1)
                if np.isfinite(att_est) and np.isfinite(att_se) and att_se > 0
                else np.nan
            ),
            "weight": float(np.sum(agg_w)),
            "n_treated": float(len(att_g)),
            "n_control": np.nan,
        }
    return {"att": att, "att_g": att_g, "att_gt": att_gt}


def _empty_policy_frame(group_cols: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=list(group_cols)
        + [
            "estimate",
            "se",
            "p_value",
            "weight",
            "n_treated",
            "n_control",
            "support_states",
            "support_state_time_cells",
        ]
    )


def _policy_target_stats(
    att_st: pd.DataFrame,
    panel_df: pd.DataFrame,
    state_col: str,
    pg_col: Optional[str],
    weighting: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not pg_col or pg_col not in panel_df.columns or att_st.empty:
        return _empty_policy_frame(["pvar"]), _empty_policy_frame(["pvar", "t"])

    state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
    pg_map = state_pg.set_index(state_col)[pg_col].to_dict()

    work = att_st.copy()
    work["pvar"] = work["state"].map(lambda s: _pg_pvar(pg_map.get(s, "0")))
    work = work.loc[work["pvar"] != "0"].copy()
    if work.empty:
        return _empty_policy_frame(["pvar"]), _empty_policy_frame(["pvar", "t"])

    def _aggregate(block: pd.DataFrame, keys: Dict[str, Any]) -> Dict[str, Any]:
        est_vals = block["estimate"].to_numpy(dtype=float)
        if weighting in {"both", "att"}:
            w_vals = block["weight"].to_numpy(dtype=float)
        else:
            w_vals = np.ones(len(block), dtype=float)
        est, se = _weighted_mean_and_se(est_vals, w_vals)
        dof = len(block) - 1
        p_val = (
            _two_sided_pvalue(est / se, dof=dof)
            if np.isfinite(est) and np.isfinite(se) and se > 0
            else np.nan
        )
        rec: Dict[str, Any] = dict(keys)
        rec.update(
            {
                "estimate": float(est),
                "se": float(se),
                "p_value": float(p_val),
                "weight": float(np.sum(w_vals)),
                "n_treated": float(len(block)),
                "n_control": np.nan,
                "support_states": float(block["state"].nunique()),
                "support_state_time_cells": float(len(block)),
            }
        )
        return rec

    pt_records: List[Dict[str, Any]] = []
    for (pvar, t_val), block in work.groupby(["pvar", "t"], sort=True):
        pt_records.append(_aggregate(block, {"pvar": pvar, "t": t_val}))

    p_records: List[Dict[str, Any]] = []
    for pvar, block in work.groupby("pvar", sort=True):
        p_records.append(_aggregate(block, {"pvar": pvar}))

    att_pt = (
        pd.DataFrame.from_records(pt_records)
        if pt_records
        else _empty_policy_frame(["pvar", "t"])
    )
    att_p = (
        pd.DataFrame.from_records(p_records)
        if p_records
        else _empty_policy_frame(["pvar"])
    )
    return att_p, att_pt


def _compute_core(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
    include_aggregation_weights: bool = True,
    include_legacy_aggregates: bool = True,
) -> Dict[str, Any]:
    rows = _build_contrast_rows(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        pg_col=pg_col,
    )

    att_s, att_st = _state_target_stats(
        rows=rows,
        panel_df=panel_df,
        state_col=state_col,
        gvar_col=gvar_col,
        weighting=weighting,
    )
    if pg_col and pg_col in panel_df.columns and not att_st.empty:
        state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
        pg_map = state_pg.set_index(state_col)[pg_col].to_dict()
        att_st = att_st.copy()
        att_st["pg"] = att_st["state"].map(lambda s: pg_map.get(s, "0"))
        att_st["pvar"] = att_st["pg"].map(_pg_pvar)

    canonical = aggregate_from_att_st(
        att_st,
        weighting=weighting,
        include_policy=True,
        include_aggregation_weights=include_aggregation_weights,
    )
    att = canonical["att"]
    att_g = canonical["att_g"]
    att_gt = canonical["att_gt"]
    att_s = canonical["att_s"]
    att_st = canonical["att_st"]
    att_p = canonical["att_p"]
    att_pt = canonical["att_pt"]
    legacy = (
        _legacy_contrast_row_aggregates(rows, weighting=weighting)
        if include_legacy_aggregates
        else {}
    )

    return {
        "rows": rows,
        "att": att,
        "att_g": att_g,
        "att_gt": att_gt,
        "att_s": att_s,
        "att_st": att_st,
        "att_p": att_p,
        "att_pt": att_pt,
        "aggregation_weights": canonical.get("aggregation_weights", pd.DataFrame()),
        "legacy_direct_aggregates": legacy,
        "aggregation_note": "ATT(s,t)-first canonical aggregation",
    }


def _extract_estimate_maps(
    core: Dict[str, Any],
) -> Tuple[
    float,
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
]:
    att_val = float(core["att"]["estimate"])

    g_map: Dict[Hashable, float] = {}
    if not core["att_g"].empty:
        for _, row in core["att_g"].iterrows():
            g_map[row["g"]] = float(row["estimate"])

    gt_map: Dict[Tuple[Any, Any], float] = {}
    if not core["att_gt"].empty:
        for _, row in core["att_gt"].iterrows():
            gt_map[(row["g"], row["t"])] = float(row["estimate"])

    s_map: Dict[Hashable, float] = {}
    if not core["att_s"].empty:
        for _, row in core["att_s"].iterrows():
            s_map[row["state"]] = float(row["estimate"])

    st_map: Dict[Tuple[Any, Any], float] = {}
    if not core["att_st"].empty:
        for _, row in core["att_st"].iterrows():
            st_map[(row["state"], row["t"])] = float(row["estimate"])

    p_map: Dict[Hashable, float] = {}
    if "att_p" in core and not core["att_p"].empty:
        for _, row in core["att_p"].iterrows():
            p_map[row["pvar"]] = float(row["estimate"])

    pt_map: Dict[Tuple[Any, Any], float] = {}
    if "att_pt" in core and not core["att_pt"].empty:
        for _, row in core["att_pt"].iterrows():
            pt_map[(row["pvar"], row["t"])] = float(row["estimate"])

    return att_val, g_map, gt_map, s_map, st_map, p_map, pt_map


def _finite_weighted_average(values: List[float], weights: List[float]) -> Tuple[float, float]:
    val_arr = np.asarray(values, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(val_arr) & np.isfinite(w_arr) & (w_arr > 0)
    if not mask.any():
        return np.nan, 0.0
    denom = float(w_arr[mask].sum())
    if denom <= 0:
        return np.nan, 0.0
    return float(np.sum(val_arr[mask] * w_arr[mask]) / denom), denom


def _compute_estimate_maps_only(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
) -> Tuple[
    float,
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
]:
    if "_y_raw" in panel_df.columns:
        return _compute_estimate_maps_only_micro(
            panel_df=panel_df,
            state_col=state_col,
            time_col=time_col,
            gvar_col=gvar_col,
            weighting=weighting,
            not_yet_treated=not_yet_treated,
            pg_col=pg_col,
        )

    rows = _build_contrast_rows(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        pg_col=pg_col,
    )
    if rows.empty:
        return np.nan, {}, {}, {}, {}, {}, {}

    pg_map: Dict[Any, str] = {}
    if pg_col and pg_col in panel_df.columns:
        state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
        pg_map = state_pg.set_index(state_col)[pg_col].to_dict()

    st_records: List[Dict[str, Any]] = []
    for (g_val, t_val), block in rows.groupby(["g", "t"], sort=True, dropna=False):
        control = block.loc[block["treat"] == 0]
        treated = block.loc[block["treat"] == 1]
        if control.empty or treated.empty:
            continue
        c_delta = control["delta"].to_numpy(dtype=float)
        c_w = control["w_diff"].to_numpy(dtype=float)
        c_mask = np.isfinite(c_delta) & np.isfinite(c_w) & (c_w > 0)
        if not c_mask.any():
            continue
        c_denom = float(c_w[c_mask].sum())
        if c_denom <= 0:
            continue
        c_mean = float(np.sum(c_delta[c_mask] * c_w[c_mask]) / c_denom)
        for tr in treated.itertuples(index=False):
            delta = float(getattr(tr, "delta"))
            if not np.isfinite(delta):
                continue
            state = getattr(tr, "state")
            rec = {
                "state": state,
                "g": float(g_val),
                "t": float(t_val),
                "estimate": float(delta - c_mean),
                "target_weight": float(getattr(tr, "target_weight")),
                "weight": float(getattr(tr, "w_att")),
            }
            if pg_map:
                rec["pvar"] = _pg_pvar(pg_map.get(state, "0"))
            st_records.append(rec)

    if not st_records:
        return np.nan, {}, {}, {}, {}, {}, {}

    st_map: Dict[Tuple[Any, Any], float] = {}
    s_groups: Dict[Hashable, Tuple[List[float], List[float], Any]] = {}
    gt_groups: Dict[Tuple[Any, Any], Tuple[List[float], List[float], List[float]]] = {}
    p_groups: Dict[Hashable, Tuple[List[float], List[float]]] = {}
    pt_groups: Dict[Tuple[Any, Any], Tuple[List[float], List[float]]] = {}

    for rec in st_records:
        state = rec["state"]
        g_val = rec["g"]
        t_val = rec["t"]
        estimate = rec["estimate"]
        target_weight = rec["target_weight"]
        avg_weight = target_weight if weighting in {"both", "att"} else 1.0

        st_map[(state, t_val)] = estimate

        s_vals, s_weights, _g = s_groups.setdefault(state, ([], [], g_val))
        s_vals.append(estimate)
        s_weights.append(avg_weight)

        gt_vals, gt_avg_weights, gt_output_weights = gt_groups.setdefault((g_val, t_val), ([], [], []))
        gt_vals.append(estimate)
        gt_avg_weights.append(avg_weight)
        gt_output_weights.append(target_weight)

        pvar = rec.get("pvar")
        if pvar and str(pvar) != "0":
            p_vals, p_weights = p_groups.setdefault(pvar, ([], []))
            p_vals.append(estimate)
            p_weights.append(avg_weight)
            pt_vals, pt_weights = pt_groups.setdefault((pvar, t_val), ([], []))
            pt_vals.append(estimate)
            pt_weights.append(avg_weight)

    s_map: Dict[Hashable, float] = {}
    for state, (values, weights, _g) in s_groups.items():
        est, _ = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            s_map[state] = est

    gt_map: Dict[Tuple[Any, Any], float] = {}
    gt_weight_map: Dict[Tuple[Any, Any], float] = {}
    for key, (values, avg_weights, output_weights) in gt_groups.items():
        est, _ = _finite_weighted_average(values, avg_weights)
        out_weight = float(np.nansum(output_weights))
        if np.isfinite(est):
            gt_map[key] = est
            gt_weight_map[key] = out_weight

    g_groups: Dict[Hashable, Tuple[List[float], List[float]]] = {}
    for (g_val, _t_val), est in gt_map.items():
        vals, weights = g_groups.setdefault(g_val, ([], []))
        vals.append(est)
        weights.append(gt_weight_map.get((g_val, _t_val), 1.0) if weighting in {"both", "att"} else 1.0)

    g_map: Dict[Hashable, float] = {}
    g_weight_map: Dict[Hashable, float] = {}
    for g_val, (values, weights) in g_groups.items():
        est, denom = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            g_map[g_val] = est
            g_weight_map[g_val] = denom

    att_val, _ = _finite_weighted_average(
        list(g_map.values()),
        [g_weight_map.get(g, 1.0) if weighting in {"both", "att"} else 1.0 for g in g_map],
    )

    p_map: Dict[Hashable, float] = {}
    for pvar, (values, weights) in p_groups.items():
        est, _ = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            p_map[pvar] = est

    pt_map: Dict[Tuple[Any, Any], float] = {}
    for key, (values, weights) in pt_groups.items():
        est, _ = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            pt_map[key] = est

    return att_val, g_map, gt_map, s_map, st_map, p_map, pt_map


def _compute_estimate_maps_only_micro(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
) -> Tuple[
    float,
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
]:
    cache = _get_micro_contrast_cache(panel_df, state_col, time_col, gvar_col, pg_col)
    times = cache["times"]
    n_wide = cache["n_wide"]
    delta_cache = cache["delta_cache"]

    state_g = panel_df[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
    g_map = state_g.set_index(state_col)[gvar_col].to_dict()
    states_index = [s for s in cache["states_index"] if s in g_map]
    cohorts = sorted(
        [float(g) for g in np.unique(np.asarray(list(g_map.values()), dtype=float)) if np.isfinite(g) and g > 0]
    )

    pg_map: Dict[Any, str] = {}
    if pg_col and pg_col in panel_df.columns:
        state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
        pg_map = state_pg.set_index(state_col)[pg_col].to_dict()

    st_records: List[Dict[str, Any]] = []
    for g_val in cohorts:
        pre = [tt for tt in times if float(tt) < g_val]
        if not pre:
            continue
        r1 = float(pre[-1])
        if r1 not in n_wide.columns:
            continue
        for t_val_raw in times:
            t_val = float(t_val_raw)
            if t_val < g_val or t_val not in n_wide.columns:
                continue

            controls_delta: List[float] = []
            controls_weight: List[float] = []
            treated_rows: List[Dict[str, Any]] = []
            n_t_col = n_wide[t_val]
            n_r1_col = n_wide[r1]

            for state in states_index:
                gv = float(g_map.get(state, np.nan))
                if gv == g_val:
                    treat = 1
                elif gv == 0 or (not_yet_treated and gv > t_val):
                    treat = 0
                else:
                    continue

                n_t_val = n_t_col.get(state, np.nan)
                n_r1_val = n_r1_col.get(state, np.nan)
                wd_val = n_t_val + n_r1_val if weighting in {"both", "diff"} else 1.0
                wa_val = n_t_val if weighting in {"both", "att"} else 1.0
                d_val = delta_cache.get((state, r1, t_val), np.nan)
                if not (
                    np.isfinite(d_val)
                    and np.isfinite(wd_val)
                    and np.isfinite(wa_val)
                    and wd_val > 0
                    and wa_val > 0
                ):
                    continue

                if treat == 0:
                    controls_delta.append(float(d_val))
                    controls_weight.append(float(wd_val))
                else:
                    treated_rows.append(
                        {
                            "state": state,
                            "g": g_val,
                            "t": t_val,
                            "delta": float(d_val),
                            "target_weight": float(wa_val),
                        }
                    )

            if not controls_delta or not treated_rows:
                continue
            c_mean, _ = _finite_weighted_average(controls_delta, controls_weight)
            if not np.isfinite(c_mean):
                continue
            for tr in treated_rows:
                rec = {
                    "state": tr["state"],
                    "g": tr["g"],
                    "t": tr["t"],
                    "estimate": float(tr["delta"] - c_mean),
                    "target_weight": tr["target_weight"],
                    "weight": tr["target_weight"],
                }
                if pg_map:
                    rec["pvar"] = _pg_pvar(pg_map.get(tr["state"], "0"))
                st_records.append(rec)

    return _aggregate_estimate_records(st_records, weighting=weighting)


def _aggregate_estimate_records(
    st_records: List[Dict[str, Any]],
    *,
    weighting: str,
) -> Tuple[
    float,
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
    Dict[Hashable, float],
    Dict[Tuple[Any, Any], float],
]:
    if not st_records:
        return np.nan, {}, {}, {}, {}, {}, {}

    st_map: Dict[Tuple[Any, Any], float] = {}
    s_groups: Dict[Hashable, Tuple[List[float], List[float], Any]] = {}
    gt_groups: Dict[Tuple[Any, Any], Tuple[List[float], List[float], List[float]]] = {}
    p_groups: Dict[Hashable, Tuple[List[float], List[float]]] = {}
    pt_groups: Dict[Tuple[Any, Any], Tuple[List[float], List[float]]] = {}

    for rec in st_records:
        state = rec["state"]
        g_val = rec["g"]
        t_val = rec["t"]
        estimate = rec["estimate"]
        target_weight = rec["target_weight"]
        avg_weight = target_weight if weighting in {"both", "att"} else 1.0

        st_map[(state, t_val)] = estimate

        s_vals, s_weights, _g = s_groups.setdefault(state, ([], [], g_val))
        s_vals.append(estimate)
        s_weights.append(avg_weight)

        gt_vals, gt_avg_weights, gt_output_weights = gt_groups.setdefault((g_val, t_val), ([], [], []))
        gt_vals.append(estimate)
        gt_avg_weights.append(avg_weight)
        gt_output_weights.append(target_weight)

        pvar = rec.get("pvar")
        if pvar and str(pvar) != "0":
            p_vals, p_weights = p_groups.setdefault(pvar, ([], []))
            p_vals.append(estimate)
            p_weights.append(avg_weight)
            pt_vals, pt_weights = pt_groups.setdefault((pvar, t_val), ([], []))
            pt_vals.append(estimate)
            pt_weights.append(avg_weight)

    s_map: Dict[Hashable, float] = {}
    for state, (values, weights, _g) in s_groups.items():
        est, _ = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            s_map[state] = est

    gt_map: Dict[Tuple[Any, Any], float] = {}
    gt_weight_map: Dict[Tuple[Any, Any], float] = {}
    for key, (values, avg_weights, output_weights) in gt_groups.items():
        est, _ = _finite_weighted_average(values, avg_weights)
        out_weight = float(np.nansum(output_weights))
        if np.isfinite(est):
            gt_map[key] = est
            gt_weight_map[key] = out_weight

    g_groups: Dict[Hashable, Tuple[List[float], List[float]]] = {}
    for (g_val, t_val), est in gt_map.items():
        vals, weights = g_groups.setdefault(g_val, ([], []))
        vals.append(est)
        weights.append(gt_weight_map.get((g_val, t_val), 1.0) if weighting in {"both", "att"} else 1.0)

    g_map: Dict[Hashable, float] = {}
    g_weight_map: Dict[Hashable, float] = {}
    for g_val, (values, weights) in g_groups.items():
        est, denom = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            g_map[g_val] = est
            g_weight_map[g_val] = denom

    att_val, _ = _finite_weighted_average(
        list(g_map.values()),
        [g_weight_map.get(g, 1.0) if weighting in {"both", "att"} else 1.0 for g in g_map],
    )

    p_map: Dict[Hashable, float] = {}
    for pvar, (values, weights) in p_groups.items():
        est, _ = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            p_map[pvar] = est

    pt_map: Dict[Tuple[Any, Any], float] = {}
    for key, (values, weights) in pt_groups.items():
        est, _ = _finite_weighted_average(values, weights)
        if np.isfinite(est):
            pt_map[key] = est

    return att_val, g_map, gt_map, s_map, st_map, p_map, pt_map


def _compile_numeric_assignment_engine(
    panel_df: pd.DataFrame,
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        state_g = panel_df[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
    except KeyError:
        return None
    if state_g.empty:
        return None

    states = state_g[state_col].tolist()
    observed_labels = pd.to_numeric(state_g[gvar_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    positive = sorted([float(x) for x in np.unique(observed_labels) if np.isfinite(x) and x > 0])
    if not positive:
        return None

    if "_y_raw" in panel_df.columns:
        cache = _get_micro_contrast_cache(panel_df, state_col, time_col, gvar_col, pg_col)
        times = [float(x) for x in cache["times"]]
        n_wide = cache["n_wide"].reindex(index=states, columns=times)
        n_arr = n_wide.to_numpy(dtype=float)
        delta_cache = cache["delta_cache"]

        def delta_vec(pre_t: float, post_t: float) -> np.ndarray:
            return np.asarray(
                [delta_cache.get((s, float(pre_t), float(post_t)), np.nan) for s in states],
                dtype=float,
            )

    else:
        required = {state_col, time_col, gvar_col, "lambda_st", "n_t"}
        if not required.issubset(panel_df.columns):
            return None
        work = panel_df[[state_col, time_col, gvar_col, "lambda_st", "n_t"]].copy()
        work = work.dropna(subset=[state_col, time_col, "lambda_st", "n_t"])
        work[time_col] = _to_numeric_or_raise(work[time_col], time_col)
        times = [float(x) for x in sorted(work[time_col].unique().tolist())]
        lambda_wide = work.pivot(index=state_col, columns=time_col, values="lambda_st").reindex(index=states, columns=times)
        n_wide = work.pivot(index=state_col, columns=time_col, values="n_t").reindex(index=states, columns=times)
        lambda_arr = lambda_wide.to_numpy(dtype=float)
        n_arr = n_wide.to_numpy(dtype=float)
        time_index = {float(t): i for i, t in enumerate(times)}

        def delta_vec(pre_t: float, post_t: float) -> np.ndarray:
            i_pre = time_index.get(float(pre_t))
            i_post = time_index.get(float(post_t))
            if i_pre is None or i_post is None:
                return np.full(len(states), np.nan, dtype=float)
            return lambda_arr[:, i_post] - lambda_arr[:, i_pre]

    time_index = {float(t): i for i, t in enumerate(times)}
    cells: List[Dict[str, Any]] = []
    for g_val in positive:
        pre = [tt for tt in times if float(tt) < g_val]
        if not pre:
            continue
        r1 = float(pre[-1])
        i_pre = time_index.get(r1)
        if i_pre is None:
            continue
        for t_val in times:
            if t_val < g_val:
                continue
            i_post = time_index.get(float(t_val))
            if i_post is None:
                continue
            n_t = n_arr[:, i_post]
            n_base = n_arr[:, i_pre]
            w_diff = n_t + n_base if weighting in {"both", "diff"} else np.ones(len(states), dtype=float)
            w_att = n_t if weighting in {"both", "att"} else np.ones(len(states), dtype=float)
            delta = delta_vec(r1, float(t_val))
            valid = (
                np.isfinite(delta)
                & np.isfinite(w_diff)
                & np.isfinite(w_att)
                & (w_diff > 0)
                & (w_att > 0)
            )
            cells.append(
                {
                    "g": float(g_val),
                    "t": float(t_val),
                    "delta": delta,
                    "w_diff": w_diff,
                    "w_att": w_att,
                    "valid": valid,
                }
            )

    if not cells:
        return None

    pg_labels = None
    if pg_col and pg_col in panel_df.columns:
        state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
        pg_lookup = state_pg.set_index(state_col)[pg_col].to_dict()
        pg_labels = np.asarray([str(pg_lookup.get(s, "0")) for s in states], dtype=object)

    return {
        "states": states,
        "observed_labels": observed_labels,
        "cells": cells,
        "weighting": weighting,
        "not_yet_treated": not_yet_treated,
        "pg_labels": pg_labels,
    }


def _pvar_matrix_from_pg(pg_matrix: np.ndarray) -> np.ndarray:
    out = np.full(pg_matrix.shape, "0", dtype=object)
    for label in np.unique(pg_matrix.astype(object)):
        label_str = str(label)
        out[pg_matrix == label] = _pg_pvar(label_str)
    return out


def _evaluate_assignment_matrix(
    engine: Dict[str, Any],
    label_matrix: np.ndarray,
    pg_matrix: Optional[np.ndarray] = None,
    active_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    labels = np.asarray(label_matrix, dtype=float)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    n_assign, n_states = labels.shape
    states = engine["states"]
    if n_states != len(states):
        raise ValueError("assignment matrix has wrong number of states")
    if active_matrix is None:
        active_states = np.ones_like(labels, dtype=bool)
    else:
        active_states = np.asarray(active_matrix, dtype=bool)
        if active_states.shape != labels.shape:
            raise ValueError("active matrix must match assignment matrix shape")

    weighting = engine["weighting"]
    not_yet_treated = bool(engine["not_yet_treated"])

    st_values: Dict[Tuple[Any, Any], np.ndarray] = {}
    s_num: Dict[Hashable, np.ndarray] = {}
    s_den: Dict[Hashable, np.ndarray] = {}
    gt_num: Dict[Tuple[Any, Any], np.ndarray] = {}
    gt_den: Dict[Tuple[Any, Any], np.ndarray] = {}
    p_num: Dict[Hashable, np.ndarray] = {}
    p_den: Dict[Hashable, np.ndarray] = {}
    pt_num: Dict[Tuple[Any, Any], np.ndarray] = {}
    pt_den: Dict[Tuple[Any, Any], np.ndarray] = {}

    pvars = _pvar_matrix_from_pg(pg_matrix) if pg_matrix is not None else None

    for cell in engine["cells"]:
        g_val = float(cell["g"])
        t_val = float(cell["t"])
        delta = cell["delta"]
        w_diff = cell["w_diff"]
        w_att = cell["w_att"]
        valid = cell["valid"]

        treated = (labels == g_val) & active_states
        if not_yet_treated:
            controls = ((labels == 0) | (labels > t_val)) & active_states
        else:
            controls = (labels == 0) & active_states

        control_valid = controls & valid[None, :]
        c_weight = np.where(valid, w_diff, 0.0)
        c_num = control_valid @ (c_weight * np.where(np.isfinite(delta), delta, 0.0))
        c_den = control_valid @ c_weight
        c_mean = np.divide(c_num, c_den, out=np.full(n_assign, np.nan, dtype=float), where=c_den > 0)

        treated_valid = treated & valid[None, :]
        if not treated_valid.any():
            continue

        est = delta[None, :] - c_mean[:, None]
        avg_w = w_att if weighting in {"both", "att"} else np.ones_like(w_att, dtype=float)
        gt_key = (g_val, t_val)
        gt_num.setdefault(gt_key, np.zeros(n_assign, dtype=float))
        gt_den.setdefault(gt_key, np.zeros(n_assign, dtype=float))

        contrib_mask = treated_valid & np.isfinite(est)
        gt_num[gt_key] += np.sum(np.where(contrib_mask, est * avg_w[None, :], 0.0), axis=1)
        gt_den[gt_key] += np.sum(np.where(contrib_mask, avg_w[None, :], 0.0), axis=1)

        for h, state in enumerate(states):
            active = contrib_mask[:, h]
            if not active.any():
                continue
            vals = np.where(active, est[:, h], np.nan)
            st_key = (state, t_val)
            if st_key in st_values:
                st_values[st_key] = np.where(np.isfinite(vals), vals, st_values[st_key])
            else:
                st_values[st_key] = vals

            w_h = float(avg_w[h])
            s_num.setdefault(state, np.zeros(n_assign, dtype=float))
            s_den.setdefault(state, np.zeros(n_assign, dtype=float))
            s_num[state] += np.where(active, est[:, h] * w_h, 0.0)
            s_den[state] += np.where(active, w_h, 0.0)

        if pvars is not None:
            active_pvars = sorted({str(x) for x in np.unique(pvars[treated_valid]) if str(x) != "0"})
            for pvar in active_pvars:
                p_mask = contrib_mask & (pvars == pvar)
                if not p_mask.any():
                    continue
                num = np.sum(np.where(p_mask, est * avg_w[None, :], 0.0), axis=1)
                den = np.sum(np.where(p_mask, avg_w[None, :], 0.0), axis=1)
                p_num.setdefault(pvar, np.zeros(n_assign, dtype=float))
                p_den.setdefault(pvar, np.zeros(n_assign, dtype=float))
                pt_num.setdefault((pvar, t_val), np.zeros(n_assign, dtype=float))
                pt_den.setdefault((pvar, t_val), np.zeros(n_assign, dtype=float))
                p_num[pvar] += num
                p_den[pvar] += den
                pt_num[(pvar, t_val)] += num
                pt_den[(pvar, t_val)] += den

    def div_map(num_map: Dict[Any, np.ndarray], den_map: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        out: Dict[Any, np.ndarray] = {}
        for key, num in num_map.items():
            den = den_map.get(key, np.zeros(n_assign, dtype=float))
            out[key] = np.divide(num, den, out=np.full(n_assign, np.nan, dtype=float), where=den > 0)
        return out

    gt_est = div_map(gt_num, gt_den)
    s_est = div_map(s_num, s_den)
    p_est = div_map(p_num, p_den)
    pt_est = div_map(pt_num, pt_den)

    g_num: Dict[Hashable, np.ndarray] = {}
    g_den: Dict[Hashable, np.ndarray] = {}
    for (g_val, _t_val), vals in gt_est.items():
        den = gt_den[(g_val, _t_val)] if weighting in {"both", "att"} else np.where(np.isfinite(vals), 1.0, 0.0)
        g_num.setdefault(g_val, np.zeros(n_assign, dtype=float))
        g_den.setdefault(g_val, np.zeros(n_assign, dtype=float))
        g_num[g_val] += np.where(np.isfinite(vals), vals * den, 0.0)
        g_den[g_val] += np.where(np.isfinite(vals), den, 0.0)
    g_est = div_map(g_num, g_den)

    att_num = np.zeros(n_assign, dtype=float)
    att_den = np.zeros(n_assign, dtype=float)
    for g_val, vals in g_est.items():
        den = g_den[g_val] if weighting in {"both", "att"} else np.where(np.isfinite(vals), 1.0, 0.0)
        att_num += np.where(np.isfinite(vals), vals * den, 0.0)
        att_den += np.where(np.isfinite(vals), den, 0.0)
    att = np.divide(att_num, att_den, out=np.full(n_assign, np.nan, dtype=float), where=att_den > 0)

    return {
        "att": att,
        "att_g": g_est,
        "att_gt": gt_est,
        "att_s": s_est,
        "att_st": st_values,
        "att_p": p_est,
        "att_pt": pt_est,
    }


def _assignment_maps_to_matrices(states: List[Any], maps: Iterable[Dict[Any, Any]]) -> np.ndarray:
    return np.asarray([[m[s] for s in states] for m in maps], dtype=object)


def _jackknife_se(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = arr.size
    if n < 2:
        return np.nan
    centered = arr - float(arr.mean())
    var = ((n - 1.0) / n) * float(np.sum(centered**2))
    return float(math.sqrt(max(var, 0.0)))


def _support_size_from_labels(labels: Iterable[Any]) -> int:
    counts = Counter(str(x) for x in labels)
    n = sum(counts.values())
    out = math.factorial(n)
    for count in counts.values():
        out //= math.factorial(count)
    return int(out)


def _assignment_key_from_labels(labels: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(str(x) for x in labels)


def _enumerate_label_permutations(
    states: List[Any],
    observed_labels: Iterable[Any],
) -> Iterable[Dict[Any, Any]]:
    labels = list(observed_labels)
    observed_key = _assignment_key_from_labels(labels)
    unique = sorted(Counter(labels).items(), key=lambda item: str(item[0]))

    def _rec(pos: int, counts: Dict[str, int], current: List[str]) -> Iterable[List[str]]:
        if pos == len(states):
            yield list(current)
            return
        for label in sorted(counts):
            if counts[label] <= 0:
                continue
            counts[label] -= 1
            current.append(label)
            yield from _rec(pos + 1, counts, current)
            current.pop()
            counts[label] += 1

    count_map = dict(unique)
    for permuted in _rec(0, count_map, []):
        if _assignment_key_from_labels(permuted) == observed_key:
            continue
        yield dict(zip(states, permuted))


def _random_label_permutations(
    states: List[Any],
    labels: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> Iterable[Dict[Any, Any]]:
    for _ in range(n_perm):
        yield dict(zip(states, rng.permutation(labels)))


def _ri_from_stats(
    observed: float,
    mc_stats: List[float],
    exact_stats: Optional[List[float]] = None,
    *,
    rm_support_cutoff: int = 99,
) -> float:
    return float(
        _ri_from_stats_info(
            observed,
            mc_stats,
            exact_stats,
            rm_support_cutoff=rm_support_cutoff,
        )["p_value"]
    )


def _ri_from_stats_info(
    observed: float,
    mc_stats: List[float],
    exact_stats: Optional[List[float]] = None,
    *,
    rm_support_cutoff: int = 99,
) -> Dict[str, Any]:
    base = {
        "p_value": np.nan,
        "ri_procedure": "undefined",
        "ri_support": 0.0,
        "ri_effective_assignments": 0.0,
    }
    if not np.isfinite(observed):
        return base
    exact_vals = np.asarray(
        [v for v in (exact_stats or []) if np.isfinite(v)],
        dtype=float,
    )
    if exact_stats is not None and len(exact_stats) > 0:
        base["ri_support"] = float(exact_vals.size)
        base["ri_effective_assignments"] = float(exact_vals.size)
        if exact_vals.size < 1:
            return base
        if exact_vals.size < rm_support_cutoff:
            out = kernel_smoothed_ri_pvalue(observed, exact_vals)
            base["p_value"] = float(out["p_value"])
            base["ri_procedure"] = "rm_exact"
            return base
        hits = int(np.sum(np.abs(exact_vals) >= abs(float(observed))))
        base["p_value"] = float((1 + hits) / (1 + exact_vals.size))
        base["ri_procedure"] = "ordinary_exact"
        return base

    vals = np.asarray([v for v in mc_stats if np.isfinite(v)], dtype=float)
    base["ri_support"] = float(vals.size)
    base["ri_effective_assignments"] = float(vals.size)
    if vals.size < 1:
        return base
    hits = int(np.sum(np.abs(vals) >= abs(float(observed))))
    base["p_value"] = float((1 + hits) / (1 + vals.size))
    base["ri_procedure"] = "sampled_999"
    return base


def _jackknife(
    panel_df: pd.DataFrame,
    base_core: Dict[str, Any],
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
    include_state_level_jackknife: bool = True,
    jackknife_support: str = "changing",
) -> Dict[str, Any]:
    if jackknife_support not in ("changing", "fixed"):
        raise ValueError("jackknife_support must be 'changing' or 'fixed'")
    states = panel_df[state_col].drop_duplicates().tolist()
    if len(states) < 2:
        return {
            "att": {
                "jk_se": np.nan,
                "jk_p_value": np.nan,
                "jackknife_available": 0.0,
                "jackknife_reason": "fewer than two states",
            },
            "att_g": {},
            "att_gt": {},
            "att_s": {},
            "att_st": {},
            "att_p": {},
            "att_pt": {},
        }

    (
        base_att,
        base_g_map,
        base_gt_map,
        base_s_map,
        base_st_map,
        base_p_map,
        base_pt_map,
    ) = _extract_estimate_maps(base_core)

    loo_att_values: List[float] = []
    loo_g_values: Dict[Hashable, List[float]] = {}
    loo_gt_values: Dict[Tuple[Any, Any], List[float]] = {}
    loo_s_values: Dict[Hashable, List[float]] = {}
    loo_st_values: Dict[Tuple[Any, Any], List[float]] = {}
    loo_p_values: Dict[Hashable, List[float]] = {}
    loo_pt_values: Dict[Tuple[Any, Any], List[float]] = {}

    numeric_engine = _compile_numeric_assignment_engine(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        pg_col=pg_col,
    )

    if numeric_engine is not None:
        engine_states = numeric_engine["states"]
        state_g = panel_df[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
        label_lookup = state_g.set_index(state_col)[gvar_col].to_dict()
        label_row = np.asarray([float(label_lookup.get(s, 0.0)) for s in engine_states], dtype=float)
        label_matrix = np.tile(label_row, (len(engine_states), 1))
        active_matrix = np.ones_like(label_matrix, dtype=bool)
        for idx in range(len(engine_states)):
            active_matrix[idx, idx] = False
        pg_matrix = None
        if pg_col and pg_col in panel_df.columns:
            state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
            pg_lookup = state_pg.set_index(state_col)[pg_col].to_dict()
            pg_row = np.asarray([str(pg_lookup.get(s, "0")) for s in engine_states], dtype=object)
            pg_matrix = np.tile(pg_row, (len(engine_states), 1))

        batch = _evaluate_assignment_matrix(
            numeric_engine,
            label_matrix,
            pg_matrix=pg_matrix,
            active_matrix=active_matrix,
        )
        loo_att_values.extend([float(v) for v in batch["att"] if np.isfinite(v)])
        for key in base_g_map:
            arr = np.asarray(batch["att_g"].get(key, []), dtype=float)
            loo_g_values[key] = [float(v) for v in arr if np.isfinite(v)]
        for key in base_gt_map:
            arr = np.asarray(batch["att_gt"].get(key, []), dtype=float)
            loo_gt_values[key] = [float(v) for v in arr if np.isfinite(v)]
        if include_state_level_jackknife:
            for key in base_s_map:
                arr = np.asarray(batch["att_s"].get(key, []), dtype=float)
                loo_s_values[key] = [float(v) for v in arr if np.isfinite(v)]
            for key in base_st_map:
                arr = np.asarray(batch["att_st"].get(key, []), dtype=float)
                loo_st_values[key] = [float(v) for v in arr if np.isfinite(v)]
        for key in base_p_map:
            arr = np.asarray(batch["att_p"].get(key, []), dtype=float)
            loo_p_values[key] = [float(v) for v in arr if np.isfinite(v)]
        for key in base_pt_map:
            arr = np.asarray(batch["att_pt"].get(key, []), dtype=float)
            loo_pt_values[key] = [float(v) for v in arr if np.isfinite(v)]
    else:
        for s in states:
            loo_panel = panel_df.loc[panel_df[state_col] != s]
            (
                loo_att,
                loo_g_map,
                loo_gt_map,
                loo_s_map,
                loo_st_map,
                loo_p_map,
                loo_pt_map,
            ) = _compute_estimate_maps_only(
                panel_df=loo_panel,
                state_col=state_col,
                time_col=time_col,
                gvar_col=gvar_col,
                weighting=weighting,
                not_yet_treated=not_yet_treated,
                pg_col=pg_col,
            )

            if np.isfinite(loo_att):
                loo_att_values.append(float(loo_att))

            for key, val in loo_g_map.items():
                if np.isfinite(val):
                    loo_g_values.setdefault(key, []).append(float(val))
            for key, val in loo_gt_map.items():
                if np.isfinite(val):
                    loo_gt_values.setdefault(key, []).append(float(val))
            if include_state_level_jackknife:
                for key, val in loo_s_map.items():
                    if np.isfinite(val):
                        loo_s_values.setdefault(key, []).append(float(val))
                for key, val in loo_st_map.items():
                    if np.isfinite(val):
                        loo_st_values.setdefault(key, []).append(float(val))
            for key, val in loo_p_map.items():
                if np.isfinite(val):
                    loo_p_values.setdefault(key, []).append(float(val))
            for key, val in loo_pt_map.items():
                if np.isfinite(val):
                    loo_pt_values.setdefault(key, []).append(float(val))

    cohort_labels = panel_df[[state_col, gvar_col]].drop_duplicates(state_col)[gvar_col]
    cohort_counts = cohort_labels[cohort_labels > 0].value_counts()
    singleton_cohort = bool((cohort_counts == 1).any())
    support_lost = jackknife_support == "fixed" and singleton_cohort
    jk_att_se = _jackknife_se(loo_att_values) if len(loo_att_values) == len(states) and not support_lost else np.nan
    jk_att_p = (
        _two_sided_pvalue(base_att / jk_att_se, dof=len(states) - 1)
        if np.isfinite(base_att) and np.isfinite(jk_att_se) and jk_att_se > 0
        else np.nan
    )

    def _jk_map(
        base_map: Dict[Any, float],
        val_map: Dict[Any, List[float]],
        require_all_replicates: bool = True,
    ) -> Dict[Any, Dict[str, float]]:
        out: Dict[Any, Dict[str, float]] = {}
        for key, base in base_map.items():
            vals = val_map.get(key, [])
            if require_all_replicates and len(vals) < len(states):
                out[key] = {
                    "jk_se": np.nan,
                    "jk_p_value": np.nan,
                    "jackknife_available": 0.0,
                    "jackknife_reason": "delete-one replicate removes target",
                }
                continue
            se = _jackknife_se(vals)
            p_val = (
                _two_sided_pvalue(base / se, dof=len(states) - 1)
                if np.isfinite(base) and np.isfinite(se) and se > 0
                else np.nan
            )
            out[key] = {
                "jk_se": se,
                "jk_p_value": p_val,
                "jackknife_available": float(np.isfinite(se)),
                "jackknife_reason": "" if np.isfinite(se) else "insufficient finite delete-one estimates",
            }
        return out

    return {
        "att": {
            "jk_se": jk_att_se,
            "jk_p_value": jk_att_p,
            "jackknife_available": float(np.isfinite(jk_att_se)),
            "jackknife_reason": "deletion removes a treatment cohort" if support_lost else ("" if np.isfinite(jk_att_se) else "insufficient finite delete-one estimates"),
            "jackknife_support": jackknife_support,
        },
        "att_g": _jk_map(base_g_map, loo_g_values, require_all_replicates=True),
        "att_gt": _jk_map(base_gt_map, loo_gt_values, require_all_replicates=True),
        "att_s": {
            key: {
                "jk_se": np.nan,
                "jk_p_value": np.nan,
                "jackknife_available": 0.0,
                "jackknife_reason": "focal-state deletion removes target",
            }
            for key in base_s_map
        } if include_state_level_jackknife else {},
        "att_st": {
            key: {
                "jk_se": np.nan,
                "jk_p_value": np.nan,
                "jackknife_available": 0.0,
                "jackknife_reason": "focal-state deletion removes target",
            }
            for key in base_st_map
        } if include_state_level_jackknife else {},
        "att_p": _jk_map(base_p_map, loo_p_values, require_all_replicates=True),
        "att_pt": _jk_map(base_pt_map, loo_pt_values, require_all_replicates=True),
    }


def _randomization(
    panel_df: pd.DataFrame,
    base_core: Dict[str, Any],
    n_perm: int,
    rng: np.random.Generator,
    state_col: str,
    time_col: str,
    gvar_col: str,
    weighting: str,
    not_yet_treated: bool,
    pg_col: Optional[str] = None,
    targets: str = "all",
) -> Dict[str, Any]:
    if n_perm <= 0:
        return {
            "att": np.nan,
            "att_g": {},
            "att_gt": {},
            "att_s": {},
            "att_st": {},
            "att_p": {},
            "att_pt": {},
        }

    state_g = panel_df[[state_col, gvar_col]].drop_duplicates(subset=[state_col])
    states = state_g[state_col].tolist()
    labels = state_g[gvar_col].to_numpy()
    if len(states) < 2:
        return {
            "att": np.nan,
            "att_g": {},
            "att_gt": {},
            "att_s": {},
            "att_st": {},
            "att_p": {},
            "att_pt": {},
        }

    (
        base_att,
        base_g_map,
        base_gt_map,
        base_s_map,
        base_st_map,
        base_p_map,
        base_pt_map,
    ) = _extract_estimate_maps(base_core)

    if targets not in {"all", "policy"}:
        raise ValueError("targets must be one of {'all', 'policy'}.")

    att_mc: List[float] = []
    att_exact: List[float] = []
    g_mc: Dict[Hashable, List[float]] = {k: [] for k in base_g_map}
    gt_mc: Dict[Tuple[Any, Any], List[float]] = {k: [] for k in base_gt_map}
    s_mc: Dict[Hashable, List[float]] = {k: [] for k in base_s_map}
    st_mc: Dict[Tuple[Any, Any], List[float]] = {k: [] for k in base_st_map}
    g_exact: Dict[Hashable, List[float]] = {k: [] for k in base_g_map}
    gt_exact: Dict[Tuple[Any, Any], List[float]] = {k: [] for k in base_gt_map}
    s_exact: Dict[Hashable, List[float]] = {k: [] for k in base_s_map}
    st_exact: Dict[Tuple[Any, Any], List[float]] = {k: [] for k in base_st_map}
    p_mc: Dict[Hashable, List[float]] = {k: [] for k in base_p_map}
    pt_mc: Dict[Tuple[Any, Any], List[float]] = {k: [] for k in base_pt_map}
    p_exact: Dict[Hashable, List[float]] = {k: [] for k in base_p_map}
    pt_exact: Dict[Tuple[Any, Any], List[float]] = {k: [] for k in base_pt_map}
    numeric_engine = _compile_numeric_assignment_engine(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        pg_col=pg_col,
    )

    def _record_maps(
        maps: Tuple[
            float,
            Dict[Hashable, float],
            Dict[Tuple[Any, Any], float],
            Dict[Hashable, float],
            Dict[Tuple[Any, Any], float],
            Dict[Hashable, float],
            Dict[Tuple[Any, Any], float],
        ],
        *,
        exact: bool,
        policy_only: bool = False,
    ) -> None:
        (
            perm_att,
            perm_g_map,
            perm_gt_map,
            perm_s_map,
            perm_st_map,
            perm_p_map,
            perm_pt_map,
        ) = maps
        if not policy_only:
            if np.isfinite(perm_att):
                (att_exact if exact else att_mc).append(float(perm_att))
            for key in base_g_map:
                val = perm_g_map.get(key, np.nan)
                if np.isfinite(val):
                    (g_exact if exact else g_mc)[key].append(float(val))
            for key in base_gt_map:
                val = perm_gt_map.get(key, np.nan)
                if np.isfinite(val):
                    (gt_exact if exact else gt_mc)[key].append(float(val))
            for key in base_s_map:
                val = perm_s_map.get(key, np.nan)
                if np.isfinite(val):
                    (s_exact if exact else s_mc)[key].append(float(val))
            for key in base_st_map:
                val = perm_st_map.get(key, np.nan)
                if np.isfinite(val):
                    (st_exact if exact else st_mc)[key].append(float(val))
        for key in base_p_map:
            val = perm_p_map.get(key, np.nan)
            if np.isfinite(val):
                (p_exact if exact else p_mc)[key].append(float(val))
        for key in base_pt_map:
            val = perm_pt_map.get(key, np.nan)
            if np.isfinite(val):
                (pt_exact if exact else pt_mc)[key].append(float(val))

    def _record_batch(batch: Dict[str, Any], *, exact: bool, policy_only: bool = False) -> None:
        if not policy_only:
            vals = np.asarray(batch["att"], dtype=float)
            (att_exact if exact else att_mc).extend([float(v) for v in vals if np.isfinite(v)])
            for key in base_g_map:
                arr = np.asarray(batch["att_g"].get(key, []), dtype=float)
                (g_exact if exact else g_mc)[key].extend([float(v) for v in arr if np.isfinite(v)])
            for key in base_gt_map:
                arr = np.asarray(batch["att_gt"].get(key, []), dtype=float)
                (gt_exact if exact else gt_mc)[key].extend([float(v) for v in arr if np.isfinite(v)])
            for key in base_s_map:
                arr = np.asarray(batch["att_s"].get(key, []), dtype=float)
                (s_exact if exact else s_mc)[key].extend([float(v) for v in arr if np.isfinite(v)])
            for key in base_st_map:
                arr = np.asarray(batch["att_st"].get(key, []), dtype=float)
                (st_exact if exact else st_mc)[key].extend([float(v) for v in arr if np.isfinite(v)])
        for key in base_p_map:
            arr = np.asarray(batch["att_p"].get(key, []), dtype=float)
            (p_exact if exact else p_mc)[key].extend([float(v) for v in arr if np.isfinite(v)])
        for key in base_pt_map:
            arr = np.asarray(batch["att_pt"].get(key, []), dtype=float)
            (pt_exact if exact else pt_mc)[key].extend([float(v) for v in arr if np.isfinite(v)])

    def _evaluate_label_maps(
        assignment_maps: List[Dict[Any, Any]],
        *,
        exact: bool,
        policy_only: bool = False,
        pg_assignments: bool = False,
    ) -> None:
        if not assignment_maps:
            return
        if numeric_engine is None:
            for assignment_map in assignment_maps:
                perm_panel = panel_df.copy()
                if pg_assignments and pg_col:
                    perm_panel[pg_col] = perm_panel[state_col].map(assignment_map).fillna("0").astype(str)
                    perm_panel[gvar_col] = perm_panel[pg_col].map(_pg_gvar)
                else:
                    perm_panel[gvar_col] = perm_panel[state_col].map(assignment_map)
                perm_maps = _compute_estimate_maps_only(
                    panel_df=perm_panel,
                    state_col=state_col,
                    time_col=time_col,
                    gvar_col=gvar_col,
                    weighting=weighting,
                    not_yet_treated=not_yet_treated,
                    pg_col=pg_col,
                )
                _record_maps(perm_maps, exact=exact, policy_only=policy_only)
            return

        matrix = _assignment_maps_to_matrices(states, assignment_maps)
        if pg_assignments:
            pg_matrix = matrix.astype(object)
            g_matrix = np.vectorize(_pg_gvar, otypes=[float])(pg_matrix)
            batch = _evaluate_assignment_matrix(numeric_engine, g_matrix, pg_matrix=pg_matrix)
        else:
            batch = _evaluate_assignment_matrix(numeric_engine, matrix.astype(float), pg_matrix=None)
        _record_batch(batch, exact=exact, policy_only=policy_only)

    if targets == "all":
        full_support = _support_size_from_labels(labels)
        if full_support - 1 < n_perm:
            _evaluate_label_maps(
                list(_enumerate_label_permutations(states, labels)),
                exact=True,
            )
        if full_support - 1 >= n_perm:
            _evaluate_label_maps(
                list(_random_label_permutations(states, labels, rng, n_perm)),
                exact=False,
            )

    if pg_col and pg_col in panel_df.columns and base_p_map:
        state_pg = panel_df[[state_col, pg_col]].drop_duplicates(subset=[state_col])
        pg_lookup = state_pg.set_index(state_col)[pg_col].to_dict()
        pg_labels = np.asarray([str(pg_lookup.get(s, "0")) for s in states], dtype=object)
        pg_support = _support_size_from_labels(pg_labels)
        if pg_support - 1 < n_perm:
            _evaluate_label_maps(
                list(_enumerate_label_permutations(states, pg_labels)),
                exact=True,
                policy_only=True,
                pg_assignments=True,
            )
        if pg_support - 1 >= n_perm:
            _evaluate_label_maps(
                list(_random_label_permutations(states, pg_labels, rng, n_perm)),
                exact=False,
                policy_only=True,
                pg_assignments=True,
            )

    def _finalize(
        base_map: Dict[Any, float],
        mc_map: Dict[Any, List[float]],
        exact_map: Dict[Any, List[float]],
    ) -> Dict[Any, float]:
        return {
            key: _ri_from_stats(
                base_map[key],
                mc_map.get(key, []),
                exact_map.get(key, []),
            )
            for key in base_map
        }

    def _finalize_info(
        base_map: Dict[Any, float],
        mc_map: Dict[Any, List[float]],
        exact_map: Dict[Any, List[float]],
    ) -> Dict[Any, Dict[str, Any]]:
        return {
            key: _ri_from_stats_info(base_map[key], mc_map.get(key, []), exact_map.get(key, []))
            for key in base_map
        }

    att_info = _ri_from_stats_info(base_att, att_mc, att_exact)

    return {
        "att": float(att_info["p_value"]),
        "att_g": _finalize(base_g_map, g_mc, g_exact),
        "att_gt": _finalize(base_gt_map, gt_mc, gt_exact),
        "att_s": _finalize(base_s_map, s_mc, s_exact),
        "att_st": _finalize(base_st_map, st_mc, st_exact),
        "att_p": _finalize(base_p_map, p_mc, p_exact),
        "att_pt": _finalize(base_pt_map, pt_mc, pt_exact),
        "ri_info": {
            "att": att_info,
            "att_g": _finalize_info(base_g_map, g_mc, g_exact),
            "att_gt": _finalize_info(base_gt_map, gt_mc, gt_exact),
            "att_s": _finalize_info(base_s_map, s_mc, s_exact),
            "att_st": _finalize_info(base_st_map, st_mc, st_exact),
            "att_p": _finalize_info(base_p_map, p_mc, p_exact),
            "att_pt": _finalize_info(base_pt_map, pt_mc, pt_exact),
        },
    }


def run_undid(
    panel_df: pd.DataFrame,
    *,
    state_col: str = "state",
    time_col: str = "time",
    gvar_col: str = "gvar",
    pg_col: Optional[str] = None,
    weighting: str = "both",
    not_yet_treated: bool = False,
    n_perm: int = 399,
    do_jackknife: bool = True,
    include_state_level_jackknife: bool = True,
    randomization_targets: str = "all",
    random_state: Optional[Any] = None,
    include_aggregation_weights: bool = True,
    include_legacy_aggregates: bool = True,
    jackknife_support: str = "changing",
) -> Dict[str, Any]:
    """
    Estimate pooled UNDiD ATT, ATTg, ATTgt, ATTs, ATTst.
    """
    core = _compute_core(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        pg_col=pg_col,
        include_aggregation_weights=include_aggregation_weights,
        include_legacy_aggregates=include_legacy_aggregates,
    )

    att = dict(core["att"])
    att["jk_se"] = np.nan
    att["jk_p_value"] = np.nan
    att["ri_p_value"] = np.nan
    att["ri_procedure"] = ""
    att["ri_support"] = np.nan
    att["ri_effective_assignments"] = np.nan
    att["jackknife_available"] = np.nan
    att["jackknife_reason"] = ""

    att_g = core["att_g"].copy()
    att_gt = core["att_gt"].copy()
    att_s = core["att_s"].copy()
    att_st = core["att_st"].copy()
    att_p = core["att_p"].copy()
    att_pt = core["att_pt"].copy()

    for frame in (att_g, att_gt, att_s, att_st, att_p, att_pt):
        for col in ("jk_se", "jk_p_value", "ri_p_value"):
            frame[col] = np.nan
        frame["ri_procedure"] = ""
        frame["ri_support"] = np.nan
        frame["ri_effective_assignments"] = np.nan
        frame["jackknife_available"] = np.nan
        frame["jackknife_reason"] = ""

    if do_jackknife:
        jk = _jackknife(
            panel_df=panel_df,
            base_core=core,
            state_col=state_col,
            time_col=time_col,
            gvar_col=gvar_col,
            weighting=weighting,
            not_yet_treated=not_yet_treated,
            pg_col=pg_col,
            include_state_level_jackknife=include_state_level_jackknife,
            jackknife_support=jackknife_support,
        )
        att["jk_se"] = float(jk["att"]["jk_se"])
        att["jk_p_value"] = float(jk["att"]["jk_p_value"])
        att["jackknife_available"] = float(jk["att"].get("jackknife_available", np.nan))
        att["jackknife_reason"] = str(jk["att"].get("jackknife_reason", ""))

        if not att_g.empty:
            att_g["jk_se"] = att_g["g"].map(lambda x: jk["att_g"].get(x, {}).get("jk_se", np.nan))
            att_g["jk_p_value"] = att_g["g"].map(lambda x: jk["att_g"].get(x, {}).get("jk_p_value", np.nan))
            att_g["jackknife_available"] = att_g["g"].map(
                lambda x: jk["att_g"].get(x, {}).get("jackknife_available", np.nan)
            )
            att_g["jackknife_reason"] = att_g["g"].map(
                lambda x: jk["att_g"].get(x, {}).get("jackknife_reason", "")
            )

        if not att_gt.empty:
            att_gt["jk_se"] = att_gt.apply(
                lambda r: jk["att_gt"].get((r["g"], r["t"]), {}).get("jk_se", np.nan),
                axis=1,
            )
            att_gt["jk_p_value"] = att_gt.apply(
                lambda r: jk["att_gt"].get((r["g"], r["t"]), {}).get("jk_p_value", np.nan),
                axis=1,
            )
            att_gt["jackknife_available"] = att_gt.apply(
                lambda r: jk["att_gt"].get((r["g"], r["t"]), {}).get("jackknife_available", np.nan),
                axis=1,
            )
            att_gt["jackknife_reason"] = att_gt.apply(
                lambda r: jk["att_gt"].get((r["g"], r["t"]), {}).get("jackknife_reason", ""),
                axis=1,
            )

        if include_state_level_jackknife and not att_s.empty:
            att_s["jk_se"] = att_s["state"].map(
                lambda x: jk["att_s"].get(x, {}).get("jk_se", np.nan)
            )
            att_s["jk_p_value"] = att_s["state"].map(
                lambda x: jk["att_s"].get(x, {}).get("jk_p_value", np.nan)
            )
            att_s["jackknife_available"] = att_s["state"].map(
                lambda x: jk["att_s"].get(x, {}).get("jackknife_available", np.nan)
            )
            att_s["jackknife_reason"] = att_s["state"].map(
                lambda x: jk["att_s"].get(x, {}).get("jackknife_reason", "")
            )

        if include_state_level_jackknife and not att_st.empty:
            att_st["jk_se"] = att_st.apply(
                lambda r: jk["att_st"].get((r["state"], r["t"]), {}).get("jk_se", np.nan),
                axis=1,
            )
            att_st["jk_p_value"] = att_st.apply(
                lambda r: jk["att_st"].get((r["state"], r["t"]), {}).get("jk_p_value", np.nan),
                axis=1,
            )
            att_st["jackknife_available"] = att_st.apply(
                lambda r: jk["att_st"].get((r["state"], r["t"]), {}).get("jackknife_available", np.nan),
                axis=1,
            )
            att_st["jackknife_reason"] = att_st.apply(
                lambda r: jk["att_st"].get((r["state"], r["t"]), {}).get("jackknife_reason", ""),
                axis=1,
            )
        if not att_p.empty:
            att_p["jk_se"] = att_p["pvar"].map(
                lambda x: jk["att_p"].get(x, {}).get("jk_se", np.nan)
            )
            att_p["jk_p_value"] = att_p["pvar"].map(
                lambda x: jk["att_p"].get(x, {}).get("jk_p_value", np.nan)
            )
            att_p["jackknife_available"] = att_p["pvar"].map(
                lambda x: jk["att_p"].get(x, {}).get("jackknife_available", np.nan)
            )
            att_p["jackknife_reason"] = att_p["pvar"].map(
                lambda x: jk["att_p"].get(x, {}).get("jackknife_reason", "")
            )
        if not att_pt.empty:
            att_pt["jk_se"] = att_pt.apply(
                lambda r: jk["att_pt"].get((r["pvar"], r["t"]), {}).get("jk_se", np.nan),
                axis=1,
            )
            att_pt["jk_p_value"] = att_pt.apply(
                lambda r: jk["att_pt"].get((r["pvar"], r["t"]), {}).get("jk_p_value", np.nan),
                axis=1,
            )
            att_pt["jackknife_available"] = att_pt.apply(
                lambda r: jk["att_pt"].get((r["pvar"], r["t"]), {}).get("jackknife_available", np.nan),
                axis=1,
            )
            att_pt["jackknife_reason"] = att_pt.apply(
                lambda r: jk["att_pt"].get((r["pvar"], r["t"]), {}).get("jackknife_reason", ""),
                axis=1,
            )

    if n_perm > 0:
        rng = _as_generator(random_state)
        ri = _randomization(
            panel_df=panel_df,
            base_core=core,
            n_perm=n_perm,
            rng=rng,
            state_col=state_col,
            time_col=time_col,
            gvar_col=gvar_col,
            weighting=weighting,
            not_yet_treated=not_yet_treated,
            pg_col=pg_col,
            targets=randomization_targets,
        )
        att["ri_p_value"] = float(ri["att"])
        ri_info = ri.get("ri_info", {})
        att_info = ri_info.get("att", {})
        att["ri_procedure"] = str(att_info.get("ri_procedure", ""))
        att["ri_support"] = float(att_info.get("ri_support", np.nan))
        att["ri_effective_assignments"] = float(att_info.get("ri_effective_assignments", np.nan))

        def _info_value(family: str, key: Any, field: str, default: Any = np.nan) -> Any:
            return ri_info.get(family, {}).get(key, {}).get(field, default)

        if not att_g.empty:
            att_g["ri_p_value"] = att_g["g"].map(lambda x: ri["att_g"].get(x, np.nan))
            att_g["ri_procedure"] = att_g["g"].map(lambda x: _info_value("att_g", x, "ri_procedure", ""))
            att_g["ri_support"] = att_g["g"].map(lambda x: _info_value("att_g", x, "ri_support"))
            att_g["ri_effective_assignments"] = att_g["g"].map(
                lambda x: _info_value("att_g", x, "ri_effective_assignments")
            )
        if not att_gt.empty:
            att_gt["ri_p_value"] = att_gt.apply(
                lambda r: ri["att_gt"].get((r["g"], r["t"]), np.nan), axis=1
            )
            att_gt["ri_procedure"] = att_gt.apply(
                lambda r: _info_value("att_gt", (r["g"], r["t"]), "ri_procedure", ""), axis=1
            )
            att_gt["ri_support"] = att_gt.apply(
                lambda r: _info_value("att_gt", (r["g"], r["t"]), "ri_support"), axis=1
            )
            att_gt["ri_effective_assignments"] = att_gt.apply(
                lambda r: _info_value("att_gt", (r["g"], r["t"]), "ri_effective_assignments"), axis=1
            )
        if not att_s.empty:
            att_s["ri_p_value"] = att_s["state"].map(lambda x: ri["att_s"].get(x, np.nan))
            att_s["ri_procedure"] = att_s["state"].map(lambda x: _info_value("att_s", x, "ri_procedure", ""))
            att_s["ri_support"] = att_s["state"].map(lambda x: _info_value("att_s", x, "ri_support"))
            att_s["ri_effective_assignments"] = att_s["state"].map(
                lambda x: _info_value("att_s", x, "ri_effective_assignments")
            )
        if not att_st.empty:
            att_st["ri_p_value"] = att_st.apply(
                lambda r: ri["att_st"].get((r["state"], r["t"]), np.nan), axis=1
            )
            att_st["ri_procedure"] = att_st.apply(
                lambda r: _info_value("att_st", (r["state"], r["t"]), "ri_procedure", ""), axis=1
            )
            att_st["ri_support"] = att_st.apply(
                lambda r: _info_value("att_st", (r["state"], r["t"]), "ri_support"), axis=1
            )
            att_st["ri_effective_assignments"] = att_st.apply(
                lambda r: _info_value("att_st", (r["state"], r["t"]), "ri_effective_assignments"), axis=1
            )
        if not att_p.empty:
            att_p["ri_p_value"] = att_p["pvar"].map(lambda x: ri["att_p"].get(x, np.nan))
            att_p["ri_procedure"] = att_p["pvar"].map(lambda x: _info_value("att_p", x, "ri_procedure", ""))
            att_p["ri_support"] = att_p["pvar"].map(lambda x: _info_value("att_p", x, "ri_support"))
            att_p["ri_effective_assignments"] = att_p["pvar"].map(
                lambda x: _info_value("att_p", x, "ri_effective_assignments")
            )
        if not att_pt.empty:
            att_pt["ri_p_value"] = att_pt.apply(
                lambda r: ri["att_pt"].get((r["pvar"], r["t"]), np.nan), axis=1
            )
            att_pt["ri_procedure"] = att_pt.apply(
                lambda r: _info_value("att_pt", (r["pvar"], r["t"]), "ri_procedure", ""), axis=1
            )
            att_pt["ri_support"] = att_pt.apply(
                lambda r: _info_value("att_pt", (r["pvar"], r["t"]), "ri_support"), axis=1
            )
            att_pt["ri_effective_assignments"] = att_pt.apply(
                lambda r: _info_value("att_pt", (r["pvar"], r["t"]), "ri_effective_assignments"), axis=1
            )

    return {
        "method": "undid",
        "weighting": weighting,
        "not_yet_treated": not_yet_treated,
        "att": att,
        "att_g": att_g.sort_values("g").reset_index(drop=True),
        "att_gt": att_gt.sort_values(["g", "t"]).reset_index(drop=True),
        "att_s": att_s.sort_values(["g", "state"]).reset_index(drop=True),
        "att_st": att_st.sort_values(["g", "state", "t"]).reset_index(drop=True),
        "att_p": att_p.sort_values(["pvar"]).reset_index(drop=True),
        "att_pt": att_pt.sort_values(["pvar", "t"]).reset_index(drop=True),
        "aggregation_weights": core.get("aggregation_weights", pd.DataFrame()).reset_index(drop=True),
        "legacy_direct_aggregates": core.get("legacy_direct_aggregates", {}),
        "aggregation_note": core.get("aggregation_note", ""),
    }


def run_undid_from_raw(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    slope_sample: str = "full",
    weighting: str = "both",
    not_yet_treated: bool = False,
    n_perm: int = 399,
    do_jackknife: bool = True,
    include_state_level_jackknife: bool = True,
    randomization_targets: str = "all",
    random_state: Optional[Any] = None,
    include_aggregation_weights: bool = True,
    include_legacy_aggregates: bool = True,
    jackknife_support: str = "changing",
) -> Dict[str, Any]:
    panel_df = prepare_panel_data(
        df=df,
        outcome_col=outcome_col,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        covariates=covariates,
        slope_sample=slope_sample,
    )
    return run_undid(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        n_perm=n_perm,
        do_jackknife=do_jackknife,
        include_state_level_jackknife=include_state_level_jackknife,
        randomization_targets=randomization_targets,
        random_state=random_state,
        include_aggregation_weights=include_aggregation_weights,
        include_legacy_aggregates=include_legacy_aggregates,
        jackknife_support=jackknife_support,
    )



