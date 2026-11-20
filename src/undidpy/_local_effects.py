"""State-level comparisons on the identified support of each treatment assignment."""
import numpy as np
from ._inference import InferenceError


def coefficients(x, y):
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    use = s > np.finfo(float).eps * max(x.shape) * s[0]
    basis = vt[use]
    return basis.T @ ((u[:, use].T @ y) / s[use]), basis.T @ basis


def identified(target, projection):
    return np.max(np.abs(target - target @ projection)) <= 1e-9 * max(1., np.max(np.abs(target)))


def local_effects(design, assignment, active, *, cache=True):
    times = design.times
    n = design.counts
    z = np.column_stack((np.ones(len(design.data)), design.data[design.covariates].to_numpy(float)))
    y = design.data[design.outcome].to_numpy(float)
    memory = design._nuisance_cache if cache else {}
    if "rows" not in memory:
        order = np.argsort(design._rowcell, kind="stable")
        memory["rows"] = np.split(order, np.cumsum(design.n.astype(int))[:-1])
    rows = memory["rows"]
    controls = np.flatnonzero(active & (assignment == 0))
    estimates, masses, support = [], [], []
    for group in np.unique(assignment[active & (assignment > 0)]):
        before = np.flatnonzero(times < group)
        if not len(before):
            continue
        base = before[-1]
        for post in np.flatnonzero(times >= group):
            values, treated_post, treated_pair = [], [], []
            for state in np.flatnonzero(active & (assignment == group)):
                post_rows = rows[state * len(times) + post]
                if not len(post_rows):
                    continue
                target = z[post_rows].mean(axis=0)
                changes, weights = [], []
                for h in [state, *controls]:
                    b, t = h * len(times) + base, h * len(times) + post
                    if not len(rows[b]) or not len(rows[t]):
                        value = np.nan
                    elif design.kind == "undid":
                        key = (h, base, post)
                        if key not in memory:
                            idx = np.r_[rows[b], rows[t]]
                            x = np.column_stack((z[idx], design._rowtime[idx] == post))
                            beta, projection = coefficients(x, y[idx])
                            memory[key] = beta[-1] if identified(np.eye(len(beta))[-1], projection) else np.nan
                        value = memory[key]
                    else:
                        for cell in (b, t):
                            if cell not in memory:
                                memory[cell] = coefficients(z[rows[cell]], y[rows[cell]])
                        bb, pb = memory[b]
                        bt, pt = memory[t]
                        value = target @ (bt - bb) if identified(target, pb) and identified(target, pt) else np.nan
                    if h == state:
                        own = value
                        if not np.isfinite(own):
                            break
                    elif np.isfinite(value):
                        changes.append(value)
                        weights.append(n[h, base] + n[h, post])
                if np.isfinite(own) and weights:
                    values.append(own - np.average(changes, weights=weights))
                    treated_post.append(n[state, post])
                    treated_pair.append(n[state, post] + n[state, base])
                    support.append((design.states[state], times[post], len(weights)))
            if values:
                comparison = getattr(design, "aggregation_weights", "post") == "comparison"
                estimates.append(np.average(values, weights=treated_pair if comparison else treated_post))
                masses.append(sum(treated_post))
    if not masses:
        raise InferenceError("undefined_estimand", "no identified treated-state comparison")
    design.last_support = tuple(support)
    return np.average(estimates, weights=masses)
