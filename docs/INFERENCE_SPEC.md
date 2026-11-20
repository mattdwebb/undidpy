# undidpy inference specification

`Design(data, outcome=..., cluster=..., time=..., assignment=state_to_cohort, covariates=[])` rejects missing or nonfinite required row values, constructs estimates from identified state-period comparisons, and uses 0 for never treated. Retained observations and cohort assignment are fixed. Sampling weights are not supported by this new adapter. Historical public APIs remain separate.

Point target: state-first contrasts: control changes weighted by n_pre+n_post, ATT(s,t) weighted by n_post; never-treated controls. DID-INT uses the production fitted ccc=int path. Covariate UN-DID/DID-INT call the preserved reference implementation; fast cells are restricted to no covariates.

Jackknife refits all retained-state estimates, in sorted state order. The SE is sqrt((G-1)/G times sum((theta_minus_h - mean(theta_minus))^2)), i.e. mean-centered CV3J. P-values always use Student t(G-1), including G=40 -> df=39. A deletion may remove a cohort when a surviving aggregate remains identified. Comparisons and weights are rebuilt on each retained sample; no required deletion is silently dropped. This definition deliberately distinguishes the full-estimate-centered CV3 also provided in the R reference. The new cross-package comparison uses mean-centered CV3J consistently.

RI uses the raw aggregate, inclusive absolute tails, fixed cohort counts, observed assignment excluded, (1+hits)/(B+1). Nonzero RI nulls use homogeneous-sharp-null outcome adjustment Y-null*D. Streamed multiset enumeration is available for support <=100000; bounded rank sampling uses no replacement and no uniqueness rejection loop. Failed draws stay in the results; any failure makes the calibrated p-value unavailable, with counts/reasons.

Replace re-estimates under placebo timing with Y fixed. Subtract uses Y-theta_hat*D_actual, with theta_hat the full-sample scalar, fixed for every inner jackknife. Its homogeneous scalar interpretation is explicit; no claim is made for arbitrary heterogeneous-effect nulls. Both calibrate the actual jackknife p-value against inner jackknife t p-values with inclusive lower tails and plus-one correction.

Add is intentionally unsupported and is not part of this estimator's intended API. The legacy placebo_add placeholder raises NotImplementedError; use Replace or Subtract. Optional smoothing is deferred.

Fast paths are cell sums for UN-DID/DID-INT without covariates; cell crossproducts for BJS and TWFE, with cached fixed nuisance inverses across TWFE assignments/deletions. BJS recomputes the assignment-dependent untreated first stage. Direct microdata refits are available via engine="reference". Crossproducts may be less stable for ill-conditioned covariates; rank checks fail explicitly rather than using a pseudoinverse. No global data cache is used. Independent Design instances are suitable for separate workers; no parallel executor is launched by the package.

_inference.py and shared parts of api.py are intentionally duplicated (did-inference-v1), with no cross-repository imports. Tests validate each local copy. Low-level assignment combinatorics and scalar jackknife summaries could eventually form a common library, but none is created here.

## Numerical ties

Inclusive RI tails treat absolute statistics within 64 machine eps times max(1, both magnitudes) as ties. Placebo p-value comparisons use 64 machine eps times the larger p-value (no absolute floor). This makes exact symmetric assignments agree across algebraically equivalent solvers without rounding tiny p-values to zero. Hit indicators are returned explicitly. A four-state exact-support regression fixture demonstrated an untreated-regression sign-reversal tie differing by roughly 1e-15; strict floating comparisons previously produced 1/6 versus 2/6. The inclusive result is 2/6.
