# Runtime dependencies

No external econometrics package is required. Python uses NumPy and pandas (data preparation and structured outputs). SciPy is a validation comparator only: internal standard-library Student-t/normal/chi-square routines and NumPy algebra replace its runtime uses. Probability routines are checked against base R across central/tail probabilities and small/large df; the exact Cauchy formula independently checks df=1. Installed SciPy 1.18.0 has an approximately 1.56e-9 discrepancy near t=1e-8, df=1; base R and the analytic formula agree with the internal implementation.

The standalone standard-library FE module is experimental; this does not make the complete estimator package strictly dependency-free. Existing inference definitions, t(G-1), assignment support and bootstrap rules are preserved. External packages and datasets used by tests are not runtime dependencies.
