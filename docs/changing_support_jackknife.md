# Aggregate jackknife support

The default `jackknife_support="changing"` evaluates the aggregate ATT on
each remaining sample, recomputing its treated-post mass weights. Deleting a
singleton adoption cohort is allowed when another treated cohort and eligible
controls remain. This convention does not establish identification or validity
of the jackknife; finite-sample behavior must be assessed separately.

Use `jackknife_support="fixed"` to reproduce the earlier singleton-cohort
failure rule. Every required state deletion must be finite. No treated units,
no eligible controls, rank failure, or a numerically zero standard error still
precludes inference. Disaggregate targets that disappear after deletion remain
undefined. Centering is the mean of all delete-one estimates, and aggregate
tests use Student t with the original number of clusters minus one degrees of
freedom. Subtract-placebo designs retain the selected support convention.
