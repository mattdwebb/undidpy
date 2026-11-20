# Validation record: 20 November 2026

The current candidate was checked in isolated Windows environments after the
publication audit corrections. The synthetic school-policy example retains its
recorded estimates and standard errors. Agreement across all nine estimator
families refers to that matched teaching example, not arbitrary empirical data.
UN-DID and DID-INT do not report a separate native analytical variance for their
supported aggregates. All required deletion estimates must exist for a reported
jackknife; unsupported samples are reported explicitly.

The complete local pytest suite and bundled README examples passed using an
isolated installation. The suite's local integration run included all nine
backends and R for its optional comparison. Hosted core CI and any optional
integration workflow are separate evidence; a skipped workflow is not a pass.
Run `python -m pytest -q` from this package to repeat available checks.

Across this release, new targeted checks cover weighted, unbalanced stacked samples with singleton
cohorts; suite FLEX nonzero-null Student references; and CSDID cohort and
comparison-period aggregation with regression adjustment and DRIPW, both control
rules, and every cluster deletion. These checks complement each package's
retained reference tests. They do not imply that every estimator supports every
panel structure, overlap pattern, or covariate specification.

The source checkouts under review remain separate from the original development
directories. Earlier preservation hashes cover the 32 software worktrees, not
all research simulations. Only the UN-DID pkgdown configuration was repaired;
the author's main website was not updated. Separate panel README examples are
provided for LP-DiD and the suite; other introductory examples use repeated cross
sections. CCC testing is outside this release's inference validation.
