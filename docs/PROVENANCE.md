# Software provenance audit

Audit time: 2026-09-04T19:03:35.366600+00:00

No pre-existing target repository was found. Research sources are read-only evidence; none was modified. Source manifests include paths, byte sizes, SHA-256, UTC mtimes, imports and entry points. Bytecode headers are evidence only; all required Python source exists. No estimator is reconstructed from bytecode.

## undidpy
Candidate chosen: `<historical-local-path>/undid.py`, with `canonical_aggregation.py` and `exact_ri.py`. No enclosing Git repository. Production `run_undid_fastrcsdid_density.py` explicitly imports this module and calls `prepare_panel_data` and `_compute_core`. Its no-additional-covariate path and microdata pair-specific covariate adjustment must remain distinct. `undid_common.py` is a separate common-effect implementation, not a replacement. Dependencies: numpy, pandas; existing optional scipy/mpmath inference fallbacks require later correction to enforce t inference. Public API: run_undid, run_undid_from_raw, prepare_panel_data. Tests: test_canonical_aggregation, test_exact_ri, test_final_inference_rules. Production configuration identifies module provenance but has no source hashes; selected numerical replay is still required to link current bytes to stored values.

## didintpy
Primary production path: `<historical-local-path>/fitted_didint.py`, prepare_fitted_first_stage and run_fitted_core, ccc=int, weight=N_s,t. run_expanded_didint_pilot.py imports it explicitly. Preserve didint.py as the intercept-based public implementation, plus its local copy of undid.py and aggregation/RI helpers; this is intentional duplicated code, not a dependency on undidpy. Earlier shareable/three-stage/FE variants and the didint_sunny_delivery_20260825 package are competing historical variants inventoried separately. No destructive selection. Dependencies: numpy/pandas; scipy for strict new t inference. Existing tests: test_didint_reconciliation and run_didint_refactor_validation. Source Git absent. Production run_metadata git_hash=null and its purpose says diagnostic/preproduction despite the directory name. Exact source identity cannot be established from directory naming alone.

## fastrcsdid
Nominated candidate: `<historical-local-path>/fastrcsdid-fast-ipw-dr-jk`. This is an R package, not an existing Python package. HEAD `02f2623d41fff51335163c0e9d7f8aa21ec40af8` is not the complete current candidate: uncommitted modifications and untracked RI/subtract modules are recorded in the manifest. The separate `<historical-local-path>/fastrcsdid` is the path named by both production runs. Both versions are preserved by manifests and must be retained as references. The user explicitly confirmed that fastrcsdid should remain R. It was migrated as an R repository; no Python port was made. R entry points: fast_csdid, collapse_rcs, evaluate_assignment, aggregate_att, jackknife/RI helpers; see DESCRIPTION for dependencies and tests/testthat for tests. Python source search in csjack found simulation helpers, not a Python CSDID package. The current working R source was overlaid on a history-preserving single-branch clone. Source/copy equality and none/reg fast-versus-brute deletions passed; existing RA/IPW/DR functionality is retained.

## fastbjspy
Chosen: `<historical-local-path>/fast_bjs.py`, with cache/result and canonical aggregation modules. estimate_bjs fits state/time effects and optional ordinary covariates on untreated observations, imputes untreated outcomes, and aggregates ATT(s,t). fast_flex.py imports BJS design/OLS helpers but estimates a different pooled regression with group-time treatment interactions. FLEX is supporting/historical source, not BJS and not a sixth package. Both source docstrings explicitly withhold official Stata certification. The expanded production runner imports both. Dependencies numpy/pandas. Existing test_fast_estimators has additive-effect, reassignment, enumeration, and deletion checks. Preserve provisional certification status.

## fasttwfepy
New Python package: no old authoritative package implementation. Direct dummy-variable OLS is the numerical oracle; any fast implementation must match it with and without ordinary covariates.

## Production and CPS provenance
`<historical-local-path>/undid_fastrcsdid_10000_final`: run_config.json records seed 20260821, R=8/16/24, J=L=2, point-estimation only; CSDID is native simple cohort aggregation and UN-DID state-first aggregation. Source CPS is `<historical-local-path>/MORG_cleaned.dta`. generate_matched_cps_manifest.py uses complete cases for state/year/log_wage/age/age_sq/educ1/educ2/educ4, eight-year windows with treatment in positions 5 and 7. The required 40/10/10/20 draw is a new single validation draw using this timing rule, not a historical production replication.

`<historical-local-path>/didint_expanded_10000rep_production`: integrated fitted DID-INT, BJS, FLEX and R CSDID appear in metadata; source hashes are not recorded there. No 10,000-replication output is copied into package history or rerun.

## Gate status
Python source choices have reasonable provenance confidence, conditional on baseline numerical replay. Update: all four imported estimators passed selected production numerical replays (maximum difference 2.23e-16). All five now have validated fast scalar inference and a completed CPS report. Add is defined only for TWFE; Add is intentionally outside the intended API for the other estimators. R fastrcsdid is retained by explicit user instruction. See JSON manifests for exact timestamps, hashes and all discovered bytecode filenames.

## Final source integrity

Historical research replay artifacts are excluded from this distribution. The packaged tests and RELEASE_VALIDATION.md document the supported estimator checks.
