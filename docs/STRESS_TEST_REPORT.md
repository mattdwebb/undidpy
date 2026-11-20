# Stress-test report

{
  "package": "undidpy",
  "candidate": false,
  "counts": {
    "pass": 25,
    "expected_failure": 3,
    "not_applicable": 2
  },
  "maxima": {
    "point_gap": 4.440892098500626e-16,
    "component_gap": 1.3322676295501878e-15,
    "weight_gap": 0.0,
    "delete_gap": 6.661338147750939e-16,
    "jk_gap": 5.329070518200751e-15,
    "ri_gap": 5.551115123125783e-16,
    "placebo_gap": 1.9984014443252818e-15
  }
}

See stress_test_results.csv for each design and stress_test_draws.json for fixed-assignment statistics and inner p-values. Missing cells are an explicit inference-adapter restriction; unsupported weights are N/A. Singleton-cohort deletion failures retain G and all deletions. Heterogeneous effects are diagnostic, not a validity proof. Reference point checks gate inference.

Reproduce from the repository root: `python tests/run_stress_validation.py undidpy`. Ordinary unit suite: `python -m unittest discover -s tests`. The stress runner and deterministic fixtures are included in this repository.
