# undidpy

**UN-DID comparisons in Python.** Estimate a policy effect from school-level changes and comparisons, with a delete-school jackknife for the aggregate.

UN-DID forms changes within treated schools and compares them with eligible untreated schools. With covariates it adjusts those comparisons before aggregating the supported school-period effects. This `Design` interface works on one local dataset; it is not the original multi-silo file-exchange workflow.

## Install and run

Use Python 3.10 or newer. Install directly from GitHub; this package is not distributed on PyPI. Clone the repository
(or download and extract its ZIP), then run these commands from its root:

```sh
git clone https://github.com/mattdwebb/undidpy.git
cd undidpy
python -m pip install "undidpy[examples] @ git+https://github.com/mattdwebb/undidpy.git"
python examples/quickstart.py
```

The installer supplies the numerical dependencies and pandas for this example.
No R or Stata installation is required. A virtual environment is recommended
when installing this package. Record the source commit for reproducible work.

## A school-policy example

The example has **960 pupils in 12 schools over eight years**. Different pupils are observed each year. The outcome is a test score and the covariate is a predetermined baseline score. Four schools adopt a policy in 2013, four in 2015, and four never adopt. The outcome is in score points, and the inference cluster is the school. These are fictional teaching data, not evidence about an actual school policy. The [data guide](examples/DATA_GUIDE.md) defines every column and links the deterministic generator.

The complete calculation in `examples/quickstart.py` is:

```python
import pandas as pd
from undidpy import Design

data = pd.read_csv("examples/school_policy.csv")
adoption = data.groupby("school")["first_treat"].first().to_dict()
model = Design(data, outcome="test_score", cluster="school", time="year",
               assignment=adoption, covariates=["baseline_score"])
jackknife = model.jackknife()
estimate = model.point()
native_se = None
jk_se = jackknife["standard_error"]
jk_p = jackknife["p_value"]
print(f"Estimate: {estimate:.6f} score points")
print("Native clustered SE: " + (f"{native_se:.6f}" if native_se is not None else "not available"))
print(f"Delete-school jackknife SE: {jk_se:.6f}")
print(f"Jackknife p-value: {jk_p:.6f}")
print("Schools: 12; jackknife reference: t(11)")
```

Actual output (rounded to six decimal places):

```text
Estimate: 2.391928 score points
Native clustered SE: not available
Delete-school jackknife SE: 0.523918
Jackknife p-value: 0.000809
Schools: 12; jackknife reference: t(11)
```

The adjusted estimate is **2.39 score points**. The delete-school standard error is **0.52 points**: it measures how sensitive the estimate is to leaving out schools one at a time. The reported jackknife p-value tests a zero effect using Student t with 11 degrees of freedom. This implementation does not supply a separate native analytic standard error for the aggregate. “Not available” is intentional; the jackknife is not relabelled as native inference.

The generator adds a two-point policy effect, along with school effects, year
effects and varying disturbances. The fitted estimate need not be exactly two.
This example teaches the interface; it does not measure test size or establish
the causal assumptions needed for an application.

## Adapt the analysis

Change `covariates=["baseline_score"]` to `covariates=[]` for an unadjusted specification. For your own data, replace the CSV and column names, keep a constant adoption year within each school, and check the supported input rules below. Numeric covariates must be prepared before fitting.

Save the compact result, for example:

```python
pd.DataFrame([{"estimate": estimate, "native_se": native_se,
               "jackknife_se": jk_se, "jackknife_p": jk_p}]).to_csv(
    "policy_results.csv", index=False)
```

Ordinary coefficient randomization inference is available after the fit:

```python
ri = model.randomization_inference(reps=999, seed=123)
print(ri["p_value"])
```

This reassigns treatment timing across schools while preserving cohort counts.
It excludes the observed assignment and uses the inclusive two-sided plus-one
p-value. The assignment mechanism must be appropriate for the study; reassignment
does not establish that a real policy was randomized.


## Supported scope and failures

The default uses never-treated controls and treated post-period sample-size weights. `aggregation_weights="comparison"` selects the documented comparison-period weighting instead. Required row values must be finite and nonmissing. Only identified school-period comparisons enter the aggregate; unavailable comparisons are reported through the retained support. A singular nuisance coefficient vector is acceptable only when the requested contrast is still identified. There is no separate native analytic covariance for this aggregate, and observation weights are not exposed by this interface.

The jackknife uses every input school deletion. If a cohort disappears, the
estimator recomputes the supported aggregate and its weights over surviving
estimable effects. It does not discard failed deletions to produce a standard
error. A remaining aggregate must still be identified. Estimates from different
DiD methods need not target the same weighted effect when treatment effects vary.

## Help, references and checks

The [release-candidate validation record](RELEASE_VALIDATION.md) gives the exact

See [estimator provenance](docs/PROVENANCE.md), [inference definitions](docs/INFERENCE_SPEC.md), and [surviving-support rule](docs/changing_support_jackknife.md). These pages contain the detailed estimator and inference
conventions. Report reproducible problems through the repository’s
[issue tracker](https://github.com/mattdwebb/undidpy/issues), including package
version, a small synthetic example and the full error message.

To run the numerical tests and the executable README check:

```sh
git clone https://github.com/mattdwebb/undidpy.git
cd undidpy
python -m pip install "undidpy[test,examples] @ git+https://github.com/mattdwebb/undidpy.git"
python -m pytest -q
```

The [MIT license](LICENSE) applies to this implementation; linked provenance
notes identify external methods and validation software. Cite the underlying
estimator as well as the software revision used.
