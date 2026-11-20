from pathlib import Path
import pandas as pd
from undidpy import Design

data = pd.read_csv(Path(__file__).with_name("school_policy.csv"))
adoption = data.groupby("school")["first_treat"].first().to_dict()
model = Design(data, outcome="test_score", cluster="school", time="year",
               assignment=adoption, covariates=["baseline_score"])
jackknife = model.jackknife()
estimate = model.point()
native_se = None
jk_se = jackknife["standard_error"]
jk_p = jackknife["p_value"]
assert jackknife["diagnostics"]["status"] == "ok"
print(f"Estimate: {estimate:.6f} score points")
print("Native clustered SE: " + (f"{native_se:.6f}" if native_se is not None else "not available"))
print(f"Delete-school jackknife SE: {jk_se:.6f}")
print(f"Jackknife p-value: {jk_p:.6f}")
print("Schools: 12; jackknife reference: t(11)")
