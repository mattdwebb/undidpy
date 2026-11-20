# School-policy teaching data

These are fictional data for learning the software. They are not evidence about
a real education policy. `generate.py` recreates both CSV files using only the
Python standard library and deterministic formulas; it does not use random draws.

`school_policy.csv` has 960 observations: ten pupils in each of 12 schools in
each year from 2010 through 2017. Schools 1–4 adopt a policy in 2013, schools 5–8
adopt in 2015, and schools 9–12 never adopt. The policy remains in place thereafter.

| Column | Meaning |
|---|---|
| `school` | School identifier and inference cluster. |
| `year` | Calendar year, 2010–2017. |
| `pupil` | Row identifier within a school-year; not a longitudinal pupil ID. |
| `first_treat` | First policy year, or zero for never-treated schools. |
| `policy` | One after adoption, zero otherwise. |
| `baseline_score` | Continuous, predetermined score covariate in artificial units. |
| `test_score` | Outcome in test-score points. |

Pupils are repeated cross sections: do not link the pupil numbers across years.
The artificial policy adds two points to the outcome. School/year effects, a
baseline-score contribution, and smooth varying disturbances make estimates and
standard errors nontrivial. This is a teaching example, not a statistical coverage
experiment or proof of identification.

`school_panel.csv` has one row per school-year and contains the means of the two
score variables. It is a genuine panel of the same 12 schools; the panel unit is
the school. This second file permits a small LP-DiD example without inventing
individual pupil histories. Its units and target differ from an individual panel.

Use the same observation definitions within an analysis. The standalone stacked
DiD API uses missing/positive infinity for never treated, whereas the suite and
these CSV files use zero; example code performs the required explicit conversion.
