"""Generate fictional teaching data; no research data or random state required."""
from pathlib import Path
import csv
import math

ROOT = Path(__file__).resolve().parent
FIELDS = ["school", "year", "pupil", "first_treat", "policy", "baseline_score", "test_score"]
rows = []
for school in range(1, 13):
    first = 2013 if school <= 4 else (2015 if school <= 8 else 0)
    for year in range(2010, 2018):
        period = year - 2009
        policy = int(first > 0 and year >= first)
        for pupil in range(1, 11):
            baseline = math.sin(school*.73 + period*.31 + pupil*.47) + .7*math.cos(pupil*.9)
            score = (50 + .4*school + .6*period + 1.2*baseline + 2*policy
                     + .8*math.sin(school*.6 + period*.8)
                     + .25*math.cos(pupil*school + period*.3))
            rows.append(dict(school=school, year=year, pupil=pupil,
                             first_treat=first, policy=policy,
                             baseline_score=round(baseline, 10), test_score=round(score, 10)))

def write(name, fields, records):
    with (ROOT/name).open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

write("school_policy.csv", FIELDS, rows)
panel=[]
for school in range(1,13):
    for year in range(2010,2018):
        cell=[r for r in rows if r["school"]==school and r["year"]==year]
        record={k:cell[0][k] for k in ["school","year","first_treat","policy"]}
        for key in ["baseline_score","test_score"]:
            record[key]=round(sum(r[key] for r in cell)/len(cell),10)
        panel.append(record)
write("school_panel.csv", ["school","year","first_treat","policy","baseline_score","test_score"], panel)
print("Wrote 960 pupil observations and 96 school-year means.")
