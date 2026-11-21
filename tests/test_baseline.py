from pathlib import Path
import json, unittest
import numpy as np
import pandas as pd
from undidpy import undid

def estimate(data, covariates):
    panel=undid.prepare_panel_data(data,outcome_col="y",state_col="state",time_col="year",gvar_col="g",covariates=covariates)
    return float(undid._compute_core(panel,"state","year","g","both",False)["att"]["estimate"])

class BaselineTests(unittest.TestCase):
    def test_frozen_research_values(self):
        base=Path(__file__).parent/'fixtures'
        for case in json.loads((base/'expected.json').read_text()):
            data=pd.read_csv(base/case['file'])
            value=estimate(data,case['covariates'])
            np.testing.assert_allclose(value,case['estimate'],rtol=2e-12,atol=2e-12)
    def test_row_and_state_label_invariance(self):
        base=Path(__file__).parent/'fixtures'
        data=pd.read_csv(base/'unequal.csv')
        value=estimate(data,[])
        data=data.sample(frac=1,random_state=33)
        data['state']=data['state'].map({s:s*17+41 for s in data['state'].unique()})
        np.testing.assert_allclose(estimate(data,[]),value,rtol=2e-12,atol=2e-12)

if __name__=='__main__':unittest.main()
