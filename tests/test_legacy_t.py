import unittest
import numpy as np
from scipy.stats import t
from undidpy import undid

class LegacyTTests(unittest.TestCase):
    def test_t39_and_t7(self):
        for df in (7,39):
            self.assertAlmostEqual(undid._two_sided_pvalue(2.,df),2*t.sf(2.,df),places=14)

    def test_aggregate_rejects_a_removed_cohort(self):
        from test_inference import fixture
        data=fixture()
        data.loc[data.state==58,'g']=0
        panel=undid.prepare_panel_data(data,outcome_col='y',state_col='state',time_col='year',gvar_col='g',covariates=[])
        result=undid.run_undid(panel,state_col='state',time_col='year',gvar_col='g',n_perm=0,include_state_level_jackknife=False,jackknife_support='fixed')
        self.assertTrue(np.isnan(result['att']['jk_p_value']))
        self.assertIn('cohort',result['att']['jackknife_reason'])
