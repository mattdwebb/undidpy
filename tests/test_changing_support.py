"""Aggregate remaining-sample ATT versus explicit fixed-support jackknife."""
import unittest
import numpy as np
import pandas as pd
from undidpy import Design, InferenceError
from undidpy.undid import prepare_panel_data, run_undid

class ChangingSupport(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21891)
        self.data = pd.DataFrame([
            (s, t, x, s*.1+t*.2+(.2+s*.1+t*.05)*x+rng.normal())
            for s in range(6) for t in range(1,6)
            for x in rng.normal(size=15+s)
        ], columns=['s','t','x','y'])
        self.a = {s: (3 if s==0 else 4 if s==1 else 0) for s in range(6)}

    def make(self, data=None, assignment=None, covariates=(), **kw):
        return Design(self.data if data is None else data, outcome='y',cluster='s',time='t',
                      assignment=self.a if assignment is None else assignment,
                      covariates=covariates, **kw)

    def test_all_deletions_equal_physical_refits(self):
        for covs in ([], ['x']):
            d = self.make(covariates=covs)
            j = d.jackknife()
            self.assertEqual(j['diagnostics']['status'],'ok')
            physical = []
            for s in range(6):
                physical.append(self.make(self.data[self.data.s!=s],
                    {k:v for k,v in self.a.items() if k!=s},covariates=covs).point())
            np.testing.assert_allclose(j['delete_estimates'],physical,atol=1e-12,rtol=0)
            se = np.sqrt(5/6*np.sum((physical-np.mean(physical))**2))
            self.assertAlmostEqual(j['standard_error'],se,12)
            self.assertEqual(j['jackknife_df'],5)

    def test_fixed_support_reproduces_failure(self):
        j = self.make(jackknife_support='fixed').jackknife()
        self.assertEqual(len(j['diagnostics']['failures']),2)
        self.assertTrue(np.isnan(j['p_value']))
        self.assertEqual(self.make(jackknife_support='fixed')._subtract(.1).jackknife_support,'fixed')

    def test_no_final_treated_or_comparison(self):
        for a in ({s:3 if s==0 else 0 for s in range(6)},
                  {s:0 if s==5 else 3 for s in range(6)}):
            j = self.make(assignment=a).jackknife()
            self.assertEqual(len(j['diagnostics']['failures']),1)
            self.assertTrue(np.isnan(j['p_value']))

    def test_invalid_convention(self):
        with self.assertRaises(ValueError):
            self.make(jackknife_support='ignore')

    def test_low_level_aggregate_only(self):
        raw=self.data.assign(g=self.data.s.map(self.a))
        panel=prepare_panel_data(raw,outcome_col='y',state_col='s',time_col='t',gvar_col='g')
        args=dict(state_col='s',time_col='t',gvar_col='g',n_perm=0,
                  include_aggregation_weights=False,include_legacy_aggregates=False)
        changing=run_undid(panel,**args)
        fixed=run_undid(panel,jackknife_support='fixed',**args)
        self.assertTrue(np.isfinite(changing['att']['jk_se']))
        self.assertTrue(np.isnan(fixed['att']['jk_se']))
        self.assertTrue(changing['att_g']['jk_se'].isna().all())
        self.assertTrue(changing['att_s']['jk_se'].isna().all())

if __name__ == '__main__':
    unittest.main()
