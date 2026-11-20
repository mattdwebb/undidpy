import unittest
import numpy as np
from undidpy import Design,assignment_draws
from certification_fixtures import fixture

class CertificationRegressions(unittest.TestCase):
    def test_exact_complement_assignment_is_an_inclusive_tie(self):
        data,_=fixture('one_cohort')
        data=data.loc[data.state.isin([41,58,109,126])].copy()
        d=Design(data,outcome='y',cluster='state',time='year',assignment=dict(data[['state','g']].drop_duplicates().itertuples(index=False,name=None)))
        assignments=list(assignment_draws(tuple(d.assignment),exact=True))
        self.assertEqual(len(assignments),5)
        f=d.randomization_inference(assignments=assignments)
        r=d.randomization_inference(assignments=assignments,engine='reference')
        self.assertEqual(f['hits'],1)
        self.assertEqual(f['p_value'],2/6)
        np.testing.assert_array_equal(f['hit_indicators'],r['hit_indicators'])
        self.assertEqual(f['p_value'],r['p_value'])

    def test_nearly_collinear_point_uses_stable_oracle(self):
        data,covs=fixture('near_collinear')
        d=Design(data,outcome='y',cluster='state',time='year',assignment=dict(data[['state','g']].drop_duplicates().itertuples(index=False,name=None)),covariates=covs)
        self.assertAlmostEqual(d.point(),d.point(engine='reference'),places=8)
