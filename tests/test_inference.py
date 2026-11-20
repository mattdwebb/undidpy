import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from undidpy import Design,InferenceError,assignment_draws


def fixture(unequal=False):
    rows=[]
    for h in range(8):
        g=3 if h<2 else 5 if h<4 else 0
        for tt in range(1,7):
            for j in range(4+(h+2*tt)%4 if unequal else 5):
                x=np.sin(h*tt+j*.7)+.1*j
                rows.append((h*17+41,tt,g,x,.3*h+.2*tt+1.1*(g>0 and tt>=g)+.3*x+.05*np.cos(h*tt)))
    return pd.DataFrame(rows,columns=['state','year','g','x','y'])

def design(data,covs=()):
    return Design(data,outcome='y',cluster='state',time='year',assignment=dict(data[['state','g']].drop_duplicates().itertuples(index=False,name=None)),covariates=covs)

class InferenceTests(unittest.TestCase):
    def test_fast_reference_point_deletions_t_and_relabeling(self):
        gaps=[]
        for unequal in [False,True]:
            data=fixture(unequal)
            for covs in [[],['x']]:
                d=design(data,covs)
                for a in [d.assignment,*list(assignment_draws(tuple(d.assignment),2,seed=23))]:
                    fast=d.jackknife(a);slow=d.jackknife(a,engine='reference')
                    self.assertEqual(fast['diagnostics']['status'],'ok')
                    for key in ['estimate','standard_error','statistic','p_value','delete_estimates']:
                        np.testing.assert_allclose(fast[key],slow[key],rtol=1e-8,atol=2e-10)
                    gaps.append(float(np.max(np.abs(fast['delete_estimates']-slow['delete_estimates']))))
                    self.assertEqual(fast['jackknife_df'],7)
                    self.assertAlmostEqual(fast['p_value'],2*t.sf(abs(fast['statistic']),7),14)
                shuffled=data.sample(frac=1,random_state=1).copy()
                shuffled['state']=shuffled['state'].map({s:f'label_{99-i}' for i,s in enumerate(sorted(data.state.unique()))})
                np.testing.assert_allclose(design(shuffled,covs).point(),d.point(),rtol=2e-11,atol=2e-11)
        print('MAX_DELETE_GAP',max(gaps))

    def test_fixed_ri_placebo_and_failures(self):
        for covs in [[],['x']]:
            d=design(fixture(True),covs);assignments=list(assignment_draws(tuple(d.assignment),3,seed=812))
            x=d.randomization_inference(assignments=assignments);y=d.randomization_inference(assignments=assignments,engine='reference')
            np.testing.assert_allclose(x['statistics'],y['statistics'],rtol=1e-9,atol=1e-10)
            self.assertEqual(x['hits'],y['hits']);self.assertEqual(x['p_value'],y['p_value'])
            for proc in ['replace','subtract']+(['add'] if d.kind=='twfe' else []):
                x=d.placebo(proc,assignments=assignments);y=d.placebo(proc,assignments=assignments,engine='reference')
                self.assertEqual(x['invalid_draws'],0)
                np.testing.assert_allclose(x['procedural_p_values'],y['procedural_p_values'],rtol=1e-8,atol=1e-9)
                self.assertEqual(x['hits'],y['hits']);self.assertEqual(x['p_value'],y['p_value'])
            if d.kind!='twfe':
                with self.assertRaisesRegex(NotImplementedError,'intentionally unsupported'):d.placebo_add(assignments=assignments)
            with self.assertRaisesRegex(InferenceError,'observed assignment excluded'):d.randomization_inference(assignments=[d.assignment])
            with self.assertRaisesRegex(InferenceError,'cohort counts changed'):d.randomization_inference(assignments=[np.repeat(3,8)])
        a=np.array([3,5,5,0,0,0,0,0]);x=d.jackknife(a)
        self.assertEqual(x['diagnostics']['status'],'ok')
        d.jackknife_support='fixed';x=d.jackknife(a)
        self.assertEqual(x['diagnostics']['status'],'singular_jackknife_replication')
        self.assertTrue(np.isnan(x['p_value']))

    def test_enumeration_and_large_rank_sampling(self):
        labels=(3,5,0,0)
        draws=list(assignment_draws(labels,exact=True))
        self.assertEqual(len(draws),11);self.assertEqual(len(set(draws)),11);self.assertNotIn(labels,draws)
        labels=tuple([3]*10+[5]*10+[0]*20)
        draws=list(assignment_draws(labels,999,seed=12))
        self.assertEqual(draws,list(assignment_draws(labels,999,seed=12)))
        self.assertEqual(len(set(draws)),999)
        self.assertTrue(all(a.count(3)==10 and a.count(5)==10 and a.count(0)==20 for a in draws))
