import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from undidpy import Design, InferenceError
from undidpy.undid import _pair_adjusted_delta_from_arrays as pair


class AuditRegressions(unittest.TestCase):
    def test_post_collinearity_rejected(self):
        with self.assertRaisesRegex(InferenceError,'numerical_failure'):
            pair(np.arange(5.),np.arange(5.)+1,np.zeros((5,1)),np.ones((5,1)),True)

    def test_saturated_pair_does_not_fall_back_to_unadjusted(self):
        with self.assertRaisesRegex(InferenceError,'numerical_failure'):
            pair(np.array([1.]),np.array([2.]),np.array([[0.]]),np.array([[1.]]),True)

    def test_solver_failure_does_not_fall_back(self):
        with patch('numpy.linalg.lstsq',side_effect=np.linalg.LinAlgError('test')):
            with self.assertRaisesRegex(InferenceError,'numerical_failure'):
                pair(np.arange(5.),np.arange(5.)+1,np.arange(5.)[:,None],np.arange(5.)[:,None],True)

    def test_full_rank_pair_and_no_covariates(self):
        x=np.arange(5.)[:,None]
        self.assertAlmostEqual(pair(3*x[:,0],2+3*x[:,0],x,x,True),2,12)
        self.assertAlmostEqual(pair(np.arange(5.),np.arange(5.)+2,None,None,False),2,12)

    def test_missing_values_rejected(self):
        d=pd.DataFrame([(s,t,j,float(s+t+j)) for s in range(6) for t in (1,2,3) for j in range(3)],
                       columns=['s','t','x','y'])
        d.loc[0,'x']=np.nan
        with self.assertRaisesRegex(InferenceError,'invalid_input'):
            Design(d,outcome='y',cluster='s',time='t',assignment={s:3 if s<2 else 0 for s in range(6)},covariates=['x'])

if __name__ == '__main__':
    unittest.main()
