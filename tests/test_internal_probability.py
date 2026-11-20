import csv,math,pathlib,unittest
from undidpy._probability import student_t_sf,chi_square_sf,normal_sf
class ProbabilityTests(unittest.TestCase):
    def test_against_base_R_reference(self):
        with (pathlib.Path(__file__).parent/'probability_reference.csv').open() as f:
            for row in csv.DictReader(f):
                expected=float(row['reference']);df=float(row['df']);x=float(row['value'])
                actual=student_t_sf(x,df) if row['distribution']=='t' else chi_square_sf(x,df)
                self.assertLess(abs(actual-expected),2e-12)
                if expected>1e-280:self.assertLess(abs(actual/expected-1),2e-9)
    def test_analytic_cauchy_and_extreme_tails(self):
        for x in [-10.,-1e-8,0.,1e-8,10.]:self.assertAlmostEqual(student_t_sf(x,1),.5-math.atan(x)/math.pi,places=14)
        self.assertLess(abs(student_t_sf(1e200,1)/(1/(math.pi*1e200))-1),1e-12)
        self.assertEqual(normal_sf(0),.5)
