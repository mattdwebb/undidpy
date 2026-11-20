"""Internal scalar probability routines; standard library only.

Regularized beta/gamma continued fractions: NIST DLMF 8.17 and 8.9.
Direct survival probabilities avoid subtracting a tiny upper tail from one.
"""
import math
from functools import lru_cache

@lru_cache(maxsize=256)
def _logbeta(a,b):
    if min(a,b)==.5 and max(a,b)>1000:
        z=max(a,b);return .5*math.log(math.pi)-.5*math.log(z)+1/(8*z)-1/(192*z**3)+1/(640*z**5)
    return math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)

def _betacf(a,b,x):
    tiny=1e-300;c=1.;d=1-(a+b)*x/(a+1)
    if abs(d)<tiny:d=tiny
    d=1/d;h=d
    for m in range(1,10001):
        for aa in (m*(b-m)*x/((a+2*m-1)*(a+2*m)),-(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))):
            d=1+aa*d;c=1+aa/c
            if abs(d)<tiny:d=tiny
            if abs(c)<tiny:c=tiny
            d=1/d;delta=d*c;h*=delta
        if abs(delta-1)<=4*math.ulp(1.):return h
    raise ArithmeticError('incomplete beta did not converge')

def _ibeta(a,b,x,xc=None):
    if x<=0:return 0.
    if xc is None:xc=1-x
    if xc<=0:return 1.
    if x>(a+1)/(a+b+2):return 1-_ibeta(b,a,xc,x)
    logx=math.log1p(-xc) if xc<.5 else math.log(x)
    logxc=math.log1p(-x) if x<.5 else math.log(xc)
    value=math.exp(a*logx+b*logxc-_logbeta(a,b))*_betacf(a,b,x)/a
    return min(1.,max(0.,value))

def student_t_sf(value,df):
    value=float(value);df=float(df)
    if df<=0 or math.isnan(df):raise ValueError('positive degrees of freedom required')
    if math.isnan(value):return math.nan
    if math.isinf(df):return normal_sf(value)
    if math.isinf(value):return 0. if value>0 else 1.
    ratio=abs(value)/math.sqrt(df)
    if ratio>1e150:
        p=.5*math.exp(-df*math.log(ratio)-_logbeta(df/2,.5)-math.log(df/2))
    else:
        x=1/(1+ratio*ratio);xc=ratio*ratio/(1+ratio*ratio)
        p=.5*_ibeta(df/2,.5,x,xc)
    return p if value>=0 else 1-p

def normal_sf(value):return .5*math.erfc(float(value)/math.sqrt(2.))

def chi_square_sf(value,df):
    x=float(value)/2;a=float(df)/2
    if a<=0 or not math.isfinite(a):raise ValueError('finite positive degrees of freedom required')
    if math.isnan(x):return math.nan
    if x<=0:return 1.
    if math.isinf(x):return 0.
    factor=math.exp(a*math.log(x)-x-math.lgamma(a));eps=4*math.ulp(1.)
    if x<a+1:
        term=total=1/a
        for n in range(1,100001):
            term*=x/(a+n);total+=term
            if abs(term)<=abs(total)*eps:return max(0.,1-factor*total)
    else:
        b=x+1-a;c=1e300;d=1/b;h=d
        for n in range(1,100001):
            an=-n*(n-a);b+=2;d=an*d+b;c=b+an/c
            if abs(d)<1e-300:d=1e-300
            if abs(c)<1e-300:c=1e-300
            d=1/d;delta=d*c;h*=delta
            if abs(delta-1)<=eps:return max(0.,min(1.,factor*h))
    raise ArithmeticError('incomplete gamma did not converge')

class student_t:
    sf=staticmethod(student_t_sf)
class norm:
    sf=staticmethod(normal_sf)
class chi2:
    sf=staticmethod(chi_square_sf)
