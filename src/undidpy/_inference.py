"""Independent local inference helpers, duplicated source version did-inference-v1.

No runtime dependency on another estimator package. Fixed-count multiset
rank/unrank uses bounded Floyd sampling, including supports larger than int64.
"""
import math
import random
from collections import Counter
from collections.abc import Mapping
import numpy as np
from ._probability import student_t

class InferenceError(ValueError):
    def __init__(self, reason, detail=''):
        self.reason=reason
        super().__init__(reason+(': '+detail if detail else ''))


def strict_lstsq(x, y, *, absorbed_df=0):
    """Reject unidentified/saturated designs; never choose an implicit constraint."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(y) != len(x):
        raise InferenceError('invalid_input', 'misaligned covariate design')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise InferenceError('invalid_input', 'nonfinite covariate design')
    if x.shape[0] <= x.shape[1] + absorbed_df:
        raise InferenceError('numerical_failure', 'saturated covariate design')
    try:
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=1e-10)
    except np.linalg.LinAlgError as exc:
        raise InferenceError('numerical_failure', 'least-squares solve failed') from exc
    if rank != x.shape[1]:
        raise InferenceError('numerical_failure', 'rank-deficient covariate design')
    return beta

def _multinomial(counts):
    out=math.factorial(sum(counts))
    for n in counts:out//=math.factorial(n)
    return out

def _rank(labels, values):
    counts=[labels.count(v) for v in values]; rank=0
    for value in labels:
        k=values.index(value)
        for j in range(k):
            if counts[j]:
                counts[j]-=1;rank+=_multinomial(counts);counts[j]+=1
        counts[k]-=1
    return rank

def _unrank(rank, counts, values):
    counts=list(counts);out=[]
    for _ in range(sum(counts)):
        for j,v in enumerate(values):
            if not counts[j]:continue
            counts[j]-=1;n=_multinomial(counts)
            if rank<n:out.append(v);break
            rank-=n;counts[j]+=1
    return tuple(out)

def assignment_draws(assignment, reps=999, seed=0, exact=False):
    labels=tuple(assignment);values=sorted(set(labels));counts=[labels.count(v) for v in values]
    support=_multinomial(counts)
    if support<2:raise InferenceError('constant_assignment')
    if not isinstance(reps,int) or reps<1:raise ValueError('reps must be a positive integer')
    if exact and support>100000:raise ValueError('exact support exceeds 100000; use sampling')
    B=support-1 if exact else min(reps,support-1);observed=_rank(labels,values)
    if B==support-1:ranks=range(support-1)
    else:
        rng=random.Random(seed);chosen=set();ranks=[]
        for j in range(support-1-B,support-1):
            candidate=rng.randrange(j+1);v=j if candidate in chosen else candidate
            chosen.add(v);ranks.append(v)
    # Excludes observed, with no rejection loop and no replacement draws.
    return (_unrank(r+(r>=observed),counts,values) for r in ranks)

def inclusive_upper(values, observed):
    """Inclusive tails, treating roundoff-scale differences as numerical ties."""
    values=np.asarray(values,float)
    tolerance=64*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(values),abs(observed)))
    return (values>=observed)|(np.abs(values-observed)<=tolerance)


def inclusive_lower(values, observed):
    # Relative p-value tolerance avoids rounding tiny tail probabilities to zero.
    values=np.asarray(values,float)
    tolerance=64*np.finfo(float).eps*np.maximum(np.abs(values),abs(observed))
    return (values<=observed)|(np.abs(values-observed)<=tolerance)

class InferenceMixin:
    def _assignment(self, assignment):
        if isinstance(assignment,Mapping):
            if set(assignment)!=set(self.states):raise InferenceError('invalid_timing_assignment','state keys differ from estimation sample')
            a=np.asarray([assignment[s] for s in self.states],dtype=float)
        else:a=np.asarray(assignment,dtype=float)
        if a.shape!=(len(self.states),) or not np.isfinite(a).all() or (a<0).any():
            raise InferenceError('invalid_timing_assignment','use 0 for never and finite positive treatment dates')
        for g in set(a)-{0.0}:
            if not (np.any(self.times<g) and np.any(self.times>=g)):
                raise InferenceError('invalid_timing_assignment','each cohort needs pre and post observations')
        if not np.any(a>0):raise InferenceError('constant_assignment')
        return a

    def point(self, assignment=None, *, engine='fast', omit=None, procedure='replace', observed=None):
        if engine not in ('fast','reference'):raise ValueError('engine must be fast or reference')
        a=self.assignment if assignment is None else self._assignment(assignment)
        value=float(self._estimate(a,engine,omit,procedure,observed))
        if not np.isfinite(value):raise InferenceError('nonfinite_estimate')
        return value

    def jackknife(self, assignment=None, *, null=0., engine='fast', procedure='replace', observed=None):
        a=self.assignment if assignment is None else self._assignment(assignment)
        estimate=self.point(a,engine=engine,procedure=procedure,observed=observed)
        G=len(self.states)
        if G<2:raise InferenceError('insufficient_clusters')
        deleted=[];failures=[]
        for i,s in enumerate(self.states):
            try:
                # Aggregate ATT is reweighted over remaining treated cohorts by default.
                # Fixed-support replay remains available explicitly.
                if self.jackknife_support=='fixed' and a[i]>0 and np.sum(a==a[i])==1:
                    raise InferenceError('undefined_estimand','deletion removes a cohort')
                deleted.append(self.point(a,engine=engine,omit=i,procedure=procedure,observed=observed))
            except InferenceError as e:
                deleted.append(float('nan'));failures.append({'cluster':str(s),'reason':e.reason})
        deleted=np.asarray(deleted);se=float('nan');stat=float('nan');p=float('nan');status='ok'
        if failures:status='singular_jackknife_replication'
        else:
            se=float(np.sqrt((G-1)/G*np.sum((deleted-deleted.mean())**2)))
            tolerance=64*np.finfo(float).eps*max(1.,abs(estimate),float(np.max(np.abs(deleted))))
            if se<=tolerance or not np.isfinite(se):status='zero_standard_error'
            else:stat=(estimate-null)/se;p=float(2*student_t.sf(abs(stat),G-1))
        return dict(estimate=estimate,standard_error=se,statistic=stat,p_value=p,method='cluster_jackknife',cluster_count=G,jackknife_standard_error=se,jackknife_t=stat,jackknife_df=G-1,jackknife_p_value=p,delete_estimates=deleted,diagnostics={'status':status,'failures':failures,'centering':'mean of delete-one estimates','engine':engine,'jackknife_support':self.jackknife_support})

    def _draws(self, assignments, reps, seed, exact):
        draws=assignment_draws(tuple(self.assignment),reps,seed,exact) if assignments is None else assignments
        seen=set();counts=Counter(self.assignment)
        for a in draws:
            a=self._assignment(a);key=tuple(a)
            if Counter(a)!=counts:raise InferenceError('invalid_timing_assignment','cohort counts changed')
            if key==tuple(self.assignment):raise InferenceError('invalid_placebo_assignment','observed assignment excluded')
            if key in seen:raise InferenceError('invalid_placebo_assignment','duplicate supplied assignment')
            seen.add(key);yield a

    def randomization_inference(self, *, assignments=None,reps=999,seed=0,exact=False,null=0.,engine='fast'):
        design=self if null==0 else self._subtract(null)
        actual=design.point(engine=engine);statistics=[];failures=[]
        for i,a in enumerate(self._draws(assignments,reps,seed,exact)):
            try:statistics.append(design.point(a,engine=engine))
            except InferenceError as e:statistics.append(float('nan'));failures.append({'draw':i,'reason':e.reason})
        stats=np.asarray(statistics);B=len(stats)
        if B==0:raise ValueError('at least one assignment is required')
        hit_indicators=inclusive_upper(np.abs(stats),abs(actual));hits=int(np.sum(hit_indicators));p=(1+hits)/(B+1) if not failures else float('nan')
        return dict(estimate=self.point(engine=engine),statistic=actual,p_value=p,method='randomization_inference',cluster_count=len(self.states),statistics=stats,hit_indicators=hit_indicators,hits=hits,draws=B,valid_draws=B-len(failures),invalid_draws=len(failures),diagnostics={'status':'invalid_draws' if failures else 'ok','failures':failures,'tail':'inclusive absolute statistic; numerical ties within 64 eps times max(1, magnitudes)','correction':'plus_one; observed excluded','engine':engine})

    def placebo(self, procedure, *, assignments=None,reps=999,seed=0,exact=False,null=0.,engine='fast'):
        if procedure not in ('add','replace','subtract'):raise ValueError('unknown placebo procedure')
        if null!=0:raise ValueError('placebo calibration currently specifies null=0 only')
        if procedure=='add' and self.kind!='twfe':raise NotImplementedError('Add placebo is intentionally unsupported for this estimator; use Replace or Subtract')
        actual=self.jackknife(engine=engine)
        if actual['diagnostics']['status']!='ok':raise InferenceError(actual['diagnostics']['status'])
        design=self._subtract(actual['estimate']) if procedure=='subtract' else self
        inner=[];failures=[]
        for i,a in enumerate(self._draws(assignments,reps,seed,exact)):
            try:
                fit=design.jackknife(a,engine=engine,procedure='add' if procedure=='add' else 'replace',observed=self.assignment)
                if fit['diagnostics']['status']!='ok':raise InferenceError(fit['diagnostics']['status'])
                inner.append(fit['p_value'])
            except InferenceError as e:inner.append(float('nan'));failures.append({'draw':i,'reason':e.reason})
        values=np.asarray(inner);B=len(values)
        if B==0:raise ValueError('at least one assignment required')
        hit_indicators=inclusive_lower(values,actual['p_value']);hits=int(np.sum(hit_indicators));p=(1+hits)/(B+1) if not failures else float('nan')
        return dict(estimate=actual['estimate'],p_value=p,empirical_placebo_p_value=p,method='placebo_'+procedure,cluster_count=len(self.states),actual_p_value=actual['p_value'],procedural_p_values=values,hit_indicators=hit_indicators,hits=hits,draws=B,valid_draws=B-len(failures),invalid_draws=len(failures),diagnostics={'status':'invalid_draws' if failures else 'ok','failures':failures,'tail':'inclusive lower procedural p value; relative numerical ties within 64 eps','correction':'plus_one; observed excluded','engine':engine,'subtract_fit':'full-sample scalar fixed throughout each inner jackknife'})
    def placebo_add(self,**kwargs):return self.placebo('add',**kwargs)
    def placebo_replace(self,**kwargs):return self.placebo('replace',**kwargs)
    def placebo_subtract(self,**kwargs):return self.placebo('subtract',**kwargs)
