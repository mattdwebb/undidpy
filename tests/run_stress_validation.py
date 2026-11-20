"""Point-gated deterministic reference comparison; command: python stress_python.py PACKAGE [candidate]."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import importlib,json,pathlib,sys,time,traceback
import numpy as np
import pandas as pd
from scipy.stats import t
from certification_fixtures import fixture,DESIGNS
ROOT=pathlib.Path(__file__).resolve().parents[2]

def checked_gap(x,y,atol=1e-9,rtol=1e-8):
    x=np.asarray(x,float);y=np.asarray(y,float)
    np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,equal_nan=True)
    finite=np.isfinite(x)&np.isfinite(y)
    return float(np.max(np.abs(x[finite]-y[finite]))) if np.any(finite) else 0.

def run(package,candidate=False):
    sys.path.insert(0,str(ROOT/package/'src'));m=importlib.import_module(package);Cls=m.Design
    if candidate:
        from fastbjspy._bjs import fit
        class Candidate(Cls):
            def _regression(self,a,omit,procedure,observed):return fit(self,a,omit)
        Cls=Candidate
    rows=[];raw=[]
    for name in DESIGNS:
        started=time.perf_counter();row=dict(design=name,status='pass',point_gap=0.,component_gap=0.,weight_gap=0.,delete_gap=0.,jk_gap=0.,ri_gap=0.,placebo_gap=0.,assignments=0,reason='')
        data,covs=fixture(name)
        if name.endswith('weights'):
            row.update(status='not_applicable',reason='common Python adapter does not support sampling weights');rows.append(row);continue
        try:
            try:d=Cls(data,outcome='y',cluster='state',time='year',assignment=dict(data[['state','g']].drop_duplicates().itertuples(index=False,name=None)),covariates=covs)
            except m.InferenceError as e:
                if name=='missing_cells':row.update(status='expected_failure',reason=str(e));rows.append(row);continue
                raise
            outcomes={}
            for engine in ['reference','fast']:
                try:outcomes[engine]=('ok',d.point(engine=engine))
                except (m.InferenceError,ValueError,np.linalg.LinAlgError) as e:outcomes[engine]=('failure',str(e))
            if any(v[0]!='ok' for v in outcomes.values()):
                if name in ['state_covariate','time_covariate'] and all(v[0]=='failure' for v in outcomes.values()):
                    row.update(status='expected_failure',reason=json.dumps(outcomes));rows.append(row);continue
                raise AssertionError(f'point failure mismatch/unsupported: {outcomes}')
            row['point_gap']=checked_gap(outcomes['fast'][1],outcomes['reference'][1])
            assignments=list(m.assignment_draws(tuple(d.assignment),3,seed=928))
            if assignments!=list(m.assignment_draws(tuple(d.assignment),3,seed=928)):raise AssertionError('seed mismatch')
            assert len(set(assignments))==len(assignments) and tuple(d.assignment) not in assignments
            assert all(sorted(a)==sorted(d.assignment) for a in assignments)
            if package=='fastbjspy':
                from fastbjspy._bjs import fit,raw_fit
                for a in [d.assignment,*assignments]:
                    a=np.asarray(a)
                    for omit in [None,*range(len(d.states))]:
                        if omit is not None and a[omit]>0 and sum(a==a[omit])==1:continue
                        f=fit(d,a,omit,True);r=raw_fit(d,a,omit,True);keep=np.ones(len(data),bool) if omit is None else d._rowstate!=omit
                        gaps=[checked_gap(f[k],r[k],atol=1e-8) for k in ['estimate','alpha','gamma']]
                        gaps.extend(checked_gap(f[k][keep],r[k][keep],atol=1e-8) for k in ['y0','tau'])
                        assert np.array_equal(f['target'],r['target'])
                        fw=f['target']/sum(f['target']);rw=r['target']/sum(r['target']);row['weight_gap']=max(row['weight_gap'],checked_gap(fw,rw))
                        row['component_gap']=max(row['component_gap'],*gaps)
            elif not covs and package in ['undidpy','didintpy']:
                group=data.groupby(['state','year'],sort=True).y.agg(['mean','size']).reset_index()
                row['component_gap']=checked_gap(d.means.ravel(),group['mean']);row['weight_gap']=checked_gap(d.counts.ravel(),group['size'])
            all_jk=[]
            for a in [d.assignment,*assignments]:
                f=d.jackknife(a);r=d.jackknife(a,engine='reference');assert f['diagnostics']['status']==r['diagnostics']['status'],(f,r)
                assert f['jackknife_df']==r['jackknife_df']==len(d.states)-1
                row['delete_gap']=max(row['delete_gap'],checked_gap(f['delete_estimates'],r['delete_estimates']))
                if f['diagnostics']['status']!='ok':
                    assert name == 'one_treated',f
                    assert np.isnan(f['p_value']) and len(f['delete_estimates'])==len(d.states)
                    continue
                for key in ['estimate','standard_error','statistic','p_value']:
                    gap=checked_gap(f[key],r[key],atol=1e-8);row['jk_gap']=max(row['jk_gap'],gap)
                checked_gap(f['p_value'],2*t.sf(abs(f['statistic']),len(d.states)-1),atol=2e-12,rtol=2e-10)
                all_jk.append((f,r))
            f=d.randomization_inference(assignments=assignments);r=d.randomization_inference(assignments=assignments,engine='reference')
            row['ri_gap']=checked_gap(f['statistics'],r['statistics']);assert f['hits']==r['hits'] and f['p_value']==r['p_value']
            np.testing.assert_array_equal(f['hit_indicators'],r['hit_indicators'])
            row['assignments']=len(assignments)
            details=dict(design=name,ri_fast=f,ri_reference=r)
            if name == 'one_treated':
                row.update(status='expected_failure',reason='every deletion retained; no treated cohort remains after one deletion; point/RI pass')
            else:
                for proc in ['replace','subtract']+(['add'] if package=='fasttwfepy' else []):
                    f=d.placebo(proc,assignments=assignments);r=d.placebo(proc,assignments=assignments,engine='reference')
                    assert f['invalid_draws']==r['invalid_draws'];row['placebo_gap']=max(row['placebo_gap'],checked_gap(f['procedural_p_values'],r['procedural_p_values'],atol=1e-8))
                    assert f['hits']==r['hits'] and (f['p_value']==r['p_value'] or np.isnan(f['p_value']) and np.isnan(r['p_value']))
                    np.testing.assert_array_equal(f['hit_indicators'],r['hit_indicators'])
                    details[proc+'_fast']=f;details[proc+'_reference']=r
                effect=d.point();sub=d._subtract(effect);D=(d.assignment[d._rowstate]>0)&(data.year.to_numpy()>=d.assignment[d._rowstate])
                checked_gap(sub.data.y,d.data.y-effect*D)
            if package!='fasttwfepy':
                try:d.placebo_add(assignments=assignments)
                except NotImplementedError:pass
                else:raise AssertionError('unsupported Add did not fail')
            raw.append(details)
        except Exception as e:
            row.update(status='unexpected_failure',reason=f'{type(e).__name__}: {e}');traceback.print_exc()
        row['seconds']=time.perf_counter()-started;rows.append(row);print(package,name,row['status'],row['reason'][:180],flush=True)
    # Actual statistics for every member of a small exact support, not sampler-only.
    dd,_=fixture('one_cohort');dd=dd.loc[dd.state.isin([41,58,109,126])].copy()
    d=Cls(dd,outcome='y',cluster='state',time='year',assignment=dict(dd[['state','g']].drop_duplicates().itertuples(index=False,name=None)))
    a=list(m.assignment_draws(tuple(d.assignment),exact=True));assert len(a)==5
    f=d.randomization_inference(assignments=a);r=d.randomization_inference(assignments=a,engine='reference');gap=checked_gap(f['statistics'],r['statistics']);assert f['p_value']==r['p_value']
    rows.append(dict(design='exact_support',status='pass',ri_gap=gap,assignments=5,reason='all non-observed assignments'))
    out=ROOT/package/'docs';pd.DataFrame(rows).to_csv(out/'stress_test_results.csv',index=False)
    (out/'stress_test_draws.json').write_text(json.dumps(raw,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),indent=2))
    frame=pd.DataFrame(rows);summary=dict(package=package,candidate=candidate,counts=frame.status.value_counts().to_dict(),maxima={c:float(frame[c].max()) for c in ['point_gap','component_gap','weight_gap','delete_gap','jk_gap','ri_gap','placebo_gap']})
    (out/'stress_test_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'STRESS_TEST_REPORT.md').write_text('# Stress-test report\n\n'+json.dumps(summary,indent=2)+'\n\nSee stress_test_results.csv for each design and stress_test_draws.json for fixed-assignment statistics and inner p-values. Missing cells are an explicit inference-adapter restriction; unsupported weights are N/A. Singleton-cohort deletion failures retain G and all deletions. Heterogeneous effects are diagnostic, not a validity proof. Reference point checks gate inference.\n')
    print(json.dumps(summary),flush=True)
    return int(any(frame.status=='unexpected_failure'))

if __name__=='__main__':sys.exit(run(sys.argv[1],'candidate' in sys.argv))
