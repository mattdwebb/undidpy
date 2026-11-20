"""Deterministic certification designs; no global RNG or source-directory writes."""
import numpy as np
import pandas as pd

def fixture(name='balanced', seed=904):
    rng=np.random.default_rng(seed);G=8;T=6;g=np.array([3,3,5,5,0,0,0,0]);covs=[]
    if name=='one_cohort':g=np.array([3]*4+[0]*4)
    if name=='one_treated':g=np.array([3]+[0]*7)
    if name=='two_treated':g=np.array([3,3]+[0]*6)
    if name=='many_treated':g=np.array([3]*3+[5]*3+[0]*2)
    if name=='singleton_cohort':g=np.array([3,5,5,0,0,0,0,0])
    if name=='unbalanced_cohorts':G=12;g=np.array([3]*2+[5]*6+[0]*4)
    if name=='long_pre':T=12;g=np.array([8,8,10,10,0,0,0,0])
    if name=='short_pre':g=np.array([2,2,4,4,0,0,0,0])
    if name=='long_post':T=12
    if name=='short_post':g=np.array([5,5,6,6,0,0,0,0])
    rows=[]
    for h in range(G):
        for tt in range(1,T+1):
            n=1 if name=='panel' else 3
            if name in ('unequal','unbalanced_post'):n=2+(h*7+tt*3)%8
            if name=='extreme_sizes':n=1 if h<3 else (100 if h==G-1 else 5)
            for j in range(n):
                x=rng.normal();z=rng.normal();effect=1.
                if name=='cohort_heterogeneous':effect=0.5 if g[h]==3 else 1.5
                if name in ('state_heterogeneous','weak_null'):effect=(h-1.5)*.7
                if name=='time_varying':effect=.2*tt
                tr=g[h]>0 and tt>=g[h]
                rows.append((len(rows)+1,h*17+41,tt,g[h],x,z,.3*h+.2*tt+.4*x+.3*rng.normal()+effect*tr,effect,float(tr),1.))
    d=pd.DataFrame(rows,columns=['id','state','year','g','x','z','y','effect','D','w'])
    if name=='weak_null':
        center=d.loc[d.D==1,'effect'].mean();d['y']-=center*d.D;d['effect']-=center
    if name in ('covariates','correlated','near_collinear','state_covariate','time_covariate'):covs=['x','z']
    if name=='correlated':d['z']=.9*d.x+.1*d.z
    if name=='near_collinear':d['z']=d.x+1e-7*d.z
    if name=='state_covariate':d['z']=d.state.astype(float)
    if name=='time_covariate':d['z']=d.year.astype(float)
    if name=='missing_cells':d=d.loc[~((d.state==41)&(d.year==2))].copy()
    if name=='row_shuffled':d=d.sample(frac=1,random_state=seed).reset_index(drop=True)
    if name=='string_labels':d['state']=d.state.map(lambda h:f'state_{h}')
    if name=='moderate_weights':d['w']=.2+(d.id%9)/3
    if name=='extreme_weights':d['w']=np.where(d.id%13==0,1000.,.01)
    return d,covs

DESIGNS=['balanced','panel','unequal','extreme_sizes','unbalanced_post','row_shuffled','string_labels','missing_cells','one_cohort','one_treated','two_treated','many_treated','singleton_cohort','unbalanced_cohorts','long_pre','short_pre','long_post','short_post','cohort_heterogeneous','state_heterogeneous','time_varying','weak_null','covariates','correlated','near_collinear','state_covariate','time_covariate','moderate_weights','extreme_weights']
