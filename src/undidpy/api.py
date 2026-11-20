"""Assignment-explicit estimator with transparent and cached calculation paths."""
import numpy as np
import pandas as pd
from ._local_effects import local_effects
from ._inference import InferenceMixin,InferenceError

class Design(InferenceMixin):
    kind='undid'
    def __init__(self,data,*,outcome,cluster,time,assignment,covariates=(),jackknife_support='changing',aggregation_weights='post'):
        if aggregation_weights not in ('post','comparison'):raise ValueError('aggregation_weights must be post or comparison')
        self.aggregation_weights=aggregation_weights
        if jackknife_support not in ('changing','fixed'):
            raise ValueError("jackknife_support must be 'changing' or 'fixed'")
        self.jackknife_support=jackknife_support
        self.outcome=outcome;self.cluster=cluster;self.time=time;self.covariates=list(covariates)
        columns=list(dict.fromkeys([outcome,cluster,time,*self.covariates]))
        self.data=data[columns].copy().reset_index(drop=True)
        if self.data.isna().any().any():raise InferenceError('invalid_input','missing values are not silently dropped')
        if not np.isfinite(self.data[[outcome,time,*self.covariates]].to_numpy(float)).all():raise InferenceError('nonfinite_estimate','input contains infinity')
        self.states=sorted(self.data[cluster].unique());self.times=np.sort(self.data[time].unique().astype(float))
        self.assignment=self._assignment(assignment);G=len(self.states);T=len(self.times)
        self._sindex={s:i for i,s in enumerate(self.states)}
        self._rowstate=np.array([self._sindex[s] for s in self.data[cluster]])
        self._rowtime=np.searchsorted(self.times,self.data[time].to_numpy(float))
        self._rowcell=self._rowstate*T+self._rowtime
        self.n=np.bincount(self._rowcell,minlength=G*T).astype(float)
        self.ysum=np.bincount(self._rowcell,weights=self.data[outcome],minlength=G*T)
        self.means=(self.ysum/np.maximum(self.n,1)).reshape(G,T);self.counts=self.n.reshape(G,T)
        self._cellstate=np.repeat(np.arange(G),T);self._celltime=np.tile(self.times,G)
        self._nuisance_cache={}
        if self.kind in ('twfe','bjs'):self._compile_regression()

    def _compile_regression(self):
        G=len(self.states);T=len(self.times);k=len(self.covariates);C=G*T
        base=np.column_stack([np.eye(G)[self._cellstate],np.tile(np.eye(T)[:,1:],(G,1))]);b=base.shape[1]
        x=self.data[self.covariates].to_numpy(float);y=self.data[self.outcome].to_numpy(float)
        xs=np.zeros((C,k));xy=np.zeros((C,k));xx=np.zeros((C,k,k))
        for j in range(k):
            xs[:,j]=np.bincount(self._rowcell,weights=x[:,j],minlength=C)
            xy[:,j]=np.bincount(self._rowcell,weights=x[:,j]*y,minlength=C)
            for l in range(k):xx[:,j,l]=np.bincount(self._rowcell,weights=x[:,j]*x[:,l],minlength=C)
        self.zsum=np.column_stack([base*self.n[:,None],xs])
        self.zz=np.zeros((C,b+k,b+k));self.zy=np.column_stack([base*self.ysum[:,None],xy])
        self.zz[:,:b,:b]=self.n[:,None,None]*base[:,:,None]*base[:,None,:]
        if k:
            self.zz[:,:b,b:]=base[:,:,None]*xs[:,None,:]
            self.zz[:,b:,:b]=self.zz[:,:b,b:].transpose(0,2,1);self.zz[:,b:,b:]=xx

    def _subtract(self,value):
        data=self.data.copy();D=(self.assignment[self._rowstate]>0)&(self.data[self.time].to_numpy()>=self.assignment[self._rowstate])
        data[self.outcome]-=value*D
        return type(self)(data,outcome=self.outcome,cluster=self.cluster,time=self.time,assignment=self.assignment,covariates=self.covariates,jackknife_support=self.jackknife_support,aggregation_weights=self.aggregation_weights)

    def _estimate(self,a,engine,omit,procedure,observed):
        if procedure=='add' and self.kind!='twfe':raise NotImplementedError('Add placebo is intentionally unsupported for this estimator; use Replace or Subtract')
        active=np.ones(len(self.states),dtype=bool)
        if omit is not None:active[omit]=False
        if not np.any(active&(a==0)):raise InferenceError('empty_comparison_group')
        if not np.any(active&(a>0)):raise InferenceError('undefined_estimand','no treated cohort remains')
        if self.jackknife_support=='fixed' and any(not np.any(active&(a==g)) for g in np.unique(a[a>0])):
            raise InferenceError('undefined_estimand','deletion removes a cohort')
        return local_effects(self, a, active, cache=(engine == "fast"))

    def _regression(self,a,omit,procedure,observed):
        keep=np.ones(len(self.n),dtype=bool)
        cols=np.arange(self.zz.shape[1])
        if omit is not None:keep=self._cellstate!=omit;cols=cols[cols!=omit]
        D=(a[self._cellstate]>0)&(self._celltime>=a[self._cellstate])
        train=keep&~D if self.kind=='bjs' else keep
        if not train.any() or not (keep&D).any():raise InferenceError('empty_comparison_group')
        key=omit
        if self.kind=='twfe' and key in self._nuisance_cache:
            inv,zy=self._nuisance_cache[key]
        else:
            zz=self.zz[train].sum(axis=0)[np.ix_(cols,cols)];zy=self.zy[train].sum(axis=0)[cols]
            if np.linalg.matrix_rank(zz)<len(cols):raise InferenceError('rank_deficient','nuisance design')
            try:inv=np.linalg.inv(zz)
            except np.linalg.LinAlgError as e:raise InferenceError('rank_deficient',str(e)) from e
            if self.kind=='twfe':self._nuisance_cache[key]=(inv,zy)
        if self.kind=='bjs':
            target=keep&D;beta=inv@zy
            return (self.ysum[target].sum()-self.zsum[target][:,cols].sum(axis=0)@beta)/self.n[target].sum()
        d=D[keep,None].astype(float)
        if procedure=='add':
            obs=np.asarray(observed if observed is not None else self.assignment)
            actual=((obs[self._cellstate]>0)&(self._celltime>=obs[self._cellstate]))[keep]
            d=np.column_stack([actual,d])
        zd=self.zsum[keep][:,cols].T@d;dy=d.T@self.ysum[keep]
        schur=d.T@(self.n[keep,None]*d)-zd.T@inv@zd
        if np.linalg.eigvalsh(schur).min()<=1e-10*max(1.,float(np.max(np.diag(d.T@(self.n[keep,None]*d))))):raise InferenceError('rank_deficient','treatment conditional on nuisance')
        try:return (np.linalg.solve(schur,dy-zd.T@inv@zy))[-1]
        except np.linalg.LinAlgError as e:raise InferenceError('rank_deficient',str(e)) from e

    def _reference(self,a,omit,procedure,observed):
        data=self.data if omit is None else self.data.loc[self._rowstate!=omit].copy()
        schedule=dict(zip(self.states,a));data=data.copy();data['__timing']=data[self.cluster].map(schedule)
        if self.kind=='undid':
            from . import undid
            panel=undid.prepare_panel_data(data,outcome_col=self.outcome,state_col=self.cluster,time_col=self.time,gvar_col='__timing',covariates=self.covariates)
            return undid._compute_core(panel,self.cluster,self.time,'__timing','both',False)['att']['estimate']
        if self.kind=='didint':
            from . import fitted_didint
            fit=fitted_didint.prepare_fitted_first_stage(data,outcome_col=self.outcome,state_col=self.cluster,time_col=self.time,gvar_col='__timing',covariates=self.covariates,ccc='int')
            return fitted_didint.run_fitted_core(fit,fit.panel)['att']['estimate']
        if self.kind=='bjs':
            from .estimators import EstimatorCache,estimate_bjs
            cache=EstimatorCache.from_data(data,outcome_col=self.outcome,state_col=self.cluster,time_col=self.time,covariates=self.covariates)
            result=estimate_bjs(cache,schedule)
            if result.status!='ok':raise InferenceError('rank_deficient',str(result.diagnostics))
            return result.att
        z=np.column_stack([pd.get_dummies(data[self.cluster],dtype=float),pd.get_dummies(data[self.time],drop_first=True,dtype=float),data[self.covariates].to_numpy(float)])
        d=((data['__timing']>0)&(data[self.time]>=data['__timing'])).to_numpy(float)
        if procedure=='add':
            obs=dict(zip(self.states,observed if observed is not None else self.assignment));g=data[self.cluster].map(obs)
            z=np.column_stack([z,((g>0)&(data[self.time]>=g)).to_numpy(float)])
        x=np.column_stack([z,d]);beta,_,rank,_=np.linalg.lstsq(x,data[self.outcome].to_numpy(float),rcond=None)
        if rank<x.shape[1]:raise InferenceError('rank_deficient')
        return beta[-1]
