from __future__ import annotations
import importlib.util, itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch  # noqa: E402  (imported after the base module's LightGBM import)
# Workspace root is the cwd when scripts are run from the restructured layout;
# otherwise fall back to the script's parent (legacy flat layout).
ROOT=Path.cwd() if (Path.cwd()/'artifact'/'scripts').is_dir() else Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'/'current'; RESULTS.mkdir(parents=True,exist_ok=True)
base_path=ROOT/'artifact'/'scripts'/'run_physical_sampler_study.py'
if not base_path.exists(): base_path=ROOT/'artifact'/'run_physical_sampler_study.py'
spec=importlib.util.spec_from_file_location('base',base_path); base=importlib.util.module_from_spec(spec); sys.modules['base']=base; spec.loader.exec_module(base)
SEEDS=(11,23,37,51,71)

def boot_diff(V,S,nboot=2000,seed=20261026):
    rng=np.random.default_rng(seed); V=np.asarray(V,float); S=np.asarray(S,float); nv,ne=V.shape; ns=S.shape[0]
    out=np.empty(nboot)
    for i in range(nboot):
        e=rng.integers(0,ne,ne); vi=rng.integers(0,nv,nv); si=rng.integers(0,ns,ns)
        out[i]=V[vi][:,e].mean()-S[si][:,e].mean()
    return np.quantile(out,[.025,.975])

def run(cfg,data_seed):
    cust,merch,tx,labels=base.generate(cfg,data_seed); buy,sell=base.incident_lists(tx,len(cust)); y=labels.target.to_numpy()
    needed=[('ORIG_H1','ORIG',1),('ORIG_H2','ORIG',2),('R1_H1','R1',1),('R2_H1','R2',1),('R3_H2','R3',2)]
    raw={}; cov={}
    for nm,v,h in needed: raw[nm],cov[nm]=base.sample_view(v,h,cust,merch,tx,buy,sell)
    # normalization remains pinned to complete original view
    full,_=base.sample_view('ORIG',2,cust,merch,tx,buy,sell); mu,sd=base.normalizer(full)
    F={k:base.norm(v,mu,sd) for k,v in raw.items()}
    tr0,te=train_test_split(np.arange(len(y)),test_size=.30,random_state=2026,stratify=y); tr,va=train_test_split(tr0,test_size=.20,random_state=2027,stratify=y[tr0])
    preds={}; models={}
    for mdl in ['LightGBM','TypedMPNN']:
      for view,X in F.items():
       for s0 in SEEDS:
        if mdl=='LightGBM': m,p=base.train_gbdt(X,y,tr,te,s0)
        else: m,p,_=base.train_nn(X,y,tr,va,te,s0,True)
        preds[(mdl,view,s0)]=p; models[(mdl,view,s0)]=m
    comps=[('mid-to-low (R1)','ORIG_H1','R1_H1'),('mid-to-full (R2)','ORIG_H1','R2_H1'),('full-to-mid (R3)','ORIG_H2','R3_H2')]
    rows=[]
    for mdl in ['LightGBM','TypedMPNN']:
      for lab,ref,view in comps:
        S=[]; V=[]; P=[]
        for a,b in itertools.permutations(SEEDS,2):
            pa=preds[(mdl,ref,a)]; pb=preds[(mdl,ref,b)]; pv=preds[(mdl,view,b)]
            S.append((pa>=.5)!=(pb>=.5)); V.append((pa>=.5)!=(pv>=.5))
        for a in SEEDS:
            pa=preds[(mdl,ref,a)]; pv=preds[(mdl,view,a)]; P.append((pa>=.5)!=(pv>=.5))
        S=np.asarray(S,float); V=np.asarray(V,float); P=np.asarray(P,float); lo,hi=boot_diff(V,S,seed=data_seed+sum(map(ord,mdl+lab)))
        rows.append(dict(domain=cfg.name,model=mdl,transition=lab,reference=ref,view=view,coverage_ref=cov[ref],coverage_view=cov[view],delta_seed_ind=S.mean(),delta_view_ind=V.mean(),esd_ind=V.mean()-S.mean(),esd_ci_lo=lo,esd_ci_hi=hi,delta_view_paired=P.mean()))
    # Role controls from the already-trained complete-view models.
    X=F['ORIG_H2']; Xswap=X.copy(); Xswap[:,[1,2]]=X[:,[2,1]]; Xswap[:,[3,4]]=X[:,[4,3]]; Xswap[:,[5,6]]=X[:,[6,5]]
    def oracle(Z): return .9*Z[:,0]+1.25*Z[:,1]-1.25*Z[:,2]+.45*Z[:,3]-.45*Z[:,4]+.75*Z[:,5]-.75*Z[:,6]
    od=oracle(Xswap)[te]-oracle(X)[te]; info=np.abs(od)>1e-10; role=[]
    for mdl in ['LightGBM','TypedMPNN']:
      for s0 in SEEDS:
        p0=preds[(mdl,'ORIG_H2',s0)]; m=models[(mdl,'ORIG_H2',s0)]
        if mdl=='LightGBM': ps=m.predict_proba(Xswap[te])[:,1]
        else:
            x,b,s=map(torch.tensor,base.parts(Xswap)); m.eval();
            with torch.no_grad(): ps=torch.sigmoid(m(x,b,s)).numpy()[te]
        d=ps-p0
        role.append(dict(domain=cfg.name,model=mdl,encoding='role-keyed',seed=s0,swap_concordance=np.mean(np.sign(d[info])==np.sign(od[info])),swap_response=np.mean(np.abs(d)),serialization_disagreement=0.0))
        # Same weights but no role metadata: physical slot order is now the only cue. A role-preserving serialization permutation therefore presents Xswap.
        role.append(dict(domain=cfg.name,model=mdl,encoding='anonymous-slot',seed=s0,swap_concordance=np.mean(np.sign(d[info])==np.sign(od[info])),swap_response=np.mean(np.abs(d)),serialization_disagreement=np.mean((p0>=.5)!=(ps>=.5))))
    return pd.DataFrame(rows),pd.DataFrame(role)

def main():
    cfgs=[(base.DomainCfg('Synth-A-uniform',heavy_tail=False,amount_sigma=.95,signal_scale=1.0),3107),(base.DomainCfg('Synth-B-heavy-tail',heavy_tail=True,amount_sigma=1.25,signal_scale=.95),8119)]
    A=[];R=[]
    for c,s in cfgs:
        print('run',c.name,flush=True); a,r=run(c,s); A.append(a);R.append(r)
    A=pd.concat(A,ignore_index=True);R=pd.concat(R,ignore_index=True)
    A.to_csv(RESULTS/'v7_drift_independent.csv',index=False); R.to_csv(RESULTS/'v7_role_intermediate.csv',index=False)
    RS=R.groupby(['domain','model','encoding']).agg(swap_concordance=('swap_concordance','mean'),swap_response=('swap_response','mean'),serialization_disagreement=('serialization_disagreement','mean')).reset_index(); RS.to_csv(RESULTS/'v7_role_intermediate_summary.csv',index=False)
    print('\n',A.to_string(index=False)); print('\n',RS.to_string(index=False))
if __name__=='__main__': main()
