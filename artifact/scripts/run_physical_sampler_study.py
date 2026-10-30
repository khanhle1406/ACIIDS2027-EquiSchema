from __future__ import annotations
import hashlib, itertools, json, warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
# Import LightGBM BEFORE torch: on macOS, loading both OpenMP runtimes in the
# opposite order segfaults inside LGBM's __init_from_np2d (numpy 2.4 / torch 2.12).
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
import os
# See docs/BUILD.md: torch and LightGBM each bundle their own libomp, and LightGBM's
# OpenMP worker init aborts on this macOS/Python 3.13 stack. Serial GBDT fits restore
# reproducibility and are deterministic given random_state, so results are unchanged.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
# Workspace root is the cwd when scripts are run from the restructured layout;
# otherwise fall back to the script's parent (legacy flat layout).
ROOT=Path.cwd() if (Path.cwd()/'artifact'/'scripts').is_dir() else Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'/'current'; RESULTS.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)

@dataclass
class DomainCfg:
    name:str; n_customers:int=3000; n_merchants:int=240; n_transactions:int=45000
    heavy_tail:bool=False; amount_sigma:float=1.0; signal_scale:float=1.0

def z(v):
    v=np.asarray(v,dtype=float); return (v-v.mean())/(v.std()+1e-9)

def hsh(df,keys):
    b=df.sort_values(keys).reset_index(drop=True).to_csv(index=False,date_format='%Y-%m-%dT%H:%M:%S').encode(); return hashlib.sha256(b).hexdigest()

def generate(cfg,seed):
    r=np.random.default_rng(seed); n,q,m=cfg.n_customers,cfg.n_merchants,cfg.n_transactions
    cx=r.normal(size=n); mr=r.normal(size=q)
    if cfg.heavy_tail:
        wb=r.pareto(1.7,n)+.2; ws=r.pareto(2.0,n)+.2; wm=r.pareto(1.8,q)+.2
        b=r.choice(n,m,p=wb/wb.sum()); s=r.choice(n,m,p=ws/ws.sum()); merch=r.choice(q,m,p=wm/wm.sum())
    else:
        b=r.integers(0,n,m); s=r.integers(0,n,m); merch=r.integers(0,q,m)
    same=b==s
    while same.any(): s[same]=r.integers(0,n,same.sum()); same=b==s
    base=r.lognormal(2.0,cfg.amount_sigma,m); amt=base*np.exp(.10*cx[b]-.07*cx[s]+.12*mr[merch])
    ts=np.datetime64('2023-01-01')+r.integers(0,365,m).astype('timedelta64[D]')
    cust=pd.DataFrame({'customer_id':np.arange(n,dtype=np.int64),'customer_x':cx})
    merchants=pd.DataFrame({'merchant_id':np.arange(q,dtype=np.int64),'merchant_risk':mr})
    tx=pd.DataFrame({'transaction_id':np.arange(m,dtype=np.int64),'buyer_id':b.astype(np.int64),'seller_id':s.astype(np.int64),'merchant_id':merch.astype(np.int64),'amount':amt,'timestamp':pd.to_datetime(ts)})
    risk=mr[merch]
    def agg(ids,w=None): return np.bincount(ids,weights=w,minlength=n).astype(float)
    raw=np.column_stack([cx,agg(b,amt),agg(s,amt),agg(b),agg(s),agg(b,risk),agg(s,risk)])
    F=raw.copy(); F[:,1]=z(np.log1p(F[:,1])); F[:,2]=z(np.log1p(F[:,2]));
    for j in range(3,7): F[:,j]=z(F[:,j])
    score=cfg.signal_scale*(.9*F[:,0]+1.25*F[:,1]-1.25*F[:,2]+.45*F[:,3]-.45*F[:,4]+.75*F[:,5]-.75*F[:,6])
    noisy=score+r.normal(scale=.95 if not cfg.heavy_tail else 1.10,size=n); y=(noisy>np.median(noisy)).astype(np.int64)
    labels=pd.DataFrame({'customer_id':np.arange(n),'target':y,'oracle_score':score})
    return cust,merchants,tx,labels

def certify(merchants,tx):
    orig=tx.sort_values('transaction_id').reset_index(drop=True)
    # R1 vertical decomposition
    core=orig[['transaction_id','buyer_id','seller_id','merchant_id']].copy(); pay=orig[['transaction_id','amount','timestamp']].copy()
    b1=core.merge(pay,on='transaction_id',validate='one_to_one')[orig.columns].sort_values('transaction_id').reset_index(drop=True); pd.testing.assert_frame_equal(orig,b1,check_dtype=False)
    # R2 N:1 denormalization + unused-dimension remainder
    d=orig.merge(merchants,on='merchant_id',how='left',validate='many_to_one'); used=set(orig.merchant_id.unique()); rem=merchants[~merchants.merchant_id.isin(used)].copy()
    mback=pd.concat([d[['merchant_id','merchant_risk']].drop_duplicates(),rem],ignore_index=True).sort_values('merchant_id').reset_index(drop=True)
    m0=merchants.sort_values('merchant_id').reset_index(drop=True); pd.testing.assert_frame_equal(m0,mback,check_dtype=False)
    b2=d.drop(columns='merchant_risk')[orig.columns].sort_values('transaction_id').reset_index(drop=True); pd.testing.assert_frame_equal(orig,b2,check_dtype=False)
    # R3 role-labelled FK reification
    fact=orig.drop(columns=['buyer_id','seller_id']); bb=orig[['transaction_id','buyer_id']].rename(columns={'buyer_id':'customer_id'}); bb['role']='buyer'; ss=orig[['transaction_id','seller_id']].rename(columns={'seller_id':'customer_id'}); ss['role']='seller'; links=pd.concat([bb,ss],ignore_index=True)
    piv=links.pivot(index='transaction_id',columns='role',values='customer_id').reset_index().rename(columns={'buyer':'buyer_id','seller':'seller_id'})
    b3=fact.merge(piv,on='transaction_id',validate='one_to_one')[orig.columns].sort_values('transaction_id').reset_index(drop=True); pd.testing.assert_frame_equal(orig,b3,check_dtype=False)
    return {'r1_roundtrip_equal':True,'r2_roundtrip_equal':True,'r3_roundtrip_equal':True,'r2_unused_dimension_rows':len(rem),'orig_tx_sha256':hsh(orig,['transaction_id']),'r1_tx_sha256':hsh(b1,['transaction_id']),'r2_tx_sha256':hsh(b2,['transaction_id']),'r3_tx_sha256':hsh(b3,['transaction_id']),'orig_merchants_sha256':hsh(m0,['merchant_id']),'r2_merchants_sha256':hsh(mback,['merchant_id'])}

# Physical distance from customer seed to each logical component after each certified rewrite.
DIST={
 'ORIG':{'struct':1,'amount':1,'risk':2},
 'R1':  {'struct':1,'amount':2,'risk':2},
 'R2':  {'struct':1,'amount':1,'risk':1},
 'R3':  {'struct':1,'amount':2,'risk':3},
}

def incident_lists(tx,n):
    buy=[[] for _ in range(n)]; sell=[[] for _ in range(n)]
    for i,(b,s) in enumerate(zip(tx.buyer_id.to_numpy(),tx.seller_id.to_numpy())): buy[int(b)].append(i); sell[int(s)].append(i)
    return buy,sell

def choose(idxs,frac,seed,cid,rolecode):
    if frac>=.999999 or len(idxs)<=1: return np.asarray(idxs,dtype=np.int64)
    k=max(1,int(np.ceil(len(idxs)*frac))); r=np.random.default_rng(seed*1000003+cid*9176+rolecode*131)
    return np.asarray(idxs,dtype=np.int64)[np.sort(r.choice(len(idxs),min(k,len(idxs)),replace=False))]

def sample_view(view,H,cust,merchants,tx,buy,sell,frac=1.0,sample_seed=707):
    n=len(cust); X=np.zeros((n,7),float); X[:,0]=cust.customer_x.to_numpy(); mr=merchants.merchant_risk.to_numpy(); mer=tx.merchant_id.to_numpy(); amt=tx.amount.to_numpy()
    d=DIST[view]; cov=[]
    for cid in range(n):
        ids_b=choose(buy[cid],frac,sample_seed,cid,1); ids_s=choose(sell[cid],frac,sample_seed,cid,2)
        total=len(buy[cid])+len(sell[cid]); visible=0
        if d['struct']<=H:
            X[cid,3]=len(ids_b); X[cid,4]=len(ids_s); visible += len(ids_b)+len(ids_s)
        if d['amount']<=H:
            X[cid,1]=amt[ids_b].sum() if len(ids_b) else 0.; X[cid,2]=amt[ids_s].sum() if len(ids_s) else 0.; visible += len(ids_b)+len(ids_s)
        if d['risk']<=H:
            X[cid,5]=mr[mer[ids_b]].sum() if len(ids_b) else 0.; X[cid,6]=mr[mer[ids_s]].sum() if len(ids_s) else 0.; visible += len(ids_b)+len(ids_s)
        cov.append(1.0 if total==0 else visible/(3.0*total))
    return X,float(np.mean(cov))

def normalizer(ref):
    x=ref.copy(); x[:,1]=np.log1p(x[:,1]); x[:,2]=np.log1p(x[:,2]); return x[:,1:].mean(0),x[:,1:].std(0)+1e-9

def norm(raw,mu,sd):
    x=raw.copy(); x[:,1]=np.log1p(x[:,1]); x[:,2]=np.log1p(x[:,2]); x[:,1:]=(x[:,1:]-mu)/sd; return x.astype(np.float32)

class TypedMPNN(nn.Module):
    def __init__(self,aware=True,h=24):
        super().__init__(); self.aware=aware; self.self_lin=nn.Linear(1,h)
        if aware: self.blin=nn.Linear(3,h,bias=False); self.slin=nn.Linear(3,h,bias=False)
        else: self.rlin=nn.Linear(3,h,bias=False)
        self.head=nn.Sequential(nn.ReLU(),nn.Linear(h,12),nn.ReLU(),nn.Linear(12,1))
    def forward(self,x,b,s):
        h=self.self_lin(x); h=h+self.blin(b)+self.slin(s) if self.aware else h+self.rlin(b+s); return self.head(h).squeeze(-1)
def parts(X): return X[:,[0]],X[:,[1,3,5]],X[:,[2,4,6]]
def train_nn(X,y,tr,va,te,seed,aware=True):
    torch.manual_seed(seed); np.random.seed(seed); model=TypedMPNN(aware); opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=1e-4); lossfn=nn.BCEWithLogitsLoss(); x,b,s=map(torch.tensor,parts(X)); yt=torch.tensor(y,dtype=torch.float32); trt=torch.tensor(tr); vat=torch.tensor(va)
    best=None; bestauc=-1.; bad=0
    for _ in range(80):
        model.train(); opt.zero_grad(); lg=model(x,b,s); loss=lossfn(lg[trt],yt[trt]); loss.backward(); opt.step(); model.eval()
        with torch.no_grad(): pv=torch.sigmoid(model(x,b,s)[vat]).numpy()
        auc=roc_auc_score(y[va],pv)
        if auc>bestauc+1e-5: bestauc=auc; best={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=8: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): pall=torch.sigmoid(model(x,b,s)).numpy()
    return model,pall[te],pall
def train_gbdt(X,y,tr,te,seed):
    m=LGBMClassifier(n_estimators=130,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=20,subsample=.82,subsample_freq=1,colsample_bytree=.85,reg_lambda=1.,random_state=seed,n_jobs=1,verbosity=-1); m.fit(X[tr],y[tr]); return m,m.predict_proba(X[te])[:,1]

def run_domain(cfg,seed,seeds=(11,23,37,51,71)):
    cust,merchants,tx,labels=generate(cfg,seed); cert=certify(merchants,tx); buy,sell=incident_lists(tx,len(cust))
    specs=[('ORIG_H1','ORIG',1),('ORIG_H2','ORIG',2),('R1_H1','R1',1),('R1_H2','R1',2),('R2_H1','R2',1),('R2_H2','R2',2),('R3_H1','R3',1),('R3_H2','R3',2),('R3_H3','R3',3)]
    raw={}; coverage={}
    for nm,v,h in specs: raw[nm],coverage[nm]=sample_view(v,h,cust,merchants,tx,buy,sell)
    mu,sd=normalizer(raw['ORIG_H2']); F={k:norm(v,mu,sd) for k,v in raw.items()}
    y=labels.target.to_numpy(); tr0,te=train_test_split(np.arange(len(y)),test_size=.30,random_state=2026,stratify=y); tr,va=train_test_split(tr0,test_size=.20,random_state=2027,stratify=y[tr0])
    perf=[]; preds={}; models={}
    for mdl in ['LightGBM','TypedMPNN']:
      for view,X in F.items():
       for sd0 in seeds:
        if mdl=='LightGBM': model,p=train_gbdt(X,y,tr,te,sd0)
        else: model,p,_=train_nn(X,y,tr,va,te,sd0,True)
        preds[(mdl,view,sd0)]=p; models[(mdl,view,sd0)]=model
        perf.append({'domain':cfg.name,'model':mdl,'view':view,'coverage':coverage[view],'seed':sd0,'auc':roc_auc_score(y[te],p),'accuracy':accuracy_score(y[te],p>=.5),'brier':brier_score_loss(y[te],p)})
    # drift and ESD
    dr=[]; comps=[('ORIG_H1','R1_H1'),('ORIG_H2','R1_H2'),('ORIG_H1','R2_H1'),('ORIG_H2','R2_H2'),('ORIG_H1','R3_H1'),('ORIG_H2','R3_H2'),('ORIG_H2','R3_H3')]
    for mdl in ['LightGBM','TypedMPNN']:
      for ref in ['ORIG_H1','ORIG_H2']:
       for a,b0 in itertools.combinations(seeds,2):
        pa,pb=preds[(mdl,ref,a)],preds[(mdl,ref,b0)]; dr.append({'domain':cfg.name,'model':mdl,'reference':ref,'view':ref,'kind':'seed','coverage':coverage[ref],'seed_a':a,'seed_b':b0,'disagreement':np.mean((pa>=.5)!=(pb>=.5)),'prob_delta':np.mean(np.abs(pa-pb))})
      for ref,v in comps:
       for sd0 in seeds:
        pa,pb=preds[(mdl,ref,sd0)],preds[(mdl,v,sd0)]; dr.append({'domain':cfg.name,'model':mdl,'reference':ref,'view':v,'kind':'view','coverage':coverage[v],'seed_a':sd0,'seed_b':sd0,'disagreement':np.mean((pa>=.5)!=(pb>=.5)),'prob_delta':np.mean(np.abs(pa-pb))})
    # role swap on complete logical context
    X=F['ORIG_H2']; Xs=X.copy(); Xs[:,[1,2]]=X[:,[2,1]]; Xs[:,[3,4]]=X[:,[4,3]]; Xs[:,[5,6]]=X[:,[6,5]]
    def oracle(Z): return .9*Z[:,0]+1.25*Z[:,1]-1.25*Z[:,2]+.45*Z[:,3]-.45*Z[:,4]+.75*Z[:,5]-.75*Z[:,6]
    od=oracle(Xs)[te]-oracle(X)[te]; role=[]
    for mdl in ['LightGBM','TypedMPNN']:
     for aware in [True,False]:
      for sd0 in seeds:
       if mdl=='LightGBM':
        A=X if aware else np.column_stack([X[:,0],X[:,1]+X[:,2],X[:,3]+X[:,4],X[:,5]+X[:,6]]); B=Xs if aware else A; mod,p0=train_gbdt(A,y,tr,te,sd0); p1=mod.predict_proba(B[te])[:,1]
       else:
        mod,p0,_=train_nn(X,y,tr,va,te,sd0,aware); x1,b1,s1=map(torch.tensor,parts(Xs)); mod.eval();
        with torch.no_grad(): p1=torch.sigmoid(mod(x1,b1,s1)).numpy()[te]
       d=p1-p0; info=np.abs(od)>1e-10; role.append({'domain':cfg.name,'model':mdl,'encoding':'role-aware' if aware else 'role-blind','seed':sd0,'directional_concordance':np.mean(np.sign(d[info])==np.sign(od[info])),'mean_abs_response':np.mean(np.abs(d))})
    # coverage ablation: natural fanout sampling on R1, H=2
    cov=[]; base={}
    for mdl in ['LightGBM','TypedMPNN']:
      bp={}
      for sd0 in seeds:
       bp[sd0]=preds[(mdl,'ORIG_H2',sd0)]
      ds=np.mean([np.mean((bp[a]>=.5)!=(bp[b0]>=.5)) for a,b0 in itertools.combinations(seeds,2)]); base[mdl]=(bp,ds)
    for frac in [.2,.4,.6,.8,1.0]:
      rr,cv=sample_view('R1',2,cust,merchants,tx,buy,sell,frac=frac,sample_seed=707); XX=norm(rr,mu,sd)
      for mdl in ['LightGBM','TypedMPNN']:
       bp,ds=base[mdl]
       for sd0 in seeds:
        if mdl=='LightGBM': _,pv=train_gbdt(XX,y,tr,te,sd0)
        else: _,pv,_=train_nn(XX,y,tr,va,te,sd0,True)
        dv=np.mean((bp[sd0]>=.5)!=(pv>=.5)); cov.append({'domain':cfg.name,'model':mdl,'fanout_fraction':frac,'coverage':cv,'seed':sd0,'delta_seed':ds,'delta_view':dv,'esd':dv-ds})
    return pd.DataFrame(perf),pd.DataFrame(dr),pd.DataFrame(role),pd.DataFrame(cov),cert

def main():
    cfgs=[(DomainCfg('Synth-A-uniform',heavy_tail=False,amount_sigma=.95,signal_scale=1.0),3107),(DomainCfg('Synth-B-heavy-tail',heavy_tail=True,amount_sigma=1.25,signal_scale=.95),8119)]
    P=[];D=[];R=[];C=[];cert={}
    for cfg,sd in cfgs:
        print('running',cfg.name,flush=True); p,d,r,c,k=run_domain(cfg,sd); P.append(p);D.append(d);R.append(r);C.append(c);cert[cfg.name]=k
    P=pd.concat(P);D=pd.concat(D);R=pd.concat(R);C=pd.concat(C)
    P.to_csv(RESULTS/'physical_sampler_performance.csv',index=False);D.to_csv(RESULTS/'physical_sampler_drift.csv',index=False);R.to_csv(RESULTS/'physical_sampler_role.csv',index=False);C.to_csv(RESULTS/'physical_sampler_coverage.csv',index=False);json.dump(cert,open(RESULTS/'physical_sampler_certificates.json','w'),indent=2)
    ps=P.groupby(['domain','model','view','coverage']).agg(auc_mean=('auc','mean'),auc_std=('auc','std')).reset_index()
    vs=D[D.kind=='view'].groupby(['domain','model','reference','view','coverage']).agg(delta_view=('disagreement','mean'),prob_delta=('prob_delta','mean')).reset_index(); ss=D[D.kind=='seed'].groupby(['domain','model','reference']).agg(delta_seed=('disagreement','mean')).reset_index(); ds=vs.merge(ss,on=['domain','model','reference']); ds['esd']=ds.delta_view-ds.delta_seed
    rs=R.groupby(['domain','model','encoding']).agg(directional_mean=('directional_concordance','mean'),response_mean=('mean_abs_response','mean')).reset_index(); cs=C.groupby(['domain','model','fanout_fraction','coverage']).agg(delta_seed=('delta_seed','mean'),delta_view=('delta_view','mean'),esd=('esd','mean'),esd_std=('esd','std')).reset_index()
    ps.to_csv(RESULTS/'physical_sampler_performance_summary.csv',index=False);ds.to_csv(RESULTS/'physical_sampler_drift_summary.csv',index=False);rs.to_csv(RESULTS/'physical_sampler_role_summary.csv',index=False);cs.to_csv(RESULTS/'physical_sampler_coverage_summary.csv',index=False)
    print('\nPERF\n',ps.to_string(index=False)); print('\nDRIFT\n',ds.to_string(index=False)); print('\nROLE\n',rs.to_string(index=False)); print('\nCOV\n',cs.to_string(index=False))
if __name__=='__main__': main()
