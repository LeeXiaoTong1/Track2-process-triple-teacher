#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune type-classifier-driven fusion between baseline and UFM on Dev."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
TYPE=["speech","sound","singing","music"]

def read(path):
    with open(path,"r",encoding="utf-8-sig") as f: return {r["name"].strip():r for r in csv.DictReader(f)}
def labels(path):
    rows=[]
    with open(path,"r",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): rows.append((r["name"].strip(),0 if r["label"].lower().strip()=="real" else 1,r["type"].lower().strip()))
    return rows
def score_obj(y,p,t,floor=0.95,pen=2.0):
    vals=[]
    for typ in TYPE:
        idx=[i for i,x in enumerate(t) if x==typ]
        vals.append(f1_score([y[i] for i in idx],[p[i] for i in idx],average="macro",labels=[0,1],zero_division=0))
    vals=np.array(vals); return float(vals.mean()-pen*np.maximum(0,floor-vals).mean()+0.1*vals.min()), vals

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ufm_csv",required=True); ap.add_argument("--baseline_csv",required=True); ap.add_argument("--type_csv",required=True); ap.add_argument("--label_csv",required=True); ap.add_argument("--out_json",required=True); ap.add_argument("--trials",type=int,default=12000); ap.add_argument("--seed",type=int,default=1234)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    ufm=read(args.ufm_csv); base=read(args.baseline_csv); typ=read(args.type_csv); labs=labels(args.label_csv)
    y=[]; t=[]; su=[]; sb=[]; qt=[]
    for n,lab,ty in labs:
        if n in ufm and n in base and n in typ:
            y.append(lab); t.append(ty); su.append(float(ufm[n]["score"])); sb.append(float(base[n]["score"])); qt.append([float(typ[n]["type_"+k]) for k in TYPE])
    y=np.array(y); su=np.array(su); sb=np.array(sb); qt=np.array(qt); qt=qt/np.maximum(qt.sum(1,keepdims=True),1e-8)
    best=(-9,None)
    candidates=[]
    # include hard branch candidates
    for _ in range(args.trials):
        alpha=np.array([np.random.beta(1.2,3), np.random.beta(1,5), np.random.beta(4,1.2), np.random.beta(1,4)])
        # bias and thresholds per predicted type
        theta=np.array([np.random.uniform(0.25,0.75),np.random.uniform(0.2,0.7),np.random.uniform(0.02,0.5),np.random.uniform(0.25,0.8)])
        temp=np.random.uniform(0.4,3.0)
        q=qt**temp; q=q/np.maximum(q.sum(1,keepdims=True),1e-8)
        a=(q*alpha[None,:]).sum(1)
        th=(q*theta[None,:]).sum(1)+np.random.uniform(-0.08,0.08)
        s=a*sb+(1-a)*su
        pred=np.where(s>=th,0,1)
        obj,vals=score_obj(y,pred,t)
        if obj>best[0]: best=(obj,(alpha,theta,temp,th.mean(),vals))
    alpha,theta,temp,_,vals=best[1]
    out={"type_order":TYPE,"alpha":alpha.tolist(),"theta":theta.tolist(),"temp":float(temp),"objective":float(best[0]),"dev_per_type_f1":[float(x) for x in vals],"rule":"q=type_prob**temp normalize; alpha=sum(q*alpha_t); threshold=sum(q*theta_t); score=alpha*baseline+(1-alpha)*ufm; real if score>=threshold"}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out_json,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
