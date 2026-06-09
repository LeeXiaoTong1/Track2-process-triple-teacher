#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json
from pathlib import Path
import numpy as np
TYPE=["speech","sound","singing","music"]
def read(path):
    with open(path,"r",encoding="utf-8-sig") as f: return {r["name"].strip():r for r in csv.DictReader(f)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ufm_csv",required=True); ap.add_argument("--baseline_csv",required=True); ap.add_argument("--type_csv",required=True); ap.add_argument("--calib_json",required=True); ap.add_argument("--out_csv",required=True); args=ap.parse_args()
    ufm=read(args.ufm_csv); base=read(args.baseline_csv); typ=read(args.type_csv); c=json.load(open(args.calib_json,"r",encoding="utf-8"))
    alpha=np.array(c["alpha"],dtype=np.float64); theta=np.array(c["theta"],dtype=np.float64); temp=float(c["temp"])
    Path(args.out_csv).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out_csv,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["name","predict"])
        for n in sorted(ufm.keys()):
            if n not in base or n not in typ: continue
            q=np.array([float(typ[n]["type_"+k]) for k in TYPE],dtype=np.float64); q=q**temp; q=q/max(q.sum(),1e-8)
            a=float((q*alpha).sum()); th=float((q*theta).sum())
            s=a*float(base[n]["score"])+(1-a)*float(ufm[n]["score"])
            w.writerow([n,"real" if s>=th else "fake"])
    print("Saved",args.out_csv)
if __name__=="__main__": main()
