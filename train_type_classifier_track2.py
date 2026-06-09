#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an independent Track2 audio-type classifier: speech/sound/singing/music.
This is intentionally separated from UFM real/fake detection because UFM's raw
latest_type_logits are weakly supervised and poorly calibrated.

Inputs:
  --train_audio, --train_label, --dev_audio, --dev_label
  --xlsr, --mert, --beats
Outputs:
  out_dir/type_classifier.pt
  out_dir/args.json
  out_dir/dev_type_probs.csv
"""
import argparse, csv, json, os, random, shutil
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import XLSR, MERT, OpenBEATS

TYPE2ID = {"speech":0, "sound":1, "singing":2, "music":3}
ID2TYPE = ["speech", "sound", "singing", "music"]


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def read_label_csv(path):
    rows=[]
    with open(path, "r", encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        for r in reader:
            name=r["name"].strip()
            typ=r["type"].strip().lower()
            if typ not in TYPE2ID:
                continue
            rows.append((name, TYPE2ID[typ]))
    return rows


def normalize(x):
    x=x.astype(np.float32)
    x=x-float(x.mean())
    x=x/np.sqrt(float(x.var())+1e-7)
    return x


def crop_or_repeat(wav, audio_len, mode="head"):
    L=len(wav); cut=int(audio_len)
    if L<=0:
        wav=np.zeros(cut, dtype=np.float32); L=cut
    if L<cut:
        rep=int(np.ceil(cut/max(L,1)))+1
        x=np.tile(wav, rep)[:cut]
    elif L==cut:
        x=wav.copy()
    else:
        if mode=="random":
            st=random.randint(0, L-cut)
        elif mode=="center":
            st=(L-cut)//2
        else:
            st=0
        x=wav[st:st+cut].copy()
    return normalize(x)


class TypeDataset(Dataset):
    def __init__(self, audio_dir, label_csv, audio_len=64600, crop_mode="head"):
        self.audio_dir=Path(audio_dir)
        self.rows=read_label_csv(label_csv)
        self.audio_len=int(audio_len)
        self.crop_mode=crop_mode
        if len(self.rows)==0:
            raise RuntimeError(f"No rows loaded from {label_csv}")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        name, tid = self.rows[idx]
        wav,_=librosa.load(str(self.audio_dir/name), sr=16000, mono=True)
        wav=crop_or_repeat(wav, self.audio_len, self.crop_mode)
        return torch.from_numpy(wav).float(), name, torch.tensor(tid, dtype=torch.long)


class Track2TypeClassifier(nn.Module):
    def __init__(self, xlsr_dir, mert_dir, beats_dir, device="cuda", dim=256, dropout=0.1):
        super().__init__()
        self.xlsr=XLSR(model_dir=xlsr_dir, device=device, freeze=True)
        self.mert=MERT(model_dir=mert_dir, device=device, freeze=True)
        self.beats=OpenBEATS(model_dir=beats_dir, device=device, freeze=True)
        xdim=getattr(self.xlsr.model.config, "hidden_size", 1024)
        mdim=getattr(self.mert.model.config, "hidden_size", 1024)
        bdim=1024
        self.proj_x=nn.Linear(xdim*2, dim)
        self.proj_m=nn.Linear(mdim*2, dim)
        self.proj_b=nn.Linear(bdim*2, dim)
        self.head=nn.Sequential(
            nn.LayerNorm(dim*3),
            nn.Linear(dim*3, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim//2, 4)
        )
    def train(self, mode=True):
        super().train(mode)
        self.xlsr.model.eval(); self.mert.model.eval(); self.beats.model.eval()
        for p in self.xlsr.model.parameters(): p.requires_grad=False
        for p in self.mert.model.parameters(): p.requires_grad=False
        for p in self.beats.model.parameters(): p.requires_grad=False
        return self
    def _pool(self, z):
        z=torch.nan_to_num(z.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20,20)
        return torch.cat([z.mean(1), z.std(1, unbiased=False)], dim=-1)
    def forward(self, wav):
        with torch.no_grad():
            x=self.xlsr.extract_features(wav)
            m=self.mert.extract_features(wav)
            b=self.beats.extract_features(wav)
        hx=self.proj_x(self._pool(x))
        hm=self.proj_m(self._pool(m))
        hb=self.proj_b(self._pool(b))
        return self.head(torch.cat([hx,hm,hb], dim=-1))


def eval_model(model, loader, device, out_csv=None):
    model.eval(); probs=[]; ys=[]; names=[]
    with torch.inference_mode():
        for wav, name, tid in tqdm(loader, desc="eval"):
            wav=wav.to(device, non_blocking=True); tid=tid.to(device)
            logits=model(wav)
            prob=F.softmax(logits, dim=-1)
            probs.append(prob.cpu()); ys.append(tid.cpu()); names += list(name)
    probs=torch.cat(probs).numpy(); ys=torch.cat(ys).numpy(); pred=probs.argmax(1)
    acc=accuracy_score(ys,pred); macro=f1_score(ys,pred,average="macro",labels=[0,1,2,3],zero_division=0)
    cm=confusion_matrix(ys,pred,labels=[0,1,2,3])
    print("Type Acc:", acc, "MacroF1:", macro)
    print("Confusion rows=true cols=pred [speech,sound,singing,music]:\n", cm)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv,"w",encoding="utf-8",newline="") as f:
            w=csv.writer(f); w.writerow(["name","type_speech","type_sound","type_singing","type_music","pred_type"])
            for n,p in zip(names, probs):
                w.writerow([n, float(p[0]), float(p[1]), float(p[2]), float(p[3]), ID2TYPE[int(np.argmax(p))]])
    return acc, macro


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_audio", required=True); ap.add_argument("--train_label", required=True)
    ap.add_argument("--dev_audio", required=True); ap.add_argument("--dev_label", required=True)
    ap.add_argument("--xlsr", required=True); ap.add_argument("--mert", required=True); ap.add_argument("--beats", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--audio_len", type=int, default=64600); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dim", type=int, default=256); ap.add_argument("--dropout", type=float, default=0.1)
    args=ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    out=Path(args.out_dir)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    with open(out/"args.json","w",encoding="utf-8") as f: json.dump(vars(args), f, indent=2)
    train_ds=TypeDataset(args.train_audio,args.train_label,args.audio_len,"random")
    dev_ds=TypeDataset(args.dev_audio,args.dev_label,args.audio_len,"head")
    # inverse-frequency type sampler, important because Music is small
    counts=np.bincount([tid for _,tid in train_ds.rows], minlength=4).astype(np.float64)
    weights=np.asarray([1.0/counts[tid] for _,tid in train_ds.rows], dtype=np.float64)
    sampler=WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=args.num_workers,pin_memory=True)
    dev_dl=DataLoader(dev_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    model=Track2TypeClassifier(args.xlsr,args.mert,args.beats,device=device,dim=args.dim,dropout=args.dropout).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    best=-1
    for ep in range(args.epochs):
        model.train(); losses=[]
        for wav, _, tid in tqdm(train_dl, desc=f"train ep{ep}"):
            wav=wav.to(device,non_blocking=True); tid=tid.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(wav)
            loss=F.cross_entropy(logits, tid)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.item()))
        print("epoch",ep,"loss",np.mean(losses))
        acc, macro=eval_model(model,dev_dl,device,out_csv=str(out/"dev_type_probs.csv"))
        if macro>best:
            best=macro; torch.save(model.state_dict(), out/"type_classifier.pt"); print("best updated",best)
    print("Best dev type macroF1", best)

if __name__=="__main__": main()
