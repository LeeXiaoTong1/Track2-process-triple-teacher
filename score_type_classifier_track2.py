#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, os, random
from pathlib import Path
import librosa, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_type_classifier_track2 import Track2TypeClassifier, crop_or_repeat, ID2TYPE

class EvalAudio(Dataset):
    def __init__(self, audio_dir, audio_len=64600):
        self.audio_dir=Path(audio_dir); self.audio_len=int(audio_len)
        self.files=sorted([p.name for p in self.audio_dir.iterdir() if p.suffix.lower() in [".wav",".flac",".mp3",".ogg",".m4a"]])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        name=self.files[idx]
        wav,_=librosa.load(str(self.audio_dir/name), sr=16000, mono=True)
        wav=crop_or_repeat(wav,self.audio_len,"head")
        return torch.from_numpy(wav).float(), name

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--eval_audio", required=True); ap.add_argument("--out_csv", required=True)
    ap.add_argument("--gpu", default="0"); ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--num_workers", type=int, default=8)
    args=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=json.load(open(Path(args.model_dir)/"args.json","r",encoding="utf-8"))
    model=Track2TypeClassifier(cfg["xlsr"],cfg["mert"],cfg["beats"],device=device,dim=cfg.get("dim",256),dropout=cfg.get("dropout",0.1)).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir)/"type_classifier.pt",map_location=device), strict=True)
    model.eval()
    ds=EvalAudio(args.eval_audio,cfg.get("audio_len",64600))
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["name","type_speech","type_sound","type_singing","type_music","pred_type"])
        with torch.inference_mode():
            for wav,names in tqdm(dl):
                prob=F.softmax(model(wav.to(device,non_blocking=True)),dim=-1).cpu().numpy()
                for n,p in zip(names,prob): w.writerow([n,float(p[0]),float(p[1]),float(p[2]),float(p[3]),ID2TYPE[int(np.argmax(p))]])
    print("Saved",args.out_csv)
if __name__=="__main__": main()
