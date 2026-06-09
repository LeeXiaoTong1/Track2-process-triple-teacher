#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_score_multicrop_plus.py

Deterministic multi-crop scoring script for AT-ADD Track2.

Outputs:
  name, score, logit_real, logit_fake, margin,
  type_speech, type_sound, type_singing, type_music,
  crop_var,
  expert_xlsr, expert_mert, expert_beats, expert_artifact

score = P(real)
margin = logit_real - logit_fake
decision convention:
  score >= threshold -> real
  score <  threshold -> fake
"""

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import *  # noqa


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--eval_task", type=str, default="atadd-track2")
    p.add_argument("--eval_audio", type=str, required=True)
    p.add_argument("--score_file", type=str, required=True)

    p.add_argument("--num_crops", type=int, default=5)
    p.add_argument("--audio_len", type=int, default=None)
    p.add_argument("--batch_files", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_files", type=int, default=-1)
    p.add_argument("--agg", type=str, default="mean_logit",
                   choices=["mean_logit", "mean_score", "min_real"])

    p.add_argument("--xlsr", type=str, default="")
    p.add_argument("--wavlm", type=str, default="")
    p.add_argument("--mert", type=str, default="")
    p.add_argument("--beats", type=str, default="")
    p.add_argument("--obeats", type=str, default="")
    p.add_argument("--model", type=str, default="")

    return p.parse_args()


def load_saved_args(cli):
    model_path = Path(cli.model_path)
    json_path = model_path / "args.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find args.json: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    args = SimpleNamespace(**saved)

    args.model_path = cli.model_path
    args.gpu = cli.gpu
    args.eval_task = cli.eval_task
    args.eval_audio = cli.eval_audio
    args.score_file = cli.score_file
    args.num_crops = int(cli.num_crops)
    args.batch_files = int(cli.batch_files)
    args.num_workers = int(cli.num_workers)
    args.max_files = int(cli.max_files)
    args.agg = cli.agg

    if cli.audio_len is not None:
        args.audio_len = int(cli.audio_len)
    else:
        args.audio_len = int(getattr(args, "audio_len", 64600))

    for k in ["xlsr", "wavlm", "mert", "beats", "obeats", "model"]:
        v = getattr(cli, k)
        if v:
            setattr(args, k, v)

    if not hasattr(args, "model"):
        raise ValueError("args.json has no model field. Pass --model explicitly.")

    if not hasattr(args, "xlsr"):
        args.xlsr = "huggingface/wav2vec2-xls-r-300m"
    if not hasattr(args, "wavlm"):
        args.wavlm = "huggingface/wavlm-large/"
    if not hasattr(args, "mert"):
        args.mert = "huggingface/MERT-v1-330M/"
    if not hasattr(args, "beats"):
        args.beats = getattr(args, "obeats", "huggingface/OpenBEATs-ICME")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def build_model(args):
    m = args.model

    if m == "aasist":
        model = Rawaasist()

    elif m == "specresnet":
        model = ResNet18ForAudio()

    elif m == "fr-w2v2aasist":
        model = XLSRAASIST(model_dir=args.xlsr)

    elif m == "ft-w2v2aasist":
        model = XLSRAASIST(model_dir=args.xlsr, freeze=False)

    elif m == "fr-wavlmaasist":
        model = WAVLMAASIST(model_dir=args.wavlm)

    elif m == "ft-wavlmaasist":
        model = WAVLMAASIST(model_dir=args.wavlm, freeze=False)

    elif m == "fr-mertaasist":
        model = MERTAASIST(model_dir=args.mert)

    elif m == "ft-mertaasist":
        model = MERTAASIST(model_dir=args.mert, freeze=False)

    elif m == "pt-w2v2aasist":
        model = PTW2V2AASIST(
            model_dir=args.xlsr,
            prompt_dim=getattr(args, "prompt_dim", 1024),
            num_prompt_tokens=getattr(args, "num_prompt_tokens", 10),
            dropout=getattr(args, "pt_dropout", 0.1),
        )

    elif m == "wpt-w2v2aasist":
        model = WPTW2V2AASIST(
            model_dir=args.xlsr,
            prompt_dim=getattr(args, "prompt_dim", 1024),
            num_prompt_tokens=getattr(args, "num_prompt_tokens", 10),
            num_wavelet_tokens=getattr(args, "num_wavelet_tokens", 4),
            dropout=getattr(args, "pt_dropout", 0.1),
        )

    elif m == "pt-wavlmaasist":
        model = PTWAVLMAASIST(
            model_dir=args.wavlm,
            prompt_dim=getattr(args, "prompt_dim", 1024),
            num_prompt_tokens=getattr(args, "num_prompt_tokens", 10),
            dropout=getattr(args, "pt_dropout", 0.1),
        )

    elif m == "wpt-wavlmaasist":
        model = WPTWAVLMAASIST(
            model_dir=args.wavlm,
            prompt_dim=getattr(args, "prompt_dim", 1024),
            num_prompt_tokens=getattr(args, "num_prompt_tokens", 10),
            num_wavelet_tokens=getattr(args, "num_wavelet_tokens", 4),
            dropout=getattr(args, "pt_dropout", 0.1),
        )

    elif m == "pt-mertaasist":
        model = PTMERTAASIST(
            model_dir=args.mert,
            prompt_dim=getattr(args, "prompt_dim", 1024),
            num_prompt_tokens=getattr(args, "num_prompt_tokens", 10),
            dropout=getattr(args, "pt_dropout", 0.1),
        )

    elif m == "wpt-mertaasist":
        model = WPTMERTAASIST(
            model_dir=args.mert,
            prompt_dim=getattr(args, "prompt_dim", 1024),
            num_prompt_tokens=getattr(args, "num_prompt_tokens", 10),
            num_wavelet_tokens=getattr(args, "num_wavelet_tokens", 4),
            dropout=getattr(args, "pt_dropout", 0.1),
        )

    elif m == "t2-router-xlsr-mert":
        model = TypeRoutedXLSRMERTAASIST(
            xlsr_dir=args.xlsr,
            mert_dir=args.mert,
            device=args.device,
            freeze_xlsr=getattr(args, "t2_router_freeze_xlsr", True),
            freeze_mert=getattr(args, "t2_router_freeze_mert", True),
        )

    elif m == "ufm-track2-full":
        model = UFMTrack2Full(
            xlsr_dir=args.xlsr,
            mert_dir=args.mert,
            beats_dir=args.beats,
            device=args.device,
            freeze_xlsr=getattr(args, "ufm_freeze_xlsr", True),
            freeze_mert=getattr(args, "ufm_freeze_mert", True),
            freeze_beats=getattr(args, "ufm_freeze_beats", True),
            dim=getattr(args, "ufm_dim", 512),
            mem_slots=getattr(args, "ufm_mem_slots", 16),
            heads=getattr(args, "ufm_heads", 8),
            layers=getattr(args, "ufm_layers", 0),
            dropout=getattr(args, "ufm_dropout", 0.0),
        )

    else:
        raise ValueError(f"Unsupported model type: {m}")

    return model.to(args.device)


def load_checkpoint(model, model_path, device):
    ckpt_path = Path(model_path) / "atadd_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    if isinstance(ckpt, dict):
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_ckpt[k] = v
        ckpt = new_ckpt

    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys:", len(missing))
    if len(missing) > 0:
        print("  first missing:", missing[:20])

    print("Unexpected keys:", len(unexpected))
    if len(unexpected) > 0:
        print("  first unexpected:", unexpected[:20])

    return model


def normalize_clip(x):
    x = x.astype(np.float32)
    x = x - float(x.mean())
    x = x / np.sqrt(float(x.var()) + 1e-7)
    return x.astype(np.float32)


def make_crops(wav, cut, num_crops):
    cut = int(cut)
    num_crops = max(1, int(num_crops))
    L = len(wav)

    if L <= 0:
        wav = np.zeros(cut, dtype=np.float32)
        L = cut

    if L < cut:
        rep = int(np.ceil(cut / max(L, 1))) + 1
        wav = np.tile(wav, rep)[:cut]
        crops = [wav.copy() for _ in range(num_crops)]

    elif L == cut:
        crops = [wav.copy() for _ in range(num_crops)]

    else:
        if num_crops == 1:
            starts = [0]
        else:
            starts = np.linspace(0, L - cut, num_crops).astype(np.int64).tolist()

        crops = [wav[s:s + cut].copy() for s in starts]

    crops = [normalize_clip(c) for c in crops]

    return torch.from_numpy(np.stack(crops, axis=0)).float().contiguous()


class MultiCropAudioDataset(Dataset):
    def __init__(self, audio_dir, audio_len=64600, num_crops=5, max_files=-1):
        self.audio_dir = Path(audio_dir)
        self.audio_len = int(audio_len)
        self.num_crops = int(num_crops)

        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio dir not found: {self.audio_dir}")

        self.files = sorted([
            f.name for f in self.audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
        ])

        if max_files is not None and int(max_files) > 0:
            self.files = self.files[:int(max_files)]

        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in: {self.audio_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = self.audio_dir / fn

        wav, _ = librosa.load(str(path), sr=16000, mono=True)
        crops = make_crops(
            wav,
            cut=self.audio_len,
            num_crops=self.num_crops
        )

        return crops, fn


def aggregate_logits(logits_all, agg):
    """
    logits_all: [B, C, 2]
    """
    avg_logits = logits_all.mean(dim=1)

    if agg == "mean_logit":
        score = F.softmax(avg_logits, dim=-1)[:, 0]

    elif agg == "mean_score":
        score = F.softmax(logits_all, dim=-1)[:, :, 0].mean(dim=1)

    elif agg == "min_real":
        score = F.softmax(logits_all, dim=-1)[:, :, 0].min(dim=1).values

    else:
        raise ValueError(f"Unknown agg: {agg}")

    crop_real = F.softmax(logits_all, dim=-1)[:, :, 0]
    crop_var = crop_real.var(dim=1, unbiased=False)

    return (
        score.detach().cpu().numpy(),
        avg_logits.detach().cpu().numpy(),
        crop_var.detach().cpu().numpy(),
    )


def main():
    cli = parse_args()
    args = load_saved_args(cli)

    print("Model path:", args.model_path)
    print("Model:", args.model)
    print("Eval audio:", args.eval_audio)
    print("Score file:", args.score_file)
    print("Device:", args.device)
    print("audio_len:", args.audio_len)
    print("num_crops:", args.num_crops)
    print("batch_files:", args.batch_files)
    print("actual forward batch:", args.batch_files * args.num_crops)
    print("agg:", args.agg)

    model = build_model(args)
    model = load_checkpoint(model, args.model_path, args.device)
    model.eval()

    ds = MultiCropAudioDataset(
        audio_dir=args.eval_audio,
        audio_len=args.audio_len,
        num_crops=args.num_crops,
        max_files=args.max_files,
    )

    loader_kwargs = dict(
        batch_size=args.batch_files,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )

    if args.num_workers > 0:
        loader_kwargs.update(dict(
            persistent_workers=True,
            prefetch_factor=2
        ))

    dl = DataLoader(ds, **loader_kwargs)

    out_path = Path(args.score_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "name",
        "score",
        "logit_real",
        "logit_fake",
        "margin",
        "type_speech",
        "type_sound",
        "type_singing",
        "type_music",
        "crop_var",
        "expert_xlsr",
        "expert_mert",
        "expert_beats",
        "expert_artifact",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with torch.inference_mode():
            for crops, names in tqdm(dl, total=len(dl)):
                B, C, L = crops.shape
                flat = crops.view(B * C, L).to(args.device, non_blocking=True)

                _, logits_flat = model(flat)
                logits_all = logits_flat.view(B, C, -1)

                scores, avg_logits, crop_var = aggregate_logits(logits_all, args.agg)

                if hasattr(model, "latest_type_logits") and model.latest_type_logits is not None:
                    type_logits = model.latest_type_logits.view(B, C, -1).mean(dim=1)
                    type_prob = F.softmax(type_logits, dim=-1).detach().cpu().numpy()

                    if type_prob.shape[1] < 4:
                        pad = np.ones((B, 4 - type_prob.shape[1]), dtype=np.float32) / 4.0
                        type_prob = np.concatenate([type_prob, pad], axis=1)[:, :4]
                    else:
                        type_prob = type_prob[:, :4]
                else:
                    type_prob = np.ones((B, 4), dtype=np.float32) / 4.0

                if hasattr(model, "latest_expert_weights") and model.latest_expert_weights is not None:
                    ew = model.latest_expert_weights.view(B, C, -1).mean(dim=1)
                    ew = ew / ew.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    ew_np = ew.detach().cpu().numpy()

                    if ew_np.shape[1] < 4:
                        pad = np.zeros((B, 4 - ew_np.shape[1]), dtype=np.float32)
                        ew_np = np.concatenate([ew_np, pad], axis=1)[:, :4]
                    else:
                        ew_np = ew_np[:, :4]
                else:
                    ew_np = np.zeros((B, 4), dtype=np.float32)

                for i, name in enumerate(names):
                    lr = float(avg_logits[i, 0])
                    lf = float(avg_logits[i, 1])
                    margin = lr - lf

                    writer.writerow([
                        name,
                        float(scores[i]),
                        lr,
                        lf,
                        float(margin),
                        float(type_prob[i, 0]),
                        float(type_prob[i, 1]),
                        float(type_prob[i, 2]),
                        float(type_prob[i, 3]),
                        float(crop_var[i]),
                        float(ew_np[i, 0]),
                        float(ew_np[i, 1]),
                        float(ew_np[i, 2]),
                        float(ew_np[i, 3]),
                    ])

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
