#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_three_branch_fusion_holdout.py

Three-branch Track2 fusion tuner:
  baseline branch: usually protects Speech/Singing
  ufm branch: general all-type / Sound branch
  music branch: Music-specialist branch
  independent type classifier: provides type probabilities

Fusion:
  q = normalize(type_prob ** temp)
  per-type branch weights W[t] over [baseline, ufm, music]
  score_t = W[t,0]*s_base + W[t,1]*s_ufm + W[t,2]*s_music
  score = sum_t q_t * score_t
  threshold = sum_t q_t * theta_t + offset
  pred = real if score >= threshold else fake

Selection:
  Random search on tune split, selected by holdout objective.
"""

import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

TYPE = ["speech", "sound", "singing", "music"]
TYPE2ID = {t: i for i, t in enumerate(TYPE)}
BRANCH = ["baseline", "ufm", "music"]


def read_by_name(path):
    d = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["name"].strip()] = r
    return d


def stable_hash01(s, seed=1234):
    h = hashlib.md5((str(seed) + "::" + s).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def load_arrays(baseline_csv, ufm_csv, music_csv, type_csv, label_csv):
    base = read_by_name(baseline_csv)
    ufm = read_by_name(ufm_csv)
    mus = read_by_name(music_csv)
    typ = read_by_name(type_csv)

    names, y, tid, sb, su, sm, qt = [], [], [], [], [], [], []
    missing = 0

    with open(label_csv, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = r["name"].strip()
            if name not in base or name not in ufm or name not in mus or name not in typ:
                missing += 1
                continue

            names.append(name)
            y.append(0 if r["label"].strip().lower() == "real" else 1)
            tid.append(TYPE2ID[r["type"].strip().lower()])
            sb.append(float(base[name]["score"]))
            su.append(float(ufm[name]["score"]))
            sm.append(float(mus[name]["score"]))
            qt.append([float(typ[name]["type_" + k]) for k in TYPE])

    qt = np.asarray(qt, dtype=np.float64)
    qt = qt / np.maximum(qt.sum(axis=1, keepdims=True), 1e-8)

    return (
        np.asarray(names),
        np.asarray(y, dtype=np.int64),
        np.asarray(tid, dtype=np.int64),
        np.asarray(sb, dtype=np.float64),
        np.asarray(su, dtype=np.float64),
        np.asarray(sm, dtype=np.float64),
        qt,
        missing,
    )


def macro_f1_binary(y, pred):
    y0 = (y == 0)
    y1 = ~y0
    p0 = (pred == 0)
    p1 = ~p0

    tp0 = np.count_nonzero(y0 & p0)
    fp0 = np.count_nonzero(y1 & p0)
    fn0 = np.count_nonzero(y0 & p1)

    tp1 = np.count_nonzero(y1 & p1)
    fp1 = np.count_nonzero(y0 & p1)
    fn1 = np.count_nonzero(y1 & p0)

    den0 = 2 * tp0 + fp0 + fn0
    den1 = 2 * tp1 + fp1 + fn1

    f0 = 0.0 if den0 == 0 else 2.0 * tp0 / den0
    f1 = 0.0 if den1 == 0 else 2.0 * tp1 / den1
    return 0.5 * (f0 + f1)


def per_type_f1(y, pred, tid):
    vals = []
    for t in range(4):
        m = (tid == t)
        vals.append(macro_f1_binary(y[m], pred[m]) if np.any(m) else 0.0)
    return np.asarray(vals, dtype=np.float64)


def objective(vals, floor=0.95, penalty=2.0, min_bonus=0.10, music_bonus=0.05):
    vals = np.asarray(vals, dtype=np.float64)
    return float(
        vals.mean()
        - penalty * np.maximum(0.0, floor - vals).mean()
        + min_bonus * vals.min()
        + music_bonus * vals[3]
    )


def softmax_rows(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-8)


def eval_params(y, tid, sb, su, sm, qt, W, theta, temp, offset, floor, penalty, min_bonus, music_bonus):
    q = np.power(qt, temp)
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)

    scores = np.stack([sb, su, sm], axis=1)  # [N,3]
    type_scores = scores @ W.T              # [N,4], score if considered as each type
    fused = (q * type_scores).sum(axis=1)

    th = (q * theta[None, :]).sum(axis=1) + offset
    pred = np.where(fused >= th, 0, 1)

    vals = per_type_f1(y, pred, tid)
    obj = objective(vals, floor=floor, penalty=penalty, min_bonus=min_bonus, music_bonus=music_bonus)
    return obj, vals


def sample_params(rng, mode):
    # W rows correspond to speech, sound, singing, music.
    # columns correspond to baseline, ufm, music-specialist.
    if mode == "music_boost":
        raw = np.vstack([
            [rng.normal(2.2, 0.8), rng.normal(0.4, 0.8), rng.normal(-1.0, 0.8)],  # speech: baseline
            [rng.normal(-0.8, 0.8), rng.normal(2.0, 0.8), rng.normal(0.2, 0.8)],  # sound: ufm
            [rng.normal(2.4, 0.8), rng.normal(0.2, 0.8), rng.normal(-1.0, 0.8)],  # singing: baseline
            [rng.normal(-1.0, 0.8), rng.normal(0.7, 0.9), rng.normal(2.4, 0.8)],  # music: music specialist
        ])
    elif mode == "conservative_vocal":
        raw = np.vstack([
            [rng.normal(2.7, 0.7), rng.normal(0.0, 0.7), rng.normal(-1.2, 0.7)],
            [rng.normal(-0.5, 0.8), rng.normal(2.0, 0.8), rng.normal(0.0, 0.8)],
            [rng.normal(3.0, 0.7), rng.normal(-0.1, 0.7), rng.normal(-1.2, 0.7)],
            [rng.normal(-0.9, 0.8), rng.normal(0.5, 0.9), rng.normal(2.5, 0.8)],
        ])
    else:
        raw = rng.normal(0, 1, size=(4, 3))

    W = softmax_rows(raw, axis=1)

    theta = np.array([
        rng.uniform(0.20, 0.85),  # speech
        rng.uniform(0.10, 0.85),  # sound
        rng.uniform(0.01, 0.50),  # singing
        rng.uniform(0.15, 0.90),  # music
    ], dtype=np.float64)

    temp = float(rng.uniform(0.5, 7.0))
    offset = float(rng.uniform(-0.15, 0.15))
    return W, theta, temp, offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--ufm_csv", required=True)
    ap.add_argument("--music_csv", required=True)
    ap.add_argument("--type_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--trials", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--holdout_frac", type=float, default=0.35)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--min_bonus", type=float, default=0.10)
    ap.add_argument("--music_bonus", type=float, default=0.08)
    ap.add_argument("--mode", default="music_boost", choices=["music_boost", "conservative_vocal", "free"])
    args = ap.parse_args()

    names, y, tid, sb, su, sm, qt, missing = load_arrays(
        args.baseline_csv, args.ufm_csv, args.music_csv, args.type_csv, args.label_csv
    )
    hv = np.asarray([stable_hash01(n, args.seed) for n in names])
    hold = hv < args.holdout_frac
    tune = ~hold

    print("Matched:", len(y), "Missing:", missing, "Tune:", int(tune.sum()), "Holdout:", int(hold.sum()))

    rng = np.random.default_rng(args.seed)
    best = (-1e18, None)

    # sensible initial candidates
    init = [
        (
            np.asarray([[0.90,0.10,0.00],[0.10,0.80,0.10],[0.95,0.05,0.00],[0.05,0.25,0.70]], dtype=np.float64),
            np.asarray([0.52,0.55,0.06,0.60], dtype=np.float64), 2.5, 0.0
        ),
        (
            np.asarray([[0.75,0.25,0.00],[0.05,0.85,0.10],[0.85,0.15,0.00],[0.02,0.18,0.80]], dtype=np.float64),
            np.asarray([0.55,0.55,0.08,0.65], dtype=np.float64), 3.0, 0.0
        ),
    ]

    for W, theta, temp, offset in init:
        ot, vt = eval_params(y[tune], tid[tune], sb[tune], su[tune], sm[tune], qt[tune], W, theta, temp, offset, args.floor, args.penalty, args.min_bonus, args.music_bonus)
        oh, vh = eval_params(y[hold], tid[hold], sb[hold], su[hold], sm[hold], qt[hold], W, theta, temp, offset, args.floor, args.penalty, args.min_bonus, args.music_bonus)
        score = oh - 0.25 * abs(ot - oh)
        if score > best[0]:
            best = (score, (W.copy(), theta.copy(), temp, offset, ot, vt.copy(), oh, vh.copy()))

    for _ in tqdm(range(args.trials), desc="three_branch_tune"):
        W, theta, temp, offset = sample_params(rng, args.mode)
        ot, vt = eval_params(y[tune], tid[tune], sb[tune], su[tune], sm[tune], qt[tune], W, theta, temp, offset, args.floor, args.penalty, args.min_bonus, args.music_bonus)
        oh, vh = eval_params(y[hold], tid[hold], sb[hold], su[hold], sm[hold], qt[hold], W, theta, temp, offset, args.floor, args.penalty, args.min_bonus, args.music_bonus)
        score = oh - 0.25 * abs(ot - oh)
        if score > best[0]:
            best = (score, (W.copy(), theta.copy(), temp, offset, ot, vt.copy(), oh, vh.copy()))
            print(
                f"\n[best] score={score:.6f} tune_obj={ot:.6f} hold_obj={oh:.6f} "
                f"hold_mean={vh.mean():.6f} hold_min={vh.min():.6f} "
                f"hold_per={np.round(vh,6).tolist()} temp={temp:.3f} offset={offset:.3f}"
            )

    W, theta, temp, offset, ot, vt, oh, vh = best[1]
    of, vf = eval_params(y, tid, sb, su, sm, qt, W, theta, temp, offset, args.floor, args.penalty, args.min_bonus, args.music_bonus)

    out = {
        "type_order": TYPE,
        "branch_order": BRANCH,
        "weights": W.tolist(),
        "theta": theta.tolist(),
        "temp": float(temp),
        "offset": float(offset),
        "selection": "holdout",
        "tune_objective": float(ot),
        "tune_per_type_f1": [float(x) for x in vt],
        "holdout_objective": float(oh),
        "holdout_per_type_f1": [float(x) for x in vh],
        "full_dev_objective": float(of),
        "full_dev_per_type_f1": [float(x) for x in vf],
        "full_dev_mean_f1": float(vf.mean()),
        "full_dev_min_f1": float(vf.min()),
        "floor": args.floor,
        "penalty": args.penalty,
        "min_bonus": args.min_bonus,
        "music_bonus": args.music_bonus,
        "rule": "q=normalize(type_prob**temp); per-type branch weights over [baseline,ufm,music]; score=sum_t q_t*(W_t dot branch_scores); threshold=sum_t q_t*theta_t+offset",
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nSaved:", args.out_json)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
