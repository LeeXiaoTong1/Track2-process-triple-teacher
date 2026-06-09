#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_multi_branch_fusion_holdout.py

Generic Track2 multi-branch fusion tuner.

Usage:
  python tune_multi_branch_fusion_holdout.py \
    --branch baseline:path/to/dev_baseline.csv \
    --branch speech:path/to/dev_speech.csv \
    --branch sound:path/to/dev_sound.csv \
    --branch music:path/to/dev_music.csv \
    --type_csv path/to/dev_type_probs.csv \
    --label_csv path/to/dev.csv \
    --out_json fusion.json

CSV branch format:
  name,score,...
score = P(real)

Type CSV:
  name,type_speech,type_sound,type_singing,type_music

Fusion:
  q = normalize(type_prob ** temp)
  type-specific branch weights W[type, branch]
  type_score[type] = sum_branch W[type, branch] * score_branch
  final_score = sum_type q[type] * type_score[type]
  threshold = sum_type q[type] * theta[type] + offset
  predict real if final_score >= threshold else fake

Selection:
  split Dev by stable hash into tune/holdout.
  search on tune; select by holdout objective with gap penalty.
"""

import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
from tqdm import tqdm

TYPE = ["speech", "sound", "singing", "music"]
TYPE2ID = {t:i for i,t in enumerate(TYPE)}


def read_by_name(path):
    d = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["name"].strip()] = r
    return d


def stable_hash01(s, seed=1234):
    h = hashlib.md5((str(seed) + "::" + s).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def parse_branch(s):
    if ":" not in s:
        raise ValueError("--branch must be name:path")
    name, path = s.split(":", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError("--branch must be name:path")
    return name, path


def load_arrays(branch_specs, type_csv, label_csv):
    branch_names = []
    branch_maps = []
    for spec in branch_specs:
        name, path = parse_branch(spec)
        branch_names.append(name)
        branch_maps.append(read_by_name(path))

    typ = read_by_name(type_csv)
    names, y, tid, q = [], [], [], []
    scores = []
    missing = 0

    with open(label_csv, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            nm = r["name"].strip()
            if nm not in typ or any(nm not in bm for bm in branch_maps):
                missing += 1
                continue
            names.append(nm)
            y.append(0 if r["label"].strip().lower() == "real" else 1)
            tid.append(TYPE2ID[r["type"].strip().lower()])
            scores.append([float(bm[nm]["score"]) for bm in branch_maps])
            q.append([float(typ[nm]["type_" + t]) for t in TYPE])

    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)
    return (
        branch_names,
        np.asarray(names),
        np.asarray(y, dtype=np.int64),
        np.asarray(tid, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        q,
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
    return np.asarray([
        macro_f1_binary(y[tid == t], pred[tid == t]) if np.any(tid == t) else 0.0
        for t in range(4)
    ], dtype=np.float64)


def objective(vals, floor=0.95, penalty=2.0, min_bonus=0.10, speech_bonus=0.0, sound_bonus=0.0, music_bonus=0.0):
    vals = np.asarray(vals, dtype=np.float64)
    bonus = speech_bonus * vals[0] + sound_bonus * vals[1] + music_bonus * vals[3]
    return float(vals.mean() - penalty * np.maximum(0.0, floor - vals).mean() + min_bonus * vals.min() + bonus)


def softmax_rows(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def eval_params(y, tid, scores, qraw, W, theta, temp, offset, args):
    q = np.power(qraw, temp)
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-8)
    type_scores = scores @ W.T  # [N, T]
    final_score = (q * type_scores).sum(axis=1)
    th = (q * theta[None, :]).sum(axis=1) + offset
    pred = np.where(final_score >= th, 0, 1)
    vals = per_type_f1(y, pred, tid)
    obj = objective(vals, args.floor, args.penalty, args.min_bonus, args.speech_bonus, args.sound_bonus, args.music_bonus)
    return obj, vals


def sample_weights(rng, branch_names, mode):
    B = len(branch_names)
    raw = rng.normal(0.0, 1.0, size=(4, B))

    # Initialize priors by branch name.
    for j, name in enumerate(branch_names):
        low = name.lower()
        if "baseline" in low or "teacher" in low or "xlsr" in low:
            raw[0, j] += 1.2  # speech
            raw[2, j] += 2.0  # singing
        if "speech" in low:
            raw[0, j] += 2.5
        if "sound" in low or "ufm" in low:
            raw[1, j] += 2.0
        if "music" in low:
            raw[3, j] += 2.5

    if mode == "protect_vocal":
        for j, name in enumerate(branch_names):
            low = name.lower()
            if "baseline" in low or "teacher" in low or "xlsr" in low:
                raw[0, j] += 1.0
                raw[2, j] += 1.5
    elif mode == "music_boost":
        for j, name in enumerate(branch_names):
            if "music" in name.lower():
                raw[3, j] += 1.5
    elif mode == "speech_music_boost":
        for j, name in enumerate(branch_names):
            low = name.lower()
            if "speech" in low:
                raw[0, j] += 1.8
            if "music" in low:
                raw[3, j] += 1.8

    return softmax_rows(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True, help="name:path_to_score_csv. Repeat for multiple branches.")
    ap.add_argument("--type_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--trials", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--holdout_frac", type=float, default=0.35)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--min_bonus", type=float, default=0.10)
    ap.add_argument("--speech_bonus", type=float, default=0.02)
    ap.add_argument("--sound_bonus", type=float, default=0.02)
    ap.add_argument("--music_bonus", type=float, default=0.08)
    ap.add_argument("--mode", default="speech_music_boost", choices=["free", "protect_vocal", "music_boost", "speech_music_boost"])
    args = ap.parse_args()

    branch_names, names, y, tid, scores, q, missing = load_arrays(args.branch, args.type_csv, args.label_csv)
    hv = np.asarray([stable_hash01(n, args.seed) for n in names])
    hold = hv < args.holdout_frac
    tune = ~hold
    print("Branches:", branch_names)
    print("Matched:", len(y), "Missing:", missing, "Tune:", int(tune.sum()), "Holdout:", int(hold.sum()))

    rng = np.random.default_rng(args.seed)
    best = (-1e18, None)

    # Sensible initial all-equal and name-prior candidates.
    initW = np.ones((4, len(branch_names)), dtype=np.float64) / len(branch_names)
    theta0 = np.asarray([0.50, 0.55, 0.08, 0.55], dtype=np.float64)
    for W, theta, temp, offset in [(initW, theta0, 2.0, 0.0)]:
        ot, vt = eval_params(y[tune], tid[tune], scores[tune], q[tune], W, theta, temp, offset, args)
        oh, vh = eval_params(y[hold], tid[hold], scores[hold], q[hold], W, theta, temp, offset, args)
        sc = oh - 0.25 * abs(ot - oh)
        best = (sc, (W.copy(), theta.copy(), temp, offset, ot, vt.copy(), oh, vh.copy()))

    for _ in tqdm(range(args.trials), desc="multi_branch_tune"):
        W = sample_weights(rng, branch_names, args.mode)
        theta = np.asarray([
            rng.uniform(0.20, 0.85),
            rng.uniform(0.10, 0.85),
            rng.uniform(0.01, 0.50),
            rng.uniform(0.15, 0.90),
        ], dtype=np.float64)
        temp = float(rng.uniform(0.5, 7.0))
        offset = float(rng.uniform(-0.15, 0.15))

        ot, vt = eval_params(y[tune], tid[tune], scores[tune], q[tune], W, theta, temp, offset, args)
        oh, vh = eval_params(y[hold], tid[hold], scores[hold], q[hold], W, theta, temp, offset, args)
        sc = oh - 0.25 * abs(ot - oh)
        if sc > best[0]:
            best = (sc, (W.copy(), theta.copy(), temp, offset, ot, vt.copy(), oh, vh.copy()))
            print(f"\n[best] score={sc:.6f} hold_obj={oh:.6f} hold_mean={vh.mean():.6f} hold_min={vh.min():.6f} hold_per={np.round(vh,6).tolist()} temp={temp:.3f} offset={offset:.3f}")

    W, theta, temp, offset, ot, vt, oh, vh = best[1]
    of, vf = eval_params(y, tid, scores, q, W, theta, temp, offset, args)
    out = {
        "type_order": TYPE,
        "branch_order": branch_names,
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
        "rule": "q=normalize(type_prob**temp); score=sum_t q_t*(W_t dot branch_scores); threshold=sum_t q_t*theta_t+offset; real if score>=threshold",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved:", args.out_json)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
