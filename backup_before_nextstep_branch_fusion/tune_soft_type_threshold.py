#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_soft_type_threshold.py

Tune Track2 type-posterior dynamic thresholds on DEV.

Input UFM score CSV must contain:
  name, score, type_speech, type_sound, type_singing, type_music

Decision:
  q = normalize(q ** gamma)
  dyn_threshold = sum_t q_t * theta_t + offset
  pred = real if score >= dyn_threshold else fake

This uses dev labels/types only for calibration and evaluation. Do not tune on Progress/Eval.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

TYPE_ORDER = ["speech", "sound", "singing", "music"]
TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]


def read_score(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["name"].strip()] = r
    return rows


def read_label(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "name": r["name"].strip(),
                "label": 0 if r["label"].strip().lower() == "real" else 1,
                "type": r["type"].strip().lower(),
            })
    return rows


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)


def track2_metrics(y, pred, types):
    per = []
    for t in TYPE_ORDER:
        idx = (types == t)
        if idx.sum() == 0:
            per.append(0.0)
        else:
            per.append(float(macro_f1(y[idx], pred[idx])))
    return float(np.mean(per)), per, float(np.min(per))


def normalize_type_prob(q, gamma):
    q = np.clip(q, 1e-8, 1.0)
    q = np.power(q, gamma)
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-8, None)
    return q


def predict(score, q, theta, gamma, offset):
    qg = normalize_type_prob(q, gamma)
    dyn_th = qg @ theta + offset
    return np.where(score >= dyn_th, 0, 1), dyn_th


def objective(mean_f1, per_type, floor, penalty, min_bonus):
    per = np.asarray(per_type, dtype=np.float32)
    gap = np.maximum(0.0, floor - per).mean()
    return float(mean_f1 - penalty * gap + min_bonus * per.min())


def per_type_oracle_thresholds(score, y, types):
    theta = []
    oracle_per = []
    for t in TYPE_ORDER:
        idx = (types == t)
        best_f, best_th = -1.0, 0.5
        for th in np.linspace(0.01, 0.99, 197):
            pred = np.where(score[idx] >= th, 0, 1)
            f = macro_f1(y[idx], pred)
            if f > best_f:
                best_f, best_th = f, float(th)
        theta.append(best_th)
        oracle_per.append(best_f)
    return np.asarray(theta, dtype=np.float32), oracle_per


def global_threshold_baseline(score, y, types):
    best = None
    for th in np.linspace(0.01, 0.99, 197):
        pred = np.where(score >= th, 0, 1)
        mean_f, per, min_f = track2_metrics(y, pred, types)
        item = (mean_f, min_f, th, per)
        if best is None or item[0] > best[0]:
            best = item
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--min_bonus", type=float, default=0.10)
    ap.add_argument("--random_trials", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init_radius", type=float, default=0.08)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    score_rows = read_score(args.score_csv)
    label_rows = read_label(args.label_csv)

    y, score, q, types, names = [], [], [], [], []
    missing = 0
    for r in label_rows:
        name = r["name"]
        if name not in score_rows:
            missing += 1
            continue
        sr = score_rows[name]
        names.append(name)
        y.append(int(r["label"]))
        score.append(float(sr["score"]))
        types.append(r["type"])
        qi = np.asarray([float(sr.get(c, 0.25)) for c in TYPE_COLS], dtype=np.float32)
        qi = qi / max(float(qi.sum()), 1e-8)
        q.append(qi)

    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)
    q = np.stack(q, axis=0).astype(np.float32)
    types = np.asarray(types)

    print("Matched:", len(y), "Missing:", missing)

    global_best = global_threshold_baseline(score, y, types)
    print("\n[Global threshold evaluated by Track2 metric]")
    print("mean_f1=%.6f min_f1=%.6f th=%.4f per=%s" % (
        global_best[0], global_best[1], global_best[2], [round(x, 6) for x in global_best[3]]
    ))

    oracle_theta, oracle_per = per_type_oracle_thresholds(score, y, types)
    print("\n[Oracle per-type thresholds, dev only]")
    for t, th, f in zip(TYPE_ORDER, oracle_theta, oracle_per):
        print("%-8s th=%.4f f1=%.6f" % (t, th, f))
    print("oracle_mean=%.6f oracle_min=%.6f" % (np.mean(oracle_per), np.min(oracle_per)))

    candidates = []

    # deterministic gamma sweep from oracle thresholds
    for gamma in [0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0, 2.5, 3.0]:
        for offset in np.linspace(-0.08, 0.08, 65):
            pred, dyn = predict(score, q, oracle_theta, gamma, float(offset))
            mean_f, per, min_f = track2_metrics(y, pred, types)
            obj = objective(mean_f, per, args.floor, args.penalty, args.min_bonus)
            candidates.append((obj, mean_f, min_f, per, oracle_theta.copy(), float(gamma), float(offset)))

    # random search around oracle thresholds
    for _ in range(args.random_trials):
        theta = oracle_theta + np.random.uniform(-args.init_radius, args.init_radius, size=4).astype(np.float32)
        theta = np.clip(theta, 0.01, 0.99)

        # Singing threshold is often very low in your diagnostic; sample a wider local range.
        if random.random() < 0.35:
            theta[2] = np.clip(np.random.normal(oracle_theta[2], 0.08), 0.01, 0.40)

        gamma = float(10 ** np.random.uniform(math.log10(0.25), math.log10(3.0)))
        offset = float(np.random.uniform(-0.10, 0.10))

        pred, dyn = predict(score, q, theta, gamma, offset)
        mean_f, per, min_f = track2_metrics(y, pred, types)
        obj = objective(mean_f, per, args.floor, args.penalty, args.min_bonus)
        candidates.append((obj, mean_f, min_f, per, theta.copy(), gamma, offset))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]

    obj, mean_f, min_f, per, theta, gamma, offset = best
    pred, dyn_th = predict(score, q, theta, gamma, offset)

    out = {
        "method": "soft_type_dynamic_threshold",
        "type_order": TYPE_ORDER,
        "formula": "q=normalize(q**gamma); dyn_threshold=sum_t q_t*theta_t+offset; pred=real if score>=dyn_threshold else fake",
        "theta": [float(x) for x in theta],
        "gamma": float(gamma),
        "offset": float(offset),
        "floor": float(args.floor),
        "penalty": float(args.penalty),
        "min_bonus": float(args.min_bonus),
        "dev_objective": float(obj),
        "dev_mean_f1": float(mean_f),
        "dev_min_f1": float(min_f),
        "dev_per_type_f1": [float(x) for x in per],
        "global_track2_mean_f1": float(global_best[0]),
        "global_track2_min_f1": float(global_best[1]),
        "global_threshold": float(global_best[2]),
        "global_per_type_f1": [float(x) for x in global_best[3]],
        "oracle_theta": [float(x) for x in oracle_theta],
        "oracle_per_type_f1": [float(x) for x in oracle_per],
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n[Best soft type dynamic threshold]")
    print("objective=%.6f mean=%.6f min=%.6f" % (obj, mean_f, min_f))
    print("per_type [speech, sound, singing, music]:", [round(x, 6) for x in per])
    print("theta:", [round(float(x), 6) for x in theta])
    print("gamma:", gamma, "offset:", offset)
    print("Saved:", args.out_json)


if __name__ == "__main__":
    main()
