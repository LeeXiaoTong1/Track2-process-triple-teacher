#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a lightweight score-level type calibrator, then tune predicted-type dynamic thresholds.

This is intended for AT-ADD Track2 score CSVs produced by generate_score_multicrop_plus.py.
It uses only official dev labels/types for calibration and threshold selection.
"""
import argparse, csv, json, math, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

TYPE_ORDER = ["speech", "sound", "singing", "music"]
TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]
EXPERT_COLS = ["expert_xlsr", "expert_mert", "expert_beats", "expert_artifact"]

def read_csv_rows(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["name"].strip()] = r
    return rows

def read_label_rows(path):
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

def get_float(r, k, default=0.0):
    try:
        return float(r.get(k, default))
    except Exception:
        return default

def make_feature(row):
    score = get_float(row, "score", 0.5)
    lr = get_float(row, "logit_real", 0.0)
    lf = get_float(row, "logit_fake", 0.0)
    margin = get_float(row, "margin", lr - lf)
    crop_var = get_float(row, "crop_var", 0.0)

    q = np.array([get_float(row, c, 0.25) for c in TYPE_COLS], dtype=np.float64)
    q = np.clip(q, 1e-8, None)
    q = q / q.sum()
    q_log = np.log(q)
    q_logit = np.log(q / np.clip(1.0 - q, 1e-8, None))

    ew = np.array([get_float(row, c, 0.0) for c in EXPERT_COLS], dtype=np.float64)
    if ew.sum() > 0:
        ew = np.clip(ew, 1e-8, None)
        ew = ew / ew.sum()

    # Interactions help separate singing/music where raw q may be under-confident.
    feat = []
    feat += [score, lr, lf, margin, abs(margin), crop_var]
    feat += q.tolist()
    feat += q_log.tolist()
    feat += q_logit.tolist()
    feat += ew.tolist()
    feat += (q * score).tolist()
    feat += (q * margin).tolist()
    feat += (ew * score).tolist()
    feat += (ew * margin).tolist()
    return np.asarray(feat, dtype=np.float64)

def feature_names():
    names = ["score", "logit_real", "logit_fake", "margin", "abs_margin", "crop_var"]
    names += TYPE_COLS
    names += ["log_" + c for c in TYPE_COLS]
    names += ["logit_" + c for c in TYPE_COLS]
    names += EXPERT_COLS
    names += [c + "*score" for c in TYPE_COLS]
    names += [c + "*margin" for c in TYPE_COLS]
    names += [c + "*score" for c in EXPERT_COLS]
    names += [c + "*margin" for c in EXPERT_COLS]
    return names

def load_data(score_csv, label_csv):
    scores = read_csv_rows(score_csv)
    labels = read_label_rows(label_csv)
    X, y_type, y_fake, score, names = [], [], [], [], []
    missing = 0
    for r in labels:
        name = r["name"]
        if name not in scores:
            missing += 1
            continue
        if r["type"] not in TYPE_ORDER:
            continue
        row = scores[name]
        X.append(make_feature(row))
        y_type.append(TYPE_ORDER.index(r["type"]))
        y_fake.append(int(r["label"]))
        score.append(get_float(row, "score", 0.5))
        names.append(name)
    return (np.vstack(X), np.asarray(y_type, dtype=np.int64), np.asarray(y_fake, dtype=np.int64),
            np.asarray(score, dtype=np.float64), names, missing)

def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def predict_q(X, mu, sd, coef, intercept):
    Xs = (X - mu) / sd
    logits = Xs @ coef.T + intercept[None, :]
    return softmax_np(logits)

def macro_by_type(y_fake, pred_fake, y_type):
    vals = []
    for t in range(4):
        idx = (y_type == t)
        if idx.sum() == 0:
            vals.append(0.0)
        else:
            vals.append(f1_score(y_fake[idx], pred_fake[idx], average="macro", labels=[0,1], zero_division=0))
    return float(np.mean(vals)), vals

def objective_from_per(per, floor=0.95, penalty=2.0, min_bonus=0.10):
    per = np.asarray(per, dtype=np.float64)
    deficits = np.maximum(0.0, floor - per)
    return float(per.mean() - penalty * deficits.mean() + min_bonus * per.min())

def eval_threshold(score, y_fake, y_type, q, theta, gamma, offset):
    qq = np.clip(q, 1e-9, 1.0)
    qq = qq ** gamma
    qq = qq / qq.sum(axis=1, keepdims=True)
    th = qq @ theta + offset
    th = np.clip(th, 0.01, 0.99)
    pred = np.where(score >= th, 0, 1)
    mean, per = macro_by_type(y_fake, pred, y_type)
    return mean, per, float(np.min(per))

def oracle_thresholds(score, y_fake, y_type):
    theta, per = [], []
    for t in range(4):
        idx = (y_type == t)
        best = (-1.0, 0.5)
        for th in np.linspace(0.01, 0.99, 197):
            pred = np.where(score[idx] >= th, 0, 1)
            f = f1_score(y_fake[idx], pred, average="macro", labels=[0,1], zero_division=0)
            if f > best[0]:
                best = (f, float(th))
        per.append(best[0]); theta.append(best[1])
    return np.asarray(theta, dtype=np.float64), per

def tune_dynamic_threshold(score, y_fake, y_type, q, floor, penalty, min_bonus, random_trials, seed):
    rng = np.random.default_rng(seed)
    oracle_theta, oracle_per = oracle_thresholds(score, y_fake, y_type)

    candidates = []
    # useful deterministic candidates
    for off in np.linspace(-0.2, 0.2, 41):
        candidates.append((oracle_theta.copy(), 1.0, float(off)))
    candidates.append((oracle_theta.copy(), 0.5, 0.0))
    candidates.append((oracle_theta.copy(), 2.0, 0.0))

    best = {"obj": -1e9}
    def consider(theta, gamma, offset):
        nonlocal best
        theta = np.clip(np.asarray(theta, dtype=np.float64), 0.01, 0.99)
        gamma = float(np.clip(gamma, 0.05, 8.0))
        offset = float(np.clip(offset, -0.5, 0.5))
        mean, per, mn = eval_threshold(score, y_fake, y_type, q, theta, gamma, offset)
        obj = objective_from_per(per, floor=floor, penalty=penalty, min_bonus=min_bonus)
        if obj > best["obj"]:
            best = {
                "obj": float(obj), "mean": float(mean), "min": float(mn),
                "per": [float(x) for x in per], "theta": [float(x) for x in theta],
                "gamma": gamma, "offset": offset,
            }

    for theta, gamma, off in candidates:
        consider(theta, gamma, off)

    for _ in range(int(random_trials)):
        mode = rng.random()
        if mode < 0.70:
            theta = oracle_theta + rng.normal(0.0, 0.10, size=4)
        else:
            theta = rng.uniform(0.01, 0.99, size=4)
        # singing often needs very low threshold; sample that region more often
        if rng.random() < 0.35:
            theta[2] = rng.uniform(0.01, 0.20)
        gamma = float(np.exp(rng.uniform(np.log(0.15), np.log(5.0))))
        off = float(rng.uniform(-0.25, 0.25))
        consider(theta, gamma, off)

    return best, oracle_theta, oracle_per

def fit_type_lr(X, y_type, C=1.0):
    mu, sd = standardize_fit(X)
    Xs = (X - mu) / sd
    clf = LogisticRegression(
        max_iter=2000,
        C=float(C),
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=1,
    )
    clf.fit(Xs, y_type)
    return mu, sd, clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--min_bonus", type=float, default=0.10)
    ap.add_argument("--random_trials", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--calib_ratio", type=float, default=0.70)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    X, y_type, y_fake, score, names, missing = load_data(args.score_csv, args.label_csv)
    print("Matched:", len(y_fake), "Missing:", missing, "Feature dim:", X.shape[1])

    idx_all = np.arange(len(y_fake))
    idx_cal, idx_val = train_test_split(
        idx_all,
        train_size=float(args.calib_ratio),
        random_state=int(args.seed),
        stratify=y_type,
    )

    # Held-out check: fit type calibrator on split, tune threshold on validation.
    mu, sd, coef, intercept = fit_type_lr(X[idx_cal], y_type[idx_cal], C=args.C)
    q_val = predict_q(X[idx_val], mu, sd, coef, intercept)
    ypred_type = q_val.argmax(axis=1)
    print("\n[Held-out type calibrator]")
    print("val type acc:", accuracy_score(y_type[idx_val], ypred_type))
    print("val confusion rows=true cols=pred:")
    print(confusion_matrix(y_type[idx_val], ypred_type, labels=list(range(4))))

    best_val, oracle_theta_val, oracle_per_val = tune_dynamic_threshold(
        score[idx_val], y_fake[idx_val], y_type[idx_val], q_val,
        floor=args.floor, penalty=args.penalty, min_bonus=args.min_bonus,
        random_trials=args.random_trials, seed=args.seed,
    )
    print("\n[Held-out calibrated dynamic threshold]")
    print("objective={obj:.6f} mean={mean:.6f} min={min:.6f}".format(**best_val))
    print("per_type:", [round(x, 6) for x in best_val["per"]])
    print("theta:", [round(x, 6) for x in best_val["theta"]], "gamma:", best_val["gamma"], "offset:", best_val["offset"])
    print("oracle_val_theta:", [round(float(x), 6) for x in oracle_theta_val], "oracle_val_per:", [round(float(x), 6) for x in oracle_per_val])

    # Final model: fit type calibrator on full dev, tune thresholds on full dev for final use.
    mu_f, sd_f, coef_f, intercept_f = fit_type_lr(X, y_type, C=args.C)
    q_full = predict_q(X, mu_f, sd_f, coef_f, intercept_f)
    print("\n[Full-dev type calibrator]")
    print("type acc:", accuracy_score(y_type, q_full.argmax(axis=1)))
    print("confusion rows=true cols=pred:")
    print(confusion_matrix(y_type, q_full.argmax(axis=1), labels=list(range(4))))

    best_full, oracle_theta_full, oracle_per_full = tune_dynamic_threshold(
        score, y_fake, y_type, q_full,
        floor=args.floor, penalty=args.penalty, min_bonus=args.min_bonus,
        random_trials=args.random_trials, seed=args.seed + 99,
    )
    print("\n[Full-dev calibrated dynamic threshold to save]")
    print("objective={obj:.6f} mean={mean:.6f} min={min:.6f}".format(**best_full))
    print("per_type [speech, sound, singing, music]:", [round(x, 6) for x in best_full["per"]])
    print("theta:", [round(x, 6) for x in best_full["theta"]], "gamma:", best_full["gamma"], "offset:", best_full["offset"])
    print("oracle_full_theta:", [round(float(x), 6) for x in oracle_theta_full], "oracle_full_per:", [round(float(x), 6) for x in oracle_per_full])

    out = {
        "type_order": TYPE_ORDER,
        "feature_names": feature_names(),
        "scaler_mean": mu_f.tolist(),
        "scaler_std": sd_f.tolist(),
        "coef": coef_f.tolist(),
        "intercept": intercept_f.tolist(),
        "theta": best_full["theta"],
        "gamma": best_full["gamma"],
        "offset": best_full["offset"],
        "floor": args.floor,
        "penalty": args.penalty,
        "min_bonus": args.min_bonus,
        "heldout_metrics": best_val,
        "full_dev_metrics": best_full,
        "rule": "q_cal=softmax(standardized_features @ coef.T + intercept); threshold=clip(sum(normalize(q_cal**gamma)*theta)+offset,0.01,0.99); real if score>=threshold",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()
