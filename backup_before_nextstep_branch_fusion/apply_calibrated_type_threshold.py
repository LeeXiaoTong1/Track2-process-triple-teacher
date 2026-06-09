#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, os
import numpy as np

TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]
EXPERT_COLS = ["expert_xlsr", "expert_mert", "expert_beats", "expert_artifact"]

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
    q = np.clip(q, 1e-8, None); q = q / q.sum()
    q_log = np.log(q)
    q_logit = np.log(q / np.clip(1.0 - q, 1e-8, None))
    ew = np.array([get_float(row, c, 0.0) for c in EXPERT_COLS], dtype=np.float64)
    if ew.sum() > 0:
        ew = np.clip(ew, 1e-8, None); ew = ew / ew.sum()
    feat = []
    feat += [score, lr, lf, margin, abs(margin), crop_var]
    feat += q.tolist(); feat += q_log.tolist(); feat += q_logit.tolist(); feat += ew.tolist()
    feat += (q * score).tolist(); feat += (q * margin).tolist()
    feat += (ew * score).tolist(); feat += (ew * margin).tolist()
    return np.asarray(feat, dtype=np.float64)

def softmax_np(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--calib_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--debug_csv", default="")
    args = ap.parse_args()

    with open(args.calib_json, "r", encoding="utf-8") as f:
        calib = json.load(f)
    mu = np.asarray(calib["scaler_mean"], dtype=np.float64)
    sd = np.asarray(calib["scaler_std"], dtype=np.float64)
    coef = np.asarray(calib["coef"], dtype=np.float64)
    intercept = np.asarray(calib["intercept"], dtype=np.float64)
    theta = np.asarray(calib["theta"], dtype=np.float64)
    gamma = float(calib["gamma"])
    offset = float(calib["offset"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    dbg = None
    if args.debug_csv:
        os.makedirs(os.path.dirname(args.debug_csv), exist_ok=True)
        dbg = open(args.debug_csv, "w", encoding="utf-8", newline="")
        dbg_writer = csv.writer(dbg)
        dbg_writer.writerow(["name", "score", "threshold", "predict", "q_speech", "q_sound", "q_singing", "q_music"])
    else:
        dbg_writer = None

    with open(args.score_csv, "r", encoding="utf-8-sig") as fin, open(args.out_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["name", "predict"])
        n = 0
        for r in reader:
            name = r["name"].strip()
            score = get_float(r, "score", 0.5)
            x = make_feature(r)
            xs = (x - mu) / sd
            logits = xs @ coef.T + intercept
            q = softmax_np(logits[None, :])[0]
            qg = np.clip(q, 1e-9, 1.0) ** gamma
            qg = qg / qg.sum()
            th = float(np.clip(qg @ theta + offset, 0.01, 0.99))
            pred = "real" if score >= th else "fake"
            writer.writerow([name, pred])
            if dbg_writer is not None:
                dbg_writer.writerow([name, score, th, pred, q[0], q[1], q[2], q[3]])
            n += 1
    if dbg is not None:
        dbg.close()
    print("Saved:", args.out_csv, "n=", n)
    if args.debug_csv:
        print("Saved debug:", args.debug_csv)

if __name__ == "__main__":
    main()
