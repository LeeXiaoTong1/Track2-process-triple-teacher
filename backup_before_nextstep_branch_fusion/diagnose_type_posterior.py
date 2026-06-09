#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

TYPE_ORDER = ["speech", "sound", "singing", "music"]
TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]

def read_scores(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["name"].strip()] = r
    return rows

def read_labels(path):
    out = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({
                "name": r["name"].strip(),
                "type": r["type"].strip().lower(),
                "label": r["label"].strip().lower(),
            })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    args = ap.parse_args()

    scores = read_scores(args.score_csv)
    labels = read_labels(args.label_csv)

    y_true, q_list = [], []
    missing = 0
    for r in labels:
        name = r["name"]
        if name not in scores:
            missing += 1
            continue
        if r["type"] not in TYPE_ORDER:
            continue
        sr = scores[name]
        q = np.array([float(sr.get(c, 0.25)) for c in TYPE_COLS], dtype=np.float64)
        q = np.clip(q, 1e-9, None)
        q = q / q.sum()
        q_list.append(q)
        y_true.append(TYPE_ORDER.index(r["type"]))

    y_true = np.asarray(y_true, dtype=np.int64)
    q = np.vstack(q_list)
    y_pred = q.argmax(axis=1)

    print("Matched:", len(y_true), "Missing:", missing)
    print("Raw type posterior argmax accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion matrix rows=true, cols=pred [speech, sound, singing, music]:")
    print(confusion_matrix(y_true, y_pred, labels=list(range(4))))

    print("\nPer-true-type posterior statistics:")
    for i, t in enumerate(TYPE_ORDER):
        idx = (y_true == i)
        qt = q[idx]
        pred_acc = (y_pred[idx] == i).mean() if idx.any() else float('nan')
        print(f"\n[{t}] n={idx.sum()} argmax_acc={pred_acc:.6f}")
        print("mean q:", {TYPE_ORDER[j]: round(float(qt[:, j].mean()), 6) for j in range(4)})
        for j, tj in enumerate(TYPE_ORDER):
            qs = np.quantile(qt[:, j], [0.1, 0.25, 0.5, 0.75, 0.9])
            print(f"  q_{tj:8s} quantile10/25/50/75/90:", [round(float(x), 4) for x in qs])

    print("\nHigh-confidence coverage by true type:")
    for i, t in enumerate(TYPE_ORDER):
        idx = (y_true == i)
        qi = q[idx, i]
        print(f"{t:8s}: q_true>0.5={(qi>0.5).mean():.4f}, q_true>0.7={(qi>0.7).mean():.4f}, q_true>0.9={(qi>0.9).mean():.4f}")

if __name__ == "__main__":
    main()
