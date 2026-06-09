#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply soft type dynamic threshold to Progress/Eval score CSV.

Input score_csv must contain:
  name, score, type_speech, type_sound, type_singing, type_music

Output:
  name,predict
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TYPE_COLS = ["type_speech", "type_sound", "type_singing", "type_music"]


def normalize_type_prob(q, gamma):
    q = np.clip(q, 1e-8, 1.0)
    q = np.power(q, gamma)
    q = q / max(float(q.sum()), 1e-8)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--calib_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.calib_json, "r", encoding="utf-8") as f:
        calib = json.load(f)

    theta = np.asarray(calib["theta"], dtype=np.float32)
    gamma = float(calib["gamma"])
    offset = float(calib["offset"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.score_csv, "r", encoding="utf-8-sig") as fin, \
         open(args.out_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["name", "predict"])

        for r in reader:
            name = r["name"].strip()
            score = float(r["score"])
            q = np.asarray([float(r.get(c, 0.25)) for c in TYPE_COLS], dtype=np.float32)
            q = normalize_type_prob(q, gamma)
            dyn_th = float(q @ theta + offset)
            pred = "real" if score >= dyn_th else "fake"
            writer.writerow([name, pred])
            n += 1

    print("Saved:", args.out_csv)
    print("Rows:", n)
    print("theta:", [float(x) for x in theta])
    print("gamma:", gamma, "offset:", offset)


if __name__ == "__main__":
    main()
