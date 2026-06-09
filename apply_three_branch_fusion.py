#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json
from pathlib import Path
import numpy as np

TYPE = ["speech", "sound", "singing", "music"]


def read_by_name(path):
    d = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["name"].strip()] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--ufm_csv", required=True)
    ap.add_argument("--music_csv", required=True)
    ap.add_argument("--type_csv", required=True)
    ap.add_argument("--calib_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--debug_csv", default="")
    args = ap.parse_args()

    base = read_by_name(args.baseline_csv)
    ufm = read_by_name(args.ufm_csv)
    mus = read_by_name(args.music_csv)
    typ = read_by_name(args.type_csv)

    cfg = json.load(open(args.calib_json, "r", encoding="utf-8"))
    W = np.asarray(cfg["weights"], dtype=np.float64)   # [4,3]
    theta = np.asarray(cfg["theta"], dtype=np.float64)
    temp = float(cfg["temp"])
    offset = float(cfg.get("offset", 0.0))

    names = sorted(set(base) & set(ufm) & set(mus) & set(typ))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    fout = open(args.out_csv, "w", encoding="utf-8", newline="")
    writer = csv.writer(fout)
    writer.writerow(["name", "predict"])

    dbg = None
    if args.debug_csv:
        dbg = open(args.debug_csv, "w", encoding="utf-8", newline="")
        dw = csv.writer(dbg)
        dw.writerow(["name","score","threshold","alpha_base","alpha_ufm","alpha_music","q_speech","q_sound","q_singing","q_music","pred"])

    for name in names:
        sb = float(base[name]["score"])
        su = float(ufm[name]["score"])
        sm = float(mus[name]["score"])
        q = np.asarray([float(typ[name]["type_" + t]) for t in TYPE], dtype=np.float64)
        q = np.power(q, temp)
        q = q / max(float(q.sum()), 1e-8)

        branch_scores = np.asarray([sb, su, sm], dtype=np.float64)
        type_scores = W @ branch_scores
        score = float((q * type_scores).sum())
        th = float((q * theta).sum() + offset)
        pred = "real" if score >= th else "fake"
        writer.writerow([name, pred])

        if dbg:
            alpha = q @ W
            dw.writerow([name, score, th, float(alpha[0]), float(alpha[1]), float(alpha[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3]), pred])

    fout.close()
    if dbg:
        dbg.close()
    print("Saved:", args.out_csv)
    print("Rows:", len(names))


if __name__ == "__main__":
    main()
