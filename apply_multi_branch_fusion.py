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


def parse_branch(s):
    if ":" not in s:
        raise ValueError("--branch must be name:path")
    name, path = s.split(":", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True, help="name:path_to_score_csv. Branch names must match calib_json branch_order.")
    ap.add_argument("--type_csv", required=True)
    ap.add_argument("--calib_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--debug_csv", default="")
    args = ap.parse_args()

    cfg = json.load(open(args.calib_json, "r", encoding="utf-8"))
    branch_order = cfg["branch_order"]
    W = np.asarray(cfg["weights"], dtype=np.float64)
    theta = np.asarray(cfg["theta"], dtype=np.float64)
    temp = float(cfg["temp"])
    offset = float(cfg.get("offset", 0.0))

    provided = {}
    for spec in args.branch:
        name, path = parse_branch(spec)
        provided[name] = read_by_name(path)

    missing_names = [b for b in branch_order if b not in provided]
    if missing_names:
        raise RuntimeError(f"Missing branch score csv for: {missing_names}")

    typ = read_by_name(args.type_csv)
    common = set(typ.keys())
    for b in branch_order:
        common &= set(provided[b].keys())
    names = sorted(common)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out_csv, "w", encoding="utf-8", newline="")
    writer = csv.writer(fout)
    writer.writerow(["name", "predict"])

    dbg = None
    if args.debug_csv:
        dbg = open(args.debug_csv, "w", encoding="utf-8", newline="")
        dw = csv.writer(dbg)
        dw.writerow(["name", "score", "threshold", "pred"] + [f"q_{t}" for t in TYPE] + [f"alpha_{b}" for b in branch_order])

    for nm in names:
        scores = np.asarray([float(provided[b][nm]["score"]) for b in branch_order], dtype=np.float64)
        q = np.asarray([float(typ[nm]["type_" + t]) for t in TYPE], dtype=np.float64)
        q = np.power(q, temp)
        q = q / max(float(q.sum()), 1e-8)
        type_scores = W @ scores
        final = float((q * type_scores).sum())
        th = float((q * theta).sum() + offset)
        pred = "real" if final >= th else "fake"
        writer.writerow([nm, pred])
        if dbg:
            alpha = q @ W
            dw.writerow([nm, final, th, pred] + [float(x) for x in q] + [float(x) for x in alpha])

    fout.close()
    if dbg:
        dbg.close()
    print("Saved:", args.out_csv)
    print("Rows:", len(names))


if __name__ == "__main__":
    main()
