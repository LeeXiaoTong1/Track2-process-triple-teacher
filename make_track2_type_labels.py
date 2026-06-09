#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_track2_type_labels.py

Create type-filtered train/dev label CSVs for AT-ADD Track2.
Input label CSV is expected to contain at least: name,label,type
Types: speech, sound, singing, music

Example:
python make_track2_type_labels.py \
  --train_label AT_ADD_data/Track2/label/train.csv \
  --dev_label AT_ADD_data/Track2/label/dev.csv \
  --out_dir AT_ADD_data/Track2/label_by_type
"""

import argparse
import csv
from pathlib import Path

TYPES = ["speech", "sound", "singing", "music"]


def filter_one(src, dst_dir, split):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    rows_by_type = {t: [] for t in TYPES}

    with open(src, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"Empty CSV: {src}")
        required = {"name", "label", "type"}
        miss = required - set(fieldnames)
        if miss:
            raise RuntimeError(f"{src} missing columns: {miss}")

        for r in reader:
            t = r["type"].strip().lower()
            if t not in rows_by_type:
                continue
            rows_by_type[t].append(r)

    for t, rows in rows_by_type.items():
        out = dst_dir / f"{split}_{t}.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split:5s} {t:8s}: {len(rows):6d} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_label", required=True)
    ap.add_argument("--dev_label", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    filter_one(args.train_label, args.out_dir, "train")
    filter_one(args.dev_label, args.out_dir, "dev")


if __name__ == "__main__":
    main()
