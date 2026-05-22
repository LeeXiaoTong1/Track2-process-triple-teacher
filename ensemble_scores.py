import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--wa", type=float, default=0.5)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(path):
    d = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["name"].strip()] = float(r["score"])
    return d


A = load(args.a)
B = load(args.b)

names = sorted(set(A) & set(B))

os.makedirs(os.path.dirname(args.out), exist_ok=True)

with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "score"])

    for n in names:
        s = args.wa * A[n] + (1.0 - args.wa) * B[n]
        w.writerow([n, float(s)])

print("Saved:", args.out)
print("N:", len(names))
print("wa:", args.wa)
