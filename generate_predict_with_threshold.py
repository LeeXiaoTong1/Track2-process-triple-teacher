import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("--input_csv", required=True)
ap.add_argument("--threshold", type=float, required=True)
args = ap.parse_args()

base = os.path.dirname(args.input_csv)
out = os.path.join(base, "predict.csv")

with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as fin, \
     open(out, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["name", "predict"])

    for row in reader:
        name = row["name"].strip()
        score = float(row["score"])

        # score = probability of real
        pred = "real" if score >= args.threshold else "fake"
        writer.writerow([name, pred])

print("Saved:", out)
print("threshold:", args.threshold)
