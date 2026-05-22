import argparse
import csv
import numpy as np
from sklearn.metrics import f1_score

TYPE_ORDER = ["speech", "sound", "singing", "music"]


def load_scores(path):
    d = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["name"].strip()] = float(r["score"])
    return d


def load_labels(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["name"].strip(),
                "label": 0 if r["label"].strip().lower() == "real" else 1,
                "type": r["type"].strip().lower()
            })
    return rows


def macro_by_type(y_true, y_pred, types):
    vals = []

    for t in TYPE_ORDER:
        idx = [i for i, x in enumerate(types) if x == t]

        if len(idx) == 0:
            continue

        vals.append(
            f1_score(
                [y_true[i] for i in idx],
                [y_pred[i] for i in idx],
                average="macro",
                labels=[0, 1],
                zero_division=0
            )
        )

    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label_csv", required=True)
    args = ap.parse_args()

    A = load_scores(args.a)
    B = load_scores(args.b)
    rows = load_labels(args.label_csv)

    y_true, types, sa, sb = [], [], [], []

    for r in rows:
        n = r["name"]

        if n in A and n in B:
            y_true.append(r["label"])
            types.append(r["type"])
            sa.append(A[n])
            sb.append(B[n])

    sa = np.array(sa)
    sb = np.array(sb)

    best = (-1.0, None, None, None)

    for wa in np.linspace(0.0, 1.0, 21):
        s = wa * sa + (1.0 - wa) * sb

        for th in np.linspace(0.05, 0.95, 181):
            pred = np.where(s >= th, 0, 1)
            macro, per = macro_by_type(y_true, pred.tolist(), types)

            if macro > best[0]:
                best = (macro, wa, th, per)

    print("Best macro:", best[0])
    print("Best wa for A:", best[1])
    print("Best threshold:", best[2])
    print("Per-type F1 [speech, sound, singing, music]:", best[3])


if __name__ == "__main__":
    main()
