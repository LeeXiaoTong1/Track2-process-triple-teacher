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
        reader = csv.DictReader(f)
        for r in reader:
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

        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]

        vals.append(
            f1_score(
                yt,
                yp,
                average="macro",
                labels=[0, 1],
                zero_division=0
            )
        )

    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    args = ap.parse_args()

    scores = load_scores(args.score_csv)
    rows = load_labels(args.label_csv)

    y_true, types, s = [], [], []

    for r in rows:
        name = r["name"]
        if name in scores:
            y_true.append(r["label"])
            types.append(r["type"])
            s.append(scores[name])

    y_true = np.array(y_true)
    s = np.array(s)

    best = (-1.0, None, None)

    for th in np.linspace(0.05, 0.95, 181):
        # score is probability of real.
        y_pred = np.where(s >= th, 0, 1)
        macro, per_type = macro_by_type(y_true.tolist(), y_pred.tolist(), types)

        if macro > best[0]:
            best = (macro, th, per_type)

    print("Best global macro:", best[0])
    print("Best global threshold:", best[1])
    print("Per-type F1 [speech, sound, singing, music]:", best[2])


if __name__ == "__main__":
    main()
