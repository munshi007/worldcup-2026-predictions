"""Build docs/data.json for the public GitHub Pages dashboard.

Reads every predictions_*.csv, joins to actual results, and writes a single JSON the
static site renders. Re-run after each round (the daily cycle can call it), then commit
docs/data.json so the live site updates — no code changes needed by viewers.

Usage:  python export_web.py
"""
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

import predict as P

CLASSES = ["away_win", "draw", "home_win"]
PICK_LABEL = {"home_win": "home", "draw": "draw", "away_win": "away"}


def main():
    # --- gather all predictions, keep the most recent file's row per fixture ---
    frames = []
    for path in sorted(glob.glob("predictions_*.csv")):
        d = pd.read_csv(path)
        d["_src"] = path
        frames.append(d)
    preds = pd.concat(frames, ignore_index=True)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.drop_duplicates(["date", "home_team", "away_team"], keep="last")

    # --- actual results ---
    res = P.load_data()[["date", "home_team", "away_team", "home_score", "away_score", "outcome"]]
    m = preds.merge(res, on=["date", "home_team", "away_team"], how="left")

    matches = []
    for _, r in m.sort_values("date").iterrows():
        p = [float(r["p_home_win"]), float(r["p_draw"]), float(r["p_away_win"])]
        s = sum(p) or 1.0
        p = [x / s for x in p]
        pick = ["home", "draw", "away"][int(np.argmax(p))]
        settled = isinstance(r["outcome"], str)
        row = {
            "date": r["date"].strftime("%Y-%m-%d"),
            "home": r["home_team"],
            "away": r["away_team"],
            "p_home": round(p[0], 4),
            "p_draw": round(p[1], 4),
            "p_away": round(p[2], 4),
            "pick": pick,
            "settled": settled,
        }
        if settled:
            row["actual"] = PICK_LABEL[r["outcome"]]
            row["correct"] = (PICK_LABEL[r["outcome"]] == pick)
            if pd.notna(r["home_score"]):
                row["score"] = f"{int(r['home_score'])}-{int(r['away_score'])}"
        matches.append(row)

    # --- track record on settled matches ---
    done = m.dropna(subset=["outcome"])
    record = None
    if len(done):
        y = done["outcome"].values
        prob = done[["p_away_win", "p_draw", "p_home_win"]].values
        prob = prob / prob.sum(1, keepdims=True)
        pred = np.array(CLASSES)[prob.argmax(1)]
        record = {
            "n": int(len(done)),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "logloss": round(float(log_loss(y, prob, labels=CLASSES)), 4),
            "baseline_logloss": 0.86,
        }

    latest = m["date"].max()
    out = {
        "updated": P.TODAY.strftime("%Y-%m-%d") if hasattr(P, "TODAY") else str(latest.date()),
        "record": record,
        "upcoming": [x for x in matches if not x["settled"]],
        "results": [x for x in matches if x["settled"]][::-1],  # most recent first
    }
    with open("docs/data.json", "w") as f:
        json.dump(out, f, indent=2)
    up, rz = len(out["upcoming"]), len(out["results"])
    rec = f"{record['accuracy']:.0%} acc, {record['logloss']:.3f} log-loss on {record['n']}" if record else "no results yet"
    print(f"docs/data.json written: {up} upcoming, {rz} settled | record: {rec}")


if __name__ == "__main__":
    main()
