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

# 2026 World Cup round windows [start, end) — boundaries fall in the gaps between rounds.
# The knockout rounds are always shown (as placeholders if no fixtures yet); "Group stage"
# only appears if it has matches.
ROUND_BOUNDS = [
    ("Group stage",    "1900-01-01", "2026-06-28"),
    ("Round of 32",    "2026-06-28", "2026-07-04"),
    ("Round of 16",    "2026-07-04", "2026-07-09"),
    ("Quarter-finals", "2026-07-09", "2026-07-13"),
    ("Semi-finals",    "2026-07-13", "2026-07-17"),
    ("Final",          "2026-07-17", "2100-01-01"),
]
ALWAYS_SHOW = {"Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"}


def round_of(date_str):
    for name, lo, hi in ROUND_BOUNDS:
        if lo <= date_str < hi:
            return name
    return "Other"


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

    # --- actual results (date-tolerant join) ---
    # Knockout fixtures sometimes shift a day between when we predicted them and when they
    # were played, so an exact date join wrongly marks played matches as "upcoming". Match
    # on the team pair and accept the nearest result within a few days.
    res = (P.load_data()[["date", "home_team", "away_team", "home_score", "away_score", "outcome"]]
           .dropna(subset=["outcome"]).rename(columns={"date": "rdate"}))
    j = preds.merge(res, on=["home_team", "away_team"], how="left")
    j["ddiff"] = (j["rdate"] - j["date"]).abs().dt.days
    TOL = 5
    j.loc[j["ddiff"] > TOL, ["home_score", "away_score", "outcome", "rdate", "ddiff"]] = np.nan
    m = (j.sort_values("ddiff")                       # NaN (no/too-far result) sorts last
          .drop_duplicates(["date", "home_team", "away_team"], keep="first")
          .sort_values("date"))

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
            "round": round_of(r["date"].strftime("%Y-%m-%d")),
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

    # --- group matches into tournament rounds (with placeholders for future rounds) ---
    by_round = {}
    for x in matches:
        by_round.setdefault(x["round"], []).append(x)
    rounds = []
    for name, _, _ in ROUND_BOUNDS:
        ms = sorted(by_round.get(name, []), key=lambda z: (z["date"], z["home"]))
        if not ms and name not in ALWAYS_SHOW:
            continue
        status = "pending" if not ms else ("done" if all(z["settled"] for z in ms) else "live")
        rounds.append({"name": name, "status": status, "matches": ms})

    latest = m["date"].max()
    out = {
        "updated": P.TODAY.strftime("%Y-%m-%d") if hasattr(P, "TODAY") else str(latest.date()),
        "record": record,
        "rounds": rounds,
    }
    with open("docs/data.json", "w") as f:
        json.dump(out, f, indent=2)
    settled_n = sum(1 for x in matches if x["settled"])
    rounds_desc = ", ".join(f"{r['name']}({len(r['matches'])},{r['status']})" for r in rounds)
    rec = f"{record['accuracy']:.0%} acc, {record['logloss']:.3f} log-loss on {record['n']}" if record else "no results yet"
    print(f"docs/data.json written: {len(matches)} matches, {settled_n} settled | record: {rec}")
    print(f"  rounds: {rounds_desc}")


if __name__ == "__main__":
    main()
