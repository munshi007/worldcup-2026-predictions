"""Score saved predictions against actual results, and tune the odds blend weight.

Usage:
  python score.py predictions_YYYYMMDD.csv      # log-loss + accuracy of that file vs results
  python score.py --tune predictions_*.csv      # find the market/model blend weight that
                                                 # would have minimised log-loss on settled games

This closes the loop the live-only odds feed leaves open: once matches are played we measure
real performance and tune `ODDS_WEIGHT` from evidence instead of guessing. A prediction is only
scored if the match has since been played in results.csv.
"""
import argparse
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import predict as P

CLASSES = ["away_win", "draw", "home_win"]


def settled(pred_csv):
    """Join a predictions CSV to actual outcomes; return only matches that have been played."""
    pred = pd.read_csv(pred_csv)
    pred["date"] = pd.to_datetime(pred["date"])
    res = P.load_data()[["date", "home_team", "away_team", "outcome"]].dropna(subset=["outcome"])
    m = pred.merge(res, on=["date", "home_team", "away_team"], how="inner")
    return m


def metrics(df):
    """Log-loss and accuracy of blended probabilities already in the file."""
    y = df["outcome"].values
    p = df[["p_away_win", "p_draw", "p_home_win"]].values
    p = p / p.sum(1, keepdims=True)
    pred = np.array(CLASSES)[p.argmax(1)]
    return log_loss(y, p, labels=CLASSES), accuracy_score(y, pred), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="prediction CSV(s) to score (globs ok)")
    ap.add_argument("--tune", action="store_true",
                    help="Also report the model-only column and best blend weight if odds cols exist")
    args = ap.parse_args()

    paths = sorted({p for pat in args.files for p in glob.glob(pat)})
    allrows = []
    for path in paths:
        df = settled(path)
        if not len(df):
            print(f"{path}: no settled matches yet")
            continue
        ll, acc, n = metrics(df)
        print(f"{path}: {n} settled  log-loss {ll:.4f}  accuracy {acc:.0%}")
        allrows.append(df)

    if not len(allrows):
        return
    # Same fixture can appear in several files (a later run re-predicts a fixture an
    # earlier run had only as "scheduled"). Keep the LAST = most recent file's row, which
    # carries the richest columns (model/market components added in later runs).
    comb = pd.concat(allrows, ignore_index=True).drop_duplicates(
        ["date", "home_team", "away_team"], keep="last")
    ll, acc, n = metrics(comb)
    print(f"\nALL settled ({n} matches): log-loss {ll:.4f}, accuracy {acc:.0%}")

    # --- Tune the blend weight from real results (needs model + market component columns) ---
    if args.tune and {"p_model_home", "p_market_home"}.issubset(comb.columns):
        c = comb.dropna(subset=["p_market_home"])      # only fixtures that had odds
        if not len(c):
            print("\n--tune: no settled fixtures had market odds saved.")
            return
        y = c["outcome"].values
        model = c[["p_model_away", "p_model_draw", "p_model_home"]].values
        market = c[["p_market_away", "p_market_draw", "p_market_home"]].values
        print(f"\n--tune on {len(c)} settled fixtures with odds:")
        best = (None, 1e9)
        for w in np.linspace(0, 1, 21):
            p = w * market + (1 - w) * model
            p = p / p.sum(1, keepdims=True)
            l = log_loss(y, p, labels=CLASSES)
            tag = "  <- model only" if w == 0 else ("  <- market only" if w == 1 else "")
            if w in (0, 0.5, 0.6, 0.75, 1.0):
                print(f"  weight {w:.2f} market: log-loss {l:.4f}{tag}")
            if l < best[1]:
                best = (w, l)
        print(f"  BEST weight = {best[0]:.2f} market (log-loss {best[1]:.4f}) "
              f"-> set ODDS_WEIGHT accordingly in predict.py")


if __name__ == "__main__":
    main()
