"""Predict international football fixtures with TabPFN on engineered features."""
import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss
from tabpfn_client import TabPFNClassifier, set_access_token

load_dotenv()
_token = os.environ.get("TABPFN_API_KEY")
if _token:
    set_access_token(_token)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")   # the-odds-api.com key (optional)
ODDS_WEIGHT = 0.60                               # blend weight toward de-vigged market odds

TODAY = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN = 10000
CAL_YEARS = 2          # window of recent neutral matches used to fit calibration temperature
HOME_ADV = 65.0
DATA = "results.csv"
RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played",
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "neutral", "importance",
    "val_home", "val_away", "val_diff",
]

VALUES_FILE = "team_values.csv"


def load_values():
    """Load team -> log(squad market value) from VALUES_FILE. Returns {} if the file is absent
    (the value features then become all-NaN, which TabPFN handles, falling back to baseline)."""
    if not os.path.exists(VALUES_FILE):
        return {}
    v = pd.read_csv(VALUES_FILE)
    return {t: np.log(m) for t, m in zip(v["team"], v["value_m"])}


VALUES = load_values()


def importance(t):
    """Map tournament name to an ELO K-factor weight; higher means bigger rating swings."""
    t = t.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(k in t for k in [
        "uefa euro", "copa am", "african cup", "asian cup",
        "gold cup", "nations league", "oceania nations"
        ]):
        return 45.0
    if "qualif" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


# Teams whose World Cup 2026 home matches are NOT neutral (hosts playing in their own country).
HOSTS = {"United States", "Canada", "Mexico"}
# Common fixture-source spellings -> dataset spellings.
TEAM_ALIASES = {"USA": "United States", "US": "United States", "Korea Republic": "South Korea"}


def load_fixtures(path):
    """Read an upcoming-fixtures CSV and shape it like the results feed (blank scores).

    Required columns: date, home_team, away_team. Optional: tournament (default 'FIFA World Cup'),
    neutral (default: 0 if the home team is a host playing at home, else 1). Team names are mapped
    through TEAM_ALIASES so they match the dataset's spellings."""
    fx = pd.read_csv(path)
    fx["date"] = pd.to_datetime(fx["date"])
    for col in ("home_team", "away_team"):
        fx[col] = fx[col].str.strip().replace(TEAM_ALIASES)
    fx["tournament"] = fx.get("tournament", "FIFA World Cup")
    if "neutral" not in fx.columns:
        # A World Cup match is neutral unless a host nation is the designated home side.
        fx["neutral"] = np.where(fx["home_team"].isin(HOSTS), "False", "True")
    # Match the raw feed's neutral encoding ("True"/"False" strings) so load_data parses it.
    fx["neutral"] = fx["neutral"].map(
        {1: "True", 0: "False", True: "True", False: "False"}).fillna(fx["neutral"]).astype(str)
    fx["home_score"] = np.nan
    fx["away_score"] = np.nan
    return fx


def load_data(refresh=False, fixtures=None):
    """Load and lightly clean the results CSV, downloading it if missing or refresh=True.

    If `fixtures` (path to a CSV with date,home_team,away_team[,tournament,neutral]) is given,
    those upcoming, result-less matches are appended so they get predicted alongside any the
    source feed already lists."""
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    if fixtures:
        df = pd.concat([df, load_fixtures(fixtures)], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    # Drop duplicate fixtures (e.g. a --fixtures row the source feed has since added itself),
    # keeping the played copy (has a score) over the scheduled one.
    df["_has_score"] = df["home_score"].notna() if "home_score" in df else False
    df = (df.sort_values("_has_score").drop_duplicates(["date", "home_team", "away_team"], keep="last")
            .drop(columns="_has_score").sort_values("date").reset_index(drop=True))
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def build_features(df):
    """One chronological pass: every feature uses only matches before kickoff."""
    elo = defaultdict(lambda: 1500.0)
    res = defaultdict(list)
    last_date, h2h = {}, defaultdict(list)

    def team_feats(team):
        """Return pre-match stats for a team: ELO, form averages, win rate, goal stats, streak, games played.
        Defaults represent a mid-table team with no history."""
        r = res[team]
        if not r:
            return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
        last5, last10 = r[-5:], r[-10:]
        # each entry is (points, gf, ga, won); walk back until a non-win to count winning streak
        streak = 0
        for p, *_ in reversed(r):
            if p < 1:
                break
            streak += 1
        return (elo[team],
                np.mean([p for p, *_ in last5]), np.mean([p for p, *_ in last10]),
                np.mean([w for *_, w in last10]),
                np.mean([g for _, g, _, _ in last5]), np.mean([a for _, _, a, _ in last5]),
                np.mean([g - a for _, g, a, _ in last10]), streak, len(r))

    def h2h_feats(home, away):
        """Head-to-head record between the two teams, keyed by sorted pair so order doesn't matter.
        GD is flipped for matches where home was the away side."""
        m = h2h[tuple(sorted((home, away)))]
        if not m:
            return 0, 0.5, 0.25, 0.0
        n = len(m)
        return (n,
                sum(w == home for _, _, w in m) / n,
                sum(w == "draw" for _, _, w in m) / n,
                np.mean([g if h == home else -g for h, g, _ in m]))

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = team_feats(h)
        ae, af5, af10, awr, agf, aga, agd, astk, an = team_feats(a)
        nm, h2h_wr, h2h_dr, h2h_gd = h2h_feats(h, a)
        # Squad market value (log €m) — current team quality, which results-based ELO lags.
        vh, va = VALUES.get(h, np.nan), VALUES.get(a, np.nan)
        rows.append({
            "val_home": vh, "val_away": va, "val_diff": vh - va,
            "elo_diff": he + adj - ae, "home_elo": he, "away_elo": ae,
            "form5_diff": hf5 - af5, "form10_diff": hf10 - af10,
            "home_form5": hf5, "away_form5": af5,
            "home_winrate": hwr, "away_winrate": awr,
            "home_gf5": hgf, "away_gf5": agf, "home_ga5": hga, "away_ga5": aga,
            "gd10_diff": hgd - agd, "home_streak": hstk, "away_streak": astk,
            "home_rest": min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest": min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played": hn, "away_played": an,
            "h2h_n": nm, "h2h_home_winrate": h2h_wr, "h2h_draw_rate": h2h_dr, "h2h_gd": h2h_gd,
        })

        if not np.isnan(r.home_score):
            gd = r.home_score - r.away_score
            # standard ELO expected score from home's perspective, with home-advantage baked into adj
            exp = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            # goal-difference multiplier (FIFA-style): bigger wins shift ratings more
            g = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta
            elo[a] -= delta
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append((h, gd, h if gd > 0 else (a if gd < 0 else "draw")))

    return df.join(pd.DataFrame(rows, index=df.index))


# Feature reflection for the home/away symmetry trick.
SWAP_PAIRS = [
    ("home_elo", "away_elo"), ("home_form5", "away_form5"),
    ("home_winrate", "away_winrate"), ("home_gf5", "away_gf5"),
    ("home_ga5", "away_ga5"), ("home_streak", "away_streak"),
    ("home_rest", "away_rest"), ("home_played", "away_played"),
]
NEGATE = ["elo_diff", "form5_diff", "form10_diff", "gd10_diff", "h2h_gd"]
FLIP_LABEL = {"home_win": "away_win", "away_win": "home_win", "draw": "draw"}


def mirror(frame):
    """Reflect each row to the opposite team's perspective: swap home/away stats, negate
    differences, flip the head-to-head win rate, and flip the outcome label. Exact only for
    neutral venues (elo_diff carries no home-advantage term there), which is why we only ever
    mirror neutral matches."""
    m = frame.copy()
    for x, y in SWAP_PAIRS:
        m[x], m[y] = frame[y].values, frame[x].values
    for c in NEGATE:
        m[c] = -frame[c].values
    m["h2h_home_winrate"] = 1 - frame["h2h_home_winrate"].values - frame["h2h_draw_rate"].values
    if "outcome" in frame.columns:
        m["outcome"] = frame["outcome"].map(FLIP_LABEL).values
    return m


def train(pool, augment=False):
    """Fit TabPFN on the feature matrix; ignore_pretraining_limits allows >1000 rows.
    With augment=True, append mirrored copies of neutral matches so the model sees each one
    from both sides and learns the home/away symmetry directly."""
    if augment:
        pool = pd.concat([pool, mirror(pool[pool["neutral"] == 1])], ignore_index=True)
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


def predict_proba(clf, frame, symmetric=False):
    """Return a probability matrix in clf.classes_ order. With symmetric=True, neutral rows are
    scored from both perspectives and averaged (removes the arbitrary home-label artifact and
    cuts variance); non-neutral rows keep their real home advantage."""
    cls = list(clf.classes_)
    p = clf.predict_proba(frame[FEATURES].values)
    if not symmetric or not (frame["neutral"] == 1).any():
        return p
    d = {c: p[:, i] for i, c in enumerate(cls)}
    mp = clf.predict_proba(mirror(frame)[FEATURES].values)
    md = {c: mp[:, i] for i, c in enumerate(cls)}
    n = (frame["neutral"] == 1).values
    avg = {
        "home_win": np.where(n, 0.5 * (d["home_win"] + md["away_win"]), d["home_win"]),
        "away_win": np.where(n, 0.5 * (d["away_win"] + md["home_win"]), d["away_win"]),
        "draw": np.where(n, 0.5 * (d["draw"] + md["draw"]), d["draw"]),
    }
    return np.column_stack([avg[c] for c in cls])


def fit_temperature(proba, y, classes):
    """Find the temperature T that minimises log-loss when applied to these probabilities.
    Directly optimises the competition metric on a held-out month."""
    from scipy.optimize import minimize_scalar
    logits = np.log(np.clip(proba, 1e-12, 1))

    def nll(T):
        return log_loss(y, _softmax(logits / T), labels=classes)

    return minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def apply_temperature(proba, T):
    """Temperature-scale a probability matrix (T>1 softens, T<1 sharpens)."""
    return _softmax(np.log(np.clip(proba, 1e-12, 1)) / T)


# Odds-feed spellings -> dataset spellings (in addition to TEAM_ALIASES).
ODDS_ALIASES = {**TEAM_ALIASES, "Turkiye": "Turkey", "Türkiye": "Turkey", "Czechia": "Czech Republic",
                "IR Iran": "Iran", "Korea Republic": "South Korea", "Cabo Verde": "Cape Verde",
                "Curacao": "Curaçao", "DR Congo": "DR Congo", "Côte d'Ivoire": "Ivory Coast"}
# Sharp bookmakers / exchanges, in preference order — sharper than the soft-book median.
SHARP_BOOKS = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au"]


def shin_devig(inv):
    """Convert raw inverse odds (1/decimal_odds) to true probabilities via Shin's (1992) method.

    Shin models the overround as arising from insider traders (proportion z) and removes it
    NON-uniformly — penalising longshots and shaving less off favourites, which corrects the
    favourite-longshot bias that naive 1/odds normalisation leaves in. More accurate than the
    standard booksum divide (mberk/shin, penaltyblog). Falls back to naive on any degeneracy."""
    inv = np.asarray(inv, float)
    B = inv.sum()                       # booksum > 1 (the overround)
    n = len(inv)
    if B <= 1 or n < 2:
        return inv / B

    def sum_p(z):                       # Σ true-prob at a given z; want this to equal 1
        root = np.sqrt(z * z + 4 * (1 - z) * inv * inv / B)
        return ((root - z) / (2 * (1 - z))).sum()

    lo, hi = 0.0, 0.99                  # sum_p(0) = √B > 1; sum_p decreases in z → unique root
    if (sum_p(lo) - 1) * (sum_p(hi) - 1) > 0:
        return inv / B
    for _ in range(60):                 # bisection on Σp(z) = 1
        mid = (lo + hi) / 2
        if sum_p(mid) > 1:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    p = (np.sqrt(z * z + 4 * (1 - z) * inv * inv / B) - z) / (2 * (1 - z))
    return p / p.sum()


def fetch_odds():
    """Fetch current World Cup match odds from the-odds-api, de-vig them (remove the bookmaker
    margin), and return per-fixture true probabilities. Aggregates across bookmakers by median.
    Returns an empty DataFrame if no API key, no live matches, or any network error."""
    if not ODDS_API_KEY:
        return pd.DataFrame()
    import json
    import urllib.request

    def get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    base = "https://api.the-odds-api.com/v4"
    try:
        sports = get(f"{base}/sports/?apiKey={ODDS_API_KEY}")
    except Exception as e:
        print(f"Odds: could not reach the-odds-api ({e}); using model only.")
        return pd.DataFrame()
    keys = [s["key"] for s in sports if s.get("active") and "soccer" in s["key"]
            and ("world_cup" in s["key"] or "fifa" in s["key"])]

    rows = []
    for k in keys:
        try:
            events = get(f"{base}/sports/{k}/odds/?apiKey={ODDS_API_KEY}"
                         f"&regions=eu,uk&markets=h2h&oddsFormat=decimal")
        except Exception:
            continue
        for e in events:
            home, away = e.get("home_team"), e.get("away_team")
            # Collect inverse odds per bookmaker so we can prefer the sharp ones.
            per_book = {}
            for bk in e.get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m.get("key") != "h2h":
                        continue
                    o = {x["name"]: x["price"] for x in m["outcomes"]}
                    if home in o and away in o and "Draw" in o:
                        per_book[bk["key"]] = (1 / o[home], 1 / o["Draw"], 1 / o[away])
            if not per_book:
                continue
            sharp = [per_book[k] for k in SHARP_BOOKS if k in per_book]
            used = sharp if sharp else list(per_book.values())   # sharp books, else all
            inv = np.median(np.array(used), axis=0)              # consensus inverse odds
            p = shin_devig(inv)                                  # Shin's true probabilities
            rows.append({"home_team": ODDS_ALIASES.get(home, home),
                         "away_team": ODDS_ALIASES.get(away, away),
                         "o_home": p[0], "o_draw": p[1], "o_away": p[2],
                         "odds_sharp": bool(sharp)})
    return pd.DataFrame(rows)


def blend_odds(out, odds, weight=ODDS_WEIGHT):
    """Blend model probabilities in `out` with de-vigged market odds (linear pool). Only fixtures
    found in the odds feed are adjusted; the rest keep their model probabilities. Mutates and
    returns `out`, and reports how many fixtures were matched."""
    if not len(odds):
        return out, 0
    m = out.merge(odds, on=["home_team", "away_team"], how="left")
    hit = m["o_home"].notna()
    for col, ocol in [("p_home_win", "o_home"), ("p_draw", "o_draw"), ("p_away_win", "o_away")]:
        m.loc[hit, col] = weight * m.loc[hit, ocol] + (1 - weight) * m.loc[hit, col]
    m["predicted"] = m[["p_away_win", "p_draw", "p_home_win"]].values.argmax(1)
    m["predicted"] = m["predicted"].map({0: "away_win", 1: "draw", 2: "home_win"})
    n_sharp = int(m.loc[hit, "odds_sharp"].sum()) if "odds_sharp" in m else 0
    # Keep market probabilities as columns so score.py can re-tune the blend weight after results.
    m = m.rename(columns={"o_home": "p_market_home", "o_draw": "p_market_draw", "o_away": "p_market_away"})
    drop = [c for c in ["odds_sharp"] if c in m]
    return m.drop(columns=drop), (int(hit.sum()), n_sharp)


def main():
    """Backtest on the previous calendar month, then predict all upcoming fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download dataset from source")
    parser.add_argument("--fixtures", help="CSV of upcoming fixtures to predict (date,home_team,away_team)")
    parser.add_argument("--symmetric", action="store_true",
                        help="Enable home/away symmetry augmentation + mirror-averaging (tested: no gain)")
    parser.add_argument("--no-odds", action="store_true",
                        help="Skip blending with live market odds even if ODDS_API_KEY is set")
    parser.add_argument("--odds-weight", type=float, default=ODDS_WEIGHT,
                        help=f"Blend weight toward market odds (default {ODDS_WEIGHT})")
    args = parser.parse_args()

    df = load_data(refresh=args.refresh, fixtures=args.fixtures)
    latest_played = df[df["outcome"].notna()]["date"].max()
    print(f"Latest result in dataset: {latest_played.date()}")
    print(f"Data freshness: {pd.Timestamp.now() - latest_played}")

    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    # Predict every fixture the dataset lists without a result yet (officially scheduled
    # matches the source feed has added). Re-running after --refresh self-updates to each
    # new round as results come in and the next round's fixtures are published.
    future = feats[feats["home_score"].isna() & (feats["date"] >= latest_played)].sort_values("date")

    def month_slice(p, frame):
        return frame[(frame["date"] >= p.start_time) & (frame["date"] < (p + 1).start_time)]

    labels = sorted(played["outcome"].unique())

    # --- Backtest on the previous calendar month (quick sanity check) ---
    eval_m = TODAY.to_period("M") - 1
    eval_set = month_slice(eval_m, played)
    if len(eval_set):
        bt = train(played[played["date"] < eval_m.start_time].tail(MAX_TRAIN), augment=args.symmetric)
        p = predict_proba(bt, eval_set, symmetric=args.symmetric)
        print(f"\nBacktest {eval_m} ({len(eval_set)} matches): "
              f"accuracy {accuracy_score(eval_set['outcome'], np.array(labels)[p.argmax(1)]):.0%}, "
              f"log-loss {log_loss(eval_set['outcome'], p, labels=labels):.3f}")

    # --- Temperature calibration, fit on a large held-out set of recent NEUTRAL matches ---
    # Neutral tournament football is the regime the World Cup lives in. TabPFN is already well
    # calibrated, so this is a small (~0.001 log-loss), cross-validation-checked gain that
    # directly targets the competition metric. (Home/away symmetry was tested and did NOT help,
    # so it is off by default; enable with --symmetric to reproduce.)
    temperature = 1.0
    cutoff = TODAY - pd.DateOffset(years=CAL_YEARS)
    cal_set = played[(played["date"] >= cutoff) & (played["neutral"] == 1)]
    if len(cal_set) >= 200:
        cal_model = train(played[played["date"] < cutoff].tail(MAX_TRAIN))
        temperature = fit_temperature(predict_proba(cal_model, cal_set), cal_set["outcome"].values, labels)
        print(f"Calibration: T={temperature:.2f} (fit on {len(cal_set)} held-out neutral matches)")

    if not len(future):
        print("\nNo upcoming fixtures in the dataset (it only contains played matches).")
        print("Append rows for future matches to results.csv (date in the future, "
              "blank home_score/away_score) and re-run to get predictions.")
        return

    clf = train(played.tail(MAX_TRAIN), augment=args.symmetric)
    proba = apply_temperature(predict_proba(clf, future, symmetric=args.symmetric), temperature)
    cols = {c: proba[:, i] for i, c in enumerate(clf.classes_)}

    out = future[["date", "home_team", "away_team"]].copy()
    out["predicted"] = clf.classes_[proba.argmax(1)]
    out["p_home_win"] = cols["home_win"]
    out["p_draw"] = cols["draw"]
    out["p_away_win"] = cols["away_win"]
    # Preserve the model-only probabilities so score.py --tune can reconstruct any blend weight.
    out["p_model_home"], out["p_model_draw"], out["p_model_away"] = \
        cols["home_win"], cols["draw"], cols["away_win"]

    # --- Blend with de-vigged live market odds (sharpest available signal) ---
    if not args.no_odds:
        out, (matched, n_sharp) = blend_odds(out, fetch_odds(), args.odds_weight)
        if matched:
            print(f"Odds: blended {matched}/{len(out)} fixtures with market "
                  f"({n_sharp} from sharp books; Shin de-vig; "
                  f"weight {args.odds_weight:.0%} market / {1 - args.odds_weight:.0%} model).")
        elif ODDS_API_KEY:
            print("Odds: no matching live fixtures found in the feed; using model only.")

    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    filename = f"predictions_{today_str}.csv"
    out.to_csv(filename, index=False)

    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r in out.itertuples():
        print(f"  {r.date.date()}  {r.home_team:>20} vs {r.away_team:<20}  "
              f"-> {r.predicted:<9}  H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}")


if __name__ == "__main__":
    main()
