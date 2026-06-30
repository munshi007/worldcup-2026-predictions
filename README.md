# TabPFN Football Predictions

This repository is a template to participate in Prior Labs' [World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/worldcup). It has a basic script that outputs predictions with a standard prediction template. Use this template to generate predictions. The `predict.py` script should only be a source of inspiration, feel free to fork the repo and add your own ideas.

The script predicts international football match outcomes using [TabPFN](https://github.com/PriorLabs/TabPFN) using the [client repository](https://github.com/PriorLabs/tabpfn-client). It achieves ~59% accuracy and ~0.86 log-loss on held-out data. There is a good margin of progression. We look forward to your submission!

The model is trained on engineered features: ELO ratings, recent form, head-to-head record, rest days, and tournament importance. Data comes from [martj42/international_results](https://github.com/martj42/international_results).

## Setup

```bash
git clone https://github.com/munshi007/worldcup-2026-predictions.git
cd worldcup-2026-predictions
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your TABPFN_API_KEY (and optional ODDS_API_KEY)
```

## Run

```bash
python predict.py
```

This will:

1. Download the full international results dataset (~47 000 matches) on first run
2. Build features with a single chronological pass (no leakage)
3. Run a quick backtest on the previous calendar month and print accuracy + log-loss
4. Train on up to 10 000 recent matches and predict all upcoming fixtures
5. Save predictions to `predictions_YYYYMMDD.csv` and print them to the console

To refresh the dataset from source before predicting:

```bash
python predict.py --refresh
```

## Output

```
Latest game in dataset: 2026-06-14
Data freshness: 0 days 18:32:11

Backtest 2026-05 (87 matches): accuracy 59%, log-loss 0.861

142 fixture predictions -> predictions_20260616.csv

  2026-06-18           Argentina vs Australia             -> home_win   H  72% | D  17% | A  11%
  2026-06-18              France vs Morocco              -> home_win   H  61% | D  23% | A  16%
  ...
```

## Features

| Feature | Description |
|---|---|
| `elo_diff` | ELO gap (home + home advantage - away) |
| `home_elo`, `away_elo` | Current ELO ratings |
| `form5_diff` | Difference in average points per game over last 5 matches |
| `form10_diff` | Same over last 10 matches |
| `home_winrate`, `away_winrate` | Win rate over last 10 matches |
| `home_gf5`, `away_gf5` | Goals scored per game over last 5 matches |
| `home_ga5`, `away_ga5` | Goals conceded per game over last 5 matches |
| `gd10_diff` | Difference in average goal difference over last 10 matches |
| `home_streak`, `away_streak` | Current win streak |
| `home_rest`, `away_rest` | Days since last match (capped at 90) |
| `home_played`, `away_played` | Total matches played in history |
| `h2h_n` | Number of head-to-head meetings |
| `h2h_home_winrate` | Home team win rate in head-to-head |
| `h2h_draw_rate` | Draw rate in head-to-head |
| `h2h_gd` | Average goal difference in head-to-head (from home team's perspective) |
| `neutral` | 1 if played at a neutral venue |
| `importance` | Tournament importance score (60 = World Cup, 20 = friendly) |
| `val_home`, `val_away`, `val_diff` | log squad market value (€m) per team and the gap, from `team_values.csv` — current squad quality, which results-based ELO lags. NaN for teams not in the table |

## Extras added on top of the template

- **`--fixtures FILE.csv`** — predict an upcoming-fixtures CSV (`date,home_team,away_team`; `neutral` auto-derived, names normalized via `TEAM_ALIASES`). See `worldcup_r32.csv`.
- **Squad market-value features** (`team_values.csv`) — CV-validated ~0.01 log-loss gain on competitive neutral matches; the single biggest improvement found.
- **Temperature calibration** — fit on held-out neutral matches, directly targets log-loss (small ~0.001 gain; TabPFN is already well-calibrated).
- **`--symmetric`** — home/away symmetry augmentation + mirror-averaging. Tested and did **not** help on neutral matches, so it is off by default; flag exists to reproduce.
- **Live market-odds blending** (the sharpest signal). With `ODDS_API_KEY` set (the-odds-api.com), fetches current World Cup odds, prefers **sharp books** (Pinnacle / Betfair exchange), de-vigs them with **Shin's method** (corrects favourite-longshot bias), and linear-pools market + model per fixture. Flags: `--no-odds`, `--odds-weight` (default 0.60). Predictions CSV saves model-only and market-only columns alongside the blend.
- **`score.py`** — validation harness. `python score.py predictions_*.csv` reports log-loss/accuracy vs actual results; `--tune` sweeps the blend weight to find the log-loss-minimising mix once games are played. Use it after each round to tune `ODDS_WEIGHT` from evidence.
