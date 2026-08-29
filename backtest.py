"""
Walk-forward calibration backtest.

This answers the one question that matters most: when the model says a team
has a 70% chance to win, does that actually happen about 70% of the time,
historically? Without this, "the model works" is just an assertion.

Method (walk-forward — never lets future data leak into training):
  1. Sort a league's real match history by date.
  2. Use the first BURN_IN matches only to initialize (never scored).
  3. For every match after that, fit the model on ONLY the matches that
     happened strictly before it, then predict that match's outcome.
  4. Record the model's predicted probability alongside what actually
     happened.
  5. Refit periodically (every REFIT_EVERY matches, not every single match)
     to keep this computationally reasonable — refitting after every one
     match barely changes the ratings but multiplies runtime.

Outputs two things per league:
  - Brier score and log-loss for the 1X2 market (3-outcome Brier score
    ranges 0 [perfect] to ~0.667 [uniform random guessing]; log-loss for
    uniform random guessing is ln(3) ≈ 1.099). Also computed: the same
    metrics for a "naive baseline" that just predicts each league's overall
    historical home/draw/away rate for every match — if the real model
    doesn't clearly beat this naive baseline, the model isn't adding value.
  - A calibration table: bucket every prediction by its predicted
    probability, and compare to how often that outcome actually happened
    in that bucket. A well-calibrated model has predicted% ≈ actual% in
    every bucket.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from train_and_export import fit_dixon_coles, dc_adjustment
from scipy.stats import poisson


BURN_IN = 60          # minimum matches before we trust a fit enough to score predictions
REFIT_EVERY = 10       # refit the model every N matches walked forward (speed/accuracy tradeoff)
CALIBRATION_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def score_matrix_probs(model, home, away, max_goals=8):
    """1X2 probabilities only — cheaper than the full market grid, all we need for scoring."""
    if home not in model["attack"] or away not in model["attack"]:
        return None
    lam_h = model["attack"][home] * model["defense"][away] * model["home_advantage"]
    lam_a = model["attack"][away] * model["defense"][home]
    goals = np.arange(0, max_goals + 1)
    ph, pa = poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a)
    matrix = np.outer(ph, pa)
    rho = model["rho"]
    for i in range(2):
        for j in range(2):
            matrix[i, j] *= dc_adjustment(i, j, lam_h, lam_a, rho)
    matrix = np.clip(matrix, 0, None)
    total = matrix.sum()
    if not np.isfinite(total) or total <= 0:
        return None  # degenerate fit (can happen with very sparse early-season data) — skip this prediction
    matrix /= total
    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()
    return p_home, p_draw, p_away


def walk_forward_backtest(matches, half_life_days=200):
    """matches: date, home_team, away_team, home_goals, away_goals — sorted by date already assumed."""
    matches = matches.sort_values("date").reset_index(drop=True)
    n = len(matches)
    if n < BURN_IN + 20:
        return None  # not enough history to backtest meaningfully

    predictions = []  # each: (p_home, p_draw, p_away, actual_outcome)
    model = None

    for i in range(BURN_IN, n):
        if model is None or (i - BURN_IN) % REFIT_EVERY == 0:
            train_slice = matches.iloc[:i]
            model = fit_dixon_coles(train_slice, half_life_days=half_life_days)

        row = matches.iloc[i]
        probs = score_matrix_probs(model, row["home_team"], row["away_team"])
        if probs is None:
            continue  # team not seen in training slice yet, skip
        p_home, p_draw, p_away = probs

        if row["home_goals"] > row["away_goals"]:
            actual = "H"
        elif row["home_goals"] == row["away_goals"]:
            actual = "D"
        else:
            actual = "A"

        predictions.append({
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away, "actual": actual,
        })

    return predictions


def compute_metrics(predictions):
    predictions = [p for p in predictions if np.isfinite(p["p_home"]) and np.isfinite(p["p_draw"]) and np.isfinite(p["p_away"])]
    if not predictions:
        return None

    briers, logs = [], []
    for p in predictions:
        outcome_vec = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[p["actual"]]
        pred_vec = [p["p_home"], p["p_draw"], p["p_away"]]
        brier = sum((pv - ov) ** 2 for pv, ov in zip(pred_vec, outcome_vec))
        briers.append(brier)
        actual_prob = pred_vec[outcome_vec.index(1)]
        logs.append(-np.log(max(actual_prob, 1e-9)))

    brier_score = float(np.mean(briers))
    log_loss = float(np.mean(logs))

    # Naive baseline: predict this league's overall historical H/D/A rate for
    # every single match, ignoring which teams are playing entirely. If the
    # real model can't beat this, it isn't earning its complexity.
    base_h = np.mean([1.0 if p["actual"] == "H" else 0.0 for p in predictions])
    base_d = np.mean([1.0 if p["actual"] == "D" else 0.0 for p in predictions])
    base_a = np.mean([1.0 if p["actual"] == "A" else 0.0 for p in predictions])
    base_briers, base_logs = [], []
    for p in predictions:
        outcome_vec = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[p["actual"]]
        base_vec = [base_h, base_d, base_a]
        base_briers.append(sum((pv - ov) ** 2 for pv, ov in zip(base_vec, outcome_vec)))
        base_logs.append(-np.log(max(base_vec[outcome_vec.index(1)], 1e-9)))
    naive_brier = float(np.mean(base_briers))
    naive_log_loss = float(np.mean(base_logs))

    # calibration: for the HOME WIN probability specifically (most bettable market)
    calib = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        bucket = [p for p in predictions if lo <= p["p_home"] < hi]
        if len(bucket) < 5:
            continue
        predicted_avg = np.mean([p["p_home"] for p in bucket])
        actual_freq = np.mean([1.0 if p["actual"] == "H" else 0.0 for p in bucket])
        calib.append({
            "bucket": f"{int(lo*100)}-{int(hi*100)}%",
            "predicted_avg": round(float(predicted_avg) * 100, 1),
            "actual_freq": round(float(actual_freq) * 100, 1),
            "n": len(bucket),
        })

    return {
        "n_predictions": len(predictions),
        "brier_score": round(brier_score, 4),
        "log_loss": round(log_loss, 4),
        "naive_baseline_brier": round(naive_brier, 4),
        "naive_baseline_log_loss": round(naive_log_loss, 4),
        "beats_naive_baseline": bool(brier_score < naive_brier),
        "home_win_calibration": calib,
    }


def backtest_league(league_name, matches):
    print(f"Backtesting {league_name} ({len(matches)} matches)...")
    predictions = walk_forward_backtest(matches)
    if predictions is None:
        print(f"  Not enough history to backtest {league_name} — skipped.")
        return None
    metrics = compute_metrics(predictions)
    print(f"  {league_name}: Brier={metrics['brier_score']}  LogLoss={metrics['log_loss']}  "
          f"n={metrics['n_predictions']}")
    return metrics


if __name__ == "__main__":
    # Standalone mode (only useful if you want to run just the backtest without
    # retraining) — fetches fresh, since it has no in-memory data to reuse.
    # Normal usage is via train_and_export.py, which calls the functions above
    # directly on data it already fetched, avoiding duplicate downloads.
    from train_and_export import FOOTBALL_DATA_CODES, fetch_football_data_csv, SEASONS

    results = {}
    for league_name, code in FOOTBALL_DATA_CODES.items():
        parts = [fetch_football_data_csv(code, s) for s in SEASONS]
        combined = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
        combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"])
        metrics = backtest_league(league_name, combined)
        if metrics:
            results[league_name] = metrics

    with open("backtest.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote backtest.json for {len(results)} leagues.")
