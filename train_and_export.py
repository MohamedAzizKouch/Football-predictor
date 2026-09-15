"""
Fetches real historical results for every available league and trains a
Dixon-Coles model per league. Designed to run unattended (no notebook, no
manual steps) — this is what the GitHub Action calls on a schedule.

Writes models.json, which index.html reads.

Also fetches real ClubElo.com ratings (elo_ratings.json) for European
competition (cross-league) mode — see clubelo.py for the full method.
ClubElo's ratings are calculated from real historical match results,
including actual cross-league games (Champions/Europa/Conference League),
so cross-league comparability is grounded in real evidence rather than an
approximation.
"""

import json
import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# Dixon-Coles engine
# ---------------------------------------------------------------------------

def time_decay_weight(days_ago, half_life_days=200.0):
    decay_rate = np.log(2) / half_life_days
    return np.exp(-decay_rate * days_ago)


def dc_adjustment(hg, ag, lam_h, lam_a, rho):
    if hg == 0 and ag == 0:
        return 1 - lam_h * lam_a * rho
    elif hg == 0 and ag == 1:
        return 1 + lam_h * rho
    elif hg == 1 and ag == 0:
        return 1 + lam_a * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles(matches, half_life_days=200.0, l2_reg=0.001):
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    as_of = df["date"].max()
    days_ago = (as_of - df["date"]).dt.days.clip(lower=0).values
    weights = time_decay_weight(days_ago, half_life_days)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    home_idx = df["home_team"].map(idx).values
    away_idx = df["away_team"].map(idx).values
    hg = df["home_goals"].values.astype(int)
    ag = df["away_goals"].values.astype(int)

    x0 = np.concatenate([np.ones(n), np.ones(n), [0.25], [-0.05]])

    def unpack(x):
        return x[:n], x[n:2 * n], x[2 * n], x[2 * n + 1]

    def neg_ll(x):
        a, d, h, r = unpack(x)
        a = np.clip(a, 1e-3, None)
        d = np.clip(d, 1e-3, None)
        lam_h = np.clip(a[home_idx] * d[away_idx] * np.exp(h), 1e-6, None)
        lam_a = np.clip(a[away_idx] * d[home_idx], 1e-6, None)
        ll = poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
        tau = np.array([dc_adjustment(hgi, agi, lhi, lai, r)
                         for hgi, agi, lhi, lai in zip(hg, ag, lam_h, lam_a)])
        ll = ll + np.log(np.clip(tau, 1e-6, None))
        reg = l2_reg * (np.sum((a - 1.0) ** 2) + np.sum((d - 1.0) ** 2))
        return -np.sum(weights * ll) + reg

    res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})
    a, d, h, r = unpack(res.x)
    return {
        "teams": teams,
        "attack": dict(zip(teams, a.tolist())),
        "defense": dict(zip(teams, d.tolist())),
        "home_advantage": float(np.exp(h)),
        "rho": float(r),
    }


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

FOOTBALL_DATA_CODES = {
    "Premier League": "E0",
    "Championship": "E1",
    "League One": "E2",
    "Ligue 1": "F1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "La Liga": "SP1",
    "Eredivisie": "N1",
}

SEASONS = ["2425", "2526", "2627"]


def fetch_football_data_csv(code, season, max_retries=3):
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(5 * attempt)
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            raw = pd.read_csv(StringIO(resp.text), encoding="latin1", on_bad_lines="skip")
            raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
            matches = raw.rename(columns={
                "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                "FTHG": "home_goals", "FTAG": "away_goals",
            })[["date", "home_team", "away_team", "home_goals", "away_goals"]].dropna()
            matches["home_goals"] = matches["home_goals"].astype(int)
            matches["away_goals"] = matches["away_goals"].astype(int)
            return matches
        except Exception as e:
            last_error = e
            print(f"  [{code} {season}] attempt {attempt+1}/{max_retries} failed: {e}")
    print(f"  [{code} {season}] all {max_retries} attempts failed, giving up on this file: {last_error}")
    return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])


def fetch_portugal():
    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/PPL/matches",
            headers={"X-Auth-Token": ""},
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"  [Liga Portugal] API returned {resp.status_code}, skipping")
            return None
        rows = []
        for m in resp.json().get("matches", []):
            if m["status"] != "FINISHED":
                continue
            rows.append({
                "date": m["utcDate"][:10],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": m["score"]["fullTime"]["home"],
                "away_goals": m["score"]["fullTime"]["away"],
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"  [Liga Portugal] fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from backtest import walk_forward_backtest, compute_metrics

    existing_models = {}
    existing_backtest = {}
    if os.path.exists("models.json"):
        with open("models.json") as f:
            existing_models = json.load(f)
    if os.path.exists("backtest.json"):
        with open("backtest.json") as f:
            existing_backtest = json.load(f)

    export = dict(existing_models)
    backtest_results = dict(existing_backtest)

    for league_name, code in FOOTBALL_DATA_CODES.items():
        print(f"Fetching {league_name}...")
        parts = [fetch_football_data_csv(code, s) for s in SEASONS]
        combined = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
        combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"])
        if len(combined) < 20:
            kept = "kept previous data" if league_name in existing_models else "no previous data either"
            print(f"  {league_name}: only {len(combined)} matches fetched this run — {kept}, skipping update.")
            continue

        model = fit_dixon_coles(combined, half_life_days=200)
        model["n_matches"] = len(combined)
        model["last_updated"] = str(pd.Timestamp.utcnow().date())
        export[league_name] = model
        print(f"  {league_name}: {len(combined)} matches, {len(model['teams'])} teams — trained.")

        predictions = walk_forward_backtest(combined)
        if predictions:
            metrics = compute_metrics(predictions)
            if metrics:
                backtest_results[league_name] = metrics
                print(f"  {league_name} backtest: Brier={metrics['brier_score']}  "
                      f"LogLoss={metrics['log_loss']}  n={metrics['n_predictions']}")

    print("Fetching Liga Portugal...")
    pt = fetch_portugal()
    if pt is not None and len(pt) >= 20:
        model = fit_dixon_coles(pt, half_life_days=200)
        model["n_matches"] = len(pt)
        model["last_updated"] = str(pd.Timestamp.utcnow().date())
        export["Liga Portugal"] = model
        predictions = walk_forward_backtest(pt)
        if predictions:
            metrics = compute_metrics(predictions)
            if metrics:
                backtest_results["Liga Portugal"] = metrics
    else:
        kept = "kept previous data" if "Liga Portugal" in existing_models else "no previous data either"
        print(f"  Liga Portugal: not enough data this run — {kept}.")

    if not export:
        print("\nERROR: nothing to write and no previous data. Leaving files untouched.")
        return

    with open("models.json", "w") as f:
        json.dump(export, f, indent=2)
    with open("backtest.json", "w") as f:
        json.dump(backtest_results, f, indent=2)

    print(f"\nWrote models.json with {len(export)} leagues.")
    print(f"Wrote backtest.json with {len(backtest_results)} leagues.")

    # --- ClubElo ratings for European competition (cross-league) mode ---
    # See clubelo.py for the full method and its honest limitations. This
    # replaces the earlier UEFA-coefficient approximation with real ratings
    # calculated from actual cross-league match history.
    from clubelo import fetch_clubelo_ratings_grouped
    print("\nFetching ClubElo ratings for European competition mode...")
    elo_grouped = fetch_clubelo_ratings_grouped()
    if elo_grouped:
        n_clubs = sum(len(clubs) for clubs in elo_grouped.values())
        with open("elo_ratings.json", "w") as f:
            json.dump(elo_grouped, f, indent=2)
        print(f"Wrote elo_ratings.json: {len(elo_grouped)} countries, {n_clubs} clubs.")
    else:
        print("ClubElo fetch failed this run — leaving any existing elo_ratings.json untouched.")


if __name__ == "__main__":
    main()
