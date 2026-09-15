"""
Cross-league match prediction using real ClubElo.com ratings.

This replaces the UEFA-coefficient approximation with something more
rigorous: ClubElo's ratings are calculated from actual historical match
results INCLUDING real cross-league games (Champions League, Europa League,
Conference League), with "inter-league adjustments" built into their own
methodology (documented at clubelo.com/System). That means the cross-league
comparability problem is solved using real evidence, not an aggregate
proxy like UEFA's country coefficient.

Data source: api.clubelo.com/YYYY-MM-DD returns every tracked club's current
Elo rating as CSV. Free, no key required, documented at clubelo.com/API.
Covers ~40+ countries, including nearly everything in Europa/Conference
League that our football-data.co.uk-based domestic models can't reach.

Elo -> expected goals conversion: uses the published formula from the World
Football Elo Ratings methodology (the same system eloratings.net uses for
international football), which fits win-expectancy to expected-goals via a
quartic polynomial calibrated on ~40,000 real matches. Source: "Predicting
international football results using an experience-weighted Poisson model"
methodology, as documented and re-derived in academic work replicating it
(e.g. arXiv:2502.08565).

HONEST LIMITATION: this polynomial was calibrated on INTERNATIONAL football
(different scoring patterns than club football — internationals tend to be
lower-scoring on average). Applying it to club matches is a reasonable
transfer, not a perfect one. It replaces one approximation (borrowed UEFA
coefficients) with a better-justified one (a real, published, cited formula)
— it does not make cross-league predictions as reliable as the domestic
single-league models, which are fitted directly on the relevant data.
"""

from io import StringIO

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson


CLUBELO_HOME_ADVANTAGE_ELO = 100  # standard constant used across most published football Elo systems


def fetch_clubelo_ratings(date: str = None) -> dict:
    """
    date: 'YYYY-MM-DD', defaults to today (UTC).
    Returns {club_name: elo_rating} for every club ClubElo tracks.
    """
    if date is None:
        date = str(pd.Timestamp.utcnow().date())
    url = f"http://api.clubelo.com/{date}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Documented columns: Rank, Club, Country, Level, Elo, From, To
        return dict(zip(df["Club"], df["Elo"]))
    except Exception as e:
        print(f"  ClubElo fetch failed for {date}: {e}")
        return {}


def fetch_clubelo_ratings_grouped(date: str = None) -> dict:
    """
    Same data as fetch_clubelo_ratings, but grouped by country for a
    two-step (country -> club) picker in the UI, since ClubElo tracks
    hundreds of clubs across 40+ countries — too many for one flat list.
    Returns {country: {club_name: elo}}.
    """
    if date is None:
        date = str(pd.Timestamp.utcnow().date())
    url = f"http://api.clubelo.com/{date}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        grouped = {}
        for _, row in df.iterrows():
            country = row["Country"]
            grouped.setdefault(country, {})[row["Club"]] = float(row["Elo"])
        return grouped
    except Exception as e:
        print(f"  ClubElo fetch failed for {date}: {e}")
        return {}


def elo_win_expectancy(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected-score formula. Add home advantage to elo_a beforehand if applicable."""
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))


def elo_win_expectancy_to_expected_goals(w: float) -> float:
    """
    Published quartic polynomial converting win expectancy -> expected goals,
    from the World Football Elo Ratings methodology (neutral-field version).
    See module docstring for source.

    Capped at 5.0: the polynomial was calibrated on the real range of
    international-match win expectancies. At extreme Elo gaps (giant club vs
    a part-time amateur side, which does happen in early Europa/Conference
    League rounds), it can extrapolate to implausible values — this cap
    keeps output sane without pretending we have calibration data out there.
    """
    w = min(max(w, 0.0), 1.0)
    if w <= 0.9:
        val = (3.90388 * w**4 - 0.58486 * w**3 - 2.98315 * w**2 + 3.13160 * w + 0.33193)
    else:
        x = w - 0.9
        val = (308097.45501 * x**4 - 42803.04696 * x**3 + 2116.35304 * x**2 - 9.61869 * x + 2.86899)
    return min(max(val, 0.05), 5.0)


def cross_league_expected_goals(elo_home: float, elo_away: float, home_advantage_elo: float = CLUBELO_HOME_ADVANTAGE_ELO):
    """Returns (lambda_home, lambda_away) — expected goals for a Poisson model."""
    w_home = elo_win_expectancy(elo_home + home_advantage_elo, elo_away)
    w_away = 1.0 - w_home
    lam_home = elo_win_expectancy_to_expected_goals(w_home)
    lam_away = elo_win_expectancy_to_expected_goals(w_away)
    return lam_home, lam_away


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


def cross_league_score_matrix(elo_home: float, elo_away: float, max_goals: int = 8, rho: float = -0.06):
    """
    rho default: an average, reasonable low-score correction (our domestic
    leagues fit values roughly -0.003 to -0.13) — there's no equivalent
    fitted rho for cross-league Elo matches, so this is a neutral default,
    not a fitted parameter. It only affects the 0-0/1-0/0-1/1-1 probabilities
    slightly; it doesn't materially change the headline 1X2/goals numbers.
    """
    lam_h, lam_a = cross_league_expected_goals(elo_home, elo_away)
    n = max_goals + 1
    goals = np.arange(n)
    ph, pa = poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a)
    matrix = np.outer(ph, pa)
    for i in range(2):
        for j in range(2):
            matrix[i, j] *= dc_adjustment(i, j, lam_h, lam_a, rho)
    matrix = np.clip(matrix, 0, None)
    matrix /= matrix.sum()
    return matrix, lam_h, lam_a
