"""
Football Prediction System — Streamlit interface.

Run with:  streamlit run app.py

Needs models.json in the same folder (produced by the training notebook,
Football_Prediction_Train_All_Leagues.ipynb — run that first and download
its output here).
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="Football Prediction System", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Core math (same Dixon-Coles engine as the training notebook, prediction-only)
# ---------------------------------------------------------------------------

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


def expected_goals(model, home_team, away_team):
    lam_h = model["attack"][home_team] * model["defense"][away_team] * model["home_advantage"]
    lam_a = model["attack"][away_team] * model["defense"][home_team]
    return lam_h, lam_a


def score_matrix(model, home_team, away_team, max_goals=8):
    lam_h, lam_a = expected_goals(model, home_team, away_team)
    goals = np.arange(0, max_goals + 1)
    ph, pa = poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a)
    matrix = np.outer(ph, pa)
    rho = model["rho"]
    for i in range(2):
        for j in range(2):
            matrix[i, j] *= dc_adjustment(i, j, lam_h, lam_a, rho)
    matrix = np.clip(matrix, 0, None)
    return matrix / matrix.sum(), lam_h, lam_a


def all_markets(matrix):
    n = matrix.shape[0]
    goals = np.arange(n)
    home_marg = matrix.sum(axis=1)
    away_marg = matrix.sum(axis=0)

    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()
    p_btts = matrix[1:, 1:].sum()

    out = {}
    out["1X2"] = {"Home Win (1)": p_home, "Draw (X)": p_draw, "Away Win (2)": p_away}
    out["Double Chance"] = {"1X": p_home + p_draw, "12": p_home + p_away, "X2": p_draw + p_away}
    out["BTTS"] = {"Yes (GG)": p_btts, "No (NG)": 1 - p_btts}

    total_dist = {}
    for total in range(0, 2 * n - 1):
        mask = np.zeros_like(matrix)
        for i in range(n):
            j = total - i
            if 0 <= j < n:
                mask[i, j] = 1
        total_dist[total] = (matrix * mask).sum()

    ou = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under = sum(p for t, p in total_dist.items() if t < line)
        ou[f"Over {line}"] = 1 - under
        ou[f"Under {line}"] = under
    out["Over/Under Total Goals"] = ou

    tt = {}
    for line in [0.5, 1.5, 2.5]:
        tt[f"Home Over {line}"] = home_marg[goals > line].sum()
        tt[f"Away Over {line}"] = away_marg[goals > line].sum()
    out["Team Totals"] = tt

    out["Clean Sheet"] = {"Home Clean Sheet": away_marg[0], "Away Clean Sheet": home_marg[0]}

    cs = {}
    for i in range(min(n, 6)):
        for j in range(min(n, 6)):
            cs[f"{i}-{j}"] = matrix[i, j]
    out["Correct Score (top 8)"] = dict(sorted(cs.items(), key=lambda kv: -kv[1])[:8])

    return out


# ---------------------------------------------------------------------------
# Load trained models
# ---------------------------------------------------------------------------

@st.cache_data
def load_models():
    try:
        with open("models.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


models = load_models()

st.title("⚽ Football Prediction System")

if models is None:
    st.error(
        "**models.json not found.** Run the training notebook "
        "(`Football_Prediction_Train_All_Leagues.ipynb`) first, download its "
        "output, and place `models.json` in this same folder."
    )
    st.stop()

st.caption(
    f"Loaded {len(models)} league(s), trained on real historical results. "
    "Predictions come from a Dixon-Coles model fitted to each league separately."
)

# ---------------------------------------------------------------------------
# Sidebar: league + team selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pick a fixture")

    league = st.selectbox("League", sorted(models.keys()))
    teams = sorted(models[league]["teams"])

    home_team = st.selectbox("Home team", teams, index=0)
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("Away team", away_options, index=0)

    bankroll = st.number_input("Bankroll (for stake sizing)", min_value=10, value=100, step=10)

    st.divider()
    st.subheader("Optional: compare to bookmaker odds")
    st.caption("Enter decimal odds to see if the model finds value. Leave at 0 to skip a market.")
    odds_home = st.number_input("Home Win (1) odds", min_value=0.0, value=0.0, step=0.05)
    odds_draw = st.number_input("Draw (X) odds", min_value=0.0, value=0.0, step=0.05)
    odds_away = st.number_input("Away Win (2) odds", min_value=0.0, value=0.0, step=0.05)
    odds_over25 = st.number_input("Over 2.5 odds", min_value=0.0, value=0.0, step=0.05)
    odds_under25 = st.number_input("Under 2.5 odds", min_value=0.0, value=0.0, step=0.05)
    odds_btts_yes = st.number_input("BTTS Yes odds", min_value=0.0, value=0.0, step=0.05)
    odds_btts_no = st.number_input("BTTS No odds", min_value=0.0, value=0.0, step=0.05)

    run = st.button("Get prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if not run:
    st.info("Pick a league and two teams in the sidebar, then click **Get prediction**.")
    with st.expander("Team ratings for the selected league"):
        m = models[league]
        ratings = pd.DataFrame({
            "team": m["teams"],
            "attack": [m["attack"][t] for t in m["teams"]],
            "defense": [m["defense"][t] for t in m["teams"]],
        })
        ratings["net_strength"] = ratings["attack"] - ratings["defense"]
        ratings = ratings.sort_values("net_strength", ascending=False).reset_index(drop=True)
        st.dataframe(ratings, use_container_width=True)
    st.stop()

model = models[league]
matrix, lam_h, lam_a = score_matrix(model, home_team, away_team)
markets = all_markets(matrix)

st.subheader(f"{home_team} vs {away_team} — {league}")
c1, c2 = st.columns(2)
c1.metric(f"{home_team} expected goals", f"{lam_h:.2f}")
c2.metric(f"{away_team} expected goals", f"{lam_a:.2f}")

tab1, tab2, tab3 = st.tabs(["Match Result", "Goals Markets", "Correct Score"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1X2**")
        df = pd.DataFrame([{"Selection": k, "Probability": f"{v*100:.1f}%", "Fair Odds": f"{1/v:.2f}"}
                            for k, v in markets["1X2"].items()])
        st.dataframe(df, hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Double Chance**")
        df = pd.DataFrame([{"Selection": k, "Probability": f"{v*100:.1f}%", "Fair Odds": f"{1/v:.2f}"}
                            for k, v in markets["Double Chance"].items()])
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("**Clean Sheet**")
    df = pd.DataFrame([{"Selection": k, "Probability": f"{v*100:.1f}%"} for k, v in markets["Clean Sheet"].items()])
    st.dataframe(df, hide_index=True, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**BTTS**")
        df = pd.DataFrame([{"Selection": k, "Probability": f"{v*100:.1f}%", "Fair Odds": f"{1/v:.2f}"}
                            for k, v in markets["BTTS"].items()])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown("**Team Totals**")
        df = pd.DataFrame([{"Selection": k, "Probability": f"{v*100:.1f}%"} for k, v in markets["Team Totals"].items()])
        st.dataframe(df, hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Over/Under Total Goals**")
        df = pd.DataFrame([{"Selection": k, "Probability": f"{v*100:.1f}%", "Fair Odds": f"{1/v:.2f}"}
                            for k, v in markets["Over/Under Total Goals"].items()])
        st.dataframe(df, hide_index=True, use_container_width=True)

with tab3:
    df = pd.DataFrame([{"Score": k, "Probability": f"{v*100:.1f}%"} for k, v in markets["Correct Score (top 8)"].items()])
    st.dataframe(df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Value bets vs entered odds
# ---------------------------------------------------------------------------

entered_odds = {
    "Home Win (1)": odds_home, "Draw (X)": odds_draw, "Away Win (2)": odds_away,
    "Over 2.5": odds_over25, "Under 2.5": odds_under25,
    "Yes (GG)": odds_btts_yes, "No (NG)": odds_btts_no,
}
flat_probs = {**markets["1X2"], **markets["Over/Under Total Goals"], **markets["BTTS"]}

value_rows = []
for selection, odds in entered_odds.items():
    if odds <= 1.0:
        continue
    model_prob = flat_probs.get(selection)
    if model_prob is None:
        continue
    market_implied = 1 / odds
    edge_pp = (model_prob - market_implied) * 100
    ev = model_prob * odds - 1
    b = odds - 1
    kelly = max(0.0, (b * model_prob - (1 - model_prob)) / b) if b > 0 else 0
    stake = round(bankroll * kelly * 0.25, 2)
    value_rows.append({
        "Selection": selection, "Model %": f"{model_prob*100:.1f}%",
        "Your Odds": odds, "Market Implied %": f"{market_implied*100:.1f}%",
        "Edge (pp)": f"{edge_pp:+.2f}", "EV per unit": f"{ev:+.3f}",
        "Suggested stake (25% Kelly)": stake,
    })

if value_rows:
    st.divider()
    st.subheader("Value vs the odds you entered")
    vdf = pd.DataFrame(value_rows)
    st.dataframe(vdf, hide_index=True, use_container_width=True)
    st.caption(
        "Positive edge means the model thinks this selection is more likely than the odds imply. "
        "This is not financial advice — the model can be wrong, and odds move fast."
    )
