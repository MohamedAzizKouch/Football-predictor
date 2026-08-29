"""
Injury / squad-availability awareness.

Honest scope, stated up front: there is no free data source that can pull
full injury lists for all 8 leagues every day without hitting a hard rate
limit. API-Football's free tier caps at 100 requests/day (confirmed at
build time). This module is designed around that real constraint rather
than pretending it doesn't exist:

  - Runs DAILY (not weekly like the main model training), but only checks
    2 leagues per day, cycling through all 8 over 4 days — this keeps each
    day's request count comfortably under the free-tier cap.
  - For each of those leagues, it only fetches injuries for teams that have
    a fixture in the next 7 days — not the full squad list for every team,
    which would blow the budget for no benefit (why check a team that isn't
    playing this week?).
  - The adjustment applied to a team's rating is a simple, transparent
    heuristic — NOT a scientifically fitted parameter. It reduces attack
    strength by a fixed percentage per missing key attacking player and
    increases goals-conceded risk per missing key defender, capped so no
    single team's rating can swing more than 25% in either direction. This
    is clearly labeled as a heuristic everywhere it surfaces, including in
    the UI, so it's never confused with the calibrated base model.

If you don't set up an API-Football key, this whole feature quietly stays
off — the base predictions work exactly as before, just without this layer.
"""

import json
import os
from datetime import datetime, timedelta

import requests


API_BASE = "https://v3.football.api-sports.io"

# football-data.co.uk league name -> API-Football league ID (these are
# API-Football's own stable numeric IDs, documented at /leagues)
API_FOOTBALL_LEAGUE_IDS = {
    "Premier League": 39,
    "Championship": 40,
    "League One": 41,
    "Ligue 1": 61,
    "Bundesliga": 78,
    "Serie A": 135,
    "La Liga": 140,
    "Eredivisie": 88,
}

# Cycle 2 leagues per day across a 4-day rotation, keeping each day's request
# count well under the 100/day free-tier cap.
ROTATION = [
    ["Premier League", "Bundesliga"],
    ["La Liga", "Serie A"],
    ["Ligue 1", "Eredivisie"],
    ["Championship", "League One"],
]

MAX_ATTACK_REDUCTION = 0.25   # a team's attack can be reduced by at most 25% from injuries
MAX_DEFENSE_PENALTY = 0.25    # and defense (goals-conceded risk) increased by at most 25%
PER_KEY_PLAYER_IMPACT = 0.08  # each missing "important" player moves the rating ~8%, capped above


def get_todays_leagues():
    """Which 2 leagues to check today, based on a 4-day rotation."""
    day_index = datetime.utcnow().toordinal() % len(ROTATION)
    return ROTATION[day_index]


def fetch_upcoming_fixtures(api_key, league_id, days_ahead=7):
    today = datetime.utcnow().date()
    end = today + timedelta(days=days_ahead)
    try:
        resp = requests.get(
            f"{API_BASE}/fixtures",
            headers={"x-apisports-key": api_key},
            params={"league": league_id, "season": datetime.utcnow().year,
                    "from": str(today), "to": str(end)},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("response", [])
    except Exception as e:
        print(f"    fixtures fetch failed for league {league_id}: {e}")
        return []


def fetch_team_injuries(api_key, team_id, league_id):
    try:
        resp = requests.get(
            f"{API_BASE}/injuries",
            headers={"x-apisports-key": api_key},
            params={"team": team_id, "league": league_id, "season": datetime.utcnow().year},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("response", [])
    except Exception as e:
        print(f"    injuries fetch failed for team {team_id}: {e}")
        return []


def classify_and_score_impact(injury_list):
    """
    Turns API-Football's raw injury response into a simple, honest heuristic
    adjustment. API-Football's injury entries include player position, so we
    use that as a rough (not precise) proxy for whether a missing player is
    an attacking or defensive loss.
    """
    n_attack_out = 0
    n_defense_out = 0
    named = []

    for entry in injury_list:
        player = entry.get("player", {})
        name = player.get("name", "Unknown")
        position = (player.get("type") or player.get("position") or "").lower()
        reason = entry.get("player", {}).get("reason", "injury")
        named.append({"name": name, "position": position, "reason": reason})

        if any(k in position for k in ["attack", "forward", "winger", "midfield"]):
            n_attack_out += 1
        elif any(k in position for k in ["defen", "back", "keeper", "goalkeeper"]):
            n_defense_out += 1
        else:
            n_attack_out += 0.5
            n_defense_out += 0.5

    attack_multiplier = max(1 - MAX_ATTACK_REDUCTION,
                             1 - PER_KEY_PLAYER_IMPACT * n_attack_out)
    defense_multiplier = min(1 + MAX_DEFENSE_PENALTY,
                              1 + PER_KEY_PLAYER_IMPACT * n_defense_out)

    return {
        "players_out": named,
        "attack_multiplier": round(attack_multiplier, 3),
        "defense_multiplier": round(defense_multiplier, 3),
    }


def run_daily_injury_check():
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        print("API_FOOTBALL_KEY not set — skipping injury check (this is optional, see SETUP.md).")
        return {}

    leagues_today = get_todays_leagues()
    print(f"Today's injury check covers: {leagues_today}")

    # load existing file so leagues not checked today keep yesterday's data
    existing = {}
    if os.path.exists("injuries.json"):
        with open("injuries.json") as f:
            existing = json.load(f)

    request_count = 0

    for league_name in leagues_today:
        league_id = API_FOOTBALL_LEAGUE_IDS.get(league_name)
        if league_id is None:
            continue

        print(f"  {league_name} (id {league_id})...")
        fixtures = fetch_upcoming_fixtures(api_key, league_id)
        request_count += 1

        team_ids_this_week = {}
        for fx in fixtures:
            for side in ["home", "away"]:
                team = fx["teams"][side]
                team_ids_this_week[team["name"]] = team["id"]

        print(f"    {len(team_ids_this_week)} teams playing in the next 7 days")

        league_result = existing.get(league_name, {})
        for team_name, team_id in team_ids_this_week.items():
            if request_count >= 90:  # safety margin under the 100/day cap
                print("    Approaching daily request limit — stopping early for today.")
                break
            injuries = fetch_team_injuries(api_key, team_id, league_id)
            request_count += 1
            impact = classify_and_score_impact(injuries)
            impact["last_checked"] = str(datetime.utcnow().date())
            league_result[team_name] = impact

        existing[league_name] = league_result

    print(f"Used {request_count} API requests today.")
    return existing


if __name__ == "__main__":
    result = run_daily_injury_check()
    with open("injuries.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote injuries.json covering {len(result)} leagues.")
