from pathlib import Path
import base64
import json
import math
import os
import urllib.error
import urllib.request
import warnings
import zlib

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
RESULTS_COLUMNS = ["match_id", "actual_home_goals", "actual_away_goals", "status", "notes"]
DEFAULT_RESULTS_API_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
    "scoreboard?dates=20260611-20260719&limit=200"
)
RESULTS_CACHE_TTL_SECONDS = 300
MATCH_MONTE_CARLO_RUNS = 40000
GROUP_MONTE_CARLO_RUNS = 12000
MODEL_RANDOM_STATE = 42

TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curacao",
    "Germany FR": "Germany",
    "IR Iran": "Iran",
    "rn\">United Arab Emirates": "United Arab Emirates",
    "rn\">Republic of Ireland": "Republic of Ireland",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "rn\">Trinidad and Tobago": "Trinidad and Tobago",
    "rn\">Serbia and Montenegro": "Serbia and Montenegro",
    "rn\">Bosnia and Herzegovina": "Bosnia and Herzegovina",
}

FLAG_BY_TEAM = {
    "Algeria": "ðŸ‡©ðŸ‡¿",
    "Argentina": "ðŸ‡¦ðŸ‡·",
    "Australia": "ðŸ‡¦ðŸ‡º",
    "Austria": "ðŸ‡¦ðŸ‡¹",
    "Belgium": "ðŸ‡§ðŸ‡ª",
    "Bosnia and Herzegovina": "ðŸ‡§ðŸ‡¦",
    "Brazil": "ðŸ‡§ðŸ‡·",
    "Canada": "ðŸ‡¨ðŸ‡¦",
    "Cape Verde": "ðŸ‡¨ðŸ‡»",
    "Colombia": "ðŸ‡¨ðŸ‡´",
    "Croatia": "ðŸ‡­ðŸ‡·",
    "Curacao": "ðŸ‡¨ðŸ‡¼",
    "Czech Republic": "ðŸ‡¨ðŸ‡¿",
    "DR Congo": "ðŸ‡¨ðŸ‡©",
    "Ecuador": "ðŸ‡ªðŸ‡¨",
    "Egypt": "ðŸ‡ªðŸ‡¬",
    "England": "ðŸ´",
    "France": "ðŸ‡«ðŸ‡·",
    "Germany": "ðŸ‡©ðŸ‡ª",
    "Ghana": "ðŸ‡¬ðŸ‡­",
    "Haiti": "ðŸ‡­ðŸ‡¹",
    "Iran": "ðŸ‡®ðŸ‡·",
    "Iraq": "ðŸ‡®ðŸ‡¶",
    "Ivory Coast": "ðŸ‡¨ðŸ‡®",
    "Japan": "ðŸ‡¯ðŸ‡µ",
    "Jordan": "ðŸ‡¯ðŸ‡´",
    "Mexico": "ðŸ‡²ðŸ‡½",
    "Morocco": "ðŸ‡²ðŸ‡¦",
    "Netherlands": "ðŸ‡³ðŸ‡±",
    "New Zealand": "ðŸ‡³ðŸ‡¿",
    "Norway": "ðŸ‡³ðŸ‡´",
    "Panama": "ðŸ‡µðŸ‡¦",
    "Paraguay": "ðŸ‡µðŸ‡¾",
    "Portugal": "ðŸ‡µðŸ‡¹",
    "Qatar": "ðŸ‡¶ðŸ‡¦",
    "Saudi Arabia": "ðŸ‡¸ðŸ‡¦",
    "Scotland": "ðŸ´",
    "Senegal": "ðŸ‡¸ðŸ‡³",
    "South Africa": "ðŸ‡¿ðŸ‡¦",
    "South Korea": "ðŸ‡°ðŸ‡·",
    "Spain": "ðŸ‡ªðŸ‡¸",
    "Sweden": "ðŸ‡¸ðŸ‡ª",
    "Switzerland": "ðŸ‡¨ðŸ‡­",
    "Tunisia": "ðŸ‡¹ðŸ‡³",
    "Turkey": "ðŸ‡¹ðŸ‡·",
    "United States": "ðŸ‡ºðŸ‡¸",
    "Uruguay": "ðŸ‡ºðŸ‡¾",
    "Uzbekistan": "ðŸ‡ºðŸ‡¿",
}

RESULT_COLORS = {
    "Alta": "#166534",
    "Media": "#92400e",
    "Baixa": "#991b1b",
}

COUNTRY_CODE_BY_TEAM = {
    "Algeria": "DZ",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Bosnia and Herzegovina": "BA",
    "Brazil": "BR",
    "Canada": "CA",
    "Cape Verde": "CV",
    "Colombia": "CO",
    "Croatia": "HR",
    "Curacao": "CW",
    "Czech Republic": "CZ",
    "DR Congo": "CD",
    "Ecuador": "EC",
    "Egypt": "EG",
    "England": "GB",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "Haiti": "HT",
    "Iran": "IR",
    "Iraq": "IQ",
    "Ivory Coast": "CI",
    "Japan": "JP",
    "Jordan": "JO",
    "Mexico": "MX",
    "Morocco": "MA",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Norway": "NO",
    "Panama": "PA",
    "Paraguay": "PY",
    "Portugal": "PT",
    "Qatar": "QA",
    "Saudi Arabia": "SA",
    "Scotland": "GB",
    "Senegal": "SN",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "United States": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
}


@st.cache_data(ttl=RESULTS_CACHE_TTL_SECONDS)
def load_data():
    matches = pd.read_csv(BASE_DIR / "Jogos Copas do Mundo.csv", encoding="cp1252")
    champions = pd.read_csv(BASE_DIR / "Campeoes.csv")
    teams_2026 = pd.read_csv(DATA_DIR / "world_cup_2026_teams.csv")
    fixtures = pd.read_csv(DATA_DIR / "world_cup_2026_group_stage.csv")
    results, results_source = load_results(fixtures)
    return matches, champions, teams_2026, fixtures, results, results_source


def get_config_value(name, default=None):
    value = os.environ.get(name)
    if value:
        return value

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        return default

    return default


def normalize_results(results):
    normalized = results.copy()
    for column in RESULTS_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan

    normalized = normalized[RESULTS_COLUMNS]
    normalized["match_id"] = pd.to_numeric(normalized["match_id"], errors="coerce")
    normalized = normalized.dropna(subset=["match_id"])
    normalized["match_id"] = normalized["match_id"].astype(int)
    normalized["actual_home_goals"] = pd.to_numeric(normalized["actual_home_goals"], errors="coerce")
    normalized["actual_away_goals"] = pd.to_numeric(normalized["actual_away_goals"], errors="coerce")
    normalized["status"] = normalized["status"].fillna("pendente")
    normalized["notes"] = normalized["notes"].fillna("")
    return normalized


def first_available(record, keys):
    for key in keys:
        if key in record and record[key] not in [None, ""]:
            return record[key]
    return None


def nested_score(record, side):
    for key in ["score", "goals"]:
        value = record.get(key)
        if isinstance(value, dict) and value.get(side) not in [None, ""]:
            return value.get(side)
        if isinstance(value, dict) and isinstance(value.get("ft"), list) and len(value["ft"]) >= 2:
            return value["ft"][0] if side == "home" else value["ft"][1]
    return None


def api_team_name(value):
    if isinstance(value, dict):
        return first_available(value, ["name", "label", "country", "team", "title"])
    return value


def team_match_key(value):
    if value in [None, ""]:
        return ""

    normalized = normalize_team(str(value))
    return "".join(char.lower() for char in normalized if char.isalnum())


def match_id_from_api_record(record, index, fixtures):
    match_id = first_available(record, ["match_id", "id", "fixture_id", "game_id", "match_number", "num"])
    if fixtures is None or fixtures.empty:
        return match_id

    valid_match_ids = set(fixtures["match_id"].astype(int))
    try:
        numeric_match_id = int(match_id)
    except (TypeError, ValueError):
        numeric_match_id = None
    if numeric_match_id in valid_match_ids:
        return numeric_match_id

    home_team = api_team_name(first_available(record, ["home_team", "homeTeam", "home", "team1"]))
    away_team = api_team_name(first_available(record, ["away_team", "awayTeam", "away", "team2"]))
    match_date = first_available(record, ["date", "match_date", "kickoff_date"])

    if home_team and away_team:
        candidates = fixtures[
            (fixtures["home_team"].map(team_match_key) == team_match_key(home_team))
            & (fixtures["away_team"].map(team_match_key) == team_match_key(away_team))
        ]
        if match_date:
            dated_candidates = candidates[candidates["date"].astype(str) == str(match_date)[:10]]
            if not dated_candidates.empty:
                return dated_candidates.iloc[0]["match_id"]
        if not candidates.empty:
            return candidates.iloc[0]["match_id"]

    return None


def records_from_espn_events(events):
    records = []
    for event in events:
        if not isinstance(event, dict):
            continue

        competitions = event.get("competitions", [])
        if not competitions or not isinstance(competitions[0], dict):
            continue

        competitors = competitions[0].get("competitors", [])
        home = next((team for team in competitors if team.get("homeAway") == "home"), None)
        away = next((team for team in competitors if team.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        status_type = event.get("status", {}).get("type", {})
        completed = bool(status_type.get("completed"))
        home_team = home.get("team", {})
        away_team = away.get("team", {})
        record = {
            "external_id": event.get("id"),
            "date": event.get("date"),
            "home_team": first_available(home_team, ["displayName", "shortDisplayName", "name"]),
            "away_team": first_available(away_team, ["displayName", "shortDisplayName", "name"]),
            "status": "finalizado" if completed else "pendente",
            "notes": f"ESPN | {status_type.get('description', '')}".strip(" |"),
        }
        if completed:
            record["actual_home_goals"] = home.get("score")
            record["actual_away_goals"] = away.get("score")
        records.append(record)

    return records


def records_from_payload(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            return records_from_espn_events(events)

        rounds = payload.get("rounds")
        if isinstance(rounds, list):
            records = []
            for round_data in rounds:
                if not isinstance(round_data, dict):
                    continue
                matches = round_data.get("matches", [])
                if not isinstance(matches, list):
                    continue
                for match in matches:
                    if isinstance(match, dict):
                        match = match.copy()
                        match["round"] = round_data.get("name", "")
                        records.append(match)
            return records

        for key in ["results", "matches", "fixtures", "data", "response"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

    return []


def normalize_api_records(records, fixtures=None):
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        match_id = match_id_from_api_record(record, index, fixtures)
        home_goals = first_available(record, ["actual_home_goals", "home_goals", "home_score", "score_home", "homeTeamScore", "goals_home", "score1"])
        away_goals = first_available(record, ["actual_away_goals", "away_goals", "away_score", "score_away", "awayTeamScore", "goals_away", "score2"])

        if home_goals is None:
            home_goals = nested_score(record, "home")
        if away_goals is None:
            away_goals = nested_score(record, "away")

        status = first_available(record, ["status", "match_status", "state"])
        if not status:
            status = "finalizado" if home_goals is not None and away_goals is not None else "pendente"

        rows.append(
            {
                "match_id": match_id,
                "actual_home_goals": home_goals,
                "actual_away_goals": away_goals,
                "status": status,
                "notes": first_available(record, ["notes", "note", "source"]) or "",
            }
        )

    if not rows:
        return pd.DataFrame(columns=RESULTS_COLUMNS)

    return normalize_results(pd.DataFrame(rows))


def fetch_results_from_api(fixtures=None):
    url = get_config_value("RESULTS_API_URL", DEFAULT_RESULTS_API_URL)
    if not url:
        return pd.DataFrame(columns=RESULTS_COLUMNS), "API não configurada"

    api_key = get_config_value("RESULTS_API_KEY")
    auth_header = get_config_value("RESULTS_API_AUTH_HEADER", "Authorization")
    auth_prefix = get_config_value("RESULTS_API_AUTH_PREFIX", "Bearer")
    headers = {
        "Accept": "application/json",
        "User-Agent": "CopaDosDados/2026",
    }
    if api_key:
        headers[auth_header] = f"{auth_prefix} {api_key}".strip()

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=RESULTS_COLUMNS), "API de resultados indisponível"

    api_results = normalize_api_records(records_from_payload(payload), fixtures)
    if api_results.empty:
        return pd.DataFrame(columns=RESULTS_COLUMNS), "ESPN - Copa do Mundo 2026"

    source = "ESPN - Copa do Mundo 2026" if url == DEFAULT_RESULTS_API_URL else "API"
    return api_results, source


def load_results(fixtures=None):
    return fetch_results_from_api(fixtures)


def normalize_team(team):
    if pd.isna(team):
        return team
    team = str(team).strip()
    return TEAM_ALIASES.get(team, team)


MODEL_FEATURES = [
    "points_per_game",
    "goals_for_per_game",
    "goals_against_per_game",
    "win_rate",
    "draw_rate",
    "recent_points_per_game",
    "recent_goals_for",
    "recent_goals_against",
    "experience",
]


def empty_team_state():
    return {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "points": 0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "recent_points": [],
        "recent_goals_for": [],
        "recent_goals_against": [],
    }


def team_state_features(state):
    games = max(int(state["games"]), 1)
    recent_games = max(len(state["recent_points"]), 1)
    return np.array(
        [
            state["points"] / games if state["games"] else 1.0,
            state["goals_for"] / games if state["games"] else 1.25,
            state["goals_against"] / games if state["games"] else 1.25,
            state["wins"] / games if state["games"] else 0.30,
            state["draws"] / games if state["games"] else 0.25,
            sum(state["recent_points"]) / recent_games if state["recent_points"] else 1.0,
            sum(state["recent_goals_for"]) / recent_games if state["recent_goals_for"] else 1.25,
            sum(state["recent_goals_against"]) / recent_games if state["recent_goals_against"] else 1.25,
            np.log1p(state["games"]),
        ],
        dtype=float,
    )


def match_feature_vector(home_features, away_features):
    difference = home_features - away_features
    return np.concatenate([home_features, away_features, difference])


def update_team_state(state, goals_for, goals_against):
    state["games"] += 1
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    if goals_for > goals_against:
        points = 3
        state["wins"] += 1
    elif goals_for == goals_against:
        points = 1
        state["draws"] += 1
    else:
        points = 0
    state["points"] += points
    state["recent_points"] = (state["recent_points"] + [points])[-5:]
    state["recent_goals_for"] = (state["recent_goals_for"] + [goals_for])[-5:]
    state["recent_goals_against"] = (state["recent_goals_against"] + [goals_against])[-5:]


def prepare_neural_training_data(matches):
    ordered = matches.copy()
    ordered["_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["Ano", "_order"])
    states = {}
    features = []
    targets = []

    for _, match in ordered.iterrows():
        home = normalize_team(match["TimeDaCasa"])
        away = normalize_team(match["TimeVisitante"])
        home_state = states.setdefault(home, empty_team_state())
        away_state = states.setdefault(away, empty_team_state())
        features.append(match_feature_vector(team_state_features(home_state), team_state_features(away_state)))

        home_goals = float(match["GolsTimeDaCasa"])
        away_goals = float(match["GolsTimeVisitante"])
        targets.append(np.log1p([home_goals, away_goals]))
        update_team_state(home_state, home_goals, away_goals)
        update_team_state(away_state, away_goals, home_goals)

    return np.asarray(features), np.asarray(targets), states


def fit_neural_goal_model(features, targets):
    split = max(int(len(features) * 0.85), 1)
    scaler = StandardScaler().fit(features[:split])
    train_x = scaler.transform(features[:split])
    validation_x = scaler.transform(features[split:])

    model_settings = {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.015,
        "learning_rate_init": 0.003,
        "max_iter": 1200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 45,
        "random_state": MODEL_RANDOM_STATE,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        validation_model = MLPRegressor(**model_settings).fit(train_x, targets[:split])

    validation_predictions = np.clip(np.expm1(validation_model.predict(validation_x)), 0, 6)
    validation_actual = np.expm1(targets[split:])
    mae = float(np.mean(np.abs(validation_predictions - validation_actual))) if len(validation_actual) else np.nan

    final_scaler = StandardScaler().fit(features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model = MLPRegressor(**model_settings).fit(final_scaler.transform(features), targets)
    return final_model, final_scaler, mae


@st.cache_resource
def build_team_ratings(matches, champions, teams_2026):
    normalized_matches = matches.copy()
    normalized_matches["TimeDaCasa"] = normalized_matches["TimeDaCasa"].map(normalize_team)
    normalized_matches["TimeVisitante"] = normalized_matches["TimeVisitante"].map(normalize_team)
    features, targets, historical_states = prepare_neural_training_data(normalized_matches)
    neural_model, feature_scaler, validation_mae = fit_neural_goal_model(features, targets)

    champion_counts = champions["Vencedor"].map(normalize_team).value_counts()
    runner_up_counts = champions["Segundo"].map(normalize_team).value_counts()
    rows = []
    for team in sorted(teams_2026["team"].unique()):
        state = historical_states.get(team, empty_team_state())
        values = team_state_features(state)
        titles = int(champion_counts.get(team, 0))
        finals = int(titles + runner_up_counts.get(team, 0))
        rating = 50 + values[0] * 16 + (values[1] - values[2]) * 9 + values[5] * 5 + np.log1p(titles * 3 + finals) * 6
        row = {
            "team": team,
            "played": int(state["games"]),
            "wins": int(state["wins"]),
            "draws": int(state["draws"]),
            "goals_for": float(state["goals_for"]),
            "goals_against": float(state["goals_against"]),
            "goals_for_per_match": round(float(values[1]), 2),
            "goals_against_per_match": round(float(values[2]), 2),
            "titles": titles,
            "finals": finals,
            "rating": round(float(rating), 2),
        }
        row.update(dict(zip(MODEL_FEATURES, values)))
        rows.append(row)

    ratings = pd.DataFrame(rows).merge(teams_2026, on="team", how="left")
    numeric_features = MODEL_FEATURES + [
        "goals_for_per_match",
        "goals_against_per_match",
        "rating",
    ]
    for column in numeric_features:
        confederation_median = ratings.groupby("confederation")[column].transform("median")
        global_median = ratings[column].replace(0, np.nan).median()
        ratings[column] = ratings[column].where(ratings["played"] > 0, confederation_median)
        ratings[column] = ratings[column].fillna(global_median)

    ratings.attrs["neural_model"] = neural_model
    ratings.attrs["feature_scaler"] = feature_scaler
    ratings.attrs["validation_mae"] = validation_mae
    ratings.attrs["model_name"] = "Rede neural MLP + Monte Carlo"
    ratings.attrs["simulation_cache"] = {}
    return ratings.sort_values("rating", ascending=False)


def model_team_features(team, ratings):
    row = ratings.loc[ratings["team"] == team]
    if row.empty:
        return np.array([1.0, 1.25, 1.25, 0.30, 0.25, 1.0, 1.25, 1.25, 0.0])
    return row.iloc[0][MODEL_FEATURES].astype(float).to_numpy()


def estimate_goals(team_a, team_b, ratings):
    cache = ratings.attrs.setdefault("simulation_cache", {})
    cache_key = ("goals", team_a, team_b)
    if cache_key in cache:
        return cache[cache_key]

    feature_vector = match_feature_vector(
        model_team_features(team_a, ratings),
        model_team_features(team_b, ratings),
    ).reshape(1, -1)
    model = ratings.attrs["neural_model"]
    scaler = ratings.attrs["feature_scaler"]
    prediction = np.expm1(model.predict(scaler.transform(feature_vector))[0])
    expected_home = float(np.clip(prediction[0], 0.15, 4.2))
    expected_away = float(np.clip(prediction[1], 0.15, 4.2))
    cache[cache_key] = (expected_home, expected_away)
    return expected_home, expected_away


def monte_carlo_scores(expected_home, expected_away, simulations=MATCH_MONTE_CARLO_RUNS):
    seed_text = f"{expected_home:.6f}|{expected_away:.6f}|{simulations}"
    rng = np.random.default_rng(zlib.crc32(seed_text.encode("utf-8")))
    home_goals = rng.poisson(expected_home, simulations)
    away_goals = rng.poisson(expected_away, simulations)
    return home_goals, away_goals


def match_probabilities(team_a, team_b, ratings, neutral_factor=0.08):
    expected_home, expected_away = estimate_goals(team_a, team_b, ratings)
    home_goals, away_goals = monte_carlo_scores(expected_home, expected_away)
    return {
        "home": float(np.mean(home_goals > away_goals)),
        "draw": float(np.mean(home_goals == away_goals)),
        "away": float(np.mean(home_goals < away_goals)),
    }


def goal_markets(expected_home, expected_away):
    home_goals, away_goals = monte_carlo_scores(expected_home, expected_away)
    total_goals = home_goals + away_goals
    score_counts = (
        pd.DataFrame({"home_goals": home_goals, "away_goals": away_goals})
        .value_counts()
        .rename("count")
        .reset_index()
    )
    score_counts["probability"] = score_counts["count"] / len(home_goals)
    score_counts["total_goals"] = score_counts["home_goals"] + score_counts["away_goals"]
    return {
        "over_1_5": float(np.mean(total_goals > 1.5)),
        "over_2_5": float(np.mean(total_goals > 2.5)),
        "over_3_5": float(np.mean(total_goals > 3.5)),
        "both_score": float(np.mean((home_goals > 0) & (away_goals > 0))),
        "scorelines": score_counts.sort_values("probability", ascending=False).head(6),
    }


def team_label(team):
    return f"{FLAG_BY_TEAM.get(team, 'ðŸ³ï¸')} {team}"


def confidence_label(probability):
    if probability >= 0.58:
        return "Alta"
    if probability >= 0.46:
        return "Media"
    return "Baixa"


def pick_for_pool(home, away, probabilities, goals):
    outcomes = {
        home: probabilities["home"],
        "Empate": probabilities["draw"],
        away: probabilities["away"],
    }
    pick = max(outcomes, key=outcomes.get)
    confidence = confidence_label(outcomes[pick])
    best_score = goals["scorelines"].iloc[0]
    score = f"{int(best_score['home_goals'])} x {int(best_score['away_goals'])}"
    return pick, outcomes[pick], confidence, score


def simulate_group_table(group, fixtures, ratings):
    group_fixtures = fixtures[fixtures["group"] == group]
    teams = sorted(set(group_fixtures["home_team"]) | set(group_fixtures["away_team"]))
    team_index = {team: index for index, team in enumerate(teams)}
    points = np.zeros((GROUP_MONTE_CARLO_RUNS, len(teams)), dtype=float)
    goals_for = np.zeros_like(points)
    goals_against = np.zeros_like(points)
    rng = np.random.default_rng(zlib.crc32(f"group-{group}".encode("utf-8")))

    for _, match in group_fixtures.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        expected_home, expected_away = estimate_goals(home, away, ratings)
        simulated_home = rng.poisson(expected_home, GROUP_MONTE_CARLO_RUNS)
        simulated_away = rng.poisson(expected_away, GROUP_MONTE_CARLO_RUNS)
        home_index = team_index[home]
        away_index = team_index[away]

        goals_for[:, home_index] += simulated_home
        goals_against[:, home_index] += simulated_away
        goals_for[:, away_index] += simulated_away
        goals_against[:, away_index] += simulated_home
        points[:, home_index] += np.where(simulated_home > simulated_away, 3, np.where(simulated_home == simulated_away, 1, 0))
        points[:, away_index] += np.where(simulated_away > simulated_home, 3, np.where(simulated_home == simulated_away, 1, 0))

    position_counts = np.zeros((len(teams), len(teams)), dtype=int)
    goal_difference = goals_for - goals_against
    for simulation in range(GROUP_MONTE_CARLO_RUNS):
        order = np.lexsort(
            (
                -goals_for[simulation],
                -goal_difference[simulation],
                -points[simulation],
            )
        )
        for position, team_position in enumerate(order):
            position_counts[team_position, position] += 1

    rows = []
    for team, index in team_index.items():
        rows.append(
            {
                "Grupo": group,
                "team": team,
                "PJ": 3,
                "Pts. esp.": round(float(points[:, index].mean()), 2),
                "GP esp.": round(float(goals_for[:, index].mean()), 2),
                "GC esp.": round(float(goals_against[:, index].mean()), 2),
                "SG esp.": round(float(goal_difference[:, index].mean()), 2),
                "Prob. 1º": position_counts[index, 0] / GROUP_MONTE_CARLO_RUNS,
                "Prob. top 2": position_counts[index, :2].sum() / GROUP_MONTE_CARLO_RUNS,
            }
        )
    projection = pd.DataFrame(rows)
    projection = projection.sort_values(["Pts. esp.", "SG esp.", "GP esp."], ascending=False).reset_index(drop=True)
    projection["Posição"] = projection.index + 1
    return projection


def projected_group_table(group, fixtures, ratings):
    projection = simulate_group_table(group, fixtures, ratings).copy()
    projection["Seleção"] = projection["team"].map(team_label)
    projection["Prob. 1º"] = projection["Prob. 1º"].map(pct)
    projection["Prob. top 2"] = projection["Prob. top 2"].map(pct)
    return projection[
        ["Seleção", "PJ", "Pts. esp.", "GP esp.", "GC esp.", "SG esp.", "Prob. 1º", "Prob. top 2"]
    ]


def projected_group_table_raw(group, fixtures, ratings):
    return simulate_group_table(group, fixtures, ratings)


def projected_group_positions(fixtures, ratings):
    groups = sorted(fixtures["group"].unique())
    rankings = pd.concat(
        [projected_group_table_raw(group, fixtures, ratings) for group in groups],
        ignore_index=True,
    )

    positions = {}
    for _, row in rankings.iterrows():
        positions[(row["Grupo"], int(row["Posição"]))] = row.to_dict()

    third_pool = rankings[rankings["Posição"] == 3].sort_values(
        ["Pts. esp.", "SG esp.", "GP esp."],
        ascending=False,
    )
    qualified_thirds = third_pool.head(8).copy()
    return rankings, positions, qualified_thirds


ROUND_OF_32_TEMPLATE = [
    (73, ("2", "A"), ("2", "B")),
    (74, ("1", "E"), ("3", ["A", "B", "C", "D", "F"])),
    (75, ("1", "F"), ("2", "C")),
    (76, ("1", "C"), ("2", "F")),
    (77, ("1", "I"), ("3", ["C", "D", "F", "G", "H"])),
    (78, ("2", "E"), ("2", "I")),
    (79, ("1", "A"), ("3", ["C", "E", "F", "H", "I"])),
    (80, ("1", "L"), ("3", ["E", "H", "I", "J", "K"])),
    (81, ("1", "D"), ("3", ["B", "E", "F", "I", "J"])),
    (82, ("1", "G"), ("3", ["A", "E", "H", "I", "J"])),
    (83, ("2", "K"), ("2", "L")),
    (84, ("1", "H"), ("2", "J")),
    (85, ("1", "B"), ("3", ["E", "F", "G", "I", "J"])),
    (86, ("1", "J"), ("2", "H")),
    (87, ("1", "K"), ("3", ["D", "E", "I", "J", "L"])),
    (88, ("2", "D"), ("2", "G")),
]

KNOCKOUT_FLOW = [
    ("Oitavas de final", 89, 73, 75),
    ("Oitavas de final", 90, 74, 77),
    ("Oitavas de final", 91, 76, 78),
    ("Oitavas de final", 92, 79, 80),
    ("Oitavas de final", 93, 83, 84),
    ("Oitavas de final", 94, 81, 82),
    ("Oitavas de final", 95, 86, 88),
    ("Oitavas de final", 96, 85, 87),
    ("Quartas de final", 97, 89, 90),
    ("Quartas de final", 98, 93, 94),
    ("Quartas de final", 99, 91, 92),
    ("Quartas de final", 100, 95, 96),
    ("Semifinal", 101, 97, 98),
    ("Semifinal", 102, 99, 100),
    ("Final", 104, 101, 102),
]


def knockout_slot_team(slot, positions, qualified_thirds, used_thirds):
    position, group_ref = slot
    if position in ["1", "2"]:
        row = positions.get((group_ref, int(position)))
        label = f"{position}º Grupo {group_ref}"
        return row["team"], label

    candidates = group_ref
    available = qualified_thirds[
        qualified_thirds["Grupo"].isin(candidates) & ~qualified_thirds["Grupo"].isin(used_thirds)
    ]
    if available.empty:
        available = qualified_thirds[~qualified_thirds["Grupo"].isin(used_thirds)]
    if available.empty:
        return "A definir", f"3º melhor ({'/'.join(candidates)})"

    row = available.iloc[0]
    used_thirds.add(row["Grupo"])
    return row["team"], f"3º Grupo {row['Grupo']}"


def predicted_knockout_winner(team_a, team_b, ratings):
    if "A definir" in [team_a, team_b]:
        return "A definir", np.nan, "-"

    probabilities = match_probabilities(team_a, team_b, ratings)
    home_share = probabilities["home"] / (probabilities["home"] + probabilities["away"])
    winner = team_a if probabilities["home"] >= probabilities["away"] else team_b
    winner_probability = home_share if winner == team_a else 1 - home_share
    expected_home, expected_away = estimate_goals(team_a, team_b, ratings)
    goals = goal_markets(expected_home, expected_away)
    best_score = goals["scorelines"].iloc[0]
    score = f"{int(best_score['home_goals'])} x {int(best_score['away_goals'])}"
    return winner, winner_probability, score


def build_knockout_projection(fixtures, ratings):
    rankings, positions, qualified_thirds = projected_group_positions(fixtures, ratings)
    used_thirds = set()
    winners = {}
    rows = []

    def add_match(round_name, match_number, team_a, team_b, source_a, source_b):
        winner, winner_probability, score = predicted_knockout_winner(team_a, team_b, ratings)
        winners[match_number] = winner
        rows.append(
            {
                "Fase": round_name,
                "Jogo": f"Jogo {match_number}",
                "Seleção A": team_label(team_a) if team_a != "A definir" else team_a,
                "Origem A": source_a,
                "Seleção B": team_label(team_b) if team_b != "A definir" else team_b,
                "Origem B": source_b,
                "Placar provável": score,
                "Classificado projetado": team_label(winner) if winner != "A definir" else winner,
                "Prob. de classificação": pct(winner_probability).replace(".", ",") if not pd.isna(winner_probability) else "-",
            }
        )

    for match_number, slot_a, slot_b in ROUND_OF_32_TEMPLATE:
        team_a, source_a = knockout_slot_team(slot_a, positions, qualified_thirds, used_thirds)
        team_b, source_b = knockout_slot_team(slot_b, positions, qualified_thirds, used_thirds)
        add_match("Fase de 32", match_number, team_a, team_b, source_a, source_b)

    for round_name, match_number, source_match_a, source_match_b in KNOCKOUT_FLOW:
        add_match(
            round_name,
            match_number,
            winners.get(source_match_a, "A definir"),
            winners.get(source_match_b, "A definir"),
            f"Vencedor jogo {source_match_a}",
            f"Vencedor jogo {source_match_b}",
        )

    return rankings, qualified_thirds, pd.DataFrame(rows)


def result_from_goals(home_goals, away_goals, home_team, away_team):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return home_team
    if away_goals > home_goals:
        return away_team
    return "Empate"


def yes_no(value):
    return "Sim" if value else "Não"


def result_from_scoreline(score, home_team, away_team):
    try:
        home_goals, away_goals = [int(value.strip()) for value in score.split("x", maxsplit=1)]
    except (AttributeError, TypeError, ValueError):
        return None
    return result_from_goals(home_goals, away_goals, home_team, away_team)


def build_tracking_table(fixtures, results, ratings):
    tracking = fixtures.merge(results, on="match_id", how="left")
    tracking["actual_home_goals"] = pd.to_numeric(tracking["actual_home_goals"], errors="coerce")
    tracking["actual_away_goals"] = pd.to_numeric(tracking["actual_away_goals"], errors="coerce")

    rows = []
    for _, match in tracking.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        probabilities = match_probabilities(home, away, ratings)
        expected_home, expected_away = estimate_goals(home, away, ratings)
        goals = goal_markets(expected_home, expected_away)
        pick, pick_probability, confidence, score = pick_for_pool(home, away, probabilities, goals)

        actual_home = match["actual_home_goals"]
        actual_away = match["actual_away_goals"]
        played = not pd.isna(actual_home) and not pd.isna(actual_away)
        actual_result = result_from_goals(actual_home, actual_away, home, away) if played else None
        actual_score = f"{int(actual_home)} x {int(actual_away)}" if played else "Pendente"
        predicted_result = result_from_scoreline(score, home, away)

        predicted_over_2_5 = goals["over_2_5"] >= 0.5
        actual_over_2_5 = (actual_home + actual_away) > 2.5 if played else None
        predicted_both_score = goals["both_score"] >= 0.5
        actual_both_score = (actual_home > 0 and actual_away > 0) if played else None
        hit_result = predicted_result == actual_result if played else None
        hit_score = score == actual_score if played else None
        hit_goals = predicted_over_2_5 == actual_over_2_5 if played else None
        hit_both = predicted_both_score == actual_both_score if played else None

        rows.append(
            {
                "Data": match["date"],
                "Grupo": match["group"],
                "Jogo": f"{team_label(home)} x {team_label(away)}",
                "Resultado previsto": team_label(predicted_result) if predicted_result != "Empate" else "Empate",
                "Palpite para bolão": team_label(pick) if pick != "Empate" else "Empate",
                "Confiança": confidence,
                "Prob. do palpite": pct(pick_probability),
                "Placar previsto": score,
                "Resultado real": team_label(actual_result) if actual_result not in [None, "Empate"] else actual_result or "Pendente",
                "Placar real": actual_score,
                "Acertou resultado": yes_no(hit_result) if played else "Pendente",
                "Acertou placar": yes_no(hit_score) if played else "Pendente",
                "Prev. acima de 2,5": yes_no(predicted_over_2_5),
                "Real acima de 2,5": yes_no(actual_over_2_5) if played else "Pendente",
                "Acertou gols": yes_no(hit_goals) if played else "Pendente",
                "Prev. ambos marcam": yes_no(predicted_both_score),
                "Real ambos marcam": yes_no(actual_both_score) if played else "Pendente",
                "Acertou ambos": yes_no(hit_both) if played else "Pendente",
                "_hit_result": hit_result,
                "_hit_score": hit_score,
                "_hit_goals": hit_goals,
                "_hit_both": hit_both,
                "_confidence": confidence,
                "_pick_probability": pick_probability,
                "Status": "Finalizado" if played else "Pendente",
            }
        )

    return pd.DataFrame(rows)


def match_result_label(goals_for, goals_against):
    if goals_for > goals_against:
        return "Vitória"
    if goals_for < goals_against:
        return "Derrota"
    return "Empate"


def head_to_head_results(matches, home, away):
    home_norm = normalize_team(home)
    away_norm = normalize_team(away)
    history = matches.copy()
    history["home_norm"] = history["TimeDaCasa"].map(normalize_team)
    history["away_norm"] = history["TimeVisitante"].map(normalize_team)
    direct = history[
        ((history["home_norm"] == home_norm) & (history["away_norm"] == away_norm))
        | ((history["home_norm"] == away_norm) & (history["away_norm"] == home_norm))
    ].copy()
    if direct.empty:
        return pd.DataFrame()

    direct["_order"] = direct.index
    direct = direct.sort_values(["Ano", "_order"], ascending=[False, False])
    rows = []
    for _, match in direct.iterrows():
        home_goals = int(match["GolsTimeDaCasa"])
        away_goals = int(match["GolsTimeVisitante"])
        winner = result_from_goals(home_goals, away_goals, match["home_norm"], match["away_norm"])
        rows.append(
            {
                "Ano": int(match["Ano"]),
                "Data": str(match["Data"]).strip(),
                "Fase": match["Fase"],
                "Jogo": f"{team_label(match['home_norm'])} x {team_label(match['away_norm'])}",
                "Placar": f"{home_goals} x {away_goals}",
                "Vencedor": team_label(winner) if winner != "Empate" else "Empate",
            }
        )
    return pd.DataFrame(rows)


def render_recent_results_page(matches, fixtures):
    st.title("Confrontos diretos")
    st.write(
        "Veja se as seleções que se enfrentarão na fase de grupos já jogaram entre si em Copas, quando foi, qual foi o placar e quem venceu."
    )

    group_options = ["Todos"] + sorted(fixtures["group"].unique())
    selected_group = st.selectbox("Grupo", group_options, key="recent_group")
    filtered_fixtures = fixtures.copy()
    if selected_group != "Todos":
        filtered_fixtures = filtered_fixtures[filtered_fixtures["group"] == selected_group]

    fixture_labels = (
        filtered_fixtures["date"]
        + " | Grupo "
        + filtered_fixtures["group"]
        + " | "
        + filtered_fixtures["home_team"].map(team_label)
        + " x "
        + filtered_fixtures["away_team"].map(team_label)
    )
    selected_label = st.selectbox("Jogo", fixture_labels, key="recent_match")
    selected_match = filtered_fixtures.loc[fixture_labels == selected_label].iloc[0]
    home = selected_match["home_team"]
    away = selected_match["away_team"]

    st.markdown(
        f"""
        <div class="match-strip">
            <div class="team-name">{team_badge(home)}</div>
            <div class="versus">Grupo {selected_match['group']}<br>{selected_match['date']}<br>Rodada {selected_match['matchday']}</div>
            <div class="team-name">{team_badge(away)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    direct = head_to_head_results(matches, home, away)
    status = "Nunca se enfrentaram" if direct.empty else "Já se enfrentaram"
    home_wins = int((direct["Vencedor"] == team_label(home)).sum()) if not direct.empty else 0
    away_wins = int((direct["Vencedor"] == team_label(away)).sum()) if not direct.empty else 0
    draws = int((direct["Vencedor"] == "Empate").sum()) if not direct.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Status do confronto", status)
    with c2:
        metric_card("Jogos encontrados", str(len(direct)))
    with c3:
        metric_card(f"Vitórias {team_label(home)}", str(home_wins))
    with c4:
        metric_card(f"Vitórias {team_label(away)}", str(away_wins))

    if direct.empty:
        st.warning("Nunca se enfrentaram em Copas do Mundo na base histórica disponível.")
    else:
        metric_card("Empates", str(draws))
        st.subheader("Histórico de jogos entre as seleções")
        st.dataframe(direct, use_container_width=True, hide_index=True)


def render_knockout_page(fixtures, ratings):
    st.title("Mata-mata projetado")
    st.write(
        "A chave usa a rede neural e as simulações de Monte Carlo da fase de grupos para preencher a Fase de 32, oitavas, quartas, semifinais e final."
    )

    rankings, qualified_thirds, bracket = build_knockout_projection(fixtures, ratings)
    champion = bracket.loc[bracket["Fase"] == "Final", "Classificado projetado"].iloc[0]
    final_score = bracket.loc[bracket["Fase"] == "Final", "Placar provável"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Campeão projetado", champion)
    with c2:
        metric_card("Placar provável da final", final_score)
    with c3:
        metric_card("Seleções no mata-mata", "32")
    with c4:
        metric_card("Terceiros classificados", "8 de 12")

    st.subheader("Classificação projetada da fase de grupos")
    group_summary = rankings.copy()
    group_summary["Seleção"] = group_summary["team"].map(team_label)
    group_summary["Status"] = np.where(
        group_summary["Posição"] <= 2,
        "Classificado",
        np.where(
            group_summary["team"].isin(qualified_thirds["team"]),
            "Classificado entre os melhores terceiros",
            "Eliminado",
        ),
    )
    st.dataframe(
        group_summary[["Grupo", "Posição", "Seleção", "Pts. esp.", "SG esp.", "GP esp.", "Status"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Chave projetada")
    tabs = st.tabs(["Fase de 32", "Oitavas", "Quartas", "Semifinais", "Final"])
    stages = ["Fase de 32", "Oitavas de final", "Quartas de final", "Semifinal", "Final"]
    for tab, stage in zip(tabs, stages):
        with tab:
            stage_rows = bracket[bracket["Fase"] == stage]
            st.dataframe(
                stage_rows[
                    [
                        "Jogo",
                        "Seleção A",
                        "Origem A",
                        "Seleção B",
                        "Origem B",
                        "Placar provável",
                        "Classificado projetado",
                        "Prob. de classificação",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "A alocação dos melhores terceiros é uma projeção operacional: usa os oito melhores terceiros estimados e preenche as vagas compatíveis da Fase de 32."
    )


def accuracy_rate(series):
    if series.empty:
        return "—"
    return f"{series.mean():.1%}".replace(".", ",")


def accuracy_card(label, value, detail, tone="neutral"):
    st.markdown(
        f"""
        <div class="accuracy-card accuracy-card--{tone}">
            <div class="accuracy-card__label">{label}</div>
            <div class="accuracy-card__value">{value}</div>
            <div class="accuracy-card__detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def accuracy_count(series):
    if series.empty:
        return "Sem jogos avaliados"
    hits = int(series.fillna(False).sum())
    return f"{hits} de {len(series)} jogos"


def decimal_number(value, digits=2):
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}".replace(".", ",")


def render_author_panel():
    st.markdown(
        """
        <div class="author-panel">
            <div>
                <div class="author-title">Feito por Thiago Ramos de Oliveira</div>
                <div class="author-subtitle">Cientista de dados, programador python</div>
            </div>
            <a class="linkedin-cta" href="https://www.linkedin.com/in/thiago-ramos-oliveira/" target="_blank" rel="noopener noreferrer">
                <span class="linkedin-logo">in</span>
                Clique e conheça meu LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


PAGE_DETAILS = {
    "Análise dos jogos": {
        "meta": "Rede neural, Monte Carlo e palpite para bolão",
        "title": "Análise dos jogos",
        "copy": "Escolha um confronto e veja as projeções produzidas por uma rede neural treinada com o histórico das Copas e milhares de simulações de Monte Carlo.",
    },
    "Confrontos diretos": {
        "meta": "Histórico entre seleções",
        "title": "Confrontos diretos",
        "copy": "Descubra se duas seleções já se enfrentaram em Copas, quando foi o jogo, quanto terminou e quem venceu.",
    },
    "Mata-mata": {
        "meta": "Simulador da chave decisiva",
        "title": "Mata-mata projetado",
        "copy": "Veja a rota estimada da Fase de 32 até a final, com classificados projetados, placares prováveis e campeão previsto.",
    },
    "Estatísticas de acertos": {
        "meta": "Auditoria do modelo de machine learning",
        "title": "Estatísticas de acertos",
        "copy": "Compare as previsões da rede neural e das simulações com os resultados reais vindos da API.",
    },
}


def render_hero(cover_uri, selected_page):
    page = PAGE_DETAILS[selected_page]
    st.markdown(
        f"""
        <div class="report-header">
            <div>
                <div class="report-kicker">Copa dos Dados 2026</div>
                <div class="report-title">{page['title']}</div>
                <div class="report-copy">{page['copy']}</div>
            </div>
            <div class="report-status">
                <span>Página ativa</span>
                <strong>{page['meta']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = []
    for page_name, details in PAGE_DETAILS.items():
        active_class = " active" if page_name == selected_page else ""
        cards.append(
            f'<div class="nav-card{active_class}"><span>{details["meta"]}</span><strong>{page_name}</strong></div>'
        )
    st.markdown(f'<div class="nav-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_accuracy_dashboard(fixtures, results, ratings, results_source):
    heading_col, action_col = st.columns([5, 1.35], vertical_alignment="bottom")
    with heading_col:
        st.title("Estatísticas de acertos")
        st.write(
            "Acompanhe como os palpites gerados antes de cada partida se comparam aos resultados oficiais."
        )
        st.caption(f"Fonte dos resultados: {results_source} · atualização automática a cada 5 minutos")
    with action_col:
        if st.button("Atualizar resultados", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tracking = build_tracking_table(fixtures, results, ratings)
    finished = tracking[tracking["Status"] == "Finalizado"].copy()
    pending = tracking[tracking["Status"] == "Pendente"].copy()
    total_games = len(tracking)
    coverage = len(finished) / total_games if total_games else 0
    avg_confidence = finished["_pick_probability"].mean() if len(finished) else np.nan

    st.markdown('<div class="accuracy-section-label">Visão geral</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        accuracy_card(
            "Jogos avaliados",
            str(len(finished)),
            f"de {total_games} partidas da fase de grupos",
            "primary",
        )
    with c2:
        accuracy_card(
            "Jogos pendentes",
            str(len(pending)),
            "aguardando resultado oficial",
            "muted",
        )
    with c3:
        st.markdown(
            f"""
            <div class="coverage-card">
                <div class="coverage-card__top">
                    <span>Cobertura da competição</span>
                    <strong>{pct(coverage).replace(".", ",")}</strong>
                </div>
                <div class="coverage-card__track">
                    <span style="width:{coverage:.2%}"></span>
                </div>
                <div class="coverage-card__detail">
                    A leitura fica mais representativa conforme novos jogos são concluídos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="accuracy-section-label">Desempenho do modelo</div>', unsafe_allow_html=True)
    c4, c5, c6, c7, c8 = st.columns(5)
    with c4:
        accuracy_card(
            "Resultado 1X2",
            accuracy_rate(finished["_hit_result"]),
            accuracy_count(finished["_hit_result"]),
            "success",
        )
    with c5:
        accuracy_card(
            "Placar exato",
            accuracy_rate(finished["_hit_score"]),
            accuracy_count(finished["_hit_score"]),
        )
    with c6:
        accuracy_card(
            "Acima de 2,5 gols",
            accuracy_rate(finished["_hit_goals"]),
            accuracy_count(finished["_hit_goals"]),
        )
    with c7:
        accuracy_card(
            "Ambos marcam",
            accuracy_rate(finished["_hit_both"]),
            accuracy_count(finished["_hit_both"]),
        )
    with c8:
        accuracy_card(
            "Confiança média",
            pct(avg_confidence).replace(".", ",") if len(finished) else "—",
            "probabilidade dos palpites",
            "info",
        )

    if finished.empty:
        st.info(
            "Ainda não há jogos finalizados para avaliar. As taxas aparecerão quando a fonte oficial publicar os primeiros resultados."
        )
    else:
        if len(finished) < 10:
            st.warning(
                f"Amostra inicial: somente {len(finished)} jogo(s) avaliado(s). "
                "Percentuais extremos ainda não representam o desempenho geral do modelo."
            )

        hit_rows = finished[finished["_hit_result"] == True]
        miss_rows = finished[finished["_hit_result"] == False]
        performance_summary = pd.DataFrame(
            [
                {"Métrica": "Resultado 1X2", "Taxa de acerto": accuracy_rate(finished["_hit_result"])},
                {"Métrica": "Placar exato", "Taxa de acerto": accuracy_rate(finished["_hit_score"])},
                {"Métrica": "Acima de 2,5 gols", "Taxa de acerto": accuracy_rate(finished["_hit_goals"])},
                {"Métrica": "Ambos marcam", "Taxa de acerto": accuracy_rate(finished["_hit_both"])},
                {
                    "Métrica": "Confiança média nos acertos",
                    "Taxa de acerto": pct(hit_rows["_pick_probability"].mean()).replace(".", ",") if len(hit_rows) else "—",
                },
                {
                    "Métrica": "Confiança média nos erros",
                    "Taxa de acerto": pct(miss_rows["_pick_probability"].mean()).replace(".", ",") if len(miss_rows) else "—",
                },
            ]
        )
        st.subheader("Resumo de desempenho")
        st.dataframe(performance_summary, use_container_width=True, hide_index=True)

        group_performance = (
            finished.groupby("Grupo", as_index=False)
            .agg(
                Jogos=("Grupo", "size"),
                Acerto_resultado=("_hit_result", "mean"),
                Acerto_placar=("_hit_score", "mean"),
                Acerto_gols=("_hit_goals", "mean"),
                Acerto_ambos=("_hit_both", "mean"),
            )
            .rename(
                columns={
                    "Acerto_resultado": "Resultado 1X2",
                    "Acerto_placar": "Placar exato",
                    "Acerto_gols": "Acima de 2,5",
                    "Acerto_ambos": "Ambos marcam",
                }
            )
        )
        for column in ["Resultado 1X2", "Placar exato", "Acima de 2,5", "Ambos marcam"]:
            group_performance[column] = group_performance[column].map(lambda value: pct(value).replace(".", ","))
        st.subheader("Desempenho por grupo")
        st.dataframe(group_performance, use_container_width=True, hide_index=True)

        confidence_performance = (
            finished.groupby("Confiança", as_index=False)
            .agg(
                Jogos=("Confiança", "size"),
                Acerto_resultado=("_hit_result", "mean"),
                Probabilidade_media=("_pick_probability", "mean"),
            )
            .rename(
                columns={
                    "Acerto_resultado": "Acerto do resultado",
                    "Probabilidade_media": "Probabilidade média do palpite",
                }
            )
        )
        confidence_performance["Acerto do resultado"] = confidence_performance["Acerto do resultado"].map(
            lambda value: pct(value).replace(".", ",")
        )
        confidence_performance["Probabilidade média do palpite"] = confidence_performance[
            "Probabilidade média do palpite"
        ].map(lambda value: pct(value).replace(".", ","))
        st.subheader("Desempenho por nível de confiança")
        st.dataframe(confidence_performance, use_container_width=True, hide_index=True)

    st.subheader("Auditoria jogo a jogo")
    finished_tab, pending_tab = st.tabs(
        [f"Finalizados ({len(finished)})", f"Pendentes ({len(pending)})"]
    )
    with finished_tab:
        if finished.empty:
            st.caption("Nenhum resultado disponível para auditoria.")
        else:
            st.dataframe(
                finished[
                    [
                        "Data",
                        "Grupo",
                        "Jogo",
                        "Resultado previsto",
                        "Placar previsto",
                        "Resultado real",
                        "Placar real",
                        "Acertou resultado",
                        "Acertou placar",
                        "Acertou gols",
                        "Acertou ambos",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    with pending_tab:
        st.dataframe(
            pending[
                [
                    "Data",
                    "Grupo",
                    "Jogo",
                    "Palpite para bolão",
                    "Confiança",
                    "Prob. do palpite",
                    "Placar previsto",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def apply_theme():
    st.markdown(
        """
        <style>
            :root {
                --ink: #252423;
                --muted: #605e5c;
                --line: #d2d0ce;
                --surface: #ffffff;
                --surface-soft: #f3f2f1;
                --green: #107c10;
                --blue: #0078d4;
                --gold: #f2c811;
                --red: #d13438;
                --cyan: #00b7c3;
                --navy: #201f1e;
            }

            .stApp {
                background: #f3f2f1;
            }

            .main .block-container {
                max-width: 1260px;
                padding-bottom: 2rem;
                padding-top: 1rem;
            }

            h1, h2, h3 {
                color: var(--ink);
                letter-spacing: 0;
            }

            h1 {
                font-size: 30px;
                font-weight: 850;
                margin-top: 4px;
            }

            h2, h3 {
                font-weight: 800;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--line);
            }

            div[role="radiogroup"] {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 3px;
                box-shadow: none;
                display: flex;
                gap: 0;
                padding: 0;
            }

            div[role="radiogroup"] label {
                border-right: 1px solid var(--line);
                border-radius: 0;
                margin: 0;
                padding: 8px 14px;
            }

            div[role="radiogroup"] label:has(input:checked) {
                background: #f2c811;
                box-shadow: inset 0 -3px 0 #252423;
                color: var(--ink);
            }

            div[role="radiogroup"] label:has(input:checked) p,
            div[role="radiogroup"] label:has(input:checked) span {
                color: var(--ink) !important;
                font-weight: 700;
            }

            .report-header {
                align-items: center;
                background: #ffffff;
                border: 1px solid var(--line);
                border-left: 8px solid var(--gold);
                border-radius: 3px;
                display: flex;
                gap: 22px;
                justify-content: space-between;
                margin: 12px 0 10px;
                padding: 18px 20px;
            }

            .report-kicker {
                color: var(--muted);
                font-size: 12px;
                font-weight: 700;
                letter-spacing: .06em;
                margin-bottom: 4px;
                text-transform: uppercase;
            }

            .report-title {
                color: var(--ink);
                font-size: 28px;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 6px;
            }

            .report-copy {
                color: var(--muted);
                font-size: 14px;
                line-height: 1.35;
                max-width: 760px;
            }

            .report-status {
                background: #201f1e;
                border-radius: 3px;
                color: #ffffff;
                min-width: 220px;
                padding: 12px 14px;
            }

            .report-status span {
                color: rgba(255, 255, 255, .72);
                display: block;
                font-size: 11px;
                font-weight: 700;
                margin-bottom: 4px;
                text-transform: uppercase;
            }

            .report-status strong {
                color: #ffffff;
                display: block;
                font-size: 15px;
                line-height: 1.2;
            }

            .hero {
                border: 1px solid rgba(255, 255, 255, .42);
                background: var(--navy);
                box-shadow: 0 24px 70px rgba(7, 24, 39, .22);
                min-height: 330px;
                border-radius: 8px;
                margin: 16px 0 14px;
                overflow: hidden;
                position: relative;
            }

            .hero:after {
                background: linear-gradient(90deg, var(--green), var(--gold), var(--red), var(--blue));
                bottom: 0;
                content: "";
                height: 5px;
                left: 0;
                position: absolute;
                right: 0;
            }

            .hero-img {
                height: 100%;
                inset: 0;
                object-fit: cover;
                position: absolute;
                width: 100%;
            }

            .hero-overlay {
                background:
                    linear-gradient(90deg, rgba(7, 17, 31, .96) 0%, rgba(7, 17, 31, .78) 45%, rgba(7, 17, 31, .18) 100%),
                    linear-gradient(135deg, rgba(0, 87, 255, .50), rgba(0, 168, 107, .22) 45%, rgba(255, 176, 0, .16));
                inset: 0;
                position: absolute;
            }

            .hero-content {
                padding: 42px 38px;
                position: relative;
                z-index: 1;
            }

            .hero-title {
                font-size: 44px;
                font-weight: 900;
                color: #ffffff;
                line-height: 1.02;
                margin-bottom: 12px;
                max-width: 620px;
            }

            .hero-copy {
                color: rgba(255, 255, 255, .86);
                font-size: 17px;
                line-height: 1.5;
                max-width: 670px;
            }

            .hero-meta {
                color: #c8ff4d;
                font-size: 13px;
                font-weight: 850;
                letter-spacing: .06em;
                margin-bottom: 10px;
                text-transform: uppercase;
            }

            .hero-scorecard {
                background: rgba(255, 255, 255, .92);
                border: 1px solid rgba(255, 255, 255, .72);
                border-radius: 8px;
                bottom: 28px;
                box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
                min-width: 245px;
                padding: 16px 18px;
                position: absolute;
                right: 28px;
                z-index: 2;
            }

            .hero-scorecard span {
                color: var(--muted);
                display: block;
                font-size: 12px;
                font-weight: 800;
                margin-bottom: 6px;
                text-transform: uppercase;
            }

            .hero-scorecard strong {
                color: var(--ink);
                font-size: 22px;
                line-height: 1.1;
            }

            .nav-card-grid {
                display: grid;
                gap: 10px;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin: 0 0 12px;
            }

            .nav-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 3px;
                box-shadow: none;
                min-height: 74px;
                padding: 12px 14px;
            }

            .nav-card span {
                color: var(--muted);
                display: block;
                font-size: 11px;
                font-weight: 800;
                margin-bottom: 7px;
                text-transform: uppercase;
            }

            .nav-card strong {
                color: var(--ink);
                display: block;
                font-size: 17px;
                line-height: 1.15;
            }

            .nav-card.active {
                background: #201f1e;
                border-color: #201f1e;
            }

            .nav-card.active span,
            .nav-card.active strong {
                color: #ffffff;
            }

            .match-strip {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                border: 1px solid var(--line);
                border-left: 6px solid var(--blue);
                border-radius: 3px;
                box-shadow: none;
                padding: 16px 18px;
                background: var(--surface);
                margin: 12px 0 14px;
            }

            .team-name {
                align-items: center;
                display: flex;
                gap: 10px;
                font-size: 24px;
                font-weight: 750;
                color: var(--ink);
            }

            .flag-img {
                border: 1px solid rgba(16, 24, 40, .12);
                border-radius: 3px;
                height: 24px;
                object-fit: cover;
                width: 34px;
            }

            .versus {
                background: #f2c811;
                border-radius: 3px;
                color: var(--ink);
                font-size: 13px;
                font-weight: 800;
                padding: 9px 12px;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: .08em;
            }

            .decision-grid {
                display: grid;
                gap: 14px;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin-bottom: 18px;
            }

            .decision-card {
                border: 1px solid var(--line);
                border-radius: 3px;
                background: var(--surface);
                box-shadow: none;
                padding: 15px 16px;
                min-height: 118px;
            }

            .decision-card.primary {
                background: #201f1e;
                border-bottom: 5px solid var(--gold);
                border-color: #201f1e;
                color: #ffffff;
            }

            .decision-card.primary .card-label,
            .decision-card.primary .card-sub {
                color: rgba(255, 255, 255, .78);
            }

            .card-label {
                color: var(--muted);
                font-size: 12px;
                font-weight: 750;
                letter-spacing: .07em;
                margin-bottom: 8px;
                text-transform: uppercase;
            }

            .card-value {
                align-items: center;
                color: inherit;
                display: flex;
                gap: 10px;
                font-size: 28px;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 6px;
            }

            .card-sub {
                color: var(--muted);
                font-size: 13px;
                line-height: 1.35;
            }

            .pick-card {
                border: 1px solid var(--line);
                border-left: 6px solid var(--green);
                border-radius: 3px;
                background: var(--surface);
                box-shadow: none;
                padding: 18px 20px;
                min-height: 154px;
            }

            .pick-label {
                color: var(--muted);
                font-size: 13px;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: .06em;
                margin-bottom: 6px;
            }

            .pick-main {
                color: var(--ink);
                font-size: 28px;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 10px;
            }

            .pick-meta {
                color: var(--muted);
                font-size: 14px;
            }

            .confidence-pill {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                color: #fff;
                font-size: 13px;
                font-weight: 700;
            }

            .accuracy-section-label {
                color: var(--muted);
                font-size: 12px;
                font-weight: 800;
                letter-spacing: .09em;
                margin: 22px 0 8px;
                text-transform: uppercase;
            }

            .accuracy-card,
            .coverage-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                min-height: 128px;
                padding: 17px 18px;
            }

            .accuracy-card {
                border-top: 4px solid #8a8886;
            }

            .accuracy-card--primary {
                border-top-color: var(--gold);
            }

            .accuracy-card--muted {
                background: #faf9f8;
                border-top-color: #a19f9d;
            }

            .accuracy-card--success {
                border-top-color: var(--green);
            }

            .accuracy-card--info {
                border-top-color: var(--blue);
            }

            .accuracy-card__label {
                color: var(--muted);
                font-size: 13px;
                font-weight: 750;
                line-height: 1.25;
                margin-bottom: 10px;
            }

            .accuracy-card__value {
                color: var(--ink);
                font-size: 34px;
                font-weight: 850;
                letter-spacing: -.03em;
                line-height: 1;
                margin-bottom: 9px;
            }

            .accuracy-card__detail,
            .coverage-card__detail {
                color: var(--muted);
                font-size: 12px;
                line-height: 1.4;
            }

            .coverage-card__top {
                align-items: center;
                color: var(--muted);
                display: flex;
                font-size: 13px;
                font-weight: 750;
                justify-content: space-between;
                margin-bottom: 18px;
            }

            .coverage-card__top strong {
                color: var(--ink);
                font-size: 24px;
            }

            .coverage-card__track {
                background: #edebe9;
                border-radius: 999px;
                height: 9px;
                margin-bottom: 13px;
                overflow: hidden;
            }

            .coverage-card__track span {
                background: linear-gradient(90deg, var(--gold), #d9a900);
                border-radius: inherit;
                display: block;
                height: 100%;
                min-width: 3px;
            }

            div[data-testid="stMetric"] {
                border: 1px solid var(--line);
                border-top: 4px solid var(--gold);
                background: #ffffff;
                box-shadow: none;
                padding: 12px 14px;
                border-radius: 3px;
            }

            div[data-testid="stMetricLabel"] {
                color: var(--muted);
                font-weight: 800;
            }

            div[data-testid="stMetricValue"] {
                color: var(--ink);
                font-weight: 900;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 3px;
                box-shadow: none;
                overflow: hidden;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 3px;
                font-weight: 800;
                padding: 8px 14px;
            }

            .stTabs [aria-selected="true"] {
                background: var(--gold);
                color: var(--ink);
            }

            .author-panel {
                align-items: center;
                background: linear-gradient(135deg, #071827, #102a43 58%, #0057ff);
                border-radius: 8px;
                box-shadow: 0 18px 45px rgba(7, 24, 39, .18);
                color: #ffffff;
                display: flex;
                gap: 18px;
                justify-content: space-between;
                margin-top: 18px;
                padding: 20px 22px;
            }

            .author-title {
                font-size: 20px;
                font-weight: 800;
                margin-bottom: 4px;
            }

            .author-subtitle {
                color: rgba(255, 255, 255, .72);
                font-size: 14px;
            }

            .linkedin-cta {
                align-items: center;
                background: linear-gradient(135deg, #0a66c2, #00c2ff);
                border-radius: 8px;
                color: #ffffff !important;
                display: inline-flex;
                font-weight: 750;
                gap: 10px;
                padding: 12px 16px;
                text-decoration: none !important;
                white-space: nowrap;
            }

            .linkedin-logo {
                align-items: center;
                background: #ffffff;
                border-radius: 4px;
                color: #0a66c2;
                display: inline-flex;
                font-size: 18px;
                font-weight: 900;
                height: 24px;
                justify-content: center;
                width: 24px;
            }

            @media (max-width: 900px) {
                .accuracy-card,
                .coverage-card {
                    min-height: auto;
                }
                .decision-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .nav-card-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .hero-scorecard {
                    bottom: auto;
                    left: 24px;
                    right: 24px;
                    top: 210px;
                }
                .match-strip {
                    align-items: flex-start;
                    flex-direction: column;
                }
                .versus {
                    text-align: left;
                }
                .author-panel {
                    align-items: flex-start;
                    flex-direction: column;
                }
            }

            @media (max-width: 640px) {
                .hero {
                    min-height: 390px;
                }
                .hero-title {
                    font-size: 34px;
                }
                .nav-card-grid,
                .decision-grid {
                    grid-template-columns: 1fr;
                }
                div[role="radiogroup"] {
                    align-items: stretch;
                    flex-direction: column;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def flag_emoji(team):
    code = COUNTRY_CODE_BY_TEAM.get(team)
    if not code:
        return ""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in code.upper())


def team_label(team):
    flag = flag_emoji(team)
    return f"{flag} {team}".strip()


def team_badge(team):
    code = COUNTRY_CODE_BY_TEAM.get(team, "")
    src = f"https://flagcdn.com/w40/{code.lower()}.png" if code else ""
    image = f'<img src="{src}" alt="{team}" class="flag-img">' if src else ""
    return f'{image}<span>{team}</span>'


def pct(value):
    return f"{value:.1%}"


def image_data_uri(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def main():
    st.set_page_config(page_title="Copa dos Dados 2026", page_icon="WC", layout="wide")
    apply_theme()

    matches, champions, teams_2026, fixtures, results, results_source = load_data()
    ratings = build_team_ratings(matches, champions, teams_2026)
    cover_uri = image_data_uri(ASSETS_DIR / "copa-dados-cover.png")

    selected_page = st.radio(
        "Navegação principal",
        list(PAGE_DETAILS.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="primary_navigation",
    )
    render_hero(cover_uri, selected_page)

    if selected_page == "Estatísticas de acertos":
        render_accuracy_dashboard(fixtures, results, ratings, results_source)
        st.caption(
            "Dados de seleções e grupos da Copa 2026 atualizados em maio de 2026 a partir do calendário oficial da FIFA e da consolidação pública da competição."
        )
        render_author_panel()
        return

    if selected_page == "Confrontos diretos":
        render_recent_results_page(matches, fixtures)
        st.caption(
            "Dados históricos baseados em jogos de Copas do Mundo; quando não houver jogo entre as duas seleções, a página sinaliza que elas nunca se enfrentaram na base."
        )
        render_author_panel()
        return

    if selected_page == "Mata-mata":
        render_knockout_page(fixtures, ratings)
        st.caption(
            "A estrutura considera o formato da Copa 2026 com 12 grupos, top 2 de cada grupo e os 8 melhores terceiros avançando ao mata-mata."
        )
        render_author_panel()
        return

    st.sidebar.title("Filtros")
    group_options = ["Todos"] + sorted(fixtures["group"].unique())
    selected_group = st.sidebar.selectbox("Grupo", group_options)

    filtered_fixtures = fixtures.copy()
    if selected_group != "Todos":
        filtered_fixtures = filtered_fixtures[filtered_fixtures["group"] == selected_group]

    fixture_labels = (
        filtered_fixtures["date"]
        + " | Grupo "
        + filtered_fixtures["group"]
        + " | "
        + filtered_fixtures["home_team"].map(team_label)
        + " x "
        + filtered_fixtures["away_team"].map(team_label)
    )
    selected_label = st.sidebar.selectbox("Jogo", fixture_labels)
    selected_match = filtered_fixtures.loc[fixture_labels == selected_label].iloc[0]

    home = selected_match["home_team"]
    away = selected_match["away_team"]
    probabilities = match_probabilities(home, away, ratings)
    expected_home, expected_away = estimate_goals(home, away, ratings)
    goals = goal_markets(expected_home, expected_away)
    pick, pick_probability, confidence, score = pick_for_pool(home, away, probabilities, goals)
    pick_label = "Empate" if pick == "Empate" else team_label(pick)
    pick_badge = "Empate" if pick == "Empate" else team_badge(pick)
    goals_signal = "Tende a ter gols" if goals["over_2_5"] >= 0.54 else "Tende a ser um jogo travado"
    both_score_signal = "Ambos marcam forte" if goals["both_score"] >= 0.52 else "Ambos marcam moderado"

    st.markdown(
        f"""
        <div class="match-strip">
            <div class="team-name">{team_badge(home)}</div>
            <div class="versus">Grupo {selected_match['group']}<br>{selected_match['date']}<br>Rodada {selected_match['matchday']}</div>
            <div class="team-name">{team_badge(away)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="decision-grid">
            <div class="decision-card primary">
                <div class="card-label">Palpite do bolão</div>
                <div class="card-value">{pick_badge}</div>
                <div class="card-sub">{pct(pick_probability)} de probabilidade | confiança {confidence}</div>
            </div>
            <div class="decision-card">
                <div class="card-label">Placar modal</div>
                <div class="card-value">{score}</div>
                <div class="card-sub">Resultado mais provável na matriz de gols</div>
            </div>
            <div class="decision-card">
                <div class="card-label">Gols esperados</div>
                <div class="card-value">{expected_home + expected_away:.2f}</div>
                <div class="card-sub">{goals_signal} | acima de 2,5 gols: {pct(goals['over_2_5'])}</div>
            </div>
            <div class="decision-card">
                <div class="card-label">Ambos marcam</div>
                <div class="card-value">{pct(goals['both_score'])}</div>
                <div class="card-sub">{both_score_signal}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pick_col, detail_col = st.columns([1, 1])
    with pick_col:
        st.markdown(
            f"""
            <div class="pick-card">
                <div class="pick-label">Recomendação do modelo de machine learning</div>
                <div class="pick-main">{pick_label}</div>
                <div class="pick-meta">Probabilidade do palpite: <strong>{pct(pick_probability)}</strong></div>
                <div class="pick-meta">Placar mais provável: <strong>{score}</strong></div>
                <div class="pick-meta" style="margin-top:10px;">
                    Confiança:
                    <span class="confidence-pill" style="background:{RESULT_COLORS[confidence]};">{confidence}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_col:
        st.subheader("Como ler o jogo")
        if pick == "Empate":
            st.write(
                "O confronto aparece equilibrado no rating e na distribuição de gols. Para bolão, o empate ganha peso quando a diferença de força é pequena."
            )
        else:
            st.write(
                f"{team_label(pick)} tem a maior probabilidade projetada pela rede neural após {MATCH_MONTE_CARLO_RUNS:,} simulações de Monte Carlo."
            )
        st.write(
            f"O modelo de machine learning projeta **{expected_home + expected_away:.2f} gols** na partida, com placar modal **{score}**."
        )

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(f"Vitória {team_label(home)}", f"{probabilities['home']:.1%}")
    with c2:
        metric_card("Empate", f"{probabilities['draw']:.1%}")
    with c3:
        metric_card(f"Vitória {team_label(away)}", f"{probabilities['away']:.1%}")

    st.divider()

    goals_col, score_col = st.columns([1, 1])
    with goals_col:
        st.subheader("Gols esperados")
        g1, g2, g3 = st.columns(3)
        with g1:
            metric_card(team_label(home), f"{expected_home:.2f}")
        with g2:
            metric_card(team_label(away), f"{expected_away:.2f}")
        with g3:
            metric_card("Total", f"{expected_home + expected_away:.2f}")

        goal_rows = pd.DataFrame(
            [
                {"Indicador": "Acima de 1,5 gol", "Probabilidade": f"{goals['over_1_5']:.1%}"},
                {"Indicador": "Acima de 2,5 gols", "Probabilidade": f"{goals['over_2_5']:.1%}"},
                {"Indicador": "Acima de 3,5 gols", "Probabilidade": f"{goals['over_3_5']:.1%}"},
                {"Indicador": "Ambos marcam", "Probabilidade": f"{goals['both_score']:.1%}"},
            ]
        )
        st.dataframe(goal_rows, use_container_width=True, hide_index=True)

    with score_col:
        st.subheader("Placares mais prováveis")
        scorelines = goals["scorelines"].copy()
        scorelines["Placar"] = (
            scorelines["home_goals"].astype(str)
            + " x "
            + scorelines["away_goals"].astype(str)
        )
        scorelines["Probabilidade"] = scorelines["probability"].map(lambda value: f"{value:.1%}")
        st.dataframe(scorelines[["Placar", "Probabilidade"]], use_container_width=True, hide_index=True)

        match_summary = pd.DataFrame(
            [
                {
                    "Seleção": team_label(home),
                    "Índice do modelo": float(ratings.loc[ratings["team"] == home, "rating"].iloc[0]),
                    "Gols pro/jogo": float(
                        ratings.loc[ratings["team"] == home, "goals_for_per_match"].iloc[0]
                    ),
                    "Gols contra/jogo": float(
                        ratings.loc[ratings["team"] == home, "goals_against_per_match"].iloc[0]
                    ),
                },
                {
                    "Seleção": team_label(away),
                    "Índice do modelo": float(ratings.loc[ratings["team"] == away, "rating"].iloc[0]),
                    "Gols pro/jogo": float(
                        ratings.loc[ratings["team"] == away, "goals_for_per_match"].iloc[0]
                    ),
                    "Gols contra/jogo": float(
                        ratings.loc[ratings["team"] == away, "goals_against_per_match"].iloc[0]
                    ),
                },
            ]
        )
        st.dataframe(match_summary, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader(f"Projeção do Grupo {selected_match['group']}")
    st.dataframe(
        projected_group_table(selected_match["group"], fixtures, ratings),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Resumo do jogo")
    summary_rows = pd.DataFrame(
        [
            {
                "Resultado": team_label(home),
                "Probabilidade": f"{probabilities['home']:.1%}",
                "Gols esperados": round(expected_home, 2),
            }
            ,
            {
                "Resultado": "Empate",
                "Probabilidade": f"{probabilities['draw']:.1%}",
                "Gols esperados": "-",
            },
            {
                "Resultado": team_label(away),
                "Probabilidade": f"{probabilities['away']:.1%}",
                "Gols esperados": round(expected_away, 2),
            },
        ]
    )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Jogos da fase de grupos")
    if True:
        display_fixtures = fixtures.copy()
        display_fixtures["Jogo"] = (
            display_fixtures["home_team"].map(team_label)
            + " x "
            + display_fixtures["away_team"].map(team_label)
        )
        display_fixtures = display_fixtures.rename(
            columns={"date": "Data", "group": "Grupo", "matchday": "Rodada"}
        )
        st.dataframe(
            display_fixtures[["Data", "Grupo", "Rodada", "Jogo"]],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Dados de seleções e grupos da Copa 2026 atualizados em maio de 2026 a partir do calendário oficial da FIFA e da consolidação pública da competição."
    )

    st.markdown(
        """
        <div class="author-panel">
            <div>
                <div class="author-title">Feito por Thiago Ramos de Oliveira</div>
                <div class="author-subtitle">Cientista de dados, programador python</div>
            </div>
            <a class="linkedin-cta" href="https://www.linkedin.com/in/thiago-ramos-oliveira/" target="_blank" rel="noopener noreferrer">
                <span class="linkedin-logo">in</span>
                Clique e conheça meu LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

