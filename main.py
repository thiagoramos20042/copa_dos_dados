from pathlib import Path
import base64
import json
import math
import os
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
RESULTS_COLUMNS = ["match_id", "actual_home_goals", "actual_away_goals", "status", "notes"]

TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
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


@st.cache_data
def load_data():
    matches = pd.read_csv(BASE_DIR / "Jogos Copas do Mundo.csv", encoding="cp1252")
    champions = pd.read_csv(BASE_DIR / "Campeoes.csv")
    teams_2026 = pd.read_csv(DATA_DIR / "world_cup_2026_teams.csv")
    fixtures = pd.read_csv(DATA_DIR / "world_cup_2026_group_stage.csv")
    results, results_source = load_results()
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


def read_local_results():
    results = pd.read_csv(DATA_DIR / "world_cup_2026_results.csv")
    return normalize_results(results)


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
    return None


def records_from_payload(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["results", "matches", "fixtures", "data", "response"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

    return []


def normalize_api_records(records):
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue

        match_id = first_available(record, ["match_id", "id", "fixture_id", "game_id"])
        home_goals = first_available(record, ["actual_home_goals", "home_goals", "home_score", "score_home", "homeTeamScore", "goals_home"])
        away_goals = first_available(record, ["actual_away_goals", "away_goals", "away_score", "score_away", "awayTeamScore", "goals_away"])

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


def fetch_results_from_api():
    url = get_config_value("RESULTS_API_URL")
    if not url:
        return pd.DataFrame(columns=RESULTS_COLUMNS), "CSV local"

    api_key = get_config_value("RESULTS_API_KEY")
    auth_header = get_config_value("RESULTS_API_AUTH_HEADER", "Authorization")
    auth_prefix = get_config_value("RESULTS_API_AUTH_PREFIX", "Bearer")
    headers = {"Accept": "application/json"}
    if api_key:
        headers[auth_header] = f"{auth_prefix} {api_key}".strip()

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=RESULTS_COLUMNS), "CSV local"

    api_results = normalize_api_records(records_from_payload(payload))
    if api_results.empty:
        return pd.DataFrame(columns=RESULTS_COLUMNS), "CSV local"

    return api_results, "API"


def load_results():
    local_results = read_local_results()
    api_results, source = fetch_results_from_api()
    if api_results.empty:
        return local_results, "CSV local"

    merged = local_results.set_index("match_id")
    api_indexed = api_results.set_index("match_id")
    for match_id, row in api_indexed.iterrows():
        if match_id in merged.index:
            merged.loc[match_id, ["actual_home_goals", "actual_away_goals", "status", "notes"]] = row[["actual_home_goals", "actual_away_goals", "status", "notes"]]
        else:
            merged.loc[match_id] = row

    return normalize_results(merged.reset_index()), source
def normalize_team(team):
    if pd.isna(team):
        return team
    team = str(team).strip()
    return TEAM_ALIASES.get(team, team)


def build_team_ratings(matches, champions, teams_2026):
    matches = matches.copy()
    matches["TimeDaCasa"] = matches["TimeDaCasa"].map(normalize_team)
    matches["TimeVisitante"] = matches["TimeVisitante"].map(normalize_team)

    teams = sorted(teams_2026["team"].unique())
    rows = []

    champion_counts = champions["Vencedor"].map(normalize_team).value_counts()
    runner_up_counts = champions["Segundo"].map(normalize_team).value_counts()

    for team in teams:
        home = matches[matches["TimeDaCasa"] == team]
        away = matches[matches["TimeVisitante"] == team]

        played = len(home) + len(away)
        wins = int((home["GolsTimeDaCasa"] > home["GolsTimeVisitante"]).sum())
        wins += int((away["GolsTimeVisitante"] > away["GolsTimeDaCasa"]).sum())
        draws = int((home["GolsTimeDaCasa"] == home["GolsTimeVisitante"]).sum())
        draws += int((away["GolsTimeVisitante"] == away["GolsTimeDaCasa"]).sum())

        goals_for = float(home["GolsTimeDaCasa"].sum() + away["GolsTimeVisitante"].sum())
        goals_against = float(home["GolsTimeVisitante"].sum() + away["GolsTimeDaCasa"].sum())
        points = wins * 3 + draws

        titles = int(champion_counts.get(team, 0))
        finals = int(titles + runner_up_counts.get(team, 0))
        recent = matches[
            ((matches["TimeDaCasa"] == team) | (matches["TimeVisitante"] == team))
            & (matches["Ano"] >= 2006)
        ]

        if played:
            points_per_match = points / played
            goal_balance = (goals_for - goals_against) / played
            attack = goals_for / played
            defense = goals_against / played
        else:
            points_per_match = 0.65
            goal_balance = -0.25
            attack = 0.75
            defense = 1.35

        recency_bonus = min(len(recent), 24) / 24
        pedigree = np.log1p(titles * 3 + finals)
        rating = (
            50
            + points_per_match * 18
            + goal_balance * 10
            + attack * 3
            - defense * 2
            + pedigree * 7
            + recency_bonus * 6
        )

        rows.append(
            {
                "team": team,
                "played": played,
                "wins": wins,
                "draws": draws,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goals_for_per_match": round(float(attack), 2),
                "goals_against_per_match": round(float(defense), 2),
                "titles": titles,
                "finals": finals,
                "rating": round(float(rating), 2),
            }
        )

    ratings = pd.DataFrame(rows).merge(teams_2026, on="team", how="left")
    confederation_strength = ratings.groupby("confederation")["rating"].transform("median")
    ratings["rating"] = ratings["rating"].where(ratings["played"] > 0, confederation_strength - 4)
    ratings["rating"] = ratings["rating"].round(2)
    return ratings.sort_values("rating", ascending=False)


def match_probabilities(team_a, team_b, ratings, neutral_factor=0.08):
    rating_a = float(ratings.loc[ratings["team"] == team_a, "rating"].iloc[0])
    rating_b = float(ratings.loc[ratings["team"] == team_b, "rating"].iloc[0])
    diff = rating_a - rating_b

    draw_base = 0.24 + max(0, 0.08 - abs(diff) / 500)
    win_a_raw = 1 / (1 + np.exp(-diff / 15))
    win_a = (1 - draw_base) * win_a_raw
    win_b = (1 - draw_base) * (1 - win_a_raw)

    win_a = win_a * (1 - neutral_factor) + neutral_factor / 2
    win_b = win_b * (1 - neutral_factor) + neutral_factor / 2
    draw = max(0.05, 1 - win_a - win_b)

    total = win_a + draw + win_b
    return {
        "home": win_a / total,
        "draw": draw / total,
        "away": win_b / total,
    }


def poisson_probability(mean, goals):
    return float(np.exp(-mean) * (mean**goals) / math.factorial(goals))


def estimate_goals(team_a, team_b, ratings):
    team_a_row = ratings.loc[ratings["team"] == team_a].iloc[0]
    team_b_row = ratings.loc[ratings["team"] == team_b].iloc[0]
    tournament_goal_avg = 2.65

    attack_a = float(team_a_row["goals_for_per_match"]) or tournament_goal_avg / 2
    attack_b = float(team_b_row["goals_for_per_match"]) or tournament_goal_avg / 2
    defense_a = float(team_a_row["goals_against_per_match"]) or tournament_goal_avg / 2
    defense_b = float(team_b_row["goals_against_per_match"]) or tournament_goal_avg / 2

    rating_a = float(team_a_row["rating"])
    rating_b = float(team_b_row["rating"])
    strength_adjustment_a = np.clip(1 + (rating_a - rating_b) / 180, 0.72, 1.35)
    strength_adjustment_b = np.clip(1 + (rating_b - rating_a) / 180, 0.72, 1.35)

    expected_a = ((attack_a + defense_b) / 2) * strength_adjustment_a
    expected_b = ((attack_b + defense_a) / 2) * strength_adjustment_b

    expected_a = float(np.clip(expected_a, 0.25, 3.4))
    expected_b = float(np.clip(expected_b, 0.25, 3.4))
    return expected_a, expected_b


def goal_markets(expected_home, expected_away):
    max_goals = 8
    score_rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = poisson_probability(expected_home, home_goals) * poisson_probability(
                expected_away, away_goals
            )
            score_rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "total_goals": home_goals + away_goals,
                    "probability": probability,
                }
            )

    scores = pd.DataFrame(score_rows)
    return {
        "over_1_5": float(scores.loc[scores["total_goals"] > 1.5, "probability"].sum()),
        "over_2_5": float(scores.loc[scores["total_goals"] > 2.5, "probability"].sum()),
        "over_3_5": float(scores.loc[scores["total_goals"] > 3.5, "probability"].sum()),
        "both_score": float(
            scores.loc[
                (scores["home_goals"] > 0) & (scores["away_goals"] > 0), "probability"
            ].sum()
        ),
        "scorelines": scores.sort_values("probability", ascending=False).head(6),
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


def projected_group_table(group, fixtures, ratings):
    group_fixtures = fixtures[fixtures["group"] == group]
    teams = sorted(set(group_fixtures["home_team"]) | set(group_fixtures["away_team"]))
    table = {
        team: {"Seleção": team_label(team), "PJ": 3, "Pts. esp.": 0.0, "GP esp.": 0.0, "GC esp.": 0.0}
        for team in teams
    }

    for _, match in group_fixtures.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        probabilities = match_probabilities(home, away, ratings)
        expected_home, expected_away = estimate_goals(home, away, ratings)

        table[home]["Pts. esp."] += probabilities["home"] * 3 + probabilities["draw"]
        table[away]["Pts. esp."] += probabilities["away"] * 3 + probabilities["draw"]
        table[home]["GP esp."] += expected_home
        table[home]["GC esp."] += expected_away
        table[away]["GP esp."] += expected_away
        table[away]["GC esp."] += expected_home

    projection = pd.DataFrame(table.values())
    projection["SG esp."] = projection["GP esp."] - projection["GC esp."]
    numeric_cols = ["Pts. esp.", "GP esp.", "GC esp.", "SG esp."]
    projection[numeric_cols] = projection[numeric_cols].round(2)
    return projection.sort_values(["Pts. esp.", "SG esp.", "GP esp."], ascending=False)


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

        predicted_over_2_5 = goals["over_2_5"] >= 0.5
        actual_over_2_5 = (actual_home + actual_away) > 2.5 if played else None
        predicted_both_score = goals["both_score"] >= 0.5
        actual_both_score = (actual_home > 0 and actual_away > 0) if played else None
        hit_result = pick == actual_result if played else None
        hit_score = score == actual_score if played else None
        hit_goals = predicted_over_2_5 == actual_over_2_5 if played else None
        hit_both = predicted_both_score == actual_both_score if played else None

        rows.append(
            {
                "Data": match["date"],
                "Grupo": match["group"],
                "Jogo": f"{team_label(home)} x {team_label(away)}",
                "Palpite estatístico": team_label(pick) if pick != "Empate" else "Empate",
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


def accuracy_rate(series):
    if series.empty:
        return "0,0%"
    return f"{series.mean():.1%}".replace(".", ",")


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


def render_accuracy_dashboard(fixtures, results, ratings, results_source):
    st.title("Estatísticas de acertos")
    st.write(
        "Compare o que o modelo estatístico indicou antes dos jogos com o que aconteceu de fato. "
        f"A fonte atual dos resultados é **{results_source}**. Quando uma API estiver configurada, "
        "os placares serão atualizados automaticamente; caso contrário, o CSV local será usado como fallback."
    )

    tracking = build_tracking_table(fixtures, results, ratings)
    finished = tracking[tracking["Status"] == "Finalizado"].copy()
    pending = tracking[tracking["Status"] == "Pendente"].copy()
    total_games = len(tracking)
    coverage = len(finished) / total_games if total_games else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Jogos avaliados", str(len(finished)))
    with c2:
        metric_card("Cobertura da base", pct(coverage).replace(".", ","))
    with c3:
        metric_card("Acerto do resultado", accuracy_rate(finished["_hit_result"]) if len(finished) else "0,0%")
    with c4:
        metric_card("Jogos pendentes", str(len(pending)))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Placar exato", accuracy_rate(finished["_hit_score"]) if len(finished) else "0,0%")
    with c6:
        metric_card("Acima de 2,5 gols", accuracy_rate(finished["_hit_goals"]) if len(finished) else "0,0%")
    with c7:
        metric_card("Ambos marcam", accuracy_rate(finished["_hit_both"]) if len(finished) else "0,0%")
    with c8:
        avg_confidence = finished["_pick_probability"].mean() if len(finished) else np.nan
        metric_card("Confiança média", pct(avg_confidence).replace(".", ",") if len(finished) else "0,0%")

    if finished.empty:
        st.info(
            "Ainda não há jogos finalizados na base. A Copa 2026 ainda não começou; preencha os placares reais no CSV de resultados quando os jogos acontecerem."
        )
    else:
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
                    "Taxa de acerto": pct(hit_rows["_pick_probability"].mean()).replace(".", ",") if len(hit_rows) else "0,0%",
                },
                {
                    "Métrica": "Confiança média nos erros",
                    "Taxa de acerto": pct(miss_rows["_pick_probability"].mean()).replace(".", ",") if len(miss_rows) else "0,0%",
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
    st.dataframe(
        tracking[
            [
                "Data",
                "Grupo",
                "Jogo",
                "Palpite estatístico",
                "Confiança",
                "Prob. do palpite",
                "Placar previsto",
                "Resultado real",
                "Placar real",
                "Acertou resultado",
                "Acertou placar",
                "Acertou gols",
                "Acertou ambos",
                "Status",
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
                --ink: #101828;
                --muted: #667085;
                --line: #d0d5dd;
                --surface: #ffffff;
                --surface-soft: #f8fafc;
                --green: #067647;
                --blue: #175cd3;
                --gold: #b54708;
                --red: #b42318;
            }

            .main .block-container {
                padding-top: 2rem;
                max-width: 1220px;
            }

            h1, h2, h3 {
                color: var(--ink);
                letter-spacing: 0;
            }

            [data-testid="stSidebar"] {
                background: #f7fafc;
                border-right: 1px solid var(--line);
            }

            .hero {
                border: 1px solid var(--line);
                background: #071827;
                min-height: 285px;
                border-radius: 8px;
                margin-bottom: 20px;
                overflow: hidden;
                position: relative;
            }

            .hero-img {
                height: 100%;
                inset: 0;
                object-fit: cover;
                position: absolute;
                width: 100%;
            }

            .hero-overlay {
                background: linear-gradient(90deg, rgba(6, 24, 38, .94) 0%, rgba(6, 24, 38, .74) 48%, rgba(6, 24, 38, .18) 100%);
                inset: 0;
                position: absolute;
            }

            .hero-content {
                padding: 32px 34px;
                position: relative;
                z-index: 1;
            }

            .hero-title {
                font-size: 34px;
                font-weight: 750;
                color: #ffffff;
                margin-bottom: 6px;
                max-width: 620px;
            }

            .hero-copy {
                color: rgba(255, 255, 255, .82);
                font-size: 16px;
                max-width: 760px;
            }

            .hero-meta {
                color: #a7f3d0;
                font-size: 13px;
                font-weight: 750;
                letter-spacing: .06em;
                margin-bottom: 8px;
                text-transform: uppercase;
            }

            .match-strip {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 18px 20px;
                background: var(--surface);
                margin: 12px 0 18px;
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
                color: var(--muted);
                font-size: 13px;
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
                border-radius: 8px;
                background: var(--surface);
                padding: 16px 18px;
                min-height: 124px;
            }

            .decision-card.primary {
                background: #063f2f;
                border-color: #063f2f;
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
                border-left: 5px solid var(--green);
                border-radius: 8px;
                background: var(--surface);
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

            div[data-testid="stMetric"] {
                border: 1px solid var(--line);
                background: var(--surface);
                padding: 14px 16px;
                border-radius: 8px;
            }

            div[data-testid="stMetricLabel"] {
                color: var(--muted);
            }

            .author-panel {
                align-items: center;
                background: #101828;
                border-radius: 8px;
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
                background: #0a66c2;
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
                .decision-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
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

    st.markdown(
        f"""
        <div class="hero">
            <img class="hero-img" src="{cover_uri}" alt="Capa com futebol e ciência de dados">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-meta">Modelo estatístico para bolão</div>
                <div class="hero-title">Copa dos Dados 2026</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Navegação")
    selected_page = st.sidebar.radio(
        "Página",
        ["Análise dos jogos", "Estatísticas de acertos"],
        label_visibility="collapsed",
    )

    if selected_page == "Estatísticas de acertos":
        render_accuracy_dashboard(fixtures, results, ratings, results_source)
        st.caption(
            "Dados de seleções e grupos da Copa 2026 atualizados em maio de 2026 a partir do calendário oficial da FIFA e da consolidação pública da competição."
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
                <div class="pick-label">Recomendação estatística</div>
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
                f"{team_label(pick)} tem a maior probabilidade projetada, combinando rating histórico, força ofensiva e gols esperados."
            )
        st.write(
            f"O modelo projeta **{expected_home + expected_away:.2f} gols** na partida, com placar modal **{score}**."
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
                    "Rating": float(ratings.loc[ratings["team"] == home, "rating"].iloc[0]),
                    "Gols pro/jogo": float(
                        ratings.loc[ratings["team"] == home, "goals_for_per_match"].iloc[0]
                    ),
                    "Gols contra/jogo": float(
                        ratings.loc[ratings["team"] == home, "goals_against_per_match"].iloc[0]
                    ),
                },
                {
                    "Seleção": team_label(away),
                    "Rating": float(ratings.loc[ratings["team"] == away, "rating"].iloc[0]),
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

