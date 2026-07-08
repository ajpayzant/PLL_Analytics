import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
from html import escape
from shared.db import query_df, schedule_display_table, table_exists, filter_values
from shared.ui import (
    apply_css, stat_card, safe_bar_chart, display_table, display_comparison_matrix,
    fmt_value, pretty_col, profile_header, profile_summary_cards,
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Matchup Preview · PLL Analytics", page_icon="🥍", layout="wide")
apply_css()

import os
from shared.db import DB_PATH
if not os.path.exists(DB_PATH):
    st.error(f"DuckDB warehouse not found: {DB_PATH}")
    st.stop()

try:
    seasons, teams_df, players_df, positions, selected_seasons, selected_teams, selected_positions, min_games = render_sidebar_filters()
except Exception as e:
    st.error("Failed to load PLL warehouse.")
    st.exception(e)
    st.stop()


# ============================================================
# LOCAL HELPER FUNCTIONS (Matchup-only)
# ============================================================

def mmss_from_seconds(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        seconds = int(round(float(x)))
    except Exception:
        return "—"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{sign}{minutes}:{secs:02d}"


def format_pct_safe(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def render_matchup_scoreboard(matchup, away_name, home_name):
    away_score_raw = matchup.get("away_score", np.nan)
    home_score_raw = matchup.get("home_score", np.nan)
    status = str(matchup.get("status_display", matchup.get("event_status_label", "—"))).title()
    game_num = fmt_value(matchup.get("game_number", np.nan), 0)
    game_date = matchup.get("game_date_display", "—")
    away_score = fmt_value(away_score_raw, 0)
    home_score = fmt_value(home_score_raw, 0)
    away_score_numeric = pd.to_numeric(pd.Series([away_score_raw]), errors="coerce").iloc[0]
    home_score_numeric = pd.to_numeric(pd.Series([home_score_raw]), errors="coerce").iloc[0]
    away_class = ""
    home_class = ""
    if pd.notna(away_score_numeric) and pd.notna(home_score_numeric):
        if away_score_numeric > home_score_numeric:
            away_class = "winner"
        elif home_score_numeric > away_score_numeric:
            home_class = "winner"
    st.markdown(
        f"""
        <style>
            .matchup-scoreboard {{
                display: grid;
                grid-template-columns: 1fr auto 1fr;
                gap: 14px;
                align-items: stretch;
                margin: 8px 0 18px 0;
            }}
            .matchup-team-card {{
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 18px;
                padding: 18px 20px;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            }}
            .matchup-team-card.winner {{
                border-color: rgba(22, 163, 74, 0.55);
                box-shadow: 0 10px 28px rgba(22, 163, 74, 0.14);
            }}
            .matchup-team-label {{
                font-size: 0.75rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                color: #64748b;
                margin-bottom: 6px;
            }}
            .matchup-team-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
            }}
            .matchup-team-name {{
                font-size: 1.35rem;
                font-weight: 850;
                color: #0f172a;
                line-height: 1.1;
            }}
            .matchup-score {{
                font-size: 2.1rem;
                font-weight: 950;
                color: #0f172a;
                min-width: 64px;
                text-align: right;
            }}
            .matchup-vs-card {{
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 76px;
                border-radius: 18px;
                background: #0f172a;
                color: white;
                font-weight: 900;
                letter-spacing: 0.08em;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16);
            }}
            .matchup-meta {{
                font-size: 0.82rem;
                color: #475569;
                margin-top: 8px;
                font-weight: 600;
            }}
        </style>
        <div class="matchup-scoreboard">
            <div class="matchup-team-card {away_class}">
                <div class="matchup-team-label">Away</div>
                <div class="matchup-team-row">
                    <div class="matchup-team-name">{escape(str(away_name))}</div>
                    <div class="matchup-score">{escape(str(away_score))}</div>
                </div>
                <div class="matchup-meta">Game {escape(str(game_num))} · {escape(str(game_date))}</div>
            </div>
            <div class="matchup-vs-card">AT</div>
            <div class="matchup-team-card {home_class}">
                <div class="matchup-team-label">Home</div>
                <div class="matchup-team-row">
                    <div class="matchup-team-name">{escape(str(home_name))}</div>
                    <div class="matchup-score">{escape(str(home_score))}</div>
                </div>
                <div class="matchup-meta">Status: {escape(str(status))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def build_team_boxscore_matrix(team_box_df, away_id, away_name, home_id, home_name):
    if team_box_df is None or len(team_box_df) == 0:
        return pd.DataFrame()
    away = team_box_df[team_box_df["team_id"] == away_id]
    home = team_box_df[team_box_df["team_id"] == home_id]
    if len(away) == 0 or len(home) == 0:
        return pd.DataFrame()
    away = away.iloc[0]
    home = home.iloc[0]
    stat_specs = [
        ("Score", "scores", "number"),
        ("Goals", "goals", "number"),
        ("1PT Goals", "one_point_goals", "number"),
        ("2PT Goals", "two_point_goals", "number"),
        ("Assists", "assists", "number"),
        ("Shots", "shots", "number"),
        ("Shots on Goal", "shots_on_goal", "number"),
        ("Shot %", "shot_pct", "pct"),
        ("Ground Balls", "ground_balls", "number"),
        ("Turnovers", "turnovers", "number"),
        ("Caused Turnovers", "caused_turnovers", "number"),
        ("Faceoffs Won", "faceoffs_won", "number"),
        ("Faceoffs Lost", "faceoffs_lost", "number"),
        ("Faceoff %", "faceoff_pct", "pct"),
        ("Saves", "saves", "number"),
        ("Save %", "save_pct", "pct"),
        ("Penalties", "num_penalties", "number"),
        ("PIM", "pim", "number"),
        ("Touches", "touches", "number"),
        ("Passes", "total_passes", "number"),
        ("Possession Time", "time_in_possession", "time"),
        ("Possession %", "time_in_possession_pct", "pct"),
        ("Official Possessions", "official_total_possessions", "number"),
        ("Offensive Sequences", "offensive_sequence_proxy", "number"),
    ]
    rows = []
    for label, col, kind in stat_specs:
        if col not in team_box_df.columns:
            continue
        away_val = away.get(col, np.nan)
        home_val = home.get(col, np.nan)
        if kind == "time":
            away_fmt = mmss_from_seconds(away_val)
            home_fmt = mmss_from_seconds(home_val)
        elif kind == "pct":
            away_fmt = format_pct_safe(away_val)
            home_fmt = format_pct_safe(home_val)
        else:
            away_fmt = fmt_value(away_val, 2)
            home_fmt = fmt_value(home_val, 2)
        rows.append({away_name: away_fmt, "Stat": label, home_name: home_fmt})
    return pd.DataFrame(rows)


def render_completed_game_review(matchup, away_id, away_name, home_id, home_name):
    status = str(matchup.get("status_display", "")).lower()
    if status != "final":
        return
    selected_game_id = matchup.get("event_id", None)
    if selected_game_id is None or pd.isna(selected_game_id):
        st.info("No completed-game ID was found for this matchup.")
        return
    st.markdown("### Completed Game Review")
    team_box = query_df("""
        SELECT
            team_id, team_name, opponent_team_id, opponent_team_name, result,
            scores, scores_against, goals, one_point_goals, two_point_goals,
            assists, shots, shot_pct, shots_on_goal, ground_balls, turnovers,
            caused_turnovers, faceoffs, faceoffs_won, faceoffs_lost, faceoff_pct,
            saves, save_pct, goals_against, num_penalties, pim, touches,
            total_passes, time_in_possession, time_in_possession_pct,
            official_total_possessions, offensive_sequence_proxy, possession_data_status
        FROM clean.team_game_stats
        WHERE game_id = ?
        ORDER BY CASE WHEN team_id = ? THEN 0 WHEN team_id = ? THEN 1 ELSE 2 END
    """, [selected_game_id, away_id, home_id])
    box_matrix = build_team_boxscore_matrix(team_box, away_id, away_name, home_id, home_name)
    if len(box_matrix) > 0:
        st.markdown("#### Team Box Score")
        display_table(box_matrix, height=520)
    else:
        st.info("No team box score rows found for this completed game.")
    if len(team_box) > 0 and "possession_data_status" in team_box.columns:
        statuses = sorted(team_box["possession_data_status"].dropna().astype(str).unique().tolist())
        if statuses:
            st.caption("Possession data status: " + ", ".join(statuses) + ". Possession time is shown as MM:SS.")
    player_box = query_df("""
        SELECT
            team_id, team_name, position, position_name, full_name,
            points, scoring_points, one_point_goals, two_point_goals, goals, assists,
            shots, shot_pct, shots_on_goal, shots_on_goal_rate, ground_balls,
            turnovers, caused_turnovers, num_penalties, pim, touches, total_passes,
            fo_record, faceoff_pct, faceoffs, faceoffs_won, faceoffs_lost,
            scores_against, goals_against, saves, save_pct, clean_saves, messy_saves
        FROM clean.player_game_stats
        WHERE game_id = ?
        ORDER BY team_name, points DESC NULLS LAST, goals DESC NULLS LAST, assists DESC NULLS LAST, shots DESC NULLS LAST
    """, [selected_game_id])
    if len(player_box) == 0:
        st.info("No player box score rows found for this completed game.")
        return
    st.markdown("#### Player Box Score")
    team_options = [(away_name, away_id), (home_name, home_id)]
    selected_team_label = st.radio(
        "Player box score team",
        options=[x[0] for x in team_options],
        horizontal=True,
        key=f"completed_box_team_{selected_game_id}"
    )
    selected_team_id = dict(team_options)[selected_team_label]
    selected_players = player_box[player_box["team_id"] == selected_team_id].copy()
    if len(selected_players) == 0:
        st.info("No players found for selected team.")
        return
    offensive_cols = ["full_name", "position", "points", "scoring_points", "goals", "one_point_goals", "two_point_goals", "assists", "shots", "shot_pct", "shots_on_goal", "ground_balls", "turnovers", "touches", "total_passes"]
    defensive_cols = ["full_name", "position", "caused_turnovers", "ground_balls", "points", "num_penalties", "pim", "shots", "touches", "total_passes"]
    faceoff_cols = ["full_name", "position", "fo_record", "faceoff_pct", "faceoffs", "faceoffs_won", "faceoffs_lost", "points", "assists", "ground_balls", "shots", "touches"]
    goalie_cols = ["full_name", "position", "scores_against", "goals_against", "save_pct", "saves", "clean_saves", "messy_saves", "touches", "total_passes"]
    _box_view = st.radio(
        "Player box score view",
        options=["Offense", "Defense", "Faceoff", "Goalie"],
        horizontal=True,
        key=f"completed_box_view_{selected_game_id}",
    )
    if _box_view == "Offense":
        off_df = selected_players[[c for c in offensive_cols if c in selected_players.columns]].sort_values(
            [c for c in ["points", "goals", "assists", "shots"] if c in selected_players.columns], ascending=False
        )
        display_table(off_df, height=420)
    elif _box_view == "Defense":
        def_df = selected_players.copy()
        if "caused_turnovers" in def_df.columns:
            def_df = def_df[
                (pd.to_numeric(def_df["caused_turnovers"], errors="coerce").fillna(0) > 0)
                | (def_df["position"].astype(str).isin(["D", "LSM", "SSDM", "G"]))
                | (pd.to_numeric(def_df.get("ground_balls", 0), errors="coerce").fillna(0) > 0)
            ]
        def_df = def_df[[c for c in defensive_cols if c in def_df.columns]].sort_values(
            [c for c in ["caused_turnovers", "ground_balls", "touches"] if c in def_df.columns], ascending=False
        )
        display_table(def_df, height=420)
    elif _box_view == "Faceoff":
        fo_df = selected_players.copy()
        if "faceoffs" in fo_df.columns:
            fo_df = fo_df[
                (fo_df["position"].astype(str).isin(["FO", "FOS"]))
                | (pd.to_numeric(fo_df["faceoffs"], errors="coerce").fillna(0) > 0)
            ]
        fo_df = fo_df[[c for c in faceoff_cols if c in fo_df.columns]].sort_values(
            [c for c in ["faceoffs", "faceoffs_won", "ground_balls"] if c in fo_df.columns], ascending=False
        )
        display_table(fo_df, height=360)
    else:
        goalie_df = selected_players.copy()
        if "saves" in goalie_df.columns:
            goalie_df = goalie_df[
                (goalie_df["position"].astype(str) == "G")
                | (pd.to_numeric(goalie_df["saves"], errors="coerce").fillna(0) > 0)
                | (pd.to_numeric(goalie_df.get("scores_against", 0), errors="coerce").fillna(0) > 0)
            ]
        goalie_df = goalie_df[[c for c in goalie_cols if c in goalie_df.columns]].sort_values(
            [c for c in ["saves", "save_pct"] if c in goalie_df.columns], ascending=False
        )
        display_table(goalie_df, height=320)


def pll_safe_number(x):
    if x is None or pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def pll_clock_from_seconds(x, total=False):
    val = pll_safe_number(x)
    if pd.isna(val):
        return "—"
    seconds = int(round(val))
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if total and hours > 0:
        return f"{sign}{hours}:{minutes:02d}:{secs:02d}"
    return f"{sign}{minutes}:{secs:02d}"


def pll_fmt_profile_value(row, col, kind="number", decimals=2):
    if row is None or col not in row.index:
        return "—"
    val = row.get(col, np.nan)
    if val is None or pd.isna(val):
        return "—"
    if kind == "time_pg":
        return pll_clock_from_seconds(val, total=False)
    if kind == "time_total":
        return pll_clock_from_seconds(val, total=True)
    if kind == "pct":
        try:
            return f"{float(val):.{decimals}%}"
        except Exception:
            return "—"
    if kind == "int":
        try:
            return f"{int(round(float(val))):,}"
        except Exception:
            return "—"
    if kind == "record":
        return str(val)
    try:
        num = float(val)
        if abs(num - round(num)) < 0.0000001:
            return f"{int(round(num)):,}"
        return f"{num:,.{decimals}f}"
    except Exception:
        return str(val)


def pll_add_team_profile_derived_cols(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    numeric_cols = [
        "games", "wins", "losses", "scores", "goals", "one_point_goals", "two_point_goals",
        "assists", "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers",
        "faceoffs", "faceoffs_won", "faceoffs_lost", "saves", "clean_saves", "messy_saves",
        "num_penalties", "pim", "touches", "total_passes", "time_in_possession",
        "official_total_possessions", "offensive_sequence_proxy"
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "games" in out.columns:
        games = out["games"].replace(0, np.nan)
        for total_col in [
            "scores", "goals", "one_point_goals", "two_point_goals", "assists",
            "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers",
            "faceoffs", "faceoffs_won", "faceoffs_lost", "saves", "clean_saves", "messy_saves",
            "num_penalties", "pim", "touches", "total_passes", "time_in_possession",
            "official_total_possessions", "offensive_sequence_proxy"
        ]:
            pg_col = f"{total_col}_per_game"
            if total_col in out.columns and pg_col not in out.columns:
                out[pg_col] = out[total_col] / games
    if "wins" in out.columns and "losses" in out.columns:
        out["record_display"] = (
            out["wins"].fillna(0).round(0).astype(int).astype(str)
            + "-"
            + out["losses"].fillna(0).round(0).astype(int).astype(str)
        )
    if "win_pct" not in out.columns and {"wins", "games"}.issubset(out.columns):
        out["win_pct"] = out["wins"] / out["games"].replace(0, np.nan)
    if "shot_pct_calc" not in out.columns and {"goals", "shots"}.issubset(out.columns):
        out["shot_pct_calc"] = out["goals"] / out["shots"].replace(0, np.nan)
    if "shots_on_goal_rate_calc" not in out.columns and {"shots_on_goal", "shots"}.issubset(out.columns):
        out["shots_on_goal_rate_calc"] = out["shots_on_goal"] / out["shots"].replace(0, np.nan)
    if "faceoff_pct_calc" not in out.columns and {"faceoffs_won", "faceoffs"}.issubset(out.columns):
        out["faceoff_pct_calc"] = out["faceoffs_won"] / out["faceoffs"].replace(0, np.nan)
    if "save_pct_calc" not in out.columns and {"saves", "goals_against"}.issubset(out.columns):
        out["save_pct_calc"] = out["saves"] / (out["saves"] + out["goals_against"]).replace(0, np.nan)
    if "passes_per_touch" not in out.columns and {"total_passes", "touches"}.issubset(out.columns):
        out["passes_per_touch"] = out["total_passes"] / out["touches"].replace(0, np.nan)
    if "seconds_possession_per_touch" not in out.columns and {"time_in_possession", "touches"}.issubset(out.columns):
        out["seconds_possession_per_touch"] = out["time_in_possession"] / out["touches"].replace(0, np.nan)
    if "touches_per_offensive_sequence_proxy" not in out.columns and {"touches", "offensive_sequence_proxy"}.issubset(out.columns):
        out["touches_per_offensive_sequence_proxy"] = out["touches"] / out["offensive_sequence_proxy"].replace(0, np.nan)
    if "passes_per_offensive_sequence_proxy" not in out.columns and {"total_passes", "offensive_sequence_proxy"}.issubset(out.columns):
        out["passes_per_offensive_sequence_proxy"] = out["total_passes"] / out["offensive_sequence_proxy"].replace(0, np.nan)
    return out


def pll_profile_matrix(profile_df, away_id, away_name, home_id, home_name, specs):
    if profile_df is None or len(profile_df) == 0:
        return pd.DataFrame()
    away_row = profile_df[profile_df["team_id"].astype(str) == str(away_id)]
    home_row = profile_df[profile_df["team_id"].astype(str) == str(home_id)]
    if len(away_row) == 0 or len(home_row) == 0:
        return pd.DataFrame()
    away_row = away_row.iloc[0]
    home_row = home_row.iloc[0]
    rows = []
    for label, col, kind, decimals in specs:
        rows.append({
            away_name: pll_fmt_profile_value(away_row, col, kind=kind, decimals=decimals),
            "Stat": label,
            home_name: pll_fmt_profile_value(home_row, col, kind=kind, decimals=decimals),
        })
    return pd.DataFrame(rows)


def render_clean_matchup_team_profile(matchup_season, away_id, away_name, home_id, home_name):
    st.markdown("### Team Season Profile")
    profile_context = f"{matchup_season} Season"
    season_profiles = query_df("""
        SELECT * FROM marts.team_season_stats
        WHERE season = ? AND team_id IN (?, ?)
        ORDER BY team_name
    """, [matchup_season, away_id, home_id])
    if len(season_profiles) < 2:
        profile_context = "Career"
        season_profiles = query_df("""
            SELECT * FROM marts.team_career_stats
            WHERE team_id IN (?, ?)
            ORDER BY team_name
        """, [away_id, home_id])
    if len(season_profiles) == 0:
        st.info("No team profile data found for this matchup.")
        return
    season_profiles = pll_add_team_profile_derived_cols(season_profiles)
    away_profile = season_profiles[season_profiles["team_id"].astype(str) == str(away_id)]
    home_profile = season_profiles[season_profiles["team_id"].astype(str) == str(home_id)]
    if len(away_profile) == 0 or len(home_profile) == 0:
        st.info("Could not find both teams in the selected season profile. Try using career context.")
        return
    away_profile = away_profile.iloc[0]
    home_profile = home_profile.iloc[0]
    st.caption(f"Context: {profile_context}. Possession time is displayed as clock time, not raw seconds.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card(f"{away_name} Record", pll_fmt_profile_value(away_profile, "record_display", kind="record"))
    with c2:
        stat_card(f"{away_name} Scores/G", pll_fmt_profile_value(away_profile, "scores_per_game", decimals=2))
    with c3:
        stat_card(f"{home_name} Record", pll_fmt_profile_value(home_profile, "record_display", kind="record"))
    with c4:
        stat_card(f"{home_name} Scores/G", pll_fmt_profile_value(home_profile, "scores_per_game", decimals=2))
    per_game_specs = [
        ("Games", "games", "int", 0), ("Scores/G", "scores_per_game", "number", 2),
        ("Goals/G", "goals_per_game", "number", 2), ("1PT Goals/G", "one_point_goals_per_game", "number", 2),
        ("2PT Goals/G", "two_point_goals_per_game", "number", 2), ("Assists/G", "assists_per_game", "number", 2),
        ("Shots/G", "shots_per_game", "number", 2), ("Shots on Goal/G", "shots_on_goal_per_game", "number", 2),
        ("Ground Balls/G", "ground_balls_per_game", "number", 2), ("Turnovers/G", "turnovers_per_game", "number", 2),
        ("Caused Turnovers/G", "caused_turnovers_per_game", "number", 2), ("Saves/G", "saves_per_game", "number", 2),
        ("Touches/G", "touches_per_game", "number", 2), ("Passes/G", "total_passes_per_game", "number", 2),
        ("Possession/G", "time_in_possession_per_game", "time_pg", 0),
        ("Official Possessions/G", "official_total_possessions_per_game", "number", 2),
        ("Offensive Sequences/G", "offensive_sequence_proxy_per_game", "number", 2),
    ]
    total_specs = [
        ("Games", "games", "int", 0), ("Wins", "wins", "int", 0), ("Losses", "losses", "int", 0),
        ("Scores", "scores", "int", 0), ("Goals", "goals", "int", 0), ("1PT Goals", "one_point_goals", "int", 0),
        ("2PT Goals", "two_point_goals", "int", 0), ("Assists", "assists", "int", 0), ("Shots", "shots", "int", 0),
        ("Shots on Goal", "shots_on_goal", "int", 0), ("Ground Balls", "ground_balls", "int", 0),
        ("Turnovers", "turnovers", "int", 0), ("Caused Turnovers", "caused_turnovers", "int", 0),
        ("Faceoffs", "faceoffs", "int", 0), ("Faceoffs Won", "faceoffs_won", "int", 0),
        ("Faceoffs Lost", "faceoffs_lost", "int", 0), ("Saves", "saves", "int", 0),
        ("Touches", "touches", "int", 0), ("Passes", "total_passes", "int", 0),
        ("Total Possession Time", "time_in_possession", "time_total", 0),
        ("Official Possessions", "official_total_possessions", "number", 0),
        ("Offensive Sequences", "offensive_sequence_proxy", "int", 0),
    ]
    rate_specs = [
        ("Win %", "win_pct", "pct", 1), ("Shot %", "shot_pct_calc", "pct", 1),
        ("SOG Rate", "shots_on_goal_rate_calc", "pct", 1), ("Faceoff %", "faceoff_pct_calc", "pct", 1),
        ("Clear %", "clear_pct_calc", "pct", 1), ("Save %", "save_pct_calc", "pct", 1),
        ("Passes/Touch", "passes_per_touch", "number", 2),
        ("Touches/Sequence", "touches_per_offensive_sequence_proxy", "number", 2),
        ("Passes/Sequence", "passes_per_offensive_sequence_proxy", "number", 2),
        ("Seconds/Touch", "seconds_possession_per_touch", "number", 2),
    ]
    possession_specs = [
        ("Touches/G", "touches_per_game", "number", 2), ("Total Touches", "touches", "int", 0),
        ("Passes/G", "total_passes_per_game", "number", 2), ("Total Passes", "total_passes", "int", 0),
        ("Possession/G", "time_in_possession_per_game", "time_pg", 0),
        ("Total Possession Time", "time_in_possession", "time_total", 0),
        ("Official Possessions/G", "official_total_possessions_per_game", "number", 2),
        ("Official Possessions", "official_total_possessions", "number", 0),
        ("Offensive Sequences/G", "offensive_sequence_proxy_per_game", "number", 2),
        ("Offensive Sequences", "offensive_sequence_proxy", "int", 0),
        ("Passes/Touch", "passes_per_touch", "number", 2),
        ("Touches/Sequence", "touches_per_offensive_sequence_proxy", "number", 2),
        ("Seconds/Touch", "seconds_possession_per_touch", "number", 2),
    ]
    _profile_view = st.radio(
        "Team profile view",
        options=["Per Game", "Totals", "Rates", "Possession / Touches"],
        horizontal=True,
        key=f"matchup_team_profile_view_{away_id}_{home_id}",
    )
    if _profile_view == "Per Game":
        display_table(pll_profile_matrix(season_profiles, away_id, away_name, home_id, home_name, per_game_specs), height=520)
    elif _profile_view == "Totals":
        display_table(pll_profile_matrix(season_profiles, away_id, away_name, home_id, home_name, total_specs), height=560)
    elif _profile_view == "Rates":
        display_table(pll_profile_matrix(season_profiles, away_id, away_name, home_id, home_name, rate_specs), height=420)
    else:
        display_table(pll_profile_matrix(season_profiles, away_id, away_name, home_id, home_name, possession_specs), height=520)
        st.caption(
            "Possession time is based on the provider possession-time field converted from seconds. "
            "For games with missing or unusual provider possession timing, review the Possession Data QC section."
        )


def filter_team_string(df, team_id_col_value):
    if df is None or len(df) == 0:
        return df
    if "teams" not in df.columns:
        return df
    return df[df["teams"].fillna("").astype(str).str.contains(str(team_id_col_value), regex=False)].copy()


# ============================================================
# PAGE CONTENT
# ============================================================

st.subheader("Matchup Preview")
st.markdown('<div class="section-note">Select a scheduled or completed game and compare team form, season profile, head-to-head history, and key players.</div>', unsafe_allow_html=True)

schedule_fixed = schedule_display_table().copy()
schedule_fixed["game_date_display"] = pd.to_datetime(schedule_fixed["game_date_guess"], errors="coerce").dt.strftime("%Y-%m-%d")

matchup_status_filter = st.radio(
    "Game group",
    options=["Upcoming / Scheduled", "Completed / Final", "All Games"],
    horizontal=True,
    key="matchup_status_filter"
)

matchup_season = st.selectbox(
    "Matchup season",
    options=seasons,
    index=len(seasons) - 1 if seasons else 0,
    key="matchup_season"
)

matchup_pool = schedule_fixed[schedule_fixed["season"] == matchup_season].copy()

if matchup_status_filter == "Upcoming / Scheduled":
    matchup_pool = matchup_pool[matchup_pool["status_display"] != "final"].copy()
elif matchup_status_filter == "Completed / Final":
    matchup_pool = matchup_pool[matchup_pool["status_display"] == "final"].copy()

if len(matchup_pool) == 0:
    st.info("No games available for this filter.")
else:
    matchup_pool = matchup_pool.sort_values(["game_number", "game_date_guess"]).reset_index(drop=True)
    matchup_pool["matchup_label"] = (
        matchup_pool["season"].astype(str)
        + " G"
        + matchup_pool["game_number"].astype(str)
        + ": "
        + matchup_pool["away_team_name"].astype(str)
        + " at "
        + matchup_pool["home_team_name"].astype(str)
        + " — "
        + matchup_pool["game_date_display"].astype(str)
        + " — "
        + matchup_pool["status_display"].astype(str)
    )

    selected_matchup_label = st.selectbox(
        "Select game",
        options=matchup_pool["matchup_label"].tolist(),
        index=0,
        key="selected_matchup_label"
    )

    matchup = matchup_pool[matchup_pool["matchup_label"] == selected_matchup_label].iloc[0]

    away_id = matchup["away_team_id"]
    home_id = matchup["home_team_id"]
    away_name = matchup["away_team_name"]
    home_name = matchup["home_team_name"]

    profile_header(
        f"{away_name} at {home_name}",
        f"{matchup.get('game_date_display', '—')} | Season {matchup_season} | Game {fmt_value(matchup.get('game_number', np.nan), 0)} | Status: {matchup.get('status_display', '—')}"
    )

    render_matchup_scoreboard(matchup, away_name, home_name)
    render_completed_game_review(matchup, away_id, away_name, home_id, home_name)
    render_clean_matchup_team_profile(matchup_season, away_id, away_name, home_id, home_name)

    st.markdown("### Defense / Opponent Allowance Profile")

    if table_exists("marts", "team_defense_season_stats"):
        defense_profiles = query_df("""
            SELECT * FROM marts.team_defense_season_stats
            WHERE season = ? AND team_id IN (?, ?)
            ORDER BY team_name
        """, [matchup_season, away_id, home_id])

        if len(defense_profiles) < 2:
            defense_profiles = query_df("""
                SELECT * FROM marts.team_defense_career_stats
                WHERE team_id IN (?, ?)
                ORDER BY team_name
            """, [away_id, home_id])

        defense_matchup_metrics = [
            "games", "scores_allowed_per_game", "goals_allowed_per_game",
            "opponent_shots_per_game", "opponent_goal_pct", "opponent_sog_rate",
            "save_pct_proxy", "caused_turnovers_for_per_game",
            "opponent_turnovers_per_game", "ct_per_opponent_turnover", "score_margin_per_game",
        ]

        profile_summary_cards(
            defense_profiles,
            title_col="team_name",
            specs=[
                ("Scores Allowed/G", "scores_allowed_per_game"),
                ("Opp Shots/G", "opponent_shots_per_game"),
                ("Opp Goal %", "opponent_goal_pct", True),
                ("Save % Proxy", "save_pct_proxy", True),
                ("CT/G", "caused_turnovers_for_per_game"),
            ],
            columns=2
        )

        display_comparison_matrix(defense_profiles, "team_name", defense_matchup_metrics, height=420)

        matchup_defense_metric_options = [m for m in defense_matchup_metrics if m in defense_profiles.columns]

        if matchup_defense_metric_options:
            matchup_defense_metric = st.selectbox(
                "Defense matchup chart metric",
                options=matchup_defense_metric_options,
                index=1 if "scores_allowed_per_game" in matchup_defense_metric_options else 0,
                format_func=pretty_col,
                key="matchup_defense_metric"
            )

            safe_bar_chart(
                defense_profiles.sort_values(matchup_defense_metric),
                x_col="team_name",
                y_col=matchup_defense_metric,
                color_col="team_name",
                title=f"Matchup Defensive Comparison — {pretty_col(matchup_defense_metric)}",
                orientation="h"
            )
    else:
        st.info("Defensive/opponent marts are not available in the warehouse yet.")

    st.markdown("### Current Form")

    last5 = query_df("""
        SELECT * FROM marts.team_last5_stats
        WHERE team_id IN (?, ?)
        ORDER BY team_name
    """, [away_id, home_id])

    last10 = query_df("""
        SELECT * FROM marts.team_last10_stats
        WHERE team_id IN (?, ?)
        ORDER BY team_name
    """, [away_id, home_id])

    form_context = st.radio(
        "Form window",
        options=["Last 5", "Last 10"],
        horizontal=True,
        key="matchup_form_window"
    )

    form_df = last5 if form_context == "Last 5" else last10

    form_metrics = [
        "games", "scores_per_game", "shots_per_game", "turnovers_per_game",
        "saves_per_game", "faceoff_pct_calc", "clear_pct_calc",
        "touches_per_game", "time_in_possession_per_game",
        "offensive_sequence_proxy_per_game"
    ]

    display_comparison_matrix(form_df, "team_name", form_metrics, height=420)

    form_chart_metric = st.selectbox(
        "Form chart metric",
        options=[m for m in form_metrics if m in form_df.columns],
        index=1 if "scores_per_game" in form_df.columns else 0,
        format_func=pretty_col,
        key="matchup_form_metric"
    )

    safe_bar_chart(
        form_df.sort_values(form_chart_metric),
        x_col="team_name",
        y_col=form_chart_metric,
        color_col="team_name",
        title=f"{form_context} Matchup Comparison — {pretty_col(form_chart_metric)}",
        orientation="h"
    )

    st.markdown("### Head-to-Head History")

    h2h = query_df("""
        SELECT
            team_name, opponent_team_name, games, scores, scores_per_game,
            goals, assists, shots, shots_per_game, saves, turnovers,
            turnovers_per_game, ground_balls, caused_turnovers,
            faceoff_pct_calc, clear_pct_calc
        FROM marts.team_vs_opponent_stats
        WHERE (team_id = ? AND opponent_team_id = ?)
           OR (team_id = ? AND opponent_team_id = ?)
        ORDER BY team_name
    """, [away_id, home_id, home_id, away_id])

    h2h_metrics = [
        "games", "scores", "scores_per_game", "goals", "assists",
        "shots", "shots_per_game", "saves", "turnovers", "turnovers_per_game",
        "ground_balls", "caused_turnovers", "faceoff_pct_calc", "clear_pct_calc"
    ]

    display_comparison_matrix(h2h, "team_name", h2h_metrics, height=360)

    h2h_games = query_df("""
        WITH a AS (
            SELECT * FROM clean.team_game_stats WHERE team_id = ? AND opponent_team_id = ?
        ),
        b AS (
            SELECT * FROM clean.team_game_stats WHERE team_id = ? AND opponent_team_id = ?
        )
        SELECT
            a.season, a.game_number, a.game_date_utc,
            a.team_name AS team_a, a.scores AS team_a_score,
            b.team_name AS team_b, b.scores AS team_b_score,
            CASE WHEN a.scores > b.scores THEN a.team_name WHEN b.scores > a.scores THEN b.team_name ELSE 'Tie' END AS winner,
            a.shots AS team_a_shots, b.shots AS team_b_shots,
            a.turnovers AS team_a_turnovers, b.turnovers AS team_b_turnovers,
            a.ground_balls AS team_a_ground_balls, b.ground_balls AS team_b_ground_balls,
            a.caused_turnovers AS team_a_caused_turnovers, b.caused_turnovers AS team_b_caused_turnovers,
            a.time_in_possession AS team_a_possession, b.time_in_possession AS team_b_possession
        FROM a
        INNER JOIN b ON a.game_id = b.game_id
        ORDER BY a.season DESC, a.game_number DESC
    """, [away_id, home_id, home_id, away_id])

    with st.expander("Show head-to-head game log", expanded=False):
        display_table(h2h_games, height=320)

    st.markdown("### Key Player Form")

    player_form_source = st.radio(
        "Player form source",
        options=["Season", "Last 5", "Last 10"],
        horizontal=True,
        key="matchup_key_player_source"
    )

    player_form_metric = st.selectbox(
        "Key player metric",
        options=["points_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game"],
        index=0,
        format_func=pretty_col,
        key="matchup_key_player_metric"
    )

    if player_form_source == "Season":
        key_players = query_df("""
            SELECT full_name, position, teams, games, points, goals, assists, shots,
                   ground_balls, caused_turnovers, points_per_game, goals_per_game,
                   assists_per_game, shots_per_game, ground_balls_per_game, caused_turnovers_per_game
            FROM marts.player_season_stats
            WHERE season = ? AND games >= 1
            ORDER BY points_per_game DESC NULLS LAST
        """, [matchup_season])
    elif player_form_source == "Last 5":
        key_players = query_df("""
            SELECT full_name, position, teams, games, points, goals, assists, shots,
                   ground_balls, caused_turnovers, points_per_game, goals_per_game,
                   assists_per_game, shots_per_game, ground_balls_per_game, caused_turnovers_per_game
            FROM marts.player_last5_stats
            WHERE games >= 1
            ORDER BY points_per_game DESC NULLS LAST
        """)
    else:
        key_players = query_df("""
            SELECT full_name, position, teams, games, points, goals, assists, shots,
                   ground_balls, caused_turnovers, points_per_game, goals_per_game,
                   assists_per_game, shots_per_game, ground_balls_per_game, caused_turnovers_per_game
            FROM marts.player_last10_stats
            WHERE games >= 1
            ORDER BY points_per_game DESC NULLS LAST
        """)

    away_players = filter_team_string(key_players, away_id).sort_values(player_form_metric, ascending=False).head(10)
    home_players = filter_team_string(key_players, home_id).sort_values(player_form_metric, ascending=False).head(10)

    kp1, kp2 = st.columns(2)

    with kp1:
        st.markdown(f"#### {away_name} Key Players")
        safe_bar_chart(
            away_players.sort_values(player_form_metric),
            x_col="full_name",
            y_col=player_form_metric,
            color_col="position",
            title=f"{away_name} — {pretty_col(player_form_metric)}",
            orientation="h"
        )
        display_table(away_players, height=320)

    with kp2:
        st.markdown(f"#### {home_name} Key Players")
        safe_bar_chart(
            home_players.sort_values(player_form_metric),
            x_col="full_name",
            y_col=player_form_metric,
            color_col="position",
            title=f"{home_name} — {pretty_col(player_form_metric)}",
            orientation="h"
        )
        display_table(home_players, height=320)
