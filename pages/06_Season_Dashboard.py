import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from shared.db import query_df, schedule_display_table, table_exists, filter_values
from shared.ui import (
    apply_css, stat_card, safe_bar_chart, display_table, fmt_value, pretty_col,
    _pll_select_existing, _pll_safe_sort, _pll_add_possession_mmss, _pll_page_note
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Season Dashboard · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Season Dashboard")
st.markdown(
    '<div class="section-note">Review season-level team performance, player production, defensive context, and schedule status. Current-season views only include completed, stat-available games.</div>',
    unsafe_allow_html=True
)

selected_season_page = st.selectbox(
    "Select season",
    options=seasons,
    index=len(seasons) - 1 if seasons else 0,
    key="season_page_season"
)

schedule_fixed = schedule_display_table()
season_schedule = schedule_fixed[schedule_fixed["season"] == selected_season_page].copy()

season_completed = query_df("""
    SELECT COUNT(DISTINCT game_id) AS games
    FROM clean.game_manifest
    WHERE season = ?
""", [selected_season_page])["games"].iloc[0]

season_team_stats = query_df("""
    SELECT *
    FROM marts.team_season_stats
    WHERE season = ?
    ORDER BY scores_per_game DESC NULLS LAST
""", [selected_season_page])

season_team_stats = _pll_add_possession_mmss(season_team_stats)

season_player_stats = query_df("""
    SELECT *
    FROM marts.player_season_stats
    WHERE season = ?
    ORDER BY points DESC NULLS LAST
""", [selected_season_page])

k1, k2, k3, k4 = st.columns(4)

with k1:
    stat_card("Completed Games", fmt_value(season_completed, 0))

with k2:
    stat_card("Scheduled Games", fmt_value(len(season_schedule), 0))

with k3:
    stat_card("Teams", fmt_value(season_team_stats["team_name"].nunique() if len(season_team_stats) else 0, 0))

with k4:
    stat_card("Players", fmt_value(season_player_stats["full_name"].nunique() if len(season_player_stats) else 0, 0))

if seasons and selected_season_page == max(seasons):
    _pll_page_note(
        "Current Season",
        "This season is in progress. Records, ranks, rates, and leaderboards update when completed games become available in the warehouse."
    )

st.markdown("### Team Rankings")
st.caption("Team-level production and efficiency using completed, stat-available games only.")

team_metric_options = [
    c for c in [
        "scores_per_game",
        "score_margin_per_game",
        "shots_per_game",
        "touches_per_game",
        "time_in_possession_per_game",
        "offensive_sequence_proxy_per_game",
        "turnovers_per_game",
        "saves_per_game",
        "faceoff_pct_calc",
        "clear_pct_calc",
    ]
    if c in season_team_stats.columns
]

team_metric = st.selectbox(
    "Team ranking metric",
    options=team_metric_options,
    index=0,
    format_func=pretty_col,
    key="season_team_metric"
)

ranked_teams = _pll_safe_sort(season_team_stats, team_metric, lower_is_better=False)

safe_bar_chart(
    ranked_teams.head(12).sort_values(team_metric),
    x_col="team_name",
    y_col=team_metric,
    color_col="team_name",
    title=f"{selected_season_page} Team Rankings — {pretty_col(team_metric)}",
    orientation="h"
)

team_summary_cols = _pll_select_existing(
    ranked_teams,
    [
        "team_name", "games", "wins", "losses", "win_pct",
        "scores_per_game", "score_margin_per_game", "shots_per_game",
        "touches_per_game", "time_in_possession_per_game_mmss",
        "turnovers_per_game", "saves_per_game", "faceoff_pct_calc", "clear_pct_calc",
    ]
)

display_table(ranked_teams[team_summary_cols], height=360)

with st.expander("Advanced team season table", expanded=False):
    team_advanced_cols = _pll_select_existing(
        ranked_teams,
        [
            "season", "team_name", "games", "wins", "losses", "win_pct",
            "scores", "scores_per_game", "goals", "assists",
            "shots", "shots_per_game", "shots_on_goal", "shots_on_goal_per_game",
            "ground_balls", "turnovers", "turnovers_per_game", "caused_turnovers",
            "saves", "saves_per_game", "faceoffs_won", "faceoffs", "faceoff_pct_calc",
            "clear_pct_calc", "touches", "touches_per_game", "total_passes", "total_passes_per_game",
            "time_in_possession", "time_in_possession_per_game_mmss", "offensive_sequence_proxy",
            "offensive_sequence_proxy_per_game"
        ]
    )
    display_table(ranked_teams[team_advanced_cols], height=420)

st.markdown("### Team Defensive / Opponent Rankings")
st.caption("Opponent-allowed metrics. For allowed/suppression metrics, lower is generally better unless noted.")

if table_exists("marts", "team_defense_season_stats"):
    season_defense_df = query_df("""
        SELECT *
        FROM marts.team_defense_season_stats
        WHERE season = ?
        ORDER BY scores_allowed_per_game ASC NULLS LAST
    """, [selected_season_page])

    defense_metric_options = [
        c for c in [
            "scores_allowed_per_game",
            "goals_allowed_per_game",
            "opponent_shots_per_game",
            "opponent_goal_pct",
            "opponent_sog_rate",
            "save_pct_proxy",
            "caused_turnovers_for_per_game",
            "opponent_turnovers_per_game",
            "ct_per_opponent_turnover",
            "score_margin_per_game",
        ]
        if c in season_defense_df.columns
    ]

    if defense_metric_options:
        defense_metric = st.selectbox(
            "Defensive ranking metric",
            options=defense_metric_options,
            index=0,
            format_func=pretty_col,
            key="season_defense_metric"
        )

        lower_is_better = {
            "scores_allowed_per_game",
            "goals_allowed_per_game",
            "opponent_shots_per_game",
            "opponent_goal_pct",
            "opponent_sog_rate",
            "opponent_scores_per_offensive_sequence_proxy",
        }

        ranked_defense = _pll_safe_sort(
            season_defense_df,
            defense_metric,
            lower_is_better=defense_metric in lower_is_better
        )

        safe_bar_chart(
            ranked_defense.head(12).sort_values(
                defense_metric,
                ascending=defense_metric not in lower_is_better
            ),
            x_col="team_name",
            y_col=defense_metric,
            color_col="team_name",
            title=f"{selected_season_page} Defensive Rankings — {pretty_col(defense_metric)}",
            orientation="h"
        )

        defense_summary_cols = _pll_select_existing(
            ranked_defense,
            [
                "team_name", "games",
                "scores_allowed_per_game", "goals_allowed_per_game",
                "opponent_shots_per_game", "opponent_goal_pct", "opponent_sog_rate",
                "save_pct_proxy", "caused_turnovers_for_per_game",
                "opponent_turnovers_per_game", "score_margin_per_game",
            ]
        )
        display_table(ranked_defense[defense_summary_cols], height=360)

        with st.expander("Advanced defensive / opponent table", expanded=False):
            display_table(ranked_defense, height=420)
    else:
        st.info("No defensive metric columns are available for this season.")
else:
    st.info("Defensive/opponent marts are not available in the warehouse yet.")

st.markdown("### Player Leaders")
st.caption("Player production leaderboard for the selected season. Use minimum games to manage early-season samples.")

player_filter_cols = st.columns([1.2, 1.0, 1.2, 0.8])

season_positions = sorted(season_player_stats["position"].dropna().unique().tolist()) if len(season_player_stats) else []

season_position_filter = player_filter_cols[0].multiselect(
    "Position filter",
    options=season_positions,
    default=[],
    key="season_page_position_filter"
)

season_min_games = player_filter_cols[1].number_input(
    "Minimum games",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
    key="season_page_min_games"
)

player_metric_options = [
    c for c in [
        "points", "points_per_game", "scoring_points", "scoring_points_per_game",
        "goals", "goals_per_game", "one_point_goals", "two_point_goals",
        "assists", "assists_per_game", "shots", "shots_per_game",
        "ground_balls", "caused_turnovers", "turnovers", "touches"
    ]
    if c in season_player_stats.columns
]

player_metric = player_filter_cols[2].selectbox(
    "Player ranking metric",
    options=player_metric_options,
    index=0,
    format_func=pretty_col,
    key="season_player_metric"
)

player_rows = player_filter_cols[3].number_input(
    "Rows",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
    key="season_player_rows"
)

filtered_season_players = season_player_stats.copy()

if "games" in filtered_season_players.columns:
    filtered_season_players = filtered_season_players[
        pd.to_numeric(filtered_season_players["games"], errors="coerce").fillna(0) >= season_min_games
    ]

if season_position_filter:
    filtered_season_players = filtered_season_players[
        filtered_season_players["position"].isin(season_position_filter)
    ]

filtered_season_players = _pll_safe_sort(filtered_season_players, player_metric, lower_is_better=False).head(int(player_rows))

safe_bar_chart(
    filtered_season_players.head(20).sort_values(player_metric),
    x_col="full_name",
    y_col=player_metric,
    color_col="position",
    title=f"{selected_season_page} Player Leaders — {pretty_col(player_metric)}",
    orientation="h"
)

player_summary_cols = _pll_select_existing(
    filtered_season_players,
    [
        "full_name", "position", "teams", "games",
        "points", "scoring_points", "one_point_goals", "two_point_goals",
        "goals", "assists", "shots", "ground_balls", "turnovers", "caused_turnovers",
        "touches", "points_per_game", "goals_per_game", "assists_per_game", "shots_per_game",
    ]
)
display_table(filtered_season_players[player_summary_cols], height=420)

with st.expander("Advanced player leader table", expanded=False):
    player_advanced_cols = _pll_select_existing(
        filtered_season_players,
        [
            "season", "full_name", "position", "teams", "games",
            "points", "points_per_game", "scoring_points", "scoring_points_per_game",
            "one_point_goals", "one_point_goals_per_game", "two_point_goals", "two_point_goals_per_game",
            "goals", "goals_per_game", "assists", "assists_per_game",
            "shots", "shots_per_game", "shots_on_goal", "shot_pct_calc",
            "ground_balls", "ground_balls_per_game", "turnovers", "turnovers_per_game",
            "caused_turnovers", "caused_turnovers_per_game", "touches", "touches_per_game"
        ]
    )
    display_table(filtered_season_players[player_advanced_cols], height=420)

st.markdown("### Season Schedule")

schedule_display = season_schedule.copy()

if len(schedule_display):
    if "away_team_name" in schedule_display.columns and "home_team_name" in schedule_display.columns:
        schedule_display["matchup"] = (
            schedule_display["away_team_name"].astype(str)
            + " at "
            + schedule_display["home_team_name"].astype(str)
        )

    if "away_score" in schedule_display.columns and "home_score" in schedule_display.columns:
        schedule_display["result"] = np.where(
            pd.to_numeric(schedule_display["away_score"], errors="coerce").notna()
            & pd.to_numeric(schedule_display["home_score"], errors="coerce").notna(),
            schedule_display["away_score"].astype(str)
            + " - "
            + schedule_display["home_score"].astype(str),
            "—"
        )

schedule_cols = _pll_select_existing(
    schedule_display,
    ["season", "game_number", "game_date_guess", "matchup", "result", "status_display", "slug"]
)

display_table(schedule_display[schedule_cols].sort_values("game_number"), height=360)
