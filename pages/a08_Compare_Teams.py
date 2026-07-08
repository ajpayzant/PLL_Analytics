import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import query_df, table_exists, filter_values
from shared.ui import (
    apply_css, safe_bar_chart, display_table, display_comparison_matrix,
    fmt_value, pretty_col, profile_summary_cards
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Compare Teams · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Compare Teams")
st.markdown('<div class="section-note">Compare teams across multi-year profile, current form, season trends, and head-to-head splits.</div>', unsafe_allow_html=True)

team_options = teams_df["team_name"].dropna().tolist()

selected_compare_teams = st.multiselect(
    "Select 2–4 teams",
    options=team_options,
    default=team_options[:2] if len(team_options) >= 2 else team_options,
    key="compare_teams"
)

if len(selected_compare_teams) < 2:
    st.info("Select at least two teams to compare.")
else:
    team_ids = teams_df[teams_df["team_name"].isin(selected_compare_teams)]["team_id"].tolist()
    placeholders = ", ".join(["?"] * len(team_ids))

    team_context = st.radio(
        "Comparison context",
        options=["Career", "Last 5", "Last 10", "Season"],
        horizontal=True,
        key="team_compare_context"
    )

    if team_context == "Career":
        compare_df = query_df(f"""
            WITH record AS (
                SELECT
                    team_id,
                    SUM(CASE WHEN scores > scores_against THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN scores < scores_against THEN 1 ELSE 0 END) AS losses,
                    CASE
                        WHEN COUNT(*) > 0
                        THEN SUM(CASE WHEN scores > scores_against THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
                        ELSE NULL
                    END AS win_pct
                FROM clean.team_game_stats
                GROUP BY team_id
            )
            SELECT c.*, r.wins, r.losses, r.win_pct
            FROM marts.team_career_stats c
            LEFT JOIN record r ON c.team_id = r.team_id
            WHERE c.team_id IN ({placeholders})
            ORDER BY c.scores_per_game DESC NULLS LAST
        """, team_ids)

    elif team_context == "Last 5":
        compare_df = query_df(f"""
            SELECT * FROM marts.team_last5_stats
            WHERE team_id IN ({placeholders})
            ORDER BY scores_per_game DESC NULLS LAST
        """, team_ids)

    elif team_context == "Last 10":
        compare_df = query_df(f"""
            SELECT * FROM marts.team_last10_stats
            WHERE team_id IN ({placeholders})
            ORDER BY scores_per_game DESC NULLS LAST
        """, team_ids)

    else:
        selected_compare_season = st.selectbox(
            "Season",
            options=seasons,
            index=len(seasons) - 1,
            key="team_compare_season"
        )
        compare_df = query_df(f"""
            SELECT * FROM marts.team_season_stats
            WHERE team_id IN ({placeholders})
              AND season = ?
            ORDER BY scores_per_game DESC NULLS LAST
        """, team_ids + [selected_compare_season])

    st.markdown("### Selected Team Snapshot")

    profile_summary_cards(
        compare_df,
        title_col="team_name",
        specs=[
            ("Games", "games"),
            ("Wins", "wins"),
            ("Losses", "losses"),
            ("Win %", "win_pct", True),
            ("Scores/G", "scores_per_game"),
            ("Shots/G", "shots_per_game"),
        ],
        columns=4
    )

    st.markdown("### Comparison Matrix")

    team_compare_metrics = [
        "games", "wins", "losses", "win_pct", "scores", "scores_per_game",
        "goals", "assists", "shots", "shots_per_game", "turnovers",
        "turnovers_per_game", "saves", "ground_balls", "caused_turnovers",
        "faceoff_pct_calc", "clear_pct_calc", "touches", "total_passes",
        "time_in_possession", "offensive_sequence_proxy",
        "offensive_sequence_proxy_per_game"
    ]

    display_comparison_matrix(compare_df, "team_name", team_compare_metrics, height=500)

    st.markdown("### Defensive Comparison Matrix")

    if table_exists("marts", "team_defense_season_stats"):
        if team_context == "Career":
            defense_compare_df = query_df(f"""
                SELECT * FROM marts.team_defense_career_stats
                WHERE team_id IN ({placeholders})
                ORDER BY scores_allowed_per_game ASC NULLS LAST
            """, team_ids)

        elif team_context == "Season":
            defense_compare_df = query_df(f"""
                SELECT * FROM marts.team_defense_season_stats
                WHERE team_id IN ({placeholders})
                  AND season = ?
                ORDER BY scores_allowed_per_game ASC NULLS LAST
            """, team_ids + [selected_compare_season])

        else:
            n_games = 5 if team_context == "Last 5" else 10
            defense_compare_df = query_df(f"""
                WITH ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY team_id
                            ORDER BY game_date_utc DESC, season DESC, game_number DESC
                        ) AS rn
                    FROM marts.team_game_opponent_context
                    WHERE team_id IN ({placeholders})
                ),
                windowed AS (SELECT * FROM ranked WHERE rn <= {n_games})
                SELECT
                    team_id,
                    ANY_VALUE(team_name) AS team_name,
                    COUNT(DISTINCT game_id) AS games,
                    SUM(team_scores) AS team_scores,
                    SUM(scores_allowed) AS scores_allowed,
                    SUM(goals_allowed) AS goals_allowed,
                    SUM(opponent_shots) AS opponent_shots,
                    SUM(opponent_shots_on_goal) AS opponent_shots_on_goal,
                    SUM(saves_for) AS saves_for,
                    SUM(caused_turnovers_for) AS caused_turnovers_for,
                    SUM(opponent_turnovers) AS opponent_turnovers,
                    SUM(opponent_touches) AS opponent_touches,
                    SUM(opponent_offensive_sequence_proxy) AS opponent_offensive_sequence_proxy,
                    SUM(score_margin) AS score_margin,
                    SUM(scores_allowed)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS scores_allowed_per_game,
                    SUM(goals_allowed)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS goals_allowed_per_game,
                    SUM(opponent_shots)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS opponent_shots_per_game,
                    SUM(opponent_shots_on_goal)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS opponent_shots_on_goal_per_game,
                    SUM(caused_turnovers_for)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS caused_turnovers_for_per_game,
                    SUM(opponent_turnovers)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS opponent_turnovers_per_game,
                    SUM(score_margin)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS score_margin_per_game,
                    SUM(goals_allowed)::DOUBLE / NULLIF(SUM(opponent_shots), 0) AS opponent_goal_pct,
                    SUM(opponent_shots_on_goal)::DOUBLE / NULLIF(SUM(opponent_shots), 0) AS opponent_sog_rate,
                    SUM(saves_for)::DOUBLE / NULLIF(SUM(saves_for) + SUM(goals_allowed), 0) AS save_pct_proxy,
                    SUM(caused_turnovers_for)::DOUBLE / NULLIF(SUM(opponent_turnovers), 0) AS ct_per_opponent_turnover,
                    SUM(scores_allowed)::DOUBLE / NULLIF(SUM(opponent_offensive_sequence_proxy), 0) AS opponent_scores_per_offensive_sequence_proxy
                FROM windowed
                GROUP BY team_id
                ORDER BY scores_allowed_per_game ASC NULLS LAST
            """, team_ids)

        defense_compare_metrics = [
            "games", "scores_allowed_per_game", "goals_allowed_per_game",
            "opponent_shots_per_game", "opponent_shots_on_goal_per_game",
            "opponent_goal_pct", "opponent_sog_rate", "save_pct_proxy",
            "caused_turnovers_for_per_game", "opponent_turnovers_per_game",
            "ct_per_opponent_turnover", "opponent_touches",
            "opponent_offensive_sequence_proxy",
            "opponent_scores_per_offensive_sequence_proxy", "score_margin_per_game",
        ]

        display_comparison_matrix(defense_compare_df, "team_name", defense_compare_metrics, height=500)

        defense_chart_options = [m for m in defense_compare_metrics if m in defense_compare_df.columns]

        if defense_chart_options:
            defense_chart_metric = st.selectbox(
                "Defensive chart metric",
                options=defense_chart_options,
                index=1 if "scores_allowed_per_game" in defense_chart_options else 0,
                format_func=pretty_col,
                key="team_compare_defense_chart_metric"
            )

            safe_bar_chart(
                defense_compare_df.sort_values(defense_chart_metric),
                x_col="team_name", y_col=defense_chart_metric, color_col="team_name",
                title=f"{team_context} Defensive Comparison — {pretty_col(defense_chart_metric)}",
                orientation="h"
            )
    else:
        st.info("Defensive/opponent marts are not available in the warehouse yet.")

    st.markdown("### Visual Comparison")

    chart_metric = st.selectbox(
        "Chart metric",
        options=[m for m in team_compare_metrics if m in compare_df.columns],
        index=5 if "scores_per_game" in compare_df.columns else 0,
        format_func=pretty_col,
        key="team_compare_chart_metric"
    )

    safe_bar_chart(
        compare_df.sort_values(chart_metric),
        x_col="team_name", y_col=chart_metric, color_col="team_name",
        title=f"{team_context} Comparison — {pretty_col(chart_metric)}",
        orientation="h"
    )
