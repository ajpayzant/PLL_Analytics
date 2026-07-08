import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
from shared.db import query_df, filter_values
from shared.ui import (
    apply_css, safe_bar_chart, display_table, display_comparison_matrix,
    fmt_value, pretty_col, profile_summary_cards, clean_chart_x, standardize_chart
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Compare Players · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Compare Players")
st.markdown('<div class="section-note">Compare players with profile cards, matrix-style summaries, trends, and recent-form splits.</div>', unsafe_allow_html=True)

player_names = players_df["full_name"].dropna().unique().tolist()

selected_compare_players = st.multiselect(
    "Select 2–6 players",
    options=player_names,
    default=player_names[:2] if len(player_names) >= 2 else player_names,
    key="compare_players"
)

if len(selected_compare_players) < 2:
    st.info("Select at least two players to compare.")
else:
    player_ids = players_df[players_df["full_name"].isin(selected_compare_players)]["player_id"].tolist()
    placeholders = ", ".join(["?"] * len(player_ids))

    compare_context = st.radio(
        "Comparison context",
        options=["Career", "Last 5", "Last 10", "Season"],
        horizontal=True,
        key="player_compare_context"
    )

    if compare_context == "Career":
        compare_df = query_df(f"""
            SELECT * FROM marts.player_career_stats
            WHERE player_id IN ({placeholders})
            ORDER BY points DESC NULLS LAST
        """, player_ids)

    elif compare_context == "Last 5":
        compare_df = query_df(f"""
            SELECT * FROM marts.player_last5_stats
            WHERE player_id IN ({placeholders})
            ORDER BY points_per_game DESC NULLS LAST
        """, player_ids)

    elif compare_context == "Last 10":
        compare_df = query_df(f"""
            SELECT * FROM marts.player_last10_stats
            WHERE player_id IN ({placeholders})
            ORDER BY points_per_game DESC NULLS LAST
        """, player_ids)

    else:
        selected_compare_season = st.selectbox(
            "Season",
            options=seasons,
            index=len(seasons) - 1,
            key="player_compare_season"
        )
        compare_df = query_df(f"""
            SELECT * FROM marts.player_season_stats
            WHERE player_id IN ({placeholders})
              AND season = ?
            ORDER BY points DESC NULLS LAST
        """, player_ids + [selected_compare_season])

    st.markdown("### Selected Player Snapshot")

    profile_summary_cards(
        compare_df,
        title_col="full_name",
        specs=[
            ("Position", "position"),
            ("Teams", "teams"),
            ("Games", "games"),
            ("Points/G", "points_per_game"),
            ("Goals/G", "goals_per_game"),
            ("Assists/G", "assists_per_game"),
        ],
        columns=3
    )

    st.markdown("### Comparison Matrix")

    player_compare_metrics = [
        "games", "points", "goals", "assists", "shots", "ground_balls",
        "turnovers", "caused_turnovers", "touches", "total_passes",
        "points_per_game", "goals_per_game", "assists_per_game",
        "shots_per_game", "ground_balls_per_game", "turnovers_per_game",
        "caused_turnovers_per_game", "shot_pct_calc", "shots_on_goal_rate_calc"
    ]

    display_comparison_matrix(compare_df, "full_name", player_compare_metrics, height=500)

    st.markdown("### Visual Comparison")

    chart_metric = st.selectbox(
        "Chart metric",
        options=[m for m in player_compare_metrics if m in compare_df.columns],
        index=0,
        format_func=pretty_col,
        key="player_compare_chart_metric"
    )

    safe_bar_chart(
        compare_df.sort_values(chart_metric),
        x_col="full_name",
        y_col=chart_metric,
        color_col="full_name",
        title=f"{compare_context} Comparison — {pretty_col(chart_metric)}",
        orientation="h"
    )

    st.markdown("### Season Trend")

    compare_seasons = query_df(f"""
        SELECT season, full_name, position, games, points, goals, assists, shots,
               ground_balls, caused_turnovers, points_per_game, goals_per_game,
               assists_per_game, shots_per_game
        FROM marts.player_season_stats
        WHERE player_id IN ({placeholders})
        ORDER BY season, full_name
    """, player_ids)

    trend_metric = st.selectbox(
        "Season trend metric",
        options=[c for c in ["points_per_game", "goals_per_game", "assists_per_game", "shots_per_game"] if c in compare_seasons.columns],
        format_func=pretty_col,
        key="player_compare_trend_metric"
    )

    if len(compare_seasons) > 0 and trend_metric:
        plot_df = clean_chart_x(compare_seasons, "season")
        fig = px.line(
            plot_df, x="season", y=trend_metric, color="full_name", markers=True,
            title=f"Player Season Trend — {pretty_col(trend_metric)}",
            labels={c: pretty_col(c) for c in plot_df.columns}
        )
        fig = standardize_chart(fig, category_x=True)
        st.plotly_chart(fig, use_container_width=True)
