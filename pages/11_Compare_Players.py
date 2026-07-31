"""
Compare Players — two to six players side by side.

The sidebar filters are not requested: every query here is driven by the player
multiselect and the context radio in the main panel.

The default selection now honours a player chosen on another page, so following a
name from the Rankings or Leaderboards lands with that player already loaded
instead of on whichever two happened to sort first.

The Last 5 / Last 10 contexts compared different eras. They read
`marts.player_last5_stats`, whose rows are each player's last five games played
in any season: 195 of the 400 players in it have windows that end before 2026 and
88 span two seasons. Putting Matt Abbott (last played August 2022) next to an
active player under the heading "Last 5" compared 2022 form against 2026 form
with nothing on screen saying so. Both contexts now read the season-scoped marts
and take a season, so the two sides of the comparison cover the same stretch of
lacrosse.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from shared import page as P
from shared.db import query_df
from shared.ui import (
    safe_bar_chart, display_comparison_matrix, pretty_col,
    profile_summary_cards, clean_chart_x, standardize_chart,
)

ctx = P.init_page(
    "Compare Players",
    "Compare players with profile cards, matrix-style summaries, trends, and "
    "recent-form splits.",
)

seasons = ctx.seasons
player_names = ctx.player_names

# A player picked elsewhere leads the default pair; otherwise fall back to the
# first two names so the page has something to show on a cold open.
incoming = P.selected_player()
if incoming in player_names:
    default_players = [incoming] + [n for n in player_names if n != incoming][:1]
else:
    default_players = player_names[:2]

selected_compare_players = st.multiselect(
    "Select 2–6 players",
    options=player_names,
    default=default_players,
    key="compare_players"
)

if len(selected_compare_players) < 2:
    st.info("Select at least two players to compare.")
else:
    player_ids = ctx.player_ids_for(selected_compare_players)
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

    elif compare_context in ("Last 5", "Last 10"):
        # Season-scoped, and the season is explicit. The league-wide last5 mart
        # holds each player's last games in any season, so this comparison used to
        # put one player's 2026 window beside another's 2022 window.
        form_season = st.selectbox(
            "Season for the form window",
            options=seasons,
            index=ctx.season_default_index(),
            key="player_compare_form_season",
            help="Both players' windows come from this season, so the comparison "
                 "covers the same stretch of lacrosse.",
        )
        form_table = ("marts.player_season_last5_stats" if compare_context == "Last 5"
                      else "marts.player_season_last10_stats")
        compare_df = query_df(f"""
            SELECT * FROM {form_table}
            WHERE player_id IN ({placeholders}) AND season = ?
            ORDER BY points_per_game DESC NULLS LAST
        """, player_ids + [form_season])

        missing = [n for n in selected_compare_players
                   if n not in set(compare_df.get("full_name", []))]
        if missing:
            st.info(f"No {form_season} games for: {', '.join(missing)}. "
                    "Pick a season they all played, or use Career.")

    else:
        selected_compare_season = st.selectbox(
            "Season",
            options=seasons,
            index=ctx.season_default_index(),
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
        "caused_turnovers_per_game", "shot_pct_calc", "shots_on_goal_rate_calc",
        # Phase 2 new metrics (present only after warehouse rebuild)
        "assist_conv_rate", "two_pt_conversion", "clean_save_rate",
        "assist_opportunities", "assist_opp_per_game",
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
        st.plotly_chart(fig, width="stretch")
