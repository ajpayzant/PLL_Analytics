import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import query_df, startup_counts, schedule_display_table, sql_in_filter
from shared.ui import apply_css, stat_card, safe_bar_chart, display_table, fmt_value, pretty_col
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Overview · PLL Analytics", page_icon="🥍", layout="wide")
apply_css()

import os
from shared.db import DB_PATH
if not os.path.exists(DB_PATH):
    st.error(f"DuckDB warehouse not found: {DB_PATH}")
    st.stop()

try:
    counts = startup_counts()
    seasons, teams_df, players_df, positions, selected_seasons, selected_teams, selected_positions, min_games = render_sidebar_filters()
except Exception as e:
    st.error("Failed to load PLL warehouse.")
    st.exception(e)
    st.stop()

st.subheader("Overview")
st.markdown('<div class="section-note">High-level warehouse status and leaguewide summary views.</div>', unsafe_allow_html=True)

cols = st.columns(5)
overview_cards = [
    ("Completed Games", counts["completed_games"]),
    ("Player-Game Rows", counts["player_game_rows"]),
    ("Team-Game Rows", counts["team_game_rows"]),
    ("Players", counts["players"]),
    ("Teams", counts["teams"]),
]

for i, (label, value) in enumerate(overview_cards):
    with cols[i]:
        stat_card(label, fmt_value(value, digits=0))

schedule_fixed = schedule_display_table()

season_counts = query_df("""
    SELECT
        season,
        COUNT(DISTINCT game_id) AS completed_stat_games
    FROM clean.game_manifest
    GROUP BY season
    ORDER BY season
""")

schedule_counts = (
    schedule_fixed
    .groupby(["season", "status_display"], dropna=False)
    .size()
    .reset_index(name="games")
    .sort_values(["season", "status_display"])
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Completed Games by Season")
    safe_bar_chart(
        season_counts,
        x_col="season",
        y_col="completed_stat_games",
        title="Completed / Stat-Available Games"
    )

with c2:
    st.markdown("### Schedule Status")
    safe_bar_chart(
        schedule_counts,
        x_col="season",
        y_col="games",
        color_col="status_display",
        title="Schedule Status by Season"
    )

season_sql, season_params = sql_in_filter("season", selected_seasons)
team_sql, team_params = sql_in_filter("team_name", selected_teams)
pos_sql, pos_params = sql_in_filter("position", selected_positions)

st.markdown("### Top Player Seasons")

player_sort_metric = st.selectbox(
    "Player season chart metric",
    options=[
        "points", "points_per_game", "goals", "goals_per_game",
        "assists", "assists_per_game", "shots", "shots_per_game",
        "ground_balls", "caused_turnovers"
    ],
    index=0,
    format_func=pretty_col,
    key="overview_player_metric"
)

top_players = query_df(f"""
    SELECT
        season,
        full_name,
        position,
        games,
        teams,
        points,
        goals,
        assists,
        shots,
        ground_balls,
        caused_turnovers,
        points_per_game,
        goals_per_game,
        assists_per_game,
        shots_per_game
    FROM marts.player_season_stats
    WHERE games >= ?
      AND {season_sql}
      AND {pos_sql}
    ORDER BY {player_sort_metric} DESC NULLS LAST
    LIMIT 25
""", [min_games] + season_params + pos_params)

safe_bar_chart(
    top_players.head(15).sort_values(player_sort_metric),
    x_col="full_name",
    y_col=player_sort_metric,
    color_col="position",
    title=f"Top Player Seasons by {pretty_col(player_sort_metric)}",
    orientation="h"
)

display_table(top_players, height=360)

st.markdown("### Top Team Seasons")

team_sort_metric = st.selectbox(
    "Team season chart metric",
    options=[
        "scores", "scores_per_game", "goals", "shots", "shots_per_game",
        "turnovers", "turnovers_per_game", "saves", "touches",
        "time_in_possession", "offensive_sequence_proxy"
    ],
    index=1,
    format_func=pretty_col,
    key="overview_team_metric"
)

top_teams = query_df(f"""
    SELECT
        season,
        team_name,
        games,
        wins,
        losses,
        win_pct,
        scores,
        scores_per_game,
        goals,
        two_point_goals,
        assists,
        shots,
        shots_per_game,
        saves,
        turnovers,
        turnovers_per_game,
        touches,
        total_passes,
        time_in_possession,
        offensive_sequence_proxy
    FROM marts.team_season_stats
    WHERE games >= ?
      AND {season_sql}
      AND {team_sql}
    ORDER BY {team_sort_metric} DESC NULLS LAST
    LIMIT 25
""", [min_games] + season_params + team_params)

safe_bar_chart(
    top_teams.head(15).sort_values(team_sort_metric),
    x_col="team_name",
    y_col=team_sort_metric,
    color_col="season",
    title=f"Top Team Seasons by {pretty_col(team_sort_metric)}",
    orientation="h"
)

display_table(top_teams, height=360)
