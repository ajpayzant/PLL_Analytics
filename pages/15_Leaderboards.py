import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from shared.db import query_df, table_exists, sql_in_filter, _pll_get_table_columns, DB_PATH
from shared.ui import (
    apply_css, stat_card, safe_bar_chart, display_table, download_csv,
    fmt_value, pretty_col, _pll_select_existing, _pll_safe_sort, _pll_add_possession_mmss
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Leaderboards · PLL Analytics", page_icon="🥍", layout="wide")
apply_css()

import os
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
# PAGE CONTENT
# ============================================================

st.subheader("Leaderboards")
st.markdown(
    '<div class="section-note">Sortable league leaderboards with cleaner default tables and advanced raw-data views available in expanders.</div>',
    unsafe_allow_html=True
)

leader_section = st.radio(
    "Leaderboard section",
    options=["Player Leaders", "Team Leaders", "Defensive / Opponent Leaders"],
    horizontal=True,
    key="leaderboard_section"
)

# ============================================================
# PLAYER LEADERS
# ============================================================

if leader_section == "Player Leaders":
    lb_cols = st.columns([1.2, 1.2, 1.0, 1.0, 0.8])

    player_scope = lb_cols[0].selectbox(
        "Player leaderboard scope",
        ["Player Seasons", "Player Career", "Player Last 5", "Player Last 10"],
        key="leader_player_scope"
    )

    if player_scope == "Player Seasons":
        table = "marts.player_season_stats"
        season_sql, season_params = sql_in_filter("season", selected_seasons)
        pos_sql, pos_params = sql_in_filter("position", selected_positions)
        where_extra = f"AND {season_sql} AND {pos_sql}"
        params_base = season_params + pos_params
    elif player_scope == "Player Career":
        table = "marts.player_career_stats"
        pos_sql, pos_params = sql_in_filter("position", selected_positions)
        where_extra = f"AND {pos_sql}"
        params_base = pos_params
    elif player_scope == "Player Last 5":
        table = "marts.player_last5_stats"
        pos_sql, pos_params = sql_in_filter("position", selected_positions)
        where_extra = f"AND {pos_sql}"
        params_base = pos_params
    else:
        table = "marts.player_last10_stats"
        pos_sql, pos_params = sql_in_filter("position", selected_positions)
        where_extra = f"AND {pos_sql}"
        params_base = pos_params

    player_cols_available = _pll_get_table_columns("marts", table.split(".")[-1])

    player_sort_options = [
        c for c in [
            "points", "points_per_game",
            "scoring_points", "scoring_points_per_game",
            "goals", "goals_per_game",
            "one_point_goals", "two_point_goals",
            "assists", "assists_per_game",
            "assist_opportunities", "assist_conv_rate",
            "two_pt_conversion",
            "shots", "shots_per_game",
            "ground_balls", "ground_balls_per_game",
            "caused_turnovers", "caused_turnovers_per_game",
            "turnovers", "turnovers_per_game",
            "saves", "save_pct_calc", "clean_save_pct",
            "faceoffs_won", "faceoff_pct_calc",
            "touches", "touches_per_game",
        ]
        if c in player_cols_available
    ]

    selected_sort = lb_cols[1].selectbox(
        "Sort by",
        player_sort_options,
        index=0,
        format_func=pretty_col,
        key="leader_player_sort"
    )

    leader_min_games = lb_cols[2].number_input(
        "Minimum games",
        min_value=1,
        max_value=100,
        value=min_games,
        step=1,
        key="leader_player_min_games"
    )

    leader_rows = lb_cols[3].number_input(
        "Rows",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key="leader_player_rows"
    )

    lower_player_metrics = {"turnovers", "turnovers_per_game", "goals_against", "goals_against_per_game"}
    lower_player = selected_sort in lower_player_metrics

    lb_cols[4].caption("Sort")
    lb_cols[4].markdown("**Low best**" if lower_player else "**High best**")

    player_select_cols = [
        c for c in [
            "season", "split_type", "full_name", "position", "teams", "games",
            "points", "points_per_game", "scoring_points", "scoring_points_per_game",
            "one_point_goals", "two_point_goals", "goals", "goals_per_game",
            "assists", "assists_per_game", "assist_opportunities", "assist_conv_rate",
            "two_pt_conversion", "shots", "shots_per_game",
            "ground_balls", "ground_balls_per_game", "turnovers", "turnovers_per_game",
            "caused_turnovers", "caused_turnovers_per_game",
            "saves", "clean_save_pct", "faceoffs_won",
            "faceoff_pct_calc", "touches", "touches_per_game", "total_passes"
        ]
        if c in player_cols_available
    ]

    leaderboard = query_df(f"""
        SELECT {", ".join(player_select_cols)}
        FROM {table}
        WHERE games >= ?
          {where_extra}
        ORDER BY {selected_sort} {"ASC" if lower_player else "DESC"} NULLS LAST
        LIMIT 200
    """, [leader_min_games] + params_base)

    leaderboard = leaderboard.head(int(leader_rows))

    safe_bar_chart(
        leaderboard.head(20).sort_values(selected_sort, ascending=not lower_player),
        x_col="full_name",
        y_col=selected_sort,
        color_col="position" if "position" in leaderboard.columns else None,
        title=f"{player_scope} — Top {min(20, len(leaderboard))} by {pretty_col(selected_sort)}",
        orientation="h"
    )

    player_summary_cols = _pll_select_existing(
        leaderboard,
        [
            "season", "split_type", "full_name", "position", "teams", "games",
            "points", "points_per_game", "scoring_points_per_game",
            "one_point_goals", "two_point_goals",
            "goals_per_game", "assists_per_game", "shots_per_game",
            "ground_balls_per_game", "caused_turnovers_per_game",
            "touches_per_game"
        ]
    )

    display_table(leaderboard[player_summary_cols], height=460)

    with st.expander("Advanced player leaderboard table", expanded=False):
        display_table(leaderboard, height=520)

    download_csv(leaderboard, "pll_player_leaderboard.csv")


# ============================================================
# TEAM LEADERS
# ============================================================

elif leader_section == "Team Leaders":
    lb_cols = st.columns([1.2, 1.2, 1.0, 1.0, 0.8])

    team_scope = lb_cols[0].selectbox(
        "Team leaderboard scope",
        ["Team Seasons", "Team Last 5", "Team Last 10"],
        key="leader_team_scope"
    )

    if team_scope == "Team Seasons":
        table = "marts.team_season_stats"
        season_sql, season_params = sql_in_filter("season", selected_seasons)
        team_sql, team_params = sql_in_filter("team_name", selected_teams)
        where_extra = f"AND {season_sql} AND {team_sql}"
        params_base = season_params + team_params
    elif team_scope == "Team Last 5":
        table = "marts.team_last5_stats"
        team_sql, team_params = sql_in_filter("team_name", selected_teams)
        where_extra = f"AND {team_sql}"
        params_base = team_params
    else:
        table = "marts.team_last10_stats"
        team_sql, team_params = sql_in_filter("team_name", selected_teams)
        where_extra = f"AND {team_sql}"
        params_base = team_params

    team_cols_available = _pll_get_table_columns("marts", table.split(".")[-1])

    team_sort_options = [
        c for c in [
            "scores_per_game", "scores", "score_margin_per_game",
            "goals", "assists", "shots_per_game", "shots",
            "touches_per_game", "touches",
            "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
            "turnovers_per_game", "saves_per_game", "faceoff_pct_calc", "clear_pct_calc"
        ]
        if c in team_cols_available
    ]

    selected_team_sort = lb_cols[1].selectbox(
        "Sort by",
        team_sort_options,
        index=0,
        format_func=pretty_col,
        key="leader_team_sort"
    )

    leader_team_min_games = lb_cols[2].number_input(
        "Minimum games",
        min_value=1,
        max_value=100,
        value=min_games,
        step=1,
        key="leader_team_min_games"
    )

    leader_team_rows = lb_cols[3].number_input(
        "Rows",
        min_value=8,
        max_value=100,
        value=25,
        step=5,
        key="leader_team_rows"
    )

    lower_team_metrics = {"turnovers", "turnovers_per_game"}
    lower_team = selected_team_sort in lower_team_metrics

    lb_cols[4].caption("Sort")
    lb_cols[4].markdown("**Low best**" if lower_team else "**High best**")

    team_select_cols = [
        c for c in [
            "season", "split_type", "team_name", "games", "wins", "losses", "win_pct",
            "scores", "scores_per_game", "goals", "assists",
            "shots", "shots_per_game", "saves", "saves_per_game",
            "turnovers", "turnovers_per_game",
            "ground_balls", "caused_turnovers", "touches", "touches_per_game",
            "total_passes", "total_passes_per_game", "time_in_possession",
            "time_in_possession_per_game", "offensive_sequence_proxy", "offensive_sequence_proxy_per_game"
        ]
        if c in team_cols_available
    ]

    team_leaderboard = query_df(f"""
        SELECT {", ".join(team_select_cols)}
        FROM {table}
        WHERE games >= ?
          {where_extra}
        ORDER BY {selected_team_sort} {"ASC" if lower_team else "DESC"} NULLS LAST
        LIMIT 200
    """, [leader_team_min_games] + params_base)

    team_leaderboard = _pll_add_possession_mmss(team_leaderboard).head(int(leader_team_rows))

    safe_bar_chart(
        team_leaderboard.head(20).sort_values(selected_team_sort, ascending=not lower_team),
        x_col="team_name",
        y_col=selected_team_sort,
        color_col="season" if "season" in team_leaderboard.columns else None,
        title=f"{team_scope} — Top {min(20, len(team_leaderboard))} by {pretty_col(selected_team_sort)}",
        orientation="h"
    )

    team_summary_cols = _pll_select_existing(
        team_leaderboard,
        [
            "season", "split_type", "team_name", "games", "wins", "losses", "win_pct",
            "scores_per_game", "shots_per_game", "touches_per_game",
            "time_in_possession_per_game_mmss", "turnovers_per_game",
            "saves_per_game", "offensive_sequence_proxy_per_game"
        ]
    )

    display_table(team_leaderboard[team_summary_cols], height=460)

    with st.expander("Advanced team leaderboard table", expanded=False):
        display_table(team_leaderboard, height=520)

    download_csv(team_leaderboard, "pll_team_leaderboard.csv")


# ============================================================
# DEFENSIVE / OPPONENT LEADERS
# ============================================================

elif leader_section == "Defensive / Opponent Leaders":
    st.markdown("### Defensive / Opponent Team Leaderboard")
    st.caption("Opponent allowance and defensive suppression metrics. Lower is better for allowed metrics.")

    if table_exists("marts", "team_defense_season_stats"):
        defense_leader_cols = st.columns([1.0, 1.2, 1.0, 0.8])

        with defense_leader_cols[0]:
            defense_scope = st.radio(
                "Scope",
                options=["Season", "Career"],
                horizontal=True,
                key="defense_leader_scope"
            )

        with defense_leader_cols[1]:
            defense_leader_metric = st.selectbox(
                "Defensive metric",
                options=[
                    "scores_allowed_per_game",
                    "goals_allowed_per_game",
                    "opponent_shots_per_game",
                    "opponent_goal_pct",
                    "opponent_sog_rate",
                    "save_pct_proxy",
                    "caused_turnovers_for_per_game",
                    "opponent_turnovers_per_game",
                    "ct_per_opponent_turnover",
                    "score_margin_per_game"
                ],
                index=0,
                format_func=pretty_col,
                key="defense_leader_metric"
            )

        with defense_leader_cols[2]:
            defense_min_games = st.number_input(
                "Minimum games",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="defense_min_games"
            )

        lower_is_better = {
            "scores_allowed_per_game",
            "goals_allowed_per_game",
            "opponent_shots_per_game",
            "opponent_goal_pct",
            "opponent_sog_rate",
            "opponent_sog_goal_pct",
            "opponent_scores_per_offensive_sequence_proxy",
        }

        defense_lower = defense_leader_metric in lower_is_better

        with defense_leader_cols[3]:
            st.caption("Sort")
            st.markdown("**Low best**" if defense_lower else "**High best**")

        if defense_scope == "Season":
            defense_leader_df = query_df("""
                SELECT *
                FROM marts.team_defense_season_stats
                WHERE games >= ?
                ORDER BY season DESC, scores_allowed_per_game ASC NULLS LAST
            """, [defense_min_games])
        else:
            defense_leader_df = query_df("""
                SELECT *
                FROM marts.team_defense_career_stats
                WHERE games >= ?
                ORDER BY scores_allowed_per_game ASC NULLS LAST
            """, [defense_min_games])

        if defense_leader_metric in defense_leader_df.columns:
            defense_leader_df = _pll_safe_sort(
                defense_leader_df,
                defense_leader_metric,
                lower_is_better=defense_lower
            )

            safe_bar_chart(
                defense_leader_df.head(20).sort_values(
                    defense_leader_metric,
                    ascending=not defense_lower
                ),
                x_col="team_name",
                y_col=defense_leader_metric,
                color_col="season" if "season" in defense_leader_df.columns else "team_name",
                title=f"Defensive Leaderboard — {pretty_col(defense_leader_metric)}",
                orientation="h"
            )

        defense_summary_cols = _pll_select_existing(
            defense_leader_df,
            [
                "season", "team_name", "games",
                "scores_allowed_per_game", "goals_allowed_per_game",
                "opponent_shots_per_game", "opponent_goal_pct",
                "save_pct_proxy", "caused_turnovers_for_per_game",
                "opponent_turnovers_per_game", "score_margin_per_game"
            ]
        )

        display_table(defense_leader_df[defense_summary_cols], height=460)

        with st.expander("Advanced defensive leaderboard table", expanded=False):
            display_table(defense_leader_df, height=520)

        download_csv(defense_leader_df, "pll_defensive_leaderboard.csv")
    else:
        st.info("Defensive/opponent marts are not available in the warehouse yet.")
