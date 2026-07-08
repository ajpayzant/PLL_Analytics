"""
shared/filters.py — Sidebar global filters rendered on every page.
"""

import streamlit as st
from shared.db import filter_values


def render_sidebar_filters():
    seasons, teams_df, players_df, positions = filter_values()

    st.sidebar.title("PLL Analytics")
    st.sidebar.caption("Interactive PLL Data Dashboard")
    st.sidebar.divider()

    default_seasons = [max(seasons)] if seasons else []
    selected_seasons = st.sidebar.multiselect(
        "Global Season Filter",
        options=seasons,
        default=default_seasons,
        key="sidebar_seasons"
    )

    team_options = teams_df["team_name"].dropna().tolist()
    selected_teams = st.sidebar.multiselect(
        "Global Team Filter",
        options=team_options,
        default=[],
        key="sidebar_teams"
    )

    selected_positions = st.sidebar.multiselect(
        "Global Position Filter",
        options=positions,
        default=[],
        key="sidebar_positions"
    )

    min_games = st.sidebar.number_input(
        "Minimum Games",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        key="sidebar_min_games"
    )

    st.sidebar.caption("Global filters primarily affect Overview and Leaderboards. Explorer pages have their own filters.")

    return seasons, teams_df, players_df, positions, selected_seasons, selected_teams, selected_positions, min_games
