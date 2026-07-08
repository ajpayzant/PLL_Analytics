import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import query_df, schedule_display_table, filter_values, DB_PATH
from shared.ui import apply_css, display_table, download_csv
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Schedule · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Schedule")
st.markdown(
    '<div class="section-note">Full schedule inventory including completed and future games.</div>',
    unsafe_allow_html=True
)

schedule_fixed = schedule_display_table()

schedule_season = st.selectbox(
    "Schedule season",
    options=seasons,
    index=len(seasons) - 1 if seasons else 0,
    key="schedule_season"
)

status_options = ["all"] + sorted(schedule_fixed["status_display"].dropna().unique().tolist())
selected_status = st.selectbox("Status", options=status_options, index=0, key="schedule_status_filter")

sched = schedule_fixed[schedule_fixed["season"] == schedule_season].copy()

if selected_status != "all":
    sched = sched[sched["status_display"] == selected_status]

sched = sched.sort_values("game_number")

display_cols = [
    "season",
    "game_number",
    "game_date_guess",
    "away_team_name",
    "home_team_name",
    "away_score",
    "home_score",
    "status_display",
    "slug"
]

display_cols = [c for c in display_cols if c in sched.columns]

display_table(sched[display_cols], height=650)
download_csv(sched[display_cols], f"pll_schedule_{schedule_season}.csv")
