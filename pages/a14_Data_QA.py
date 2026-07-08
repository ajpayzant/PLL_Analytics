import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import pandas as pd

from shared.db import query_df, schedule_display_table, table_exists, table_index, DB_PATH, ARTIFACT_INDEX_PATH
from shared.ui import apply_css, stat_card, display_table, fmt_value
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Data QA · PLL Analytics", page_icon="🥍", layout="wide")
apply_css()

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

st.subheader("Data QA")
st.markdown(
    '<div class="section-note">Warehouse validation, status repair checks, and artifact inventory.</div>',
    unsafe_allow_html=True
)

quality = query_df("""
    SELECT *
    FROM qc.quality_summary
    ORDER BY
        CASE status
            WHEN 'fail' THEN 1
            WHEN 'warning' THEN 2
            WHEN 'pass' THEN 3
            ELSE 4
        END,
        check_name
""")

fail_count = int((quality["status"] == "fail").sum()) if "status" in quality.columns else 0
warning_count = int((quality["status"] == "warning").sum()) if "status" in quality.columns else 0
pass_count = int((quality["status"] == "pass").sum()) if "status" in quality.columns else 0
total = len(quality)

q1, q2, q3, q4 = st.columns(4)

with q1:
    stat_card("Failures", fmt_value(fail_count, 0))

with q2:
    stat_card("Warnings", fmt_value(warning_count, 0))

with q3:
    stat_card("Passes", fmt_value(pass_count, 0))

with q4:
    stat_card("Total Checks", fmt_value(total, 0))

st.markdown("### Quality Checks")
display_table(quality, height=520)

st.markdown("### 2023 Schedule Status Repair Check")

status_repair = schedule_display_table()
status_repair = (
    status_repair[status_repair["season"] == 2023]
    .groupby(["event_status_label", "status_display"])
    .size()
    .reset_index(name="games")
)

display_table(status_repair, height=180)

st.markdown("### Defensive / Opponent Build QC")

if table_exists("qc", "defensive_opponent_build_quality"):
    defensive_qc_df = query_df("""
        SELECT *
        FROM qc.defensive_opponent_build_quality
        ORDER BY status, check_name
    """)
    display_table(defensive_qc_df, height=320)
else:
    st.info("No defensive/opponent QC table found.")

st.markdown("### Possession Data QC")

if table_exists("qc", "game_possession_quality"):
    possession_qc_df = query_df("""
        SELECT *
        FROM qc.game_possession_quality
        ORDER BY
            CASE possession_data_status
                WHEN 'normal' THEN 4
                WHEN 'extended_or_ot_clock' THEN 3
                WHEN 'short_or_provider_clock' THEN 2
                WHEN 'missing_possession_time' THEN 1
                ELSE 0
            END,
            season,
            game_number
    """)
    display_table(possession_qc_df, height=360)
else:
    st.info("No game possession quality table found.")

st.markdown("### Warehouse Tables")
display_table(table_index(), height=520)

st.markdown("### Artifact Index")

if os.path.exists(ARTIFACT_INDEX_PATH):
    artifact_index = pd.read_csv(ARTIFACT_INDEX_PATH)
    display_table(artifact_index, height=520)
else:
    st.warning("artifact_index.csv was not found.")
