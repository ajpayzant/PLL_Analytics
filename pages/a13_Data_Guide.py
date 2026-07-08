import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from shared.db import DB_PATH
from shared.ui import apply_css, display_table, note_box
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Data Guide · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Data Guide")
st.markdown(
    '<div class="section-note">Definitions, formulas, interpretation notes, and known data caveats for the PLL data platform.</div>',
    unsafe_allow_html=True
)

guide_section = st.radio(
    "Guide section",
    options=["Core Stats", "Goalie / Faceoff", "Rankings", "Team Style", "Data Notes"],
    horizontal=True,
    key="data_guide_section"
)

# ============================================================
# CORE STATS
# ============================================================

if guide_section == "Core Stats":
    st.markdown("### Core Scoring and Possession Terms")

    core_defs = pd.DataFrame([
        ["Scores", "Team scoreboard total. This can differ from goals because PLL 2-point goals count as two scores.", "Official / source field"],
        ["Goals", "Total made goals regardless of scoreboard value. A 2-point goal is still one goal but two scores.", "Official / source field"],
        ["Scoring Points", "Goal scoring value where 1PT goals and 2PT goals are valued by scoreboard impact when available.", "Calculated / source-dependent"],
        ["1PT Goals", "Goals scored from inside the 2-point arc.", "Official / source field"],
        ["2PT Goals", "Goals scored from beyond the 2-point arc.", "Official / source field"],
        ["Points", "Player points: goals plus assists unless otherwise specified by the source table.", "Official / source field"],
        ["Shots on Goal Rate", "Shots on goal divided by total shots.", "Calculated"],
        ["Shot %", "Goals divided by shots.", "Calculated"],
        ["Touches", "Provider-tracked player or team touches. Use as a possession/usage indicator, not as official possession count.", "Provider field"],
        ["Possession Time", "Provider-tracked time of possession. Displayed in MM:SS when used as a per-game value.", "Provider field"],
        ["Offensive Sequences", "Estimated offensive possessions/sequence proxy used when official possession counts are unavailable or inconsistent.", "Calculated proxy"],
    ], columns=["Metric", "Definition", "Source / Notes"])

    display_table(core_defs, height=420)


# ============================================================
# GOALIE / FACEOFF
# ============================================================

elif guide_section == "Goalie / Faceoff":
    st.markdown("### Goalie and Faceoff Terms")

    specialist_defs = pd.DataFrame([
        ["Save Percentage", "Saves divided by saves plus goals against. The app recalculates this for goalie pages to prevent invalid values above 100%.", "Saves / (Saves + Goals Against)"],
        ["Shots Faced", "Estimated goalie shots faced based on saves plus goals against.", "Saves + Goals Against"],
        ["Scores Against", "Opponent scoreboard scores allowed while goalie/team is credited in the source.", "Source field"],
        ["Goals Against", "Opponent goals allowed. This can differ from scores against when 2-point goals occur.", "Source field"],
        ["Clean Saves", "Provider-tracked clean saves where available.", "Source field"],
        ["Messy Saves", "Provider-tracked non-clean saves where available.", "Source field"],
        ["Faceoff Win %", "Faceoffs won divided by total faceoffs.", "FO Won / Faceoffs"],
        ["Minimum Faceoffs", "Filter used to avoid small-sample faceoff leaderboard noise.", "User-selected filter"],
    ], columns=["Metric", "Definition", "Formula / Notes"])

    display_table(specialist_defs, height=420)


# ============================================================
# RANKINGS
# ============================================================

elif guide_section == "Rankings":
    st.markdown("### Player Ranking Formula")

    st.markdown(
        """
        The official player ranking page uses **Overall Score**. The goal is to keep rankings grounded in production while also recognizing when a player is genuinely separated from comparable players in his role.

        **Role Context Value** combines three signals:

        - **50% Role Score**: the player's main role score, such as offense, defense, faceoff, or goalie.
        - **25% Role Percentile**: where the player ranks among players in the same role group.
        - **25% Peer Separation**: a robust z-score style measure of how far above or below role peers the player is.

        **Peer Separation** is the key improvement over percentile alone. A player can rank first in a role group without being dramatically better than the field; the separation score helps identify whether the gap is actually meaningful.
        """
    )

    ranking_defs = pd.DataFrame([
        ["Base Impact", "General all-around player impact score before final role-context adjustment."],
        ["Role Score", "Primary score for the player's role: offense, defense, faceoff, or goalie."],
        ["Role Percentile", "Rank-based position/role signal. Useful for order, but not enough by itself."],
        ["Peer Separation", "Magnitude-based score based on robust z-score distance from role peers."],
        ["Role Context Value", "Weighted blend of role score, role percentile, and role separation."],
        ["Role Tier", "Plain-English tier based on adjusted role separation, such as Elite or High-End."],
        ["Scoring Value", "Direct scoring value signal that includes scoring points, 1PT goals, 2PT goals, and scoring efficiency."],
        ["Playmaking Value", "Creation value signal that includes assists, assists per touch, points per touch, passing involvement, and turnover security."],
        ["Goalie Transfer Adjustment", "Overall-only adjustment that compresses goalie-specific scores toward average instead of subtracting fixed points; goalie-specific views keep full goalie value."],
    ], columns=["Ranking Term", "Definition"])

    display_table(ranking_defs, height=360)

    formula_df = pd.DataFrame([
        ["Offense", "62% Base Impact + 20% Role Context + 10% Usage + 8% Goal Value"],
        ["Defense", "58% Base Impact + 34% Role Context + 8% Usage"],
        ["Faceoff", "74% Base Impact + 14% Role Context + 7% Ground-Ball Value + 5% Usage"],
        ["Goalie", "72% transfer-adjusted Base Impact + 12% transfer-adjusted Role Context + 10% transfer-adjusted Save % + 6% transfer-adjusted Save Volume"],
    ], columns=["Role Group", "Overall Score Formula"])

    display_table(formula_df, height=240)


# ============================================================
# TEAM STYLE
# ============================================================

elif guide_section == "Team Style":
    st.markdown("### Team Style Profile Formula")

    team_style_defs = pd.DataFrame([
        ["Overall Style", "Composite team identity score combining offense, defense, possession, ball movement, and tempo."],
        ["Offensive Volume", "How much offensive activity a team generates through scores, shots, touches, and sequences."],
        ["Offensive Efficiency", "How efficiently a team converts offensive chances into scores/goals."],
        ["Ball Movement", "Passing and assist-oriented style signal."],
        ["Possession Control", "Touches, possession time, and possession-oriented team indicators."],
        ["Defensive Suppression", "How well a team limits opponent scoring, shot quality, and efficiency."],
        ["Pace / Tempo", "How quickly or actively a team plays based on possession and volume signals."],
        ["Net Scores/G", "Scores per game minus scores allowed per game."],
    ], columns=["Team Style Metric", "Definition"])

    display_table(team_style_defs, height=420)


# ============================================================
# DATA NOTES
# ============================================================

elif guide_section == "Data Notes":
    st.markdown("### Data Source and Interpretation Notes")

    note_box(
        "Completed vs Scheduled Games",
        "The app separates completed stat-available games from scheduled games. Current-season totals can be partial until new games are scraped and processed."
    )

    note_box(
        "2026 Current Season",
        "2026 is an in-progress season in the current warehouse. Early-season ranks, trends, and team profiles should be interpreted with sample size in mind."
    )

    note_box(
        "Possession Data Note",
        "PLL provider possession fields are not perfectly consistent across all historical games. The app displays possession time in MM:SS where appropriate and exposes data-quality warnings separately."
    )

    note_box(
        "Official vs Calculated Fields",
        "Some fields come directly from the source, while others are calculated to improve consistency, formatting, or interpretability."
    )
