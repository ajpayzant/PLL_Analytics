"""
Schedule — every game, played and upcoming.

Two fixes over the previous version. Unplayed games showed "0 — 0" because the
feed zero-fills the score, which reads as a scoreless draw rather than a game that
hasn't happened; they now show a dash. And a game whose stats haven't landed is
still flagged `scheduled` by the feed even after kickoff, so this page separates
"upcoming" from "awaiting stats" instead of lumping them together — the second
group is why a season's game count can trail its schedule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import metrics as M
from shared import page as P
from shared import segments
from shared import ui
from shared.db import schedule_display_table

ctx = P.init_page(
    "Schedule",
    "Full schedule inventory, including completed and future games.",
    # This page covers every segment on its own terms, so the sidebar's
    # regular/playoffs selector would be decorative here.
    scope=False,
)

schedule = schedule_display_table()

controls = st.columns([1, 1.2, 1.1])
season = controls[0].selectbox(
    "Season",
    options=ctx.seasons,
    index=P.default_index(ctx.seasons, P.selected_season(), fallback=-1),
    key="schedule_season",
)
if season is not None:
    P.select_season(season)

sched = schedule[schedule["season"] == season].copy()

# `scheduled` covers two different things once kickoff has passed. Splitting them
# is the point: the second group explains a gap between games played and games
# with stats in the warehouse.
kickoff = pd.to_datetime(sched.get("game_date_guess"), errors="coerce", utc=True)
now = pd.Timestamp.now(tz="UTC")
is_scheduled = sched.get("status_display").eq("scheduled") if "status_display" in sched else False
sched["stage"] = np.where(
    is_scheduled & (kickoff >= now), "Upcoming",
    np.where(is_scheduled, "Awaiting stats", "Final"),
)

view_options = ["All", "Final", "Upcoming", "Awaiting stats"]
view = controls[1].radio("Show", options=view_options, horizontal=True,
                         key="schedule_view")

# The segment control belongs here rather than in the sidebar: the schedule is one
# table for every segment, so it filters rows instead of switching tables. It
# appears only once a build has labelled the fixtures.
segment_choice = "All"
if "competition_type" in sched.columns:
    segment_choice = controls[2].radio(
        "Segment", options=["All", "Regular season", "Playoffs"], horizontal=True,
        key="schedule_segment",
        help="Playoff fixtures are listed as soon as the league publishes the "
             "bracket, which is before the seeding is known — so an unseeded "
             "round shows a date and a round name but no teams.",
    )
    if segment_choice != "All":
        sched = segments.filter_frame(
            sched,
            segments.REGULAR if segment_choice == "Regular season" else segments.PLAYOFFS,
        )

k = st.columns(4)
for col, stage in zip(k, ["Final", "Upcoming", "Awaiting stats"]):
    with col:
        ui.stat_card(stage, f"{int((sched['stage'] == stage).sum()):,}")
with k[3]:
    ui.stat_card("Scheduled Games", f"{len(sched):,}")

awaiting = int((sched["stage"] == "Awaiting stats").sum())
if awaiting:
    ui.note_box(
        "Games awaiting stats",
        f"{awaiting} game(s) have passed their scheduled start but have no stats in "
        "the warehouse yet. They are excluded from every average and total in this "
        "app until the feed catches up.",
    )

shown = sched if view == "All" else sched[sched["stage"] == view]

display = shown.copy()
if {"away_team_name", "home_team_name"}.issubset(display.columns):
    display["matchup"] = (display["away_team_name"].astype(str) + " at "
                          + display["home_team_name"].astype(str))
if {"away_score", "home_score"}.issubset(display.columns):
    away = pd.to_numeric(display["away_score"], errors="coerce")
    home = pd.to_numeric(display["home_score"], errors="coerce")
    # The feed zero-fills unplayed games, so a 0–0 here means "not played".
    played = display["stage"] == "Final"
    display["result"] = np.where(
        played,
        away.fillna(0).astype("Int64").astype(str) + " – "
        + home.fillna(0).astype("Int64").astype(str),
        "—",
    )

cols = M.existing(display, [
    "game_number", "game_date_guess", "competition_type", "round_label",
    "matchup", "result", "stage",
    "away_team_name", "home_team_name", "venue", "slug",
])
sort_col = "game_number" if "game_number" in display.columns else cols[0]
ui.display_table(display[cols].sort_values(sort_col), height=620,
                 date_cols=["game_date_guess"],
                 empty_message=f"No {view.lower()} games in {season}.")
# The playoff rows are the ones a reader is least likely to expect, so they are
# counted out loud rather than left to be noticed in a 23-row table.
if segment_choice == "All" and "competition_type" in sched.columns:
    playoff_games = len(segments.filter_frame(sched, segments.PLAYOFFS))
    if playoff_games:
        st.caption(f"{playoff_games} of these are playoff games.")
ui.download_csv(display[cols], f"pll_schedule_{season}.csv")

st.divider()
nav = st.columns(3)
with nav[0]:
    P.link_to("matchup", "Matchup preview →")
with nav[1]:
    P.link_to("league", "League overview →")
with nav[2]:
    P.link_to("qa", "Data QA →")
