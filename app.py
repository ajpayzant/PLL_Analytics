"""
PLL Analytics — home.

This page used to be three lines ending in "Use the sidebar to navigate between
pages", which put the whole burden of finding anything on a reader who did not yet
know what the fourteen pages contained. It now answers the two questions a reader
actually arrives with — what is the state of the season, and where do I go for the
question I have — and routes by question rather than by filename.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import analysis
from shared import metrics as M
from shared import page as P
from shared import ui
from shared.db import query_df, schedule_display_table, startup_counts

ctx = P.init_page(
    "PLL Analytics",
    "Player and team analysis for the Premier Lacrosse League — regular and "
    "advanced stats, on offence and defence.",
)

season = ctx.latest_season

# ============================================================
# SEASON STATE
# ============================================================

teams = pd.DataFrame()
if season is not None:
    teams = query_df("SELECT * FROM marts.team_season_stats WHERE season = ?", [season])
    teams = analysis.add_per_100_possessions(teams)

counts = startup_counts()
schedule = schedule_display_table()
season_schedule = (schedule[schedule["season"] == season].copy()
                   if season is not None else pd.DataFrame())

played = 0
if season is not None:
    played = int(query_df(
        "SELECT COUNT(DISTINCT game_id) AS n FROM clean.game_manifest WHERE season = ?",
        [season],
    )["n"].iloc[0])

k = st.columns(4)
with k[0]:
    ui.stat_card(
        "Season", str(season) if season is not None else "—",
        sub=(f"{played:,} of {len(season_schedule):,} games played"
             if len(season_schedule) else None),
    )
with k[1]:
    ui.stat_card("Seasons Covered", f"{len(ctx.seasons):,}",
                 sub=(f"{min(ctx.seasons)}–{max(ctx.seasons)}" if ctx.seasons else None))
with k[2]:
    ui.stat_card("Players", f"{counts['players']:,}")
with k[3]:
    league_eff = (pd.to_numeric(teams.get("scores_per_100_poss"), errors="coerce").mean()
                  if len(teams) else np.nan)
    ui.stat_card("League Scores/100 Poss",
                 M.format_value("scores_per_100_poss", league_eff),
                 sub="pace-adjusted")

# ============================================================
# WHO IS PLAYING WELL
# ============================================================

if len(teams):
    ui.section(
        f"{season} at a glance",
        "Best in the league on each side of the ball, per 100 offensive sequences "
        "where possible so pace doesn't flatter the fast teams.",
    )

    highlights = [
        ("scores_per_100_poss", "Best offence"),
        ("scores_against_per_100_poss", "Best defence"),
        ("shot_pct_calc", "Best shooting"),
        ("faceoff_pct_calc", "Best at the faceoff X"),
    ]
    available = [(m, h) for m, h in highlights if m in teams.columns]
    if available:
        cards = st.columns(len(available))
        for col, (metric, headline) in zip(cards, available):
            ranked = M.sort_df(teams, metric)
            top = ranked.iloc[0] if len(ranked) else None
            with col:
                ui.stat_card(
                    headline,
                    str(top.get("team_name", "—")) if top is not None else "—",
                    sub=(f"{M.label(metric)} "
                         f"{M.format_value(metric, top.get(metric) if top is not None else np.nan)}"),
                )

# ============================================================
# NEXT GAMES
# ============================================================

if {"status_display", "game_date_guess"}.issubset(season_schedule.columns):
    pending = season_schedule[season_schedule["status_display"] == "scheduled"].copy()
    # Two forms of the same timestamp: the feed's own offset for display (a US
    # league's fans want local time), UTC for the comparison below.
    pending["kickoff_local"] = pd.to_datetime(pending["game_date_guess"],
                                              errors="coerce")
    pending["kickoff"] = pd.to_datetime(pending["game_date_guess"], errors="coerce",
                                        utc=True)
    # Filter by date, not game number: a game whose stats have not landed yet is
    # still flagged `scheduled`, so ordering by game number surfaces games that
    # have already been played and reads as though they were next.
    now = pd.Timestamp.now(tz="UTC")
    upcoming = pending[pending["kickoff"] >= now].sort_values("kickoff")
    awaiting = pending[pending["kickoff"] < now]

    if len(upcoming):
        ui.section("Next up", "Open the Matchup Preview for a head-to-head breakdown.")
        for _, row in upcoming.head(3).iterrows():
            slot = st.columns([3, 1])
            kickoff = row.get("kickoff_local")
            when = "" if pd.isna(kickoff) else kickoff.strftime("%a %d %b, %H:%M %Z")
            slot[0].markdown(
                f"**{row.get('away_team_name', '?')}** at "
                f"**{row.get('home_team_name', '?')}**  \n"
                f"<span class='section-note'>{when}</span>",
                unsafe_allow_html=True,
            )
            with slot[1]:
                P.link_to("matchup", "Preview →", full_width=True)

        if len(awaiting):
            st.caption(
                f"{len(awaiting)} earlier game(s) are still marked scheduled — their "
                "stats have not landed in the warehouse yet, so they are excluded "
                "from every figure on this page."
            )

# ============================================================
# WHERE TO GO
# ============================================================

ui.section(
    "Start with a question",
    "Routing by question rather than by page name, because \"Team Styles\" does "
    "not tell you it is where pace and playing identity live.",
)

# (question, page key, link label, what the page actually does)
DESTINATIONS = [
    ("Who is good this season, and why?", "league", "League Overview",
     "Standings with pace- and opponent-adjusted efficiency, plus an "
     "offence-versus-defence quadrant."),
    ("How good is this player?", "players", "Player Profiles",
     "Season and career profiles with role-appropriate stats and recent form."),
    ("How good is this team?", "teams", "Team Profiles",
     "Team production and results on both sides of the ball."),
    ("Who wins this matchup?", "matchup", "Matchup Preview",
     "Head-to-head form, history and the stat edges between two teams."),
    ("Who leads the league in X?", "leaderboards", "Leaderboards",
     "Every countable and rate stat, filterable by season, position and games."),
    ("Who are the best players overall?", "rankings", "Player Rankings",
     "Composite ratings by role, with the inputs shown so you can disagree."),
    ("How does this team play?", "styles", "Team Styles",
     "Pace, possession and shot-profile identity rather than raw production."),
    ("Faceoffs, goaltending, ground balls?", "specialists", "Specialists",
     "The specialist roles that headline production stats hide."),
    ("Player A or player B?", "compare_players", "Compare Players",
     "Side-by-side comparison on a shared metric set."),
    ("Team A or team B?", "compare_teams", "Compare Teams",
     "Side-by-side team comparison, offence and defence."),
    ("What does this stat mean?", "guide", "Data Guide",
     "Definitions, how each composite is weighted, and the data's known limits."),
    ("Can I trust these numbers?", "qa", "Data QA",
     "Warehouse coverage by season and the validation checks."),
]

cols = st.columns(3)
for i, (question, key, link_label, blurb) in enumerate(DESTINATIONS):
    with cols[i % 3]:
        ui.nav_card(question, blurb)
        P.link_to(key, f"{link_label} →", full_width=True)

st.divider()
st.caption(
    f"{counts['player_game_rows']:,} player-game and {counts['team_game_rows']:,} "
    f"team-game rows across {len(ctx.seasons)} seasons. Possession-based rates "
    "depend on provider tracking — see Data QA for coverage."
)
