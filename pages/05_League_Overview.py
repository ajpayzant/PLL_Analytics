"""
League Overview — the season-level answer to "who is good, and why".

Replaces two overlapping pages. `05_Overview.py` opened with warehouse row counts
(Player-Game Rows: 6,756) — a data-engineering readout, not analysis — and then
showed top player/team seasons. `06_Season_Dashboard.py` did the same team and
player leaderboards plus defence and schedule, for one season. The row counts now
live on Data QA, where they belong, and everything analytical is here once.

What is new rather than merged:

* Standings with per-100-possession efficiency, so the table separates "scores a
  lot" from "scores efficiently". Nothing in the app was pace-adjusted before.
* Opponent-adjusted offence and defence. In an 8-team league on an unbalanced
  schedule, who you played matters and no page accounted for it.
* An offence-vs-defence quadrant chart, which answers "who is actually good" in
  one glance where two separate ranked bar charts could not.
* Special situations (power play, man-down, clears, rides, shot-clock
  expirations). Every one of these columns was already in the warehouse and
  displayed nowhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import analysis
from shared import metrics as M
from shared import page as P
from shared import ui
from shared.db import query_df, schedule_display_table, table_exists

ctx = P.init_page(
    "League Overview",
    "Season-level team and player performance, with pace- and opponent-adjusted context.",
)

# The season picker is the page's primary control, so it sits in the main panel
# rather than the sidebar.
season = st.selectbox(
    "Season",
    options=ctx.seasons,
    index=P.default_index(ctx.seasons, P.selected_season(), fallback=-1),
    key="league_season",
)
if season is not None:
    P.select_season(season)

is_current = bool(ctx.seasons) and season == max(ctx.seasons)

# ============================================================
# DATA
# ============================================================

teams = query_df("SELECT * FROM marts.team_season_stats WHERE season = ?", [season])
players = query_df("SELECT * FROM marts.player_season_stats WHERE season = ?", [season])

games = pd.DataFrame()
if table_exists("marts", "team_game_opponent_context"):
    games = query_df(
        "SELECT * FROM marts.team_game_opponent_context WHERE season = ?", [season]
    )

defense = pd.DataFrame()
if table_exists("marts", "team_defense_season_stats"):
    defense = query_df(
        "SELECT * FROM marts.team_defense_season_stats WHERE season = ?", [season]
    )

if len(teams) == 0:
    st.info(f"No team data available for {season}.")
    st.stop()

teams = analysis.add_per_100_possessions(teams)
if len(defense):
    defense = analysis.add_per_100_possessions(defense)

schedule = schedule_display_table()
season_schedule = schedule[schedule["season"] == season].copy()

# ============================================================
# HEADLINE
# ============================================================

completed = int(
    query_df(
        "SELECT COUNT(DISTINCT game_id) AS n FROM clean.game_manifest WHERE season = ?",
        [season],
    )["n"].iloc[0]
)

league_scores = pd.to_numeric(teams.get("scores_per_game"), errors="coerce").mean()
league_eff = pd.to_numeric(teams.get("scores_per_100_poss"), errors="coerce").mean()

k = st.columns(5)
with k[0]:
    ui.stat_card("Completed Games", f"{completed:,}",
                 sub=f"of {len(season_schedule):,} scheduled" if len(season_schedule) else None)
with k[1]:
    ui.stat_card("Teams", f"{teams['team_name'].nunique():,}")
with k[2]:
    ui.stat_card("Players", f"{players['full_name'].nunique():,}" if len(players) else "—")
with k[3]:
    ui.stat_card("League Scores/G", M.format_value("scores_per_game", league_scores))
with k[4]:
    ui.stat_card("League Scores/100", M.format_value("scores_per_100_poss", league_eff),
                 sub="pace-adjusted")

if is_current:
    ui.note_box(
        "Season in progress",
        "Records, rates and ranks reflect completed, stat-available games only and "
        "will move as more games land in the warehouse.",
    )

coverage = analysis.possession_coverage(teams)
if coverage < 0.98:
    st.warning(
        f"Possession data is available for {coverage:.0%} of team-seasons, so the "
        "per-100-possession columns below are based on partial data."
    )

# ============================================================
# STANDINGS
# ============================================================

ui.section(
    "Standings and efficiency",
    "Record with pace-adjusted scoring on both sides of the ball. Scores/100 Poss "
    "separates teams that score a lot from teams that score efficiently.",
)

standings = teams.copy()
if len(defense):
    join_cols = ["team_id"] + [
        c for c in ("scores_allowed_per_game", "scores_allowed_per_100_poss",
                    "opponent_shots_per_game", "save_pct_proxy")
        if c in defense.columns
    ]
    standings = standings.merge(defense[join_cols], on="team_id", how="left")

# Net efficiency is the single best "who is good" column available here, and the
# warehouse does not ship it.
if {"scores_per_100_poss", "scores_allowed_per_100_poss"}.issubset(standings.columns):
    standings["net_efficiency_per_100_poss"] = (
        pd.to_numeric(standings["scores_per_100_poss"], errors="coerce")
        - pd.to_numeric(standings["scores_allowed_per_100_poss"], errors="coerce")
    )

standings_cols = M.existing(standings, [
    "team_name", "games", "wins", "losses", "win_pct", "score_margin_per_game",
    "scores_per_game", "scores_allowed_per_game",
    "scores_per_100_poss", "scores_allowed_per_100_poss",
    "net_efficiency_per_100_poss", "offensive_sequence_proxy_per_game",
])
sort_key = ("net_efficiency_per_100_poss"
            if "net_efficiency_per_100_poss" in standings.columns else "win_pct")
ui.display_table(
    M.sort_df(standings, sort_key)[standings_cols],
    height=330,
    highlight=sort_key,
)
ui.definition_caption(["scores_per_100_poss", "scores_allowed_per_100_poss",
                       "net_efficiency_per_100_poss", "offensive_sequence_proxy_per_game"])

# ============================================================
# OFFENCE vs DEFENCE
# ============================================================

if {"scores_per_100_poss", "scores_allowed_per_100_poss"}.issubset(standings.columns):
    ui.section(
        "Offence versus defence",
        "Quadrants split at the league median. Top-left is the good corner: "
        "scoring efficiently while conceding little.",
    )
    ui.safe_scatter(
        standings,
        x_col="scores_allowed_per_100_poss",
        y_col="scores_per_100_poss",
        color_col="team_name",
        hover_col="team_name",
        title=f"{season} — efficiency on both sides of the ball",
        quadrants=True,
    )

# ============================================================
# OPPONENT-ADJUSTED
# ============================================================

if len(games) and {"team_scores", "scores_allowed"}.issubset(games.columns):
    ui.section(
        "Opponent-adjusted rating",
        "Each team's per-game average corrected for the quality of opposition it "
        "faced, using a leave-one-out estimate of what its opponents did against "
        "everyone else.",
    )

    names = games.drop_duplicates("team_id").set_index("team_id")["team_name"]
    frames = []
    for metric, side in (("team_scores", "Offence"), ("scores_allowed", "Defence")):
        adj = analysis.opponent_adjusted(games, metric)
        if len(adj) == 0:
            continue
        adj = adj.rename(columns={
            f"{metric}_raw": "raw",
            f"{metric}_adjusted": "adjusted",
        })
        adj["team_name"] = adj["team_id"].map(names)
        adj["side"] = side
        frames.append(adj[["team_name", "side", "games", "raw",
                           "schedule_strength", "adjusted"]])

    if frames:
        adjusted = pd.concat(frames, ignore_index=True)
        left, right = st.columns(2)
        for col, side in ((left, "Offence"), (right, "Defence")):
            part = adjusted[adjusted["side"] == side].copy()
            if len(part) == 0:
                continue
            ascending = side == "Defence"
            part = part.sort_values("adjusted", ascending=ascending)
            with col:
                st.markdown(f"**{side}** — scores {'conceded' if ascending else 'for'} per game")
                view = part[["team_name", "games", "raw", "schedule_strength", "adjusted"]]
                view.columns = ["Team", "Games", "Raw", "Sched. Strength", "Adjusted"]
                st.dataframe(
                    view.style.format({"Raw": "{:.2f}", "Sched. Strength": "{:+.2f}",
                                       "Adjusted": "{:.2f}"}, na_rep="—"),
                    width="stretch", hide_index=True, height=330,
                )
                st.caption(analysis.schedule_strength_note(
                    "scores_allowed" if ascending else "team_scores"
                ))

# ============================================================
# TEAM RANKINGS
# ============================================================

ui.section("Team rankings", "Pick any team metric to rank the league.")

team_metric = ui.metric_selectbox(
    "Rank teams by",
    options=M.with_data(standings, [
        "scores_per_game", "scores_per_100_poss", "score_margin_per_game",
        "scores_allowed_per_game", "scores_allowed_per_100_poss",
        "shots_per_game", "shot_pct_calc", "shots_on_goal_rate_calc",
        "two_point_goals_per_game", "assists_per_game",
        "turnovers_per_game", "turnovers_per_100_poss", "caused_turnovers_per_game",
        "ground_balls_per_game", "faceoff_pct_calc", "clear_pct_calc",
        "saves_per_game", "touches_per_game", "total_passes_per_game",
        "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
        "shot_clock_expirations_per_game", "power_play_goals_per_game",
    ]),
    key="league_team_metric",
    default="scores_per_100_poss",
)

if team_metric:
    ranked = M.sort_df(standings, team_metric)
    ui.safe_bar_chart(
        # Reverse for the horizontal bar so rank 1 sits at the top.
        ranked.iloc[::-1],
        x_col="team_name",
        y_col=team_metric,
        color_col="team_name",
        title=f"{season} — {M.label(team_metric)}",
        orientation="h",
    )

# ============================================================
# SPECIAL SITUATIONS
# ============================================================

special_keys = M.with_data(teams, [
    "power_play_goals", "power_play_shots", "power_play_goals_per_game",
    "times_man_up", "times_short_handed", "power_play_goals_against",
    "power_play_goals_against_per_game", "ride_attempts_per_game",
    "clears", "clear_attempts", "clear_pct_calc",
    "shot_clock_expirations", "shot_clock_expirations_per_game",
    "num_penalties_per_game", "pim_per_game",
])
if special_keys:
    ui.section(
        "Special situations",
        "Man-up and man-down work, clearing, riding and shot-clock discipline — "
        "all present in the warehouse and not surfaced anywhere else in the app.",
    )
    # Derived conversion rates the marts don't ship at season level.
    special = teams.copy()
    pp_shots = pd.to_numeric(special.get("power_play_shots"), errors="coerce")
    pp_goals = pd.to_numeric(special.get("power_play_goals"), errors="coerce")
    man_up = pd.to_numeric(special.get("times_man_up"), errors="coerce")
    short = pd.to_numeric(special.get("times_short_handed"), errors="coerce")
    pp_ga = pd.to_numeric(special.get("power_play_goals_against"), errors="coerce")

    if pp_goals is not None and man_up is not None:
        special["power_play_pct"] = pp_goals / man_up.replace(0, np.nan)
    if pp_ga is not None and short is not None:
        # Kill rate: the share of short-handed situations survived.
        special["man_down_pct"] = 1.0 - (pp_ga / short.replace(0, np.nan))

    ui.display_table(
        M.sort_df(special, "power_play_pct" if "power_play_pct" in special.columns
                  else special_keys[0])[
            M.existing(special, [
                "team_name", "times_man_up", "power_play_goals", "power_play_shots",
                "power_play_pct", "times_short_handed", "power_play_goals_against",
                "man_down_pct", "clears", "clear_attempts", "clear_pct_calc",
                "ride_attempts_per_game", "shot_clock_expirations",
                "num_penalties_per_game", "pim_per_game",
            ])
        ],
        height=330,
    )
    ui.definition_caption(["power_play_pct", "man_down_pct", "clear_pct_calc",
                           "ride_attempts_per_game", "shot_clock_expirations"])

# ============================================================
# PLAYER LEADERS
# ============================================================

ui.section("Player leaders", "Season production leaders. Raise the games minimum "
                             "to filter out small-sample rows.")

if len(players) == 0:
    st.info(f"No player data available for {season}.")
else:
    from shared import roles

    players = roles.add_role_column(players)
    controls = st.columns([1.1, 1.4, 0.8, 0.9])

    role_choice = controls[0].selectbox(
        "Role",
        options=["All"] + [roles.role_label(r) for r in roles.ROLE_ORDER
                           if r in set(players["role_group"])],
        key="league_player_role",
        help="Roles use one taxonomy app-wide: SSDM and LSM count as defence.",
    )

    player_metric = ui.metric_selectbox(
        "Rank players by",
        options=M.with_data(players, [
            "points", "points_per_game", "goals", "goals_per_game",
            "assists", "assists_per_game", "scoring_points_per_game",
            "two_point_goals", "shots", "shots_per_game", "shot_pct_calc",
            "ground_balls", "ground_balls_per_game",
            "caused_turnovers", "caused_turnovers_per_game",
            "turnovers_per_game", "touches", "touches_per_game",
            "faceoff_pct_calc", "faceoffs_won",
            "save_pct_calc", "saves", "saves_per_game", "saa",
        ]),
        key="league_player_metric",
        default="points_per_game",
        container=controls[1],
    )

    player_min_games = int(controls[2].number_input(
        "Min games", min_value=1, max_value=25, value=max(1, min(5, completed)),
        step=1, key="league_player_min_games",
    ))
    rows = int(controls[3].number_input(
        "Rows", min_value=10, max_value=200, value=25, step=5,
        key="league_player_rows",
    ))

    board = players.copy()
    if "games" in board.columns:
        board = board[pd.to_numeric(board["games"], errors="coerce").fillna(0)
                      >= player_min_games]
    if role_choice != "All":
        target = next(r for r in roles.ROLE_ORDER if roles.role_label(r) == role_choice)
        board = board[board["role_group"] == target]

    if player_metric and len(board):
        board = M.sort_df(board, player_metric).head(rows)
        ui.safe_bar_chart(
            board.head(20).iloc[::-1],
            x_col="full_name",
            y_col=player_metric,
            color_col="position",
            title=f"{season} — {M.label(player_metric)}",
            orientation="h",
        )

        # Show the columns that matter for the role on screen, not one fixed list.
        if role_choice == "All":
            display_keys = ["full_name", "position", "team_names", "games",
                            "points_per_game", "goals", "assists", "shots",
                            "shot_pct_calc", "ground_balls", "caused_turnovers",
                            "turnovers", "touches_per_game"]
        else:
            display_keys = (["full_name", "position", "team_names", "games"]
                            + roles.ROLE_HEADLINE_METRICS.get(target, []))
        if player_metric not in display_keys:
            display_keys.append(player_metric)

        ui.display_table(board[M.existing(board, display_keys)], height=420,
                         highlight=player_metric)
        ui.download_csv(board, f"pll_player_leaders_{season}.csv")
    elif len(board) == 0:
        st.info("No players match the current filters.")

# ============================================================
# SCHEDULE
# ============================================================

if len(season_schedule):
    with st.expander(f"{season} schedule and results", expanded=False):
        sched = season_schedule.copy()
        if {"away_team_name", "home_team_name"}.issubset(sched.columns):
            sched["matchup"] = (sched["away_team_name"].astype(str) + " at "
                                + sched["home_team_name"].astype(str))
        if {"away_score", "home_score"}.issubset(sched.columns):
            away = pd.to_numeric(sched["away_score"], errors="coerce")
            home = pd.to_numeric(sched["home_score"], errors="coerce")
            played = away.notna() & home.notna()
            sched["result"] = np.where(
                played,
                away.fillna(0).astype("Int64").astype(str) + " - "
                + home.fillna(0).astype("Int64").astype(str),
                "—",
            )
        # `status_display` is `scheduled` both for a game that has not kicked off
        # and for one that has been played but whose stats have not landed. The
        # second explains any gap between this list and the figures above, so the
        # two are separated here the way they are on the Schedule page.
        if {"status_display", "game_date_guess"}.issubset(sched.columns):
            kickoff = pd.to_datetime(sched["game_date_guess"], errors="coerce", utc=True)
            is_scheduled = sched["status_display"].eq("scheduled")
            sched["stage"] = np.where(
                is_scheduled & (kickoff >= pd.Timestamp.now(tz="UTC")), "Upcoming",
                np.where(is_scheduled, "Awaiting stats", "Final"),
            )
        cols = M.existing(sched, ["season", "game_number", "game_date_guess",
                                  "matchup", "result", "stage", "slug"])
        # Game number, not date: this is the full season in schedule order, not a
        # "what is next" list.
        sort_col = "game_number" if "game_number" in sched.columns else cols[0]
        ui.display_table(sched[cols].sort_values(sort_col), height=360)

st.divider()
nav = st.columns(4)
with nav[0]:
    P.link_to("teams", "Team profiles →")
with nav[1]:
    P.link_to("players", "Player profiles →")
with nav[2]:
    P.link_to("rankings", "Player rankings →")
with nav[3]:
    P.link_to("styles", "Team styles →")
