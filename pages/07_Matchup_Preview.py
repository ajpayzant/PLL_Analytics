"""
Matchup Preview — two teams, side by side, before or after the game.

The sidebar filters are not requested: the season, game-group and game pickers in
the main panel are what drive every query here.

Five things were wrong or missing, beyond the shared-helper cleanup.

1. The game list sorted by `game_number` and opened on index 0, so the default
   "upcoming" game was the *lowest-numbered* game still flagged `scheduled` —
   which, once a game has kicked off but its stats have not landed, is a game
   that was already played. The page called that the next matchup. Games are now
   classified by kickoff time the way `app.py` and the Schedule page do, and the
   picker opens on the next game by clock.

2. Current Form read `marts.team_last5_stats` and `marts.player_last5_stats`,
   which are league-wide "last five games played" tables — every row in them is
   from 2026. Previewing Atlas–Chaos in 2022 showed Atlas at 10.6 scores/game
   from games in May–July 2026 under the heading "Last 5"; their actual last five
   of 2022 was 12.4. The season-scoped marts (`team_season_last5_stats`,
   `player_season_last5_stats`) exist and are what this page now reads.

3. Key Players matched a team by substring-searching the pipe-delimited `teams`
   column ("ATL|WHP"), which listed a mid-season arrival under both of his teams
   with his combined totals — Matt Rambo's 2026 season showed as 5 games and 5
   points for Atlas *and* for Whipsnakes, where the split is 4 games/5 points and
   1 game/0 points. `marts.player_season_stats_by_team` carries a real `team_id`
   and one row per team, so the season view reads that.

4. Everything on the page was raw or per-game. A matchup is about one team's
   offence against the other's defence, so the page now leads with pace-adjusted
   efficiency on both sides with league rank, and the defensive marts are a
   first-class view rather than one section at the bottom.

5. The completed-game box score gated its Defense/Faceoff/Goalie views on
   hardcoded position strings, including `["D", "LSM", "SSDM", "G"]` in one place
   and `["FO", "FOS"]` in another. `roles.py` settles that taxonomy.

Local helpers removed: `mmss_from_seconds` and `format_pct_safe` (already in
`shared/ui.py`), `pll_safe_number`, `pll_clock_from_seconds`,
`pll_fmt_profile_value` and `pll_profile_matrix` (the metric registry formats and
`ui.comparison_matrix` builds the side-by-side frame),
`pll_add_team_profile_derived_cols` (the marts already ship every per-game and
`_calc` column it recomputed), `build_team_boxscore_matrix` (same), and
`render_matchup_scoreboard`, which injected its own light-theme CSS that fought
the app's card styling — `ui.scoreboard` is the themed version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import analysis
from shared import metrics as M
from shared import page as P
from shared import roles
from shared import ui
from shared.db import query_df, schedule_display_table, table_exists

ctx = P.init_page(
    "Matchup Preview",
    "Two teams side by side — efficiency, form, history and personnel.",
)

# ============================================================
# GAME PICKER
# ============================================================

schedule = schedule_display_table().copy()

controls = st.columns([1, 2])

season = controls[0].selectbox(
    "Season",
    options=ctx.seasons,
    index=P.default_index(ctx.seasons, P.selected_season(), fallback=-1),
    key="matchup_season",
)
if season is not None:
    P.select_season(season)

pool = schedule[schedule["season"] == season].copy()

# `scheduled` means two different things once kickoff has passed: a genuinely
# upcoming game, and one that has been played but whose stats have not reached
# the warehouse. Only the first is "next".
kickoff = pd.to_datetime(pool.get("game_date_guess"), errors="coerce", utc=True)
now = pd.Timestamp.now(tz="UTC")
is_final = pool["status_display"].eq("final") if "status_display" in pool else False
pool["kickoff"] = kickoff
pool["kickoff_local"] = pd.to_datetime(pool.get("game_date_guess"), errors="coerce")
pool["stage"] = np.where(
    is_final, "Final",
    np.where(kickoff >= now, "Upcoming", "Awaiting stats"),
)

group_options = ["Upcoming", "Final", "All"]
# A completed season has no upcoming games, so defaulting to Upcoming there lands
# the reader on an empty page. The default follows the season being viewed.
group = controls[1].radio(
    "Show",
    options=group_options,
    index=0 if (pool["stage"] == "Upcoming").any() else 1,
    horizontal=True,
    key=f"matchup_group_{season}",
    help="Awaiting stats — kickoff has passed but the box score has not landed "
         "in the warehouse yet — is included under All.",
)

if group == "Upcoming":
    shown = pool[pool["stage"] == "Upcoming"].sort_values("kickoff")
elif group == "Final":
    shown = pool[pool["stage"] == "Final"].sort_values("kickoff", ascending=False)
else:
    shown = pool.sort_values("kickoff")

if len(shown) == 0:
    st.info(f"No {group.lower()} games in {season}.")
    st.stop()

shown = shown.reset_index(drop=True)
shown["when"] = shown["kickoff_local"].dt.strftime("%a %d %b %Y").fillna("date TBD")
shown["label"] = (
    "G" + shown["game_number"].astype("Int64").astype(str) + " · "
    + shown["away_team_name"].astype(str) + " at " + shown["home_team_name"].astype(str)
    + " · " + shown["when"].astype(str) + " · " + shown["stage"].astype(str)
)

# On All, open on the next game by clock rather than the first row, so the page
# lands on the matchup a reader is most likely to want.
default_row = 0
if group == "All":
    upcoming_rows = shown.index[shown["stage"] == "Upcoming"].tolist()
    if upcoming_rows:
        default_row = upcoming_rows[0]

label = st.selectbox("Game", options=shown["label"].tolist(), index=default_row,
                     key="matchup_game")
game = shown[shown["label"] == label].iloc[0]

away_id, home_id = game["away_team_id"], game["home_team_id"]
away_name, home_name = str(game["away_team_name"]), str(game["home_team_name"])
game_id = game.get("event_id")
stage = str(game["stage"])
is_played = stage == "Final"

# ============================================================
# SCOREBOARD
# ============================================================

when = game.get("kickoff_local")
when_text = "date TBD" if pd.isna(when) else when.strftime("%a %d %b %Y, %H:%M %Z")

ui.scoreboard(
    away_name, game.get("away_score"), home_name, game.get("home_score"),
    meta=f"{stage.upper()}<br>{when_text}",
    away_sub="Away", home_sub="Home",
)

if stage == "Awaiting stats":
    ui.note_box(
        "Stats have not landed for this game",
        "Kickoff has passed but the box score is not in the warehouse yet, so "
        "this game contributes nothing to the season and form figures below.",
    )

# ============================================================
# DATA
# ============================================================

pair = [away_id, home_id]


def _two_team_frame(sql: str, params: list) -> pd.DataFrame:
    """Query rows for the two teams, ordered away-then-home so columns read L→R."""
    df = query_df(sql, params)
    if len(df) == 0 or "team_id" not in df.columns:
        return df
    order = {str(away_id): 0, str(home_id): 1}
    df = df.copy()
    df["_side"] = df["team_id"].astype(str).map(order)
    return df.sort_values("_side").drop(columns="_side")


league = query_df("SELECT * FROM marts.team_season_stats WHERE season = ?", [season])
league_defense = pd.DataFrame()
if table_exists("marts", "team_defense_season_stats"):
    league_defense = query_df(
        "SELECT * FROM marts.team_defense_season_stats WHERE season = ?", [season])

profile_context = f"{season} season"
if league["team_id"].astype(str).isin([str(x) for x in pair]).sum() < 2:
    # A team can be absent from a season (expansion, relocation), in which case
    # the season table cannot compare the two and career is the honest fallback.
    profile_context = "career (both teams are not in this season's table)"
    league = query_df("SELECT * FROM marts.team_career_stats")
    league_defense = (query_df("SELECT * FROM marts.team_defense_career_stats")
                      if table_exists("marts", "team_defense_career_stats")
                      else pd.DataFrame())

if len(league) == 0:
    st.info("No team data is available for this matchup.")
    st.stop()


def _merge_defense(teams: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    """
    Add the allowed/opponent columns to the offensive frame.

    Only columns the offensive mart does not already have are taken, and the
    defensive mart's `team_*` mirrors of own-team stats are dropped: they restate
    columns already present under their plain names and would double every row of
    the comparison table.
    """
    if defense is None or len(defense) == 0 or "team_id" not in defense.columns:
        return teams
    keep = ["team_id"] + [
        c for c in defense.columns
        if c not in teams.columns and not c.startswith("team_") and c != "team_id"
    ]
    return teams.merge(defense[keep], on="team_id", how="left")


league = _merge_defense(league, league_defense)
league = analysis.add_per_100_possessions(league)

# The single best "who is better" column available, and the warehouse does not
# ship it.
if {"scores_per_100_poss", "scores_allowed_per_100_poss"}.issubset(league.columns):
    league["net_efficiency_per_100_poss"] = (
        pd.to_numeric(league["scores_per_100_poss"], errors="coerce")
        - pd.to_numeric(league["scores_allowed_per_100_poss"], errors="coerce")
    )

if {"wins", "losses"}.issubset(league.columns):
    league["record_display"] = (
        pd.to_numeric(league["wins"], errors="coerce").fillna(0).astype(int).astype(str)
        + "-"
        + pd.to_numeric(league["losses"], errors="coerce").fillna(0).astype(int).astype(str)
    )

league_by_id = league.set_index(league["team_id"].astype(str))
away_row = league_by_id.loc[str(away_id)] if str(away_id) in league_by_id.index else None
home_row = league_by_id.loc[str(home_id)] if str(home_id) in league_by_id.index else None

teams_pair = league[league["team_id"].astype(str).isin([str(away_id), str(home_id)])].copy()
teams_pair["_side"] = teams_pair["team_id"].astype(str).map(
    {str(away_id): 0, str(home_id): 1})
teams_pair = teams_pair.sort_values("_side").drop(columns="_side")

st.caption(f"Team figures below cover the {profile_context}. "
           f"Ranks are within the {len(league)} teams in that table.")

# ============================================================
# THE MATCHUP
# ============================================================
#
# A preview's central question is one team's offence against the other's defence,
# so that comparison leads rather than sitting in a section at the bottom.

OFF_KEY = "scores_per_100_poss"
DEF_KEY = "scores_allowed_per_100_poss"

if away_row is not None and home_row is not None and \
        {OFF_KEY, DEF_KEY}.issubset(league.columns):
    ui.section(
        "The matchup",
        "Each team's offence against the other's defence, per 100 offensive "
        "sequences so pace does not flatter the faster team.",
    )
    edge = st.columns(4)
    for col, (label_text, row, key) in zip(edge, [
        (f"{away_name} offence", away_row, OFF_KEY),
        (f"{home_name} defence", home_row, DEF_KEY),
        (f"{home_name} offence", home_row, OFF_KEY),
        (f"{away_name} defence", away_row, DEF_KEY),
    ]):
        with col:
            ui.stat_card(
                label_text,
                M.format_value(key, row.get(key, np.nan)),
                sub=analysis.rank_text(league, key, row.name) or M.label(key),
            )

    if "net_efficiency_per_100_poss" in league.columns:
        net = st.columns(2)
        for col, (name, row) in zip(net, [(away_name, away_row), (home_name, home_row)]):
            with col:
                ui.stat_card(
                    f"{name} net efficiency",
                    M.format_value("net_efficiency_per_100_poss",
                                   row.get("net_efficiency_per_100_poss", np.nan)),
                    sub=analysis.rank_text(league, "net_efficiency_per_100_poss",
                                           row.name) or None,
                )

    ui.definition_caption([OFF_KEY, DEF_KEY, "net_efficiency_per_100_poss",
                           "offensive_sequence_proxy_per_game"])

    coverage = analysis.possession_coverage(league)
    if coverage < 0.98:
        st.warning(
            f"Possession data covers {coverage:.0%} of rows in this table, so the "
            "per-100 figures are based on partial data."
        )

# ============================================================
# TABS
# ============================================================

tab_names = ["Season profile", "Recent form", "Head to head", "Key players"]
if is_played:
    tab_names.append("Box score")
tabs = st.tabs(tab_names)

# ------------------------------------------------------------
# SEASON PROFILE
# ------------------------------------------------------------

# Metric groups, not one 98-column dump. Every list is generous on purpose:
# display_comparison_matrix drops what the table does not carry, so the career
# fallback and the season table can share these definitions.
PROFILE_VIEWS = {
    "Per game": [
        "games", "record_display", "win_pct", "scores_per_game",
        "score_margin_per_game", "goals_per_game", "one_point_goals_per_game",
        "two_point_goals_per_game", "assists_per_game", "shots_per_game",
        "shots_on_goal_per_game", "ground_balls_per_game", "turnovers_per_game",
        "caused_turnovers_per_game", "saves_per_game", "touches_per_game",
        "total_passes_per_game", "time_in_possession_per_game",
        "offensive_sequence_proxy_per_game",
    ],
    "Efficiency": [
        "scores_per_100_poss", "goals_per_100_poss", "shots_per_100_poss",
        "assists_per_100_poss", "turnovers_per_100_poss",
        "caused_turnovers_per_100_poss", "ground_balls_per_100_poss",
        "touches_per_100_poss", "scores_allowed_per_100_poss",
        "goals_allowed_per_100_poss", "net_efficiency_per_100_poss",
        "offensive_sequence_proxy_per_game",
    ],
    "Shooting and possession": [
        "shot_pct_calc", "shots_on_goal_rate_calc", "faceoff_pct_calc",
        "clear_pct_calc", "passes_per_touch", "seconds_possession_per_touch",
        "touches_per_offensive_sequence_proxy",
        "passes_per_offensive_sequence_proxy", "time_in_possession_per_game",
        "official_total_possessions_per_game",
    ],
    "Special situations": [
        "times_man_up_per_game", "power_play_goals_per_game",
        "power_play_shots_per_game", "times_short_handed_per_game",
        "power_play_goals_against_per_game", "clear_pct_calc",
        "clear_attempts_per_game", "ride_attempts_per_game",
        "shot_clock_expirations_per_game", "num_penalties_per_game",
        "pim_per_game",
    ],
    "Defence and allowed": [
        "scores_allowed_per_game", "goals_allowed_per_game",
        "assists_allowed_per_game", "one_point_goals_allowed_per_game",
        "two_point_goals_allowed_per_game", "opponent_shots_per_game",
        "opponent_shots_on_goal_per_game", "opponent_goal_pct",
        "opponent_sog_rate", "opponent_sog_goal_pct", "save_pct_proxy",
        "caused_turnovers_for_per_game", "opponent_turnovers_per_game",
        "ct_per_opponent_turnover", "opponent_touches_per_game",
        "opponent_offensive_sequence_proxy_per_game",
        "opponent_scores_per_offensive_sequence_proxy",
        "scores_allowed_per_100_poss",
    ],
    "Totals": [
        "games", "wins", "losses", "ties", "scores", "goals", "one_point_goals",
        "two_point_goals", "assists", "shots", "shots_on_goal", "ground_balls",
        "turnovers", "caused_turnovers", "faceoffs", "faceoffs_won",
        "faceoffs_lost", "saves", "clears", "clear_attempts", "ride_attempts",
        "times_man_up", "times_short_handed", "power_play_goals",
        "shot_clock_expirations", "num_penalties", "pim", "touches",
        "total_passes", "time_in_possession", "offensive_sequence_proxy",
        "official_total_possessions",
    ],
}

with tabs[0]:
    if len(teams_pair) < 2:
        st.info("Both teams are not present in this table, so they cannot be compared.")
    else:
        summary_specs = [
            ("Record", "record_display"),
            ("Scores/G", "scores_per_game"),
            ("Scores Allowed/G", "scores_allowed_per_game"),
            ("Scores/100 Poss", "scores_per_100_poss"),
            ("Shot %", "shot_pct_calc"),
            ("FO Win %", "faceoff_pct_calc"),
        ]
        ui.profile_summary_cards(teams_pair, "team_name", summary_specs, columns=2)

        view = st.radio("Profile view", options=list(PROFILE_VIEWS),
                        horizontal=True, key="matchup_profile_view")
        keys = PROFILE_VIEWS[view]
        # The "Best" column comes from each metric's direction, so a reader does
        # not have to know which end of Turnovers/G is good.
        ui.display_comparison_matrix(teams_pair, "team_name", keys, height=520)
        ui.definition_caption(M.with_data(teams_pair, keys)[:10])

        chart_options = M.with_data(league, keys)
        chart_metric = ui.metric_selectbox(
            "Chart this metric against the league", options=chart_options,
            key="matchup_profile_chart",
            default=next((k for k in ("scores_per_100_poss", "scores_per_game")
                          if k in chart_options), None),
        )
        if chart_metric:
            # The whole league, with the two teams named, answers "is that good?"
            # in a way a two-bar chart cannot.
            chart_df = league.copy()
            chart_df["side"] = np.where(
                chart_df["team_id"].astype(str) == str(away_id), away_name,
                np.where(chart_df["team_id"].astype(str) == str(home_id), home_name,
                         "Rest of league"))
            ui.safe_bar_chart(
                M.sort_df(chart_df, chart_metric),
                x_col="team_name", y_col=chart_metric, color_col="side",
                title=f"{M.label(chart_metric)} — {profile_context}",
                orientation="h", height=440,
            )

# ------------------------------------------------------------
# RECENT FORM
# ------------------------------------------------------------

with tabs[1]:
    window = st.radio("Form window", options=["Last 5", "Last 10"],
                      horizontal=True, key="matchup_form_window")
    # Season-scoped, not the league-wide last5/last10 marts: those hold only the
    # most recent games played anywhere, so previewing an older season through
    # them showed the current season's form.
    form_table = ("marts.team_season_last5_stats" if window == "Last 5"
                  else "marts.team_season_last10_stats")
    form = _two_team_frame(
        f"SELECT * FROM {form_table} WHERE season = ? AND team_id IN (?, ?)",
        [season, away_id, home_id])

    if len(form) == 0:
        st.info(f"No {window.lower()} rows for these teams in {season}.")
    else:
        form = analysis.add_per_100_possessions(form)
        st.caption(
            f"{window} completed games of the {season} season for each team, which "
            "may cover different dates if the two have played a different number "
            "of games."
        )
        form_keys = [
            "games", "first_game_date", "last_game_date", "scores_per_game",
            "scores_against_per_game", "scores_per_100_poss", "shots_per_game",
            "shot_pct_calc", "goals_per_game", "assists_per_game",
            "ground_balls_per_game", "turnovers_per_game",
            "caused_turnovers_per_game", "saves_per_game", "faceoff_pct_calc",
            "clear_pct_calc", "touches_per_game", "time_in_possession_per_game",
            "offensive_sequence_proxy_per_game",
        ]
        ui.display_comparison_matrix(form, "team_name", form_keys, height=460)

    # An aggregate hides the shape of a run, so the game-by-game series follows.
    ui.section("Game by game",
               "Both teams' completed games this season, with a rolling average.")
    trend = query_df(
        """
        SELECT season, game_number, game_date_utc, team_id, team_name,
               opponent_team_name, result, scores, scores_against, score_margin,
               shots, shot_pct, ground_balls, turnovers, caused_turnovers,
               saves, save_pct, faceoff_pct, clear_pct, touches, total_passes,
               time_in_possession, offensive_sequence_proxy
        FROM clean.team_game_stats
        WHERE season = ? AND team_id IN (?, ?)
        ORDER BY game_date_utc, game_number
        """, [season, away_id, home_id])

    if len(trend) == 0:
        st.info(f"No completed games for these teams in {season}.")
    else:
        trend_options = M.with_data(trend, [
            "scores", "scores_against", "score_margin", "shots", "shot_pct",
            "ground_balls", "turnovers", "caused_turnovers", "saves", "save_pct",
            "faceoff_pct", "clear_pct", "touches", "offensive_sequence_proxy",
        ])
        trend_metric = ui.metric_selectbox(
            "Game-by-game metric", options=trend_options, key="matchup_trend_metric",
            default="scores" if "scores" in trend_options else None)

        if trend_metric:
            # add_rolling sorts by season/game_number, so it is applied per team
            # rather than across the interleaved frame.
            rolled = pd.concat(
                [analysis.add_rolling(part, trend_metric, window=3)
                 for _, part in trend.groupby("team_id")],
                ignore_index=True,
            )
            rolled["game_label"] = "G" + rolled["game_number"].astype("Int64").astype(str)
            ui.safe_line_chart(
                rolled.sort_values(["game_number"]),
                x_col="game_label", y_cols=[trend_metric], color_col="team_name",
                title=f"{M.label(trend_metric)} by game — {season}",
            )
            st.caption(
                f"Three-game rolling average is in the table below as "
                f"{M.label(trend_metric)} Roll3."
            )
            show_cols = M.existing(rolled, [
                "team_name", "game_number", "game_date_utc", "opponent_team_name",
                "result", "scores", "scores_against", trend_metric,
                f"{trend_metric}_roll3",
            ])
            ui.display_table(rolled[show_cols], height=360,
                             date_cols=["game_date_utc"], clean_schema=True)

# ------------------------------------------------------------
# HEAD TO HEAD
# ------------------------------------------------------------

with tabs[2]:
    # The vs-opponent marts are all-time, not per season — saying so matters,
    # because the section sits under a season-scoped page.
    st.caption("All-time meetings between these two teams, across every season "
               "in the warehouse.")

    h2h = _two_team_frame(
        """
        SELECT * FROM marts.team_vs_opponent_stats
        WHERE (team_id = ? AND opponent_team_id = ?)
           OR (team_id = ? AND opponent_team_id = ?)
        """, [away_id, home_id, home_id, away_id])

    if len(h2h) == 0:
        st.info("These two teams have not met in the seasons the warehouse covers.")
    else:
        h2h = analysis.add_per_100_possessions(h2h)
        ui.display_comparison_matrix(h2h, "team_name", [
            "games", "scores_per_game", "scores_against_per_game",
            "scores_per_100_poss", "goals_per_game", "assists_per_game",
            "shots_per_game", "shot_pct_calc", "ground_balls_per_game",
            "turnovers_per_game", "caused_turnovers_per_game", "saves_per_game",
            "faceoff_pct_calc", "clear_pct_calc", "touches_per_game",
        ], height=440)

    h2h_games = query_df(
        """
        WITH a AS (
            SELECT * FROM clean.team_game_stats
            WHERE team_id = ? AND opponent_team_id = ?
        ), b AS (
            SELECT * FROM clean.team_game_stats
            WHERE team_id = ? AND opponent_team_id = ?
        )
        SELECT
            a.season, a.game_number, a.game_date_utc,
            a.team_name AS away_or_a, a.scores AS a_scores,
            b.team_name AS home_or_b, b.scores AS b_scores,
            CASE WHEN a.scores > b.scores THEN a.team_name
                 WHEN b.scores > a.scores THEN b.team_name
                 ELSE 'Tie' END AS winner,
            a.shots AS a_shots, b.shots AS b_shots,
            a.turnovers AS a_turnovers, b.turnovers AS b_turnovers,
            a.ground_balls AS a_ground_balls, b.ground_balls AS b_ground_balls,
            a.caused_turnovers AS a_caused_turnovers,
            b.caused_turnovers AS b_caused_turnovers
        FROM a INNER JOIN b ON a.game_id = b.game_id
        ORDER BY a.season DESC, a.game_number DESC
        """, [away_id, home_id, home_id, away_id])

    if len(h2h_games):
        wins = h2h_games["winner"].value_counts()
        record = st.columns(3)
        with record[0]:
            ui.stat_card(f"{away_name} wins", f"{int(wins.get(away_name, 0)):,}")
        with record[1]:
            ui.stat_card(f"{home_name} wins", f"{int(wins.get(home_name, 0)):,}")
        with record[2]:
            ui.stat_card("Meetings", f"{len(h2h_games):,}",
                         sub=f"{int(wins.get('Tie', 0))} tied"
                             if wins.get("Tie", 0) else None)

        ui.section("Meeting log")
        ui.display_table(h2h_games, height=380, date_cols=["game_date_utc"],
                         clean_schema=True)
        ui.download_csv(h2h_games,
                        f"{away_name}_vs_{home_name}_h2h.csv".replace(" ", "_").lower())

# ------------------------------------------------------------
# KEY PLAYERS
# ------------------------------------------------------------


def _players_for_team(df: pd.DataFrame, team_id) -> pd.DataFrame:
    """
    Rows for one team, whether the mart carries `team_id` or only `teams`.

    The season view uses `player_season_stats_by_team`, which has a real
    `team_id` and one row per team a player appeared for. The recent-form marts
    have neither: they carry a pipe-delimited `teams` string ("ATL|WHP") and a
    single set of totals covering both stints. Matt Rambo's 2026 season is 4
    games and 5 points for Atlas plus 1 game and 0 points for Whipsnakes, and the
    old substring match on `teams` listed him under *both* teams with the
    combined 5 games and 5 points — so a mid-season arrival was previewed as
    though he had produced his whole season for his new team. The season view now
    reads the by-team mart; the recent-form views can only split on the
    delimiter, which at least keeps a player off a team he never played for.
    """
    if df is None or len(df) == 0:
        return df
    if "team_id" in df.columns:
        return df[df["team_id"].astype(str) == str(team_id)].copy()
    if "teams" not in df.columns:
        return df.iloc[0:0].copy()
    codes = df["teams"].fillna("").astype(str).str.split("|")
    return df[codes.apply(lambda parts: str(team_id) in [p.strip() for p in parts])].copy()


with tabs[3]:
    source = st.radio("Player form source",
                      options=["Season", "Last 5", "Last 10"],
                      horizontal=True, key="matchup_player_source")

    if source == "Season":
        # by_team splits a traded player's season, so a mid-season arrival is
        # credited to the team he actually played these games for.
        key_players = query_df(
            "SELECT * FROM marts.player_season_stats_by_team WHERE season = ?",
            [season])
    else:
        split_table = ("marts.player_season_last5_stats" if source == "Last 5"
                       else "marts.player_season_last10_stats")
        key_players = query_df(f"SELECT * FROM {split_table} WHERE season = ?",
                               [season])

    if len(key_players) == 0:
        st.info(f"No {source.lower()} player rows for {season}.")
    else:
        key_players = roles.add_role_column(key_players)

        picker = st.columns([2, 1, 1])
        role_options = ["All"] + [roles.role_label(r) for r in roles.ROLE_ORDER
                                  if (key_players["role_group"] == r).any()]
        role_choice = picker[1].radio("Role", options=role_options,
                                      key="matchup_player_role")
        top_n = picker[2].number_input("Players per team", min_value=3, max_value=25,
                                      value=8, step=1, key="matchup_player_top_n")

        # Role decides which metrics are worth ranking by: a defender sorted on
        # Points/G is a list of the wrong defenders.
        if role_choice == "All":
            metric_pool = ["points_per_game", "goals_per_game", "assists_per_game",
                           "shots_per_game", "ground_balls_per_game",
                           "caused_turnovers_per_game", "touches_per_game",
                           "faceoff_pct_calc", "save_pct_calc", "saves_per_game"]
        else:
            role_key = next(r for r in roles.ROLE_ORDER
                            if roles.role_label(r) == role_choice)
            metric_pool = list(roles.ROLE_HEADLINE_METRICS.get(role_key, []))
            key_players = key_players[key_players["role_group"] == role_key]

        metric = ui.metric_selectbox(
            "Rank by", options=M.with_data(key_players, metric_pool),
            key="matchup_player_metric", container=picker[0])

        if not metric:
            st.info("No ranking metric is available for this selection.")
        else:
            sides = st.columns(2)
            for col, (name, team_id) in zip(sides, [(away_name, away_id),
                                                    (home_name, home_id)]):
                with col:
                    st.markdown(f"#### {name}")
                    side = _players_for_team(key_players, team_id)
                    side = M.sort_df(side, metric).head(int(top_n))
                    if len(side) == 0:
                        st.info(f"No {source.lower()} rows for {name}.")
                        continue
                    ui.safe_bar_chart(
                        side, x_col="full_name", y_col=metric, color_col="position",
                        title=f"{name} — {M.label(metric)}", orientation="h",
                        height=360,
                    )
                    show = M.existing(side, [
                        "full_name", "position", "games", metric,
                        "points_per_game", "goals_per_game", "assists_per_game",
                        "shots_per_game", "shot_pct_calc", "ground_balls_per_game",
                        "caused_turnovers_per_game", "turnovers_per_game",
                        "touches_per_game",
                    ])
                    ui.display_table(side[show], height=320)

# ------------------------------------------------------------
# BOX SCORE (completed games only)
# ------------------------------------------------------------

if is_played:
    with tabs[4]:
        if game_id is None or pd.isna(game_id):
            st.info("This game is marked final but carries no game ID, so its box "
                    "score cannot be located.")
        else:
            team_box = _two_team_frame(
                "SELECT * FROM clean.team_game_stats WHERE game_id = ?", [game_id])

            if len(team_box) == 0:
                st.info("No team box score rows were found for this game.")
            else:
                ui.section("Team box score")
                ui.display_comparison_matrix(team_box, "team_name", [
                    "scores", "goals", "one_point_goals", "two_point_goals",
                    "assists", "shots", "shots_on_goal", "shot_pct",
                    "shots_on_goal_pct", "two_point_shot_pct", "ground_balls",
                    "turnovers", "caused_turnovers", "faceoffs_won",
                    "faceoffs_lost", "faceoff_pct", "saves", "save_pct",
                    "clears", "clear_attempts", "clear_pct", "ride_attempts",
                    "times_man_up", "power_play_goals", "power_play_pct",
                    "times_short_handed", "man_down_pct",
                    "shot_clock_expirations", "num_penalties", "pim", "touches",
                    "total_passes", "time_in_possession", "time_in_possession_pct",
                    "official_total_possessions", "offensive_sequence_proxy",
                ], height=560)

                if "possession_data_status" in team_box.columns:
                    statuses = sorted(team_box["possession_data_status"]
                                      .dropna().astype(str).unique().tolist())
                    if statuses:
                        st.caption("Possession data status: " + ", ".join(statuses)
                                   + ". Possession time is shown as M:SS.")

            player_box = query_df(
                "SELECT * FROM clean.player_game_stats WHERE game_id = ?", [game_id])

            if len(player_box) == 0:
                st.info("No player box score rows were found for this game.")
            else:
                ui.section("Player box score")
                player_box = roles.add_role_column(player_box)

                box_controls = st.columns(2)
                side_name = box_controls[0].radio(
                    "Team", options=[away_name, home_name], horizontal=True,
                    key="matchup_box_team")
                side_id = away_id if side_name == away_name else home_id

                # Gated on roles.py, not on the hardcoded ["D","LSM","SSDM","G"]
                # and ["FO","FOS"] sets this page used to carry.
                role_views = {
                    "Offense": (roles.ROLE_OFFENSE, [
                        "full_name", "position", "points", "scoring_points",
                        "goals", "one_point_goals", "two_point_goals", "assists",
                        "assist_opportunities", "shots", "shots_on_goal",
                        "shot_pct", "ground_balls", "turnovers", "touches",
                        "total_passes"]),
                    "Defense": (roles.ROLE_DEFENSE, [
                        "full_name", "position", "caused_turnovers",
                        "ground_balls", "turnovers", "num_penalties", "pim",
                        "points", "shots", "touches", "total_passes"]),
                    "Faceoff": (roles.ROLE_FACEOFF, [
                        "full_name", "position", "fo_record", "faceoff_pct",
                        "faceoffs", "faceoffs_won", "faceoffs_lost",
                        "ground_balls", "points", "assists", "shots", "touches"]),
                    "Goalie": (roles.ROLE_GOALIE, [
                        "full_name", "position", "saves", "goals_against",
                        "scores_against", "save_pct", "clean_saves",
                        "messy_saves", "clean_save_pct", "saa", "touches",
                        "total_passes"]),
                }
                view_name = box_controls[1].radio(
                    "View", options=list(role_views), horizontal=True,
                    key="matchup_box_view")
                role_key, cols = role_views[view_name]

                side = player_box[player_box["team_id"].astype(str) == str(side_id)]
                side = side[side["role_group"] == role_key]
                if len(side) == 0:
                    st.info(f"No {view_name.lower()} players are listed for "
                            f"{side_name} in this game.")
                else:
                    sort_keys = {
                        roles.ROLE_OFFENSE: ["points", "goals", "assists"],
                        roles.ROLE_DEFENSE: ["caused_turnovers", "ground_balls"],
                        roles.ROLE_FACEOFF: ["faceoffs_won", "faceoffs"],
                        roles.ROLE_GOALIE: ["saves"],
                    }[role_key]
                    first = next((k for k in sort_keys if k in side.columns), None)
                    if first:
                        side = M.sort_df(side, first)
                    # clean_schema: clean.player_game_stats stores clean_save_pct
                    # on a 0–1 scale where the marts store 0–100.
                    ui.display_table(side[M.existing(side, cols)], height=420,
                                     clean_schema=True)

# ============================================================
# WHERE TO GO
# ============================================================

st.divider()
nav = st.columns(4)
with nav[0]:
    P.link_to("teams", f"{away_name} profile →", team=away_name)
with nav[1]:
    P.link_to("teams", f"{home_name} profile →", team=home_name)
with nav[2]:
    P.link_to("compare_teams", "Compare teams →")
with nav[3]:
    P.link_to("schedule", "Full schedule →")
