"""
Leaderboards — who leads the league in anything, on offence or defence.

Four fixes over the previous version.

The row cap was silent. Every query ended in `LIMIT 200` and the result was then
cut to the chosen row count in pandas. The displayed top-N was right, because the
sort happened in SQL, but nothing else was: `player_season_stats` has 989
qualifying rows, so 789 never reached the page, the "Rows" box maxed out at 100
while claiming a range up to 100 of an unstated total, and the CSV export silently
contained 200 rows of a 989-row league. Ranks are now computed over the full
filtered set, the cap only limits what is displayed, a caption states the total,
and the download covers every qualifying row.

Sort direction was two hardcoded name sets. They agreed with the registry on which
metrics are lower-better, but they only had two states, so seven volume metrics —
touches, possession time, offensive sequences, saves per game — were labelled
"High best" when neither direction is better: a team with the most touches is not
thereby good at anything. Direction now comes from the registry, which has a third
state for exactly this, and an undirected metric is described as volume and left
un-colour-scaled rather than being scored.

And the defensive tab ignored the sidebar entirely — no season filter, no team
filter, its own separate minimum-games box — so it showed every season at once
while the sidebar said otherwise. All three tabs now apply the filters the page
declares.

Fourth: the Last 5 / Last 10 scopes mixed eras. They read `player_last5_stats`,
whose rows are each player's last five games played in any season, so 195 of the
400 players in it have windows ending before 2026 and 88 span two seasons —
a retired player's 2022 form was ranked against an active player's 2026 form
under one "Last 5" heading, and because that mart has no `season` column the
sidebar's season filter could not narrow it. The scopes now read the
`*_season_last5/10_stats` marts, which are the same columns plus `season`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shared import metrics as M
from shared import page as P
from shared import ui
from shared.db import _pll_get_table_columns, query_df, sql_in_filter, table_exists

ctx = P.init_page(
    "Leaderboards",
    "League leaders in every countable and rate stat, filterable by season, "
    "team, position and games played.",
    filters=(P.F_SEASON, P.F_TEAM, P.F_POSITION, P.F_MIN_GAMES),
)

# Scope -> (mart table, has a season column). The season column decides whether
# the sidebar's season filter can go into SQL.
#
# The last-N scopes read the season-scoped marts, not `player_last5_stats`. That
# one holds each player's last five games played *anywhere*, so 195 of 400
# players' windows end before 2026 — a retired player's 2022 form appeared on the
# same "Last 5" board as an active player's 2026 form, and the sidebar's season
# filter could not touch it because the table has no season column. The
# `*_season_last*` marts are the same columns plus `season`.
PLAYER_SCOPES = {
    "Season": ("marts.player_season_stats", True),
    "Career": ("marts.player_career_stats", False),
    "Last 5": ("marts.player_season_last5_stats", True),
    "Last 10": ("marts.player_season_last10_stats", True),
}
TEAM_SCOPES = {
    "Season": ("marts.team_season_stats", True),
    "Last 5": ("marts.team_season_last5_stats", True),
    "Last 10": ("marts.team_season_last10_stats", True),
}
DEFENSE_SCOPES = {
    "Season": ("marts.team_defense_season_stats", True),
    "Career": ("marts.team_defense_career_stats", False),
}

PLAYER_SORTS = [
    "points", "points_per_game", "scoring_points", "scoring_points_per_game",
    "goals", "goals_per_game", "one_point_goals", "two_point_goals",
    "assists", "assists_per_game", "assist_opportunities", "assist_conv_rate",
    "two_pt_conversion", "shots", "shots_per_game", "shot_pct_calc",
    "ground_balls", "ground_balls_per_game",
    "caused_turnovers", "caused_turnovers_per_game",
    "turnovers", "turnovers_per_game",
    "saves", "saves_per_game", "save_pct_calc", "clean_save_pct",
    "faceoffs_won", "faceoff_pct_calc",
    "touches", "touches_per_game", "total_passes",
]
TEAM_SORTS = [
    "scores", "scores_per_game", "score_margin_per_game", "win_pct",
    "goals", "assists", "shots", "shots_per_game", "shot_pct_calc",
    "touches", "touches_per_game",
    "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
    "turnovers", "turnovers_per_game", "saves_per_game",
    "faceoff_pct_calc", "clear_pct_calc",
]
DEFENSE_SORTS = [
    "scores_allowed_per_game", "goals_allowed_per_game",
    "opponent_shots_per_game", "opponent_goal_pct", "opponent_sog_rate",
    "save_pct_proxy", "caused_turnovers_for_per_game",
    "opponent_turnovers_per_game", "ct_per_opponent_turnover",
    "score_margin_per_game",
]

PLAYER_SUMMARY = [
    "rank", "season", "split_type", "full_name", "position", "teams", "games",
    "points", "points_per_game", "scoring_points_per_game",
    "one_point_goals", "two_point_goals", "goals_per_game", "assists_per_game",
    "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game",
    "touches_per_game",
]
TEAM_SUMMARY = [
    "rank", "season", "split_type", "team_name", "games", "wins", "losses",
    "win_pct", "scores_per_game", "shots_per_game", "touches_per_game",
    "time_in_possession_per_game", "turnovers_per_game", "saves_per_game",
    "offensive_sequence_proxy_per_game",
]
DEFENSE_SUMMARY = [
    "rank", "season", "team_name", "games",
    "scores_allowed_per_game", "goals_allowed_per_game",
    "opponent_shots_per_game", "opponent_goal_pct", "save_pct_proxy",
    "caused_turnovers_for_per_game", "opponent_turnovers_per_game",
    "score_margin_per_game",
]


def load_leaders(table: str, sort_key: str, has_season: bool,
                 entity_filter: tuple[str, list] | None = None) -> pd.DataFrame:
    """
    Every row matching the sidebar filters, ranked on `sort_key`.

    No LIMIT: a rank has to be a rank among everyone who qualifies, not among
    whichever slice a cap happened to keep. These marts are a few hundred rows.
    """
    clauses, params = ["games >= ?"], [ctx.min_games]

    if has_season and ctx.selected_seasons:
        sql, values = sql_in_filter("season", ctx.selected_seasons)
        clauses.append(sql)
        params += values
    if entity_filter is not None:
        column, values = entity_filter
        if values:
            sql, bound = sql_in_filter(column, values)
            clauses.append(sql)
            params += bound

    ascending = M.is_lower_better(sort_key)
    df = query_df(
        f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} "
        f"ORDER BY {sort_key} {'ASC' if ascending else 'DESC'} NULLS LAST",
        params,
    )
    if len(df) and sort_key in df.columns:
        df.insert(0, "rank", M.rank_series(df, sort_key))
    return df


def filter_to_teams(df: pd.DataFrame, sort_key: str) -> pd.DataFrame:
    """
    Restrict a player leaderboard to the sidebar's teams, then re-rank.

    A traded player's teams are pipe-joined in one column, so this cannot be an IN
    clause. The season and career marts spell the names out (`team_names`) while
    the last-5/10 marts carry only abbreviations (`teams`), so both spellings are
    accepted and matched against the split parts — a substring test would let a
    longer code containing "ATL" through.
    """
    if not ctx.selected_teams or df is None or not len(df):
        return df

    wanted = {str(t) for t in ctx.selected_teams}
    # team_id in the directory is the abbreviation the last-N marts use.
    if {"team_id", "team_name"}.issubset(ctx.teams.columns):
        codes = ctx.teams[ctx.teams["team_name"].isin(wanted)]["team_id"]
        wanted |= {str(c) for c in codes.dropna()}

    column = "team_names" if "team_names" in df.columns else "teams"
    if column not in df.columns:
        return df

    keep = df[column].fillna("").apply(
        lambda value: bool(wanted & {p.strip() for p in str(value).split("|")})
    )
    out = df[keep].copy()
    # Re-rank within the selection. Keeping the league-wide rank would show a
    # team's best player at #37, which reads as a mistake next to a table holding
    # only that team. Ties still share a rank, so 1, 2, 2, 4 is expected.
    if len(out):
        out["rank"] = M.rank_series(out, sort_key)
    return out


def render(df: pd.DataFrame, sort_key: str, name_col: str, rows: int,
           summary_cols: list[str], color_col: str | None, csv_name: str,
           scope_label: str) -> None:
    """Chart, summary table, full table and CSV for one leaderboard."""
    if not len(df):
        st.info("No rows match the current filters. Try lowering the minimum games "
                "or widening the sidebar selection.")
        return
    if sort_key not in df.columns:
        st.warning(f"{M.label(sort_key)} is not available for {scope_label}.")
        return

    shown = df.head(rows)

    # Some volume metrics carry no direction — more touches is not better, just
    # more. Say "most" rather than "best" for those, and skip the colour scale,
    # since green would assert a direction the registry deliberately withholds.
    directed = M.direction(sort_key) is not None
    ordering = ("lowest first" if M.is_lower_better(sort_key)
                else "highest first")
    st.caption(
        f"{len(df):,} qualifying rows, ranked on {M.label(sort_key)}, "
        f"{ordering}"
        + ("" if directed else " — neither direction is inherently better for "
                              "this metric, so read it as volume, not quality")
        + f". Showing the top {len(shown):,}; the download covers all {len(df):,}."
    )

    ui.safe_bar_chart(
        shown.head(20).sort_values(sort_key, ascending=not M.is_lower_better(sort_key)),
        x_col=name_col, y_col=sort_key,
        color_col=color_col if color_col in shown.columns else None,
        title=f"{scope_label} — top {min(20, len(shown))} by {M.label(sort_key)}",
        orientation="h",
    )

    # The sort metric is always visible, even when it is not a summary column —
    # otherwise the table is ordered by something the reader cannot see.
    cols = M.existing(shown, summary_cols)
    if sort_key not in cols:
        cols.append(sort_key)
    ui.display_table(shown[cols], height=460,
                     highlight=sort_key if directed else None)
    ui.definition_caption(cols)

    with st.expander("All columns", expanded=False):
        ui.display_table(shown, height=520)

    ui.download_csv(df, csv_name, label="Download all qualifying rows (CSV)")


section = st.radio(
    "Leaderboard", options=["Players", "Teams", "Defence"], horizontal=True,
    key="leaderboard_section",
)

# ============================================================
# PLAYERS
# ============================================================

if section == "Players":
    controls = st.columns([1, 1.6, 0.8])
    scope = controls[0].selectbox("Scope", options=list(PLAYER_SCOPES),
                                  key="leader_player_scope")
    table, has_season = PLAYER_SCOPES[scope]

    available = _pll_get_table_columns("marts", table.split(".")[-1])
    sort_key = ui.metric_selectbox(
        "Rank by", options=[c for c in PLAYER_SORTS if c in available],
        key="leader_player_sort", container=controls[1],
    )
    rows = controls[2].number_input("Show top", min_value=5, max_value=500,
                                    value=25, step=5, key="leader_player_rows")

    if sort_key:
        leaders = load_leaders(
            table, sort_key, has_season,
            entity_filter=("position", ctx.selected_positions),
        )
        leaders = filter_to_teams(leaders, sort_key)

        render(leaders, sort_key, "full_name", int(rows), PLAYER_SUMMARY,
               "position", f"pll_player_leaders_{sort_key}.csv", f"Player {scope}")

# ============================================================
# TEAMS
# ============================================================

elif section == "Teams":
    controls = st.columns([1, 1.6, 0.8])
    scope = controls[0].selectbox("Scope", options=list(TEAM_SCOPES),
                                  key="leader_team_scope")
    table, has_season = TEAM_SCOPES[scope]

    available = _pll_get_table_columns("marts", table.split(".")[-1])
    sort_key = ui.metric_selectbox(
        "Rank by", options=[c for c in TEAM_SORTS if c in available],
        key="leader_team_sort", container=controls[1],
    )
    rows = controls[2].number_input("Show top", min_value=5, max_value=200,
                                    value=25, step=5, key="leader_team_rows")

    if sort_key:
        leaders = load_leaders(table, sort_key, has_season,
                               entity_filter=("team_name", ctx.selected_teams))
        render(leaders, sort_key, "team_name", int(rows), TEAM_SUMMARY,
               "season", f"pll_team_leaders_{sort_key}.csv", f"Team {scope}")

# ============================================================
# DEFENCE
# ============================================================

else:
    st.caption(
        "Opponent allowance and defensive suppression. Allowed metrics are ranked "
        "so the best defence is first — the registry knows which way is good, so "
        "nothing here needs reading backwards."
    )

    controls = st.columns([1, 1.6, 0.8])
    scope = controls[0].selectbox("Scope", options=list(DEFENSE_SCOPES),
                                  key="defense_leader_scope")
    table, has_season = DEFENSE_SCOPES[scope]
    schema, name = table.split(".")

    if not table_exists(schema, name):
        st.info("Defensive/opponent marts are not available in this warehouse build.")
    else:
        available = _pll_get_table_columns(schema, name)
        sort_key = ui.metric_selectbox(
            "Rank by", options=[c for c in DEFENSE_SORTS if c in available],
            key="defense_leader_metric", container=controls[1],
        )
        rows = controls[2].number_input("Show top", min_value=5, max_value=200,
                                        value=25, step=5, key="defense_leader_rows")

        if sort_key:
            leaders = load_leaders(table, sort_key, has_season,
                                   entity_filter=("team_name", ctx.selected_teams))
            render(leaders, sort_key, "team_name", int(rows), DEFENSE_SUMMARY,
                   "season", f"pll_defense_leaders_{sort_key}.csv",
                   f"Defence {scope}")

st.divider()
nav = st.columns(3)
with nav[0]:
    P.link_to("rankings", "Player rankings →")
with nav[1]:
    P.link_to("league", "League overview →")
with nav[2]:
    P.link_to("guide", "What do these mean? →")
