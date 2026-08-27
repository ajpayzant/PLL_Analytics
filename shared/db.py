"""
shared/db.py — Database connection and query helpers for PLL Analytics.
"""

import os
import duckdb
import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
DB_PATH = os.getenv("PLL_DB_PATH", os.path.join(DATA_DIR, "analytics_database", "pll_warehouse.duckdb"))
ARTIFACT_INDEX_PATH = os.getenv(
    "PLL_ARTIFACT_INDEX_PATH",
    os.path.join(DATA_DIR, "curated_data", "all_requested_seasons", "artifact_index.csv")
)

# ============================================================
# CONNECTION
# ============================================================
#
# One shared, read-only DuckDB connection for the whole app session, cached as a
# Streamlit resource. The previous implementation opened a NEW duckdb.connect()
# on EVERY query_df/read_table call; with many cached queries firing per page
# load (and on Streamlit reruns) that produced many concurrent opens against the
# same file. Under load on Streamlit Cloud that intermittently crashed the
# container — surfacing as a "400 connection error" on load and needing a
# reboot. A single cached connection removes the churn. Access is serialised
# with a lock because one DuckDB connection object is not safe to use from
# multiple threads at once (Streamlit may run work on several threads), and we
# use .cursor() per query so concurrent callers don't clobber each other's
# result cursor.

import threading

_CONN_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def get_connection():
    """Process-wide read-only DuckDB connection (opened once, reused)."""
    return duckdb.connect(DB_PATH, read_only=True)


def _run(sql, params=None):
    con = get_connection()
    with _CONN_LOCK:
        cur = con.cursor()
        try:
            if params is None:
                return cur.execute(sql).df()
            return cur.execute(sql, params).df()
        finally:
            cur.close()


# Every query goes through the segment resolver before it is cached, so the
# selected scope (regular season / playoffs / both) is part of the cache key —
# rewriting after the cache would serve one scope's rows under another's key.
# Pass scoped=False for anything that describes the warehouse itself rather than
# a sample of games: the directories, the schedule, information_schema, QC.

@st.cache_data(ttl=600, show_spinner=False)
def _query_cached(sql, params=None):
    return _run(sql, params)


def query_df(sql, params=None, scoped: bool = True):
    if scoped:
        from shared import segments
        sql = segments.resolve_sql(sql)
    return _query_cached(sql, params)


def read_table(table_name: str, scoped: bool = True) -> pd.DataFrame:
    return query_df(f"SELECT * FROM {table_name}", scoped=scoped)


# ============================================================
# STARTUP COUNTS
# ============================================================

# Not cached here: the result depends on the selected segment scope, which is
# session state rather than an argument, so an outer cache would serve one
# scope's counts under another's key. query_df caches the query itself.
def startup_counts(scoped: bool = True):
    df = query_df("""
        SELECT
            (SELECT COUNT(*) FROM clean.game_manifest) AS completed_games,
            (SELECT COUNT(*) FROM clean.player_game_stats) AS player_game_rows,
            (SELECT COUNT(*) FROM clean.team_game_stats) AS team_game_rows,
            (SELECT COUNT(*) FROM clean.player_directory) AS players,
            (SELECT COUNT(*) FROM clean.team_directory) AS teams
    """, scoped=scoped)
    row = df.iloc[0]
    return {
        "completed_games": int(row["completed_games"]),
        "player_game_rows": int(row["player_game_rows"]),
        "team_game_rows": int(row["team_game_rows"]),
        "players": int(row["players"]),
        "teams": int(row["teams"]),
    }


# ============================================================
# FILTER VALUES
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def filter_values():
    # Read unscoped throughout: a player's name, position and team are identity,
    # not a segment, so the pickers must offer every player and team whichever
    # games are being counted. Scoping them would drop a playoff-only call-up
    # from the sidebar the moment someone selected "Regular season".

    # Seasons
    seasons = []
    for table_name in ["clean.game_schedule_all", "clean.team_game_stats", "clean.player_game_stats"]:
        try:
            df = read_table(table_name, scoped=False)
            if df is not None and len(df) > 0 and "season" in df.columns:
                seasons = (
                    pd.to_numeric(df["season"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .drop_duplicates()
                    .sort_values()
                    .tolist()
                )
                if seasons:
                    break
        except Exception:
            continue

    # Teams
    try:
        teams = read_table("clean.team_directory", scoped=False)
    except Exception:
        teams = pd.DataFrame()

    if teams is None or len(teams) == 0:
        try:
            tgs = read_table("clean.team_game_stats", scoped=False)
            teams = tgs[[c for c in ["team_id", "team_name"] if c in tgs.columns]].copy()
        except Exception:
            teams = pd.DataFrame()

    if teams is None or len(teams) == 0:
        teams = pd.DataFrame(columns=["team_id", "team_name"])
    else:
        teams = teams.copy()
        if "team_id" not in teams.columns:
            for c in ["latest_team_id", "official_team_id", "current_team_id", "team_abbrev", "team"]:
                if c in teams.columns:
                    teams["team_id"] = teams[c]
                    break
        if "team_name" not in teams.columns:
            for c in ["latest_team_name", "official_team_name", "current_team_name", "team_display_name", "name"]:
                if c in teams.columns:
                    teams["team_name"] = teams[c]
                    break
        if "team_id" not in teams.columns:
            teams["team_id"] = pd.NA
        if "team_name" not in teams.columns:
            teams["team_name"] = teams["team_id"]
        teams["team_id"] = teams["team_id"].astype("string")
        teams["team_name"] = teams["team_name"].astype("string")
        teams = (
            teams[["team_id", "team_name"]]
            .dropna(subset=["team_id", "team_name"])
            .query("team_id != '' and team_name != ''")
            .drop_duplicates()
            .sort_values("team_name", na_position="last")
            .reset_index(drop=True)
        )

    # Players
    try:
        players = read_table("clean.player_directory", scoped=False)
    except Exception:
        players = pd.DataFrame()

    if players is None or len(players) == 0:
        try:
            pgs = read_table("clean.player_game_stats", scoped=False)
            candidate_cols = [
                c for c in [
                    "player_id", "full_name", "player_name", "name", "display_name",
                    "position", "position_name", "team_id", "team_name"
                ]
                if c in pgs.columns
            ]
            players = pgs[candidate_cols].copy() if candidate_cols else pd.DataFrame()
        except Exception:
            players = pd.DataFrame()

    if players is None or len(players) == 0:
        players = pd.DataFrame(columns=["player_id", "full_name", "position", "position_name", "team_id", "team_name"])
    else:
        players = players.copy()
        if "full_name" not in players.columns:
            for c in ["player_name", "name", "display_name", "latest_full_name"]:
                if c in players.columns:
                    players["full_name"] = players[c]
                    break
        if "full_name" not in players.columns:
            if "player_id" in players.columns:
                players["full_name"] = players["player_id"]
            else:
                players["full_name"] = pd.NA
        if "position" not in players.columns:
            for c in ["position_name", "primary_position", "latest_position"]:
                if c in players.columns:
                    players["position"] = players[c]
                    break
        if "position" not in players.columns:
            players["position"] = pd.NA
        if "position_name" not in players.columns:
            players["position_name"] = players["position"]
        if "team_id" not in players.columns:
            for c in ["latest_team_id", "official_team_id", "current_team_id", "team_abbrev", "team"]:
                if c in players.columns:
                    players["team_id"] = players[c]
                    break
        if "team_id" not in players.columns:
            players["team_id"] = pd.NA
        if "team_name" not in players.columns:
            for c in ["latest_team_name", "official_team_name", "current_team_name", "team_display_name"]:
                if c in players.columns:
                    players["team_name"] = players[c]
                    break
        if "team_name" not in players.columns:
            players["team_name"] = players["team_id"]
        if "player_id" not in players.columns:
            players["player_id"] = (
                players["full_name"]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9]+", "_", regex=True)
                .str.strip("_")
            )
        keep = ["player_id", "full_name", "position", "position_name", "team_id", "team_name"]
        for c in keep:
            if c not in players.columns:
                players[c] = pd.NA
        players["player_id"] = players["player_id"].astype("string")
        players["full_name"] = players["full_name"].astype("string")
        players["position"] = players["position"].astype("string")
        players["position_name"] = players["position_name"].astype("string")
        players["team_id"] = players["team_id"].astype("string")
        players["team_name"] = players["team_name"].astype("string")
        players = (
            players[keep]
            .dropna(subset=["player_id", "full_name"])
            .query("player_id != '' and full_name != ''")
            .drop_duplicates()
            .sort_values("full_name", na_position="last")
            .reset_index(drop=True)
        )

    # Positions
    if players is not None and len(players) > 0 and "position" in players.columns:
        positions = (
            players["position"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    else:
        positions = []

    return seasons, teams, players, positions


# ============================================================
# TABLE HELPERS
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def table_exists(schema_name, table_name):
    df = query_df("""
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
    """, [schema_name, table_name], scoped=False)
    return bool(len(df) > 0 and int(df["n"].iloc[0]) > 0)


@st.cache_data(ttl=600, show_spinner=False)
def schedule_display_table():
    """Every scheduled game, with a display status. Ignores the segment scope.

    The schedule is the fixture list, not a sample of games, so "Playoffs only"
    must not empty it. `final` is decided from the widest manifest the warehouse
    has (regular + playoffs) so a played playoff game isn't left looking unplayed.
    """
    from shared import segments
    manifest = (segments.resolve_table("clean", "game_manifest", segments.ALL)
                or "clean.game_manifest")
    return query_df(f"""
        WITH stat_games AS (
            SELECT DISTINCT season, game_id
            FROM {manifest}
        )
        SELECT
            s.*,
            CASE
                WHEN sg.game_id IS NOT NULL THEN 'final'
                WHEN s.event_status_label = 'unknown' AND s.season <= 2025 THEN 'final'
                ELSE s.event_status_label
            END AS status_display
        FROM clean.game_schedule_all s
        LEFT JOIN stat_games sg
            ON s.season = sg.season
           AND s.event_id = sg.game_id
    """, scoped=False)


@st.cache_data(ttl=600, show_spinner=False)
def table_index():
    return query_df("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('clean', 'marts', 'qc')
        ORDER BY table_schema, table_name
    """, scoped=False)


def sql_in_filter(column, values):
    if not values:
        return "1=1", []
    placeholders = ", ".join(["?"] * len(values))
    return f"{column} IN ({placeholders})", list(values)


def _pll_get_table_columns(schema_name, table_name):
    try:
        cols_df = query_df("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            ORDER BY ordinal_position
        """, [schema_name, table_name], scoped=False)
        return cols_df["column_name"].astype(str).tolist()
    except Exception:
        return []
