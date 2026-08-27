"""
shared/segments.py — regular season, playoffs, or both.

The warehouse now ingests postseason games alongside the regular season. Rather
than add a `competition_type` column to every mart and hope each of the ~110
hardcoded queries remembers to filter on it, the builder writes three copies of
every game-grain table and every mart derived from one:

    marts.player_season_stats            regular season only  (unchanged meaning)
    marts.player_season_stats_all        regular season + playoffs
    marts.player_season_stats_playoffs   playoffs only

Separate tables beat a segment column because a call site that forgets the filter
then shows the wrong *scope* — a number that is stale but internally consistent —
instead of triple-counting rows into a number that is simply wrong.

This module is the single place that knows which physical table a logical name
resolves to. `shared.db.query_df` rewrites every query through `resolve_sql`, so
a page gets the selected scope without naming it: the page still says
`FROM marts.player_season_stats` and the resolver decides what that means right
now. That is deliberate — 110 call sites is 110 chances to miss one.

Two fallbacks, chosen so a partially-built warehouse never lies:

* `all` with no `_all` table → the regular table. Before the postseason ingest
  ran, "regular + playoffs" *is* the regular season, so this is correct rather
  than merely convenient.
* `playoffs` with no `_playoffs` table → an empty result with the right columns.
  A missing playoffs variant means there are no playoff rows of that kind, so
  zero rows is the honest answer. Falling back to the regular table here would
  put regular-season numbers under a "Playoffs only" heading.
"""

from __future__ import annotations

import re

import streamlit as st

# ============================================================
# SCOPES
# ============================================================

REGULAR = "regular"
ALL = "all"
PLAYOFFS = "playoffs"

SCOPES = (REGULAR, ALL, PLAYOFFS)

# Must match SCOPE_SUFFIX in scripts/build_warehouse.py.
SCOPE_SUFFIX = {REGULAR: "", ALL: "_all", PLAYOFFS: "_playoffs"}

SCOPE_LABEL = {
    REGULAR: "Regular season",
    ALL: "Regular + playoffs",
    PLAYOFFS: "Playoffs only",
}

SCOPE_SHORT = {
    REGULAR: "regular season",
    ALL: "regular season and playoffs",
    PLAYOFFS: "playoff games",
}

SCOPE_HELP = (
    "Which games every stat on this page counts. Regular season is the default "
    "and matches how the league reports its leaders; playoff games are a "
    "different sample and are kept separate unless you ask for both."
)

STATE_KEY = "pll_segment_scope"

# Set per page load. A page that does not render the scope control must not be
# quietly served another page's scope, so it opts out of the resolver entirely
# rather than showing playoff tables under a warehouse-wide heading. It lives in
# session state, not a module global, because one imported module is shared by
# every browser session in the process.
SUPPRESS_KEY = "_pll_segment_scope_suppressed"

# Must match SEGMENT_CLEAN_NAMES / SEGMENT_MART_NAMES in the builder. A name
# absent here is never rewritten, which is why the directories, the schedule and
# the QC tables stay whole-warehouse whatever the scope.
SCOPED_TABLES = {
    "clean": (
        "game_manifest",
        "team_game_stats",
        "player_game_stats",
    ),
    "marts": (
        "player_season_stats_by_team",
        "player_season_stats",
        "player_career_stats",
        "player_vs_opponent_stats",
        "player_last5_stats",
        "player_last10_stats",
        "player_season_last5_stats",
        "player_season_last10_stats",
        "player_ranking_profiles",
        "team_season_stats",
        "team_career_stats",
        "team_vs_opponent_stats",
        "team_last5_stats",
        "team_last10_stats",
        "team_season_last5_stats",
        "team_season_last10_stats",
        "team_style_profiles",
        "team_game_opponent_context",
        "team_defense_season_stats",
        "team_defense_career_stats",
    ),
}

# Columns the postseason ingest added to the game-grain tables.
SEGMENT_COLS = ("competition_type", "round_label")

# The value `competition_type` carries for a regular-season game.
REGULAR_SEGMENT_VALUE = "regular"


# ============================================================
# NAME RESOLUTION
# ============================================================

# `\b` at the end matters: it stops `clean.game_manifest` from matching the
# leading characters of `clean.game_manifest_all`, so a rewritten name can never
# be rewritten twice, and the pre-existing `clean.game_schedule_all` is left
# alone because `game_schedule` is not a scoped name.
_TABLE_RE = re.compile(
    r"\b(clean|marts)\.("
    + "|".join(sorted(
        (n for names in SCOPED_TABLES.values() for n in names),
        key=len, reverse=True,
    ))
    + r")\b"
)


def is_scoped_table(schema: str, table: str) -> bool:
    return table in SCOPED_TABLES.get(schema, ())


@st.cache_data(ttl=600, show_spinner=False)
def _table_index() -> frozenset:
    """Every `schema.table` in the warehouse, as one query rather than N."""
    from shared.db import query_df

    try:
        df = query_df(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('clean', 'marts')
            """,
            scoped=False,
        )
    except Exception:
        return frozenset()
    return frozenset(
        f"{row.table_schema}.{row.table_name}" for row in df.itertuples()
    )


def variant_exists(schema: str, table: str, scope: str) -> bool:
    suffix = SCOPE_SUFFIX.get(scope, "")
    if not suffix:
        return f"{schema}.{table}" in _table_index()
    return f"{schema}.{table}{suffix}" in _table_index()


def has_segment_tables() -> bool:
    """True once a build has written segment variants — any of them.

    Checked against the expected names rather than by suffix, because
    `clean.game_schedule_all` predates this scheme and is not a variant.
    """
    index = _table_index()
    return any(
        f"{schema}.{table}{suffix}" in index
        for schema, names in SCOPED_TABLES.items()
        for table in names
        for suffix in ("_all", "_playoffs")
    )


def available_scopes() -> tuple:
    """Scopes this warehouse build can actually serve."""
    if has_segment_tables():
        return SCOPES
    return (REGULAR,)


def resolve_table(schema: str, table: str, scope: str | None = None) -> str:
    """Physical table for a logical name, or "" when the answer is no rows.

    Returning "" rather than the regular table for a missing playoffs variant is
    the point: see the module docstring.
    """
    scope = scope or current_scope()
    if scope == REGULAR or not is_scoped_table(schema, table):
        return f"{schema}.{table}"
    if variant_exists(schema, table, scope):
        return f"{schema}.{table}{SCOPE_SUFFIX[scope]}"
    if scope == ALL:
        return f"{schema}.{table}"
    return ""


def select_columns(schema: str = "clean", table: str = "player_game_stats") -> str:
    """`"competition_type, round_label, "` — but only the columns that exist.

    Both arrive with a warehouse rebuild, and the app has to keep working against
    the build that is committed right now, so a game-log query splices this in
    rather than naming columns the current DuckDB file may not have yet.
    """
    from shared.db import _pll_get_table_columns

    physical = resolve_table(schema, table) or f"{schema}.{table}"
    sch, _, tbl = physical.partition(".")
    have = set(_pll_get_table_columns(sch, tbl))
    return "".join(f"{c}, " for c in SEGMENT_COLS if c in have)


def resolve_sql(sql: str, scope: str | None = None) -> str:
    """Rewrite every scoped table reference in `sql` for the active scope."""
    if not sql:
        return sql
    scope = scope or current_scope()
    if scope == REGULAR:
        return sql

    def swap(match: re.Match) -> str:
        schema, table = match.group(1), match.group(2)
        physical = resolve_table(schema, table, scope)
        if physical:
            return physical
        # No rows, but keep the column list so `SELECT a, b FROM ...` still
        # parses and pages render their empty state instead of an exception.
        return f"(SELECT * FROM {schema}.{table} WHERE 1 = 0)"

    return _TABLE_RE.sub(swap, sql)


# ============================================================
# SELECTED SCOPE
# ============================================================

def current_scope() -> str:
    if st.session_state.get(SUPPRESS_KEY):
        return REGULAR
    scope = st.session_state.get(STATE_KEY, REGULAR)
    return scope if scope in SCOPES else REGULAR


def suppress_scope(flag: bool = True) -> None:
    """Make this page read the regular-season tables whatever is selected."""
    st.session_state[SUPPRESS_KEY] = bool(flag)


def selected_scope() -> str:
    """The scope the user picked, even on a page that ignores it."""
    scope = st.session_state.get(STATE_KEY, REGULAR)
    return scope if scope in SCOPES else REGULAR


def set_scope(scope: str) -> None:
    st.session_state[STATE_KEY] = scope if scope in SCOPES else REGULAR


def scope_label(scope: str | None = None) -> str:
    return SCOPE_LABEL.get(scope or current_scope(), SCOPE_LABEL[REGULAR])


def scope_note(scope: str | None = None) -> str:
    """One line a page can print to say what its numbers count."""
    scope = scope or current_scope()
    note = f"Counting {SCOPE_SHORT[scope]}."
    if scope == PLAYOFFS:
        # An empty page is the expected answer here for most of the year, so it is
        # explained up front rather than left to read as a broken page.
        note += (" A season whose playoffs have not been played yet has no rows "
                 "in this scope.")
    return note


def is_regular() -> bool:
    return current_scope() == REGULAR


def render_control(sidebar: bool = True) -> str:
    """Draw the scope picker and return the selected scope."""
    target = st.sidebar if sidebar else st
    options = available_scopes()

    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = REGULAR
    if st.session_state[STATE_KEY] not in options:
        # A build that lost its variants must not leave the app stuck on one.
        st.session_state[STATE_KEY] = REGULAR

    if len(options) == 1:
        target.caption(
            "Regular season only — this warehouse build has no playoff tables "
            "yet. Run the **Update PLL Data Warehouse** workflow to add them."
        )
        return REGULAR

    target.radio(
        "Games included",
        options=options,
        format_func=lambda s: SCOPE_LABEL[s],
        key=STATE_KEY,
        help=SCOPE_HELP,
    )
    return current_scope()


# ============================================================
# FRAMES THE RESOLVER CANNOT REACH
# ============================================================
#
# The schedule is one table for every segment — there is no `_playoffs` copy of a
# fixture list — so a frame read from it is narrowed by column instead of by name.

def is_postseason_value(value) -> bool:
    """Anything the ingest labelled other than 'regular' is postseason.

    Mirrors is_postseason_segment in the builder without hardcoding its env var:
    champseries and all-star games are excluded at ingest, so the only other
    label that reaches the warehouse is the postseason one.
    """
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "<na>"}:
        return False
    return text != REGULAR_SEGMENT_VALUE


def filter_frame(df, scope: str | None = None, column: str = "competition_type"):
    """Keep the rows of `df` belonging to `scope`.

    A frame with no segment column is returned untouched: that is a warehouse
    built before the postseason ingest, where every row is a regular-season row.
    """
    scope = scope or current_scope()
    if df is None or len(df) == 0 or column not in getattr(df, "columns", []):
        return df
    if scope == ALL:
        return df
    is_post = df[column].map(is_postseason_value)
    return df[is_post if scope == PLAYOFFS else ~is_post].copy()


# ============================================================
# DISPLAY
# ============================================================

def segment_display(competition_type, round_label=None) -> str:
    """A game's segment as a phrase: "Regular", "Playoffs", "Semifinal"."""
    label = str(round_label).strip() if round_label is not None else ""
    if label and label.lower() not in {"nan", "none", "<na>"}:
        return label
    value = str(competition_type or "").strip().lower()
    if not value or value == "nan":
        return ""
    if value == REGULAR_SEGMENT_VALUE:
        return "Regular"
    if value == "post":
        return "Playoffs"
    return value.replace("_", " ").title()
