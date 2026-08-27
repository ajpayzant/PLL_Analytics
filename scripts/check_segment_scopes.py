"""
scripts/check_segment_scopes.py — does the app resolve the right table for the
right scope, and do the scopes add up in the warehouse it is pointed at?

Two halves, both runnable without the API:

1. THE RESOLVER, against a pretend table index. This is the part that decides
   what `FROM marts.player_season_stats` means, so it is checked against a table
   list rather than a database: names swapped for the selected scope, names left
   alone when the scope is regular, a missing playoffs variant answered with no
   rows rather than with regular-season rows, and — the one that would be silent
   and wrong — `clean.game_schedule_all` never mistaken for a variant.

2. THE WAREHOUSE, if it has segment variants. Every resolved name is queried, and
   the counts are checked to add up: regular + playoffs = all. Run this after the
   CI rebuild lands to confirm the build the app is reading is coherent.

    python scripts/check_segment_scopes.py

Exit code 0 means every check passed. A warehouse with no variants yet skips the
second half and says so — that is not a failure, it is a build that has not run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import segments as S


# ------------------------------------------------------------
# 1. The resolver
# ------------------------------------------------------------

# Everything a fully built warehouse would hold, plus the pre-existing
# clean.game_schedule_all, which ends in _all and is not a variant of anything.
FULL_INDEX = frozenset(
    [f"{schema}.{table}{suffix}"
     for schema, names in S.SCOPED_TABLES.items()
     for table in names
     for suffix in ("", "_all", "_playoffs")]
    + ["clean.game_schedule_all", "clean.player_directory", "clean.team_directory"]
)

# One row per game or per team-game, so the three scopes partition these exactly.
# Everything else is an aggregate, where a player who played in both scopes is one
# row in each and one row in the combined table.
GAME_GRAIN = {
    "clean.game_manifest",
    "clean.team_game_stats",
    "clean.player_game_stats",
    "marts.team_game_opponent_context",
}

# A build with no postseason: only the unsuffixed tables.
BARE_INDEX = frozenset(
    [f"{schema}.{table}"
     for schema, names in S.SCOPED_TABLES.items()
     for table in names]
    + ["clean.game_schedule_all"]
)


def with_index(index):
    """Point the resolver at a pretend warehouse."""
    S._table_index = lambda: index  # noqa: SLF001 — the seam this check needs


def check_resolver() -> list[str]:
    problems: list[str] = []

    def expect(label, got, want):
        if got != want:
            problems.append(f"{label}: got {got!r}, wanted {want!r}")
            print(f"  FAIL  {label}\n          got  {got}\n          want {want}")
        else:
            print(f"  ok    {label}")

    with_index(FULL_INDEX)

    expect(
        "regular scope leaves the query alone",
        S.resolve_sql("SELECT * FROM marts.player_season_stats WHERE season = ?", S.REGULAR),
        "SELECT * FROM marts.player_season_stats WHERE season = ?",
    )
    expect(
        "all scope reads the _all table",
        S.resolve_sql("SELECT * FROM marts.player_season_stats", S.ALL),
        "SELECT * FROM marts.player_season_stats_all",
    )
    expect(
        "playoffs scope reads the _playoffs table",
        S.resolve_sql("SELECT * FROM clean.player_game_stats", S.PLAYOFFS),
        "SELECT * FROM clean.player_game_stats_playoffs",
    )
    expect(
        "every reference in a join is swapped",
        S.resolve_sql(
            "SELECT c.* FROM marts.team_career_stats c "
            "JOIN clean.team_game_stats g ON g.team_id = c.team_id",
            S.ALL,
        ),
        "SELECT c.* FROM marts.team_career_stats_all c "
        "JOIN clean.team_game_stats_all g ON g.team_id = c.team_id",
    )
    expect(
        "the schedule is not a scoped table",
        S.resolve_sql("SELECT * FROM clean.game_schedule_all", S.PLAYOFFS),
        "SELECT * FROM clean.game_schedule_all",
    )
    expect(
        "an already-scoped name is not scoped twice",
        S.resolve_sql("SELECT * FROM clean.game_manifest_all", S.ALL),
        "SELECT * FROM clean.game_manifest_all",
    )
    expect(
        "the directories are never scoped",
        S.resolve_sql("SELECT * FROM clean.player_directory", S.PLAYOFFS),
        "SELECT * FROM clean.player_directory",
    )
    expect(
        "a qc table is never scoped",
        S.resolve_sql("SELECT * FROM qc.quality_summary", S.ALL),
        "SELECT * FROM qc.quality_summary",
    )

    # The fallbacks, on a warehouse that has not been rebuilt yet.
    with_index(BARE_INDEX)
    expect(
        "all falls back to the regular table when there is no _all",
        S.resolve_sql("SELECT * FROM marts.team_season_stats", S.ALL),
        "SELECT * FROM marts.team_season_stats",
    )
    expect(
        "playoffs answers with no rows rather than regular-season rows",
        S.resolve_sql("SELECT * FROM marts.team_season_stats", S.PLAYOFFS),
        "SELECT * FROM (SELECT * FROM marts.team_season_stats WHERE 1 = 0)",
    )
    expect(
        "a warehouse with no variants offers only the regular scope",
        S.available_scopes(),
        (S.REGULAR,),
    )
    with_index(FULL_INDEX)
    expect(
        "a warehouse with variants offers all three",
        S.available_scopes(),
        S.SCOPES,
    )

    # Frames the resolver cannot reach: the schedule.
    frame = pd.DataFrame({
        "game_number": [1, 2, 3, 4],
        "competition_type": ["regular", "regular", "post", None],
    })
    expect("filter_frame keeps regular rows (a missing label is not a playoff game)",
           len(S.filter_frame(frame, S.REGULAR)), 3)
    expect("filter_frame keeps playoff rows", len(S.filter_frame(frame, S.PLAYOFFS)), 1)
    expect("filter_frame leaves the whole frame for all", len(S.filter_frame(frame, S.ALL)), 4)
    expect("regular + playoffs accounts for every schedule row",
           len(S.filter_frame(frame, S.REGULAR)) + len(S.filter_frame(frame, S.PLAYOFFS)),
           len(frame))

    untagged = pd.DataFrame({"game_number": [1, 2]})
    expect("a frame with no segment column is left alone",
           len(S.filter_frame(untagged, S.PLAYOFFS)), 2)

    expect("a round label beats the raw segment",
           S.segment_display("post", "Semifinal"), "Semifinal")
    expect("an unlabelled playoff game still reads as one",
           S.segment_display("post", None), "Playoffs")
    expect("a regular game reads as regular", S.segment_display("regular"), "Regular")
    expect("nothing known reads as nothing", S.segment_display(None, None), "")

    return problems


# ------------------------------------------------------------
# 2. The warehouse
# ------------------------------------------------------------

def check_warehouse() -> list[str]:
    """Query the real DuckDB file: do the variants exist, and do they add up?"""
    problems: list[str] = []

    import duckdb

    from shared.db import DB_PATH

    if not Path(DB_PATH).exists():
        print(f"  --    no warehouse at {DB_PATH} — skipping")
        return problems

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        index = frozenset(
            f"{schema}.{table}" for schema, table in con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema IN ('clean', 'marts')"
            ).fetchall()
        )
        with_index(index)

        if not S.has_segment_tables():
            print("  --    this build has no segment variants yet, so there is "
                  "nothing to reconcile.")
            print("        Run the 'Update PLL Data Warehouse' workflow, pull the "
                  "committed data/, and run this again.")
            return problems

        # Every name the app can ask for has to be queryable.
        for scope in S.SCOPES:
            for schema, names in S.SCOPED_TABLES.items():
                for table in names:
                    sql = S.resolve_sql(f"SELECT * FROM {schema}.{table}", scope)
                    try:
                        con.execute(sql + " LIMIT 0")
                    except Exception as exc:
                        problems.append(f"{scope}/{schema}.{table}: {exc}")
                        print(f"  FAIL  {scope}: {schema}.{table} — {exc}")
            print(f"  ok    every table resolves and queries in the '{scope}' scope")

        # And the row counts have to add up wherever all three exist.
        for schema, names in S.SCOPED_TABLES.items():
            for table in names:
                counts = {}
                for scope in S.SCOPES:
                    name = S.resolve_table(schema, table, scope)
                    if not name:
                        counts[scope] = 0
                        continue
                    counts[scope] = int(
                        con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                    )
                if not S.variant_exists(schema, table, S.ALL):
                    continue

                # One row per game (or per team-game) means the three scopes
                # partition the rows, so they must add up exactly.
                if f"{schema}.{table}" in GAME_GRAIN:
                    total = counts[S.REGULAR] + counts[S.PLAYOFFS]
                    if total != counts[S.ALL]:
                        problems.append(
                            f"{schema}.{table}: regular {counts[S.REGULAR]} + playoffs "
                            f"{counts[S.PLAYOFFS]} = {total}, but all is {counts[S.ALL]}"
                        )
                        print(f"  FAIL  {schema}.{table}: {counts[S.REGULAR]} + "
                              f"{counts[S.PLAYOFFS]} != {counts[S.ALL]}")
                    else:
                        print(f"  ok    {schema}.{table}: {counts[S.REGULAR]} regular + "
                              f"{counts[S.PLAYOFFS]} playoff = {counts[S.ALL]}")
                    continue

                # An aggregate is one row per player or team, so a man who played
                # in both scopes is one row in each and still one row in `all`:
                # the counts cannot add up, but neither scope may hold anyone the
                # combined table has never heard of.
                widest = max(counts[S.REGULAR], counts[S.PLAYOFFS])
                if counts[S.ALL] < widest:
                    problems.append(
                        f"{schema}.{table}: all has {counts[S.ALL]} rows, fewer than "
                        f"the {widest} in a single scope"
                    )
                    print(f"  FAIL  {schema}.{table}: all {counts[S.ALL]} < {widest}")
                else:
                    print(f"  ok    {schema}.{table}: {counts[S.ALL]} combined rows "
                          f"cover {counts[S.REGULAR]} regular and {counts[S.PLAYOFFS]} playoff")
    finally:
        con.close()

    return problems


def main() -> int:
    print("the resolver, against a pretend table index:")
    problems = check_resolver()

    print("\nthe warehouse this app is pointed at:")
    problems += check_warehouse()

    print()
    if problems:
        print(f"{len(problems)} problem(s) found.")
        return 1
    print("Every segment-scope check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
