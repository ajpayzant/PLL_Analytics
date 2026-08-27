"""
scripts/check_mart_functions.py — does the mart arithmetic still produce the
committed numbers?

The warehouse is built in CI against the PLL API, so a change to build_warehouse.py
normally cannot be tried out locally. This harness lifts the pure aggregation
functions out of that script with the ast module — definitions only, none of the
ingest — runs them over the game rows already committed under
data/curated_data/all_requested_seasons, and compares the result to the mart
parquet committed beside them.

That is the check the segment-scope work needed: the regular-season tables were
rebuilt from extracted functions instead of inline code, and this proves the
extraction changed no number.

    python scripts/check_mart_functions.py

Exit code 0 means every compared mart matched. Player ranking and team style
profiles are not compared: they are deterministic functions of the season and
career tables that are compared here, and they carry ~200 scored columns whose
float noise would drown the signal.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_warehouse.py"
CURATED = PROJECT_ROOT / "data" / "curated_data" / "all_requested_seasons"


# ------------------------------------------------------------
# Lift the definitions out of the builder
# ------------------------------------------------------------

def load_builder_namespace() -> dict:
    """Every top-level function and literal constant from build_warehouse.py.

    Nothing else is executed, so no token is read, no request is made and no file
    is written. A function whose own globals are missing only fails if it is called.
    """
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))

    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Decorators are retry/caching wrappers on the fetch helpers, which this
            # harness never calls. Dropping them keeps tenacity out of the namespace.
            node.decorator_list = []
            kept.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                continue
            kept.append(node)

    ns: dict = {
        "np": np,
        "pd": pd,
        "__builtins__": __builtins__,
        # Read from the environment in the builder; fixed here to the shipped default.
        "TARGET_SEASONS": [2022, 2023, 2024, 2025, 2026],
        "COMPETITION_TYPE": "regular",
        "POSTSEASON_COMPETITION_TYPES": {"post"},
        "INCLUDED_COMPETITION_TYPES": {"regular", "post"},
    }

    exec(compile(ast.Module(body=kept, type_ignores=[]), str(BUILDER), "exec"), ns)
    return ns


def read_curated(name: str) -> pd.DataFrame | None:
    path = CURATED / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

def compare(name: str, built: pd.DataFrame, committed: pd.DataFrame, keys: list[str]) -> list[str]:
    """Row-for-row, column-for-column comparison on the shared columns."""
    problems = []

    if built is None or len(built) == 0:
        return [f"{name}: built nothing"]

    if len(built) != len(committed):
        problems.append(f"{name}: {len(built)} rows built, {len(committed)} committed")

    missing = [c for c in committed.columns if c not in built.columns]
    if missing:
        problems.append(f"{name}: {len(missing)} committed columns absent from the build: {missing[:8]}")

    keys = [k for k in keys if k in built.columns and k in committed.columns]
    if not keys:
        return problems + [f"{name}: no shared key columns to align on"]

    a = built.sort_values(keys).reset_index(drop=True)
    b = committed.sort_values(keys).reset_index(drop=True)

    shared = [c for c in b.columns if c in a.columns]
    if len(a) != len(b):
        return problems

    for c in shared:
        left, right = a[c], b[c]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            lv = pd.to_numeric(left, errors="coerce").astype(float)
            rv = pd.to_numeric(right, errors="coerce").astype(float)
            bad = int((~np.isclose(lv, rv, rtol=1e-9, atol=1e-9, equal_nan=True)).sum())
        else:
            lv = left.astype("string").fillna("<na>")
            rv = right.astype("string").fillna("<na>")
            bad = int((lv != rv).sum())
        if bad:
            problems.append(f"{name}.{c}: {bad} of {len(a)} values differ")

    return problems


def relabel_tail_as_postseason(df: pd.DataFrame, games_per_season: int = 5) -> pd.DataFrame:
    """Call the last few games of each season a playoff game.

    The committed warehouse predates the postseason ingest, so nothing in it is
    labelled 'post'. Relabelling by (season, game_number) is consistent across the
    manifest, team rows and player rows, which is what makes the totals add up.
    """
    out = df.copy()
    if "game_number" not in out.columns or "season" not in out.columns:
        return out
    gn = pd.to_numeric(out["game_number"], errors="coerce")
    cut = gn.groupby(out["season"]).transform("max") - games_per_season
    out["competition_type"] = np.where(gn > cut, "post", "regular")
    return out


def check_scopes(ns: dict, player_games: pd.DataFrame, team_games: pd.DataFrame,
                 manifest: pd.DataFrame | None) -> list[str]:
    """Build all three scopes over relabelled rows and check they add up.

    Two things have to hold, and neither needs the API:
      * the 'all' scope over the relabelled rows reproduces the committed marts
        exactly — relabelling moved no row, so no total may move;
      * regular + playoffs accounts for all of it, with nothing counted twice.
    """
    problems: list[str] = []

    scope_frame = ns["scope_frame"]
    build = ns["build_scoped_tables"]

    pgs = relabel_tail_as_postseason(player_games)
    tgs = relabel_tail_as_postseason(team_games)
    gm = relabel_tail_as_postseason(manifest) if manifest is not None else pd.DataFrame()

    scoped = {}
    for scope in ("regular", "all", "playoffs"):
        scoped[scope] = build(
            scope_frame(pgs, scope, "player_game_stats"),
            scope_frame(tgs, scope, "team_game_stats"),
            scope_frame(gm, scope, "game_manifest"),
        )

    # 1. The 'all' scope must reproduce what is committed.
    for name, keys in (("player_season_stats", ["season", "player_id"]),
                       ("team_season_stats", ["season", "team_id"]),
                       ("player_career_stats", ["player_id"]),
                       ("team_defense_season_stats", ["season", "team_id"])):
        committed = read_curated(name)
        if committed is None:
            continue
        found = compare(f"{name} [all scope]", scoped["all"][name], committed, keys)
        if found:
            problems.extend(found)
            for p in found[:4]:
                print("  FAIL  " + p)
        else:
            print(f"  ok    {name} [all scope]: matches the committed table")

    # 2. Regular + playoffs must account for every game and every point.
    for name, keys, totals in (("team_season_stats", ["season", "team_id"], ["games", "scores"]),
                               ("player_season_stats", ["season", "player_id"], ["games", "points"])):
        every, regular, post = scoped["all"][name], scoped["regular"][name], scoped["playoffs"][name]
        if any(f is None or len(f) == 0 for f in (every, regular)):
            problems.append(f"{name}: a scope built nothing")
            continue

        cols = [c for c in totals if c in every.columns]
        merged = (
            every[keys + cols]
            .merge(regular[keys + cols], on=keys, how="left", suffixes=("", "_regular"))
            .merge(post[keys + cols] if len(post) else None, on=keys, how="left", suffixes=("", "_post"))
        )

        for c in cols:
            left = pd.to_numeric(merged[c], errors="coerce").fillna(0)
            right = (pd.to_numeric(merged.get(f"{c}_regular"), errors="coerce").fillna(0)
                     + pd.to_numeric(merged.get(f"{c}_post"), errors="coerce").fillna(0))
            bad = int((~np.isclose(left, right, rtol=1e-9, atol=1e-9)).sum())
            if bad:
                problems.append(f"{name}.{c}: regular + playoffs misses all on {bad} rows")
                print(f"  FAIL  {name}.{c}: regular + playoffs misses all on {bad} rows")
            else:
                print(f"  ok    {name}.{c}: regular + playoffs = all on {len(merged)} rows")

        if len(post) == 0:
            problems.append(f"{name}: the playoffs scope built nothing from relabelled rows")

    # 3. The playoffs scope must hold only playoff rows.
    post_games = scoped["playoffs"]["player_game_stats"]
    if len(post_games) and "competition_type" in post_games.columns:
        stray = int((~post_games["competition_type"].map(ns["is_postseason_segment"])).sum())
        if stray:
            problems.append(f"playoffs scope carries {stray} non-playoff game rows")
            print(f"  FAIL  playoffs scope carries {stray} non-playoff game rows")
        else:
            print(f"  ok    playoffs scope holds {len(post_games)} rows, all of them playoff rows")

    return problems


def main() -> int:
    ns = load_builder_namespace()

    player_games = read_curated("player_game_stats")
    team_games = read_curated("team_game_stats")

    if player_games is None or team_games is None:
        print("No committed game rows under", CURATED, "— nothing to check against.")
        return 1

    # The builder narrows the sum-column lists to what the game rows actually carry.
    ns["PLAYER_SUM_COLS"] = [c for c in ns["PLAYER_SUM_COLS"] if c in player_games.columns]
    ns["TEAM_SUM_COLS"] = [c for c in ns["TEAM_SUM_COLS"] if c in team_games.columns]

    # Committed clean.team_game_stats already carries the result flags; adding them
    # again is a no-op, and it keeps this harness honest about the build order.
    team_games = ns["add_score_based_team_result_flags"](team_games)

    context = ns["build_team_game_opponent_context"](team_games)

    cases = [
        ("player_season_stats_by_team",
         ns["build_player_season_stats_by_team"](player_games),
         ["season", "player_id", "team_id"]),
        ("player_season_stats",
         ns["build_player_season_stats"](player_games),
         ["season", "player_id"]),
        ("player_career_stats",
         ns["build_player_career_stats"](player_games),
         ["player_id"]),
        ("player_vs_opponent_stats",
         ns["build_player_vs_opponent_stats"](player_games),
         ["player_id", "opponent_team_id"]),
        ("player_last5_stats",
         ns["build_player_last_n_stats"](player_games, n=5, by_season=False),
         ["player_id"]),
        ("player_last10_stats",
         ns["build_player_last_n_stats"](player_games, n=10, by_season=False),
         ["player_id"]),
        ("player_season_last5_stats",
         ns["build_player_last_n_stats"](player_games, n=5, by_season=True),
         ["season", "player_id"]),
        ("player_season_last10_stats",
         ns["build_player_last_n_stats"](player_games, n=10, by_season=True),
         ["season", "player_id"]),
        ("team_season_stats",
         ns["build_team_season_stats"](team_games),
         ["season", "team_id"]),
        ("team_career_stats",
         ns["build_team_career_stats"](team_games),
         ["team_id"]),
        ("team_vs_opponent_stats",
         ns["build_team_vs_opponent_stats"](team_games),
         ["team_id", "opponent_team_id"]),
        ("team_last5_stats",
         ns["build_team_last_n_stats"](team_games, n=5, by_season=False),
         ["team_id"]),
        ("team_last10_stats",
         ns["build_team_last_n_stats"](team_games, n=10, by_season=False),
         ["team_id"]),
        ("team_season_last5_stats",
         ns["build_team_last_n_stats"](team_games, n=5, by_season=True),
         ["season", "team_id"]),
        ("team_season_last10_stats",
         ns["build_team_last_n_stats"](team_games, n=10, by_season=True),
         ["season", "team_id"]),
        ("team_game_opponent_context", context, ["game_id", "team_id"]),
        ("team_defense_season_stats",
         ns["build_team_defense_agg"](context, ["season", "team_id", "team_name"]),
         ["season", "team_id"]),
        ("team_defense_career_stats",
         ns["build_team_defense_agg"](context, ["team_id", "team_name"]),
         ["team_id"]),
    ]

    all_problems = []
    for name, built, keys in cases:
        committed = read_curated(name)
        if committed is None:
            print(f"  --    {name}: nothing committed to compare against")
            continue
        problems = compare(name, built, committed, keys)
        if problems:
            all_problems.extend(problems)
            for p in problems[:6]:
                print("  FAIL  " + p)
            if len(problems) > 6:
                print(f"        ... and {len(problems) - 6} more differences in {name}")
        else:
            print(f"  ok    {name}: {len(built)} rows match")

    # The scope helpers, on the frames themselves.
    scope_frame = ns["scope_frame"]
    regular = scope_frame(player_games, "regular", "player_game_stats")
    every = scope_frame(player_games, "all", "player_game_stats")
    playoffs = scope_frame(player_games, "playoffs", "player_game_stats")

    if "competition_type" in player_games.columns:
        if len(regular) + len(playoffs) != len(every):
            all_problems.append(
                f"scope_frame: regular {len(regular)} + playoffs {len(playoffs)} != all {len(every)}")
        else:
            print(f"  ok    scope_frame splits {len(every)} rows into "
                  f"{len(regular)} regular + {len(playoffs)} playoff")
    else:
        print("  --    committed game rows carry no competition_type column "
              "(pre-playoff warehouse), so the scopes cannot be split yet")

    print("\nsegment scopes, over game rows relabelled to fake a playoff bracket:")
    all_problems.extend(check_scopes(ns, player_games, team_games, read_curated("game_manifest")))

    print()
    if all_problems:
        print(f"{len(all_problems)} difference(s) found.")
        return 1

    print("Every compared mart matches the committed warehouse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
