"""
Rebuild `marts.player_ranking_profiles` in place, without re-scraping.

`build_warehouse.py` is a ported notebook: its module level scrapes the PLL API
and rebuilds every table, so it cannot be imported to reuse a few functions. The
ranking mart, though, is a pure function of marts that already exist
(`player_career_stats`, `player_season_stats`, `player_last5/10_stats`) — so this
loads the builder's *function definitions only* via AST, skipping every
module-level statement, and re-runs the ranking block against the warehouse.

That keeps the scoring math in exactly one place. Editing a weight or a helper in
build_warehouse.py and re-running this script is enough to see the effect on the
real board; there is no second copy of the formula here to drift out of sync.

Usage:
    python scripts/rebuild_rankings.py            # rebuild and write
    python scripts/rebuild_rankings.py --dry-run  # report, write nothing
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILDER = ROOT / "scripts" / "build_warehouse.py"
DB_PATH = ROOT / "data" / "analytics_database" / "pll_warehouse.duckdb"

# The ranking builder reaches for these notebook conveniences. The two constants
# are seeded because build_warehouse.py derives them from the environment and then
# uses them as *parameter defaults*, which Python evaluates at definition time —
# without them the scraper function definitions raise NameError.
PRELUDE = """
import os, re, json, math, datetime as dt
import numpy as np, pandas as pd
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
COMPETITION_TYPE = os.getenv("PLL_COMPETITION_TYPE", "regular").strip().lower()
def display(*a, **k):
    pass
"""

# Functions this script cannot run without; asserted after loading so a future
# refactor of build_warehouse.py fails loudly here instead of silently ranking
# with a stale copy of the math.
REQUIRED = ["_build_player_ranking_context", "_add_player_ranking_scores",
            "_rank_pct", "_sigmoid_stretch"]


def load_builder_functions() -> dict:
    """
    Exec only the top-level `def`s and constant assignments from build_warehouse.py.

    Module-level code there performs network calls and DuckDB writes, so parsing to
    an AST and keeping just the safe nodes is what makes reuse possible. Constant
    assignments have to come along because several functions use module-level names
    (COMPETITION_TYPE, TARGET_SEASONS) as parameter defaults, which are evaluated
    at definition time. Anything whose value is not a literal is skipped, so no
    Path building, no requests session, no scrape.
    """
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"), filename=str(BUILDER))

    keep: list = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Literal-only assignments: names, numbers, strings, lists of those.
            try:
                if node.value is not None:
                    ast.literal_eval(node.value)
                    keep.append(node)
            except (ValueError, TypeError, SyntaxError):
                continue

    ns: dict = {}
    exec(compile(PRELUDE, "<prelude>", "exec"), ns)

    # Node at a time: a definition that depends on a module-level name this script
    # deliberately skipped (a Path, a requests session) would abort the whole exec
    # if compiled as one module. The ranking functions do not need those, so a
    # NameError on an unrelated scraper helper is safe to note and step over.
    skipped = []
    for node in keep:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(BUILDER), "exec"), ns)
        except Exception as exc:
            skipped.append(f"{getattr(node, 'name', type(node).__name__)}: {type(exc).__name__}")

    n_funcs = sum(1 for n in keep if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    print(f"loaded {n_funcs - len(skipped)} of {n_funcs} builder functions"
          + (f" ({len(skipped)} skipped)" if skipped else ""))

    missing = [name for name in REQUIRED if name not in ns]
    if missing:
        raise SystemExit(
            "build_warehouse.py no longer exposes: " + ", ".join(missing)
            + "\nskipped during load: " + "; ".join(skipped[:10]))
    return ns


def build_contexts(con, ns) -> pd.DataFrame:
    """Re-run the ranking contexts exactly as build_warehouse.py assembles them."""
    build = ns["_build_player_ranking_context"]

    def mart(name):
        return con.execute(f"SELECT * FROM marts.{name}").df()

    frames = []

    career = mart("player_career_stats")
    if len(career):
        frames.append(build(career, "Career", "Career", 0))

    last10 = mart("player_last10_stats")
    if len(last10):
        frames.append(build(last10, "Last 10", "Last 10", 1))

    last5 = mart("player_last5_stats")
    if len(last5):
        frames.append(build(last5, "Last 5", "Last 5", 2))

    seasons = mart("player_season_stats")
    if len(seasons) and "season" in seasons.columns:
        years = sorted(pd.to_numeric(seasons["season"], errors="coerce")
                       .dropna().astype(int).unique())
        for i, season in enumerate(years):
            sdf = seasons[pd.to_numeric(seasons["season"], errors="coerce").eq(season)].copy()
            frames.append(build(sdf, "Season", f"{season} Season", 100 + i))

    if not frames:
        raise SystemExit("no ranking contexts could be built — are the player marts present?")
    return pd.concat(frames, ignore_index=True, sort=False)


def report(df: pd.DataFrame, context: str = "2026 Season") -> None:
    """Print the checks that actually catch an inverted or mis-ranked board."""
    ctx = df[df["ranking_context"] == context]
    elig = ctx[pd.to_numeric(ctx["eligible_for_default_ranking"], errors="coerce").eq(1)]
    if not len(elig):
        print(f"no eligible rows in {context}")
        return

    print(f"\n=== {context}: top 15 ===")
    cols = ["overall_rank", "full_name", "position", "role_group", "games",
            "overall_score", "points_per_game"]
    print(elig.nsmallest(15, "overall_rank")[cols].round(2).to_string(index=False))

    print(f"\nrole mix of top 25: "
          f"{elig.nsmallest(25, 'overall_rank')['role_group'].value_counts().to_dict()}")

    off = elig[elig["role_group"] == "Offense"]
    if len(off) > 2:
        r = off["overall_score"].corr(off["points_per_game"])
        print(f"corr(overall_score, points_per_game) for offense: {r:+.3f}  "
              f"(was -0.930 with the inverted percentile)")

    ranks = elig["overall_rank"].dropna().sort_values().tolist()
    contiguous = ranks[:len(ranks)] == list(range(1, len(ranks) + 1))
    print(f"eligible ranks contiguous 1..{len(ranks)}: {contiguous}")
    print(f"score range: {elig['overall_score'].min():.1f} – {elig['overall_score'].max():.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument("--context", default="2026 Season")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"warehouse not found at {DB_PATH}")

    ns = load_builder_functions()
    con = duckdb.connect(str(DB_PATH), read_only=args.dry_run)
    profiles = build_contexts(con, ns)
    print(f"rebuilt player_ranking_profiles: {profiles.shape}")

    report(profiles, args.context)

    if args.dry_run:
        print("\n--dry-run: warehouse not modified")
        return 0

    con.register("_new_profiles", profiles)
    con.execute("CREATE OR REPLACE TABLE marts.player_ranking_profiles AS "
                "SELECT * FROM _new_profiles")
    con.unregister("_new_profiles")

    # Keep the parquet the Streamlit bootstrap reads in step with the table.
    out = ROOT / "data" / "curated_data" / "all_requested_seasons" / "player_ranking_profiles.parquet"
    if out.parent.exists():
        profiles.to_parquet(out, index=False)
        print(f"wrote {out.relative_to(ROOT)}")
    csv_out = out.with_suffix(".csv")
    if csv_out.parent.exists():
        profiles.to_csv(csv_out, index=False)
        print(f"wrote {csv_out.relative_to(ROOT)}")

    con.close()
    print("\nwarehouse updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
