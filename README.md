# PLL GitHub Warehouse Builder Files

Copy these files into the root of the GitHub repository:

- `scripts/build_warehouse.py`
- `requirements.txt`
- `runtime.txt`
- `.github/workflows/update-data.yml`
- `.gitignore` optional

The builder writes the app-ready DuckDB database to:

`data/analytics_database/pll_warehouse.duckdb`

and writes exported CSV/Parquet artifacts to:

`data/curated_data/all_requested_seasons/`

This matches the GitHub-ready `app.py`, which expects the warehouse inside the repository `data/` folder.

Required GitHub secret:

`PLL_BEARER_TOKEN`

Manual workflow run:

GitHub → Actions → Update PLL Warehouse → Run workflow

Scheduled workflow runs:

- Monday 05:00 UTC, intended to represent Sunday midnight EST
- Friday 13:00 UTC, intended to represent Friday 8 AM EST

The workflow file in this repository is `.github/workflows/update-pll-data.yml`
and the workflow is named **Update PLL Data Warehouse**. It commits the rebuilt
`data/**` back to the branch, so the app picks up a new build by pulling.

## Regular season, playoffs, or both

The builder ingests postseason games alongside the regular season and writes
three copies of every game-grain table and every mart derived from one:

| Table | Games it counts |
| --- | --- |
| `marts.player_season_stats` | regular season only — unchanged meaning |
| `marts.player_season_stats_all` | regular season + playoffs |
| `marts.player_season_stats_playoffs` | playoffs only |

Champions Series and the All-Star Game are excluded. They are a different
format, and the All-Star Game sits mid-season, where including it would renumber
every game after it.

Separate tables rather than a `competition_type` filter on one table: a query
that forgets the filter then reads the wrong *scope* — stale but internally
consistent — instead of triple-counting rows into a number that is simply wrong.

The app never names those physical tables. Pages still say
`FROM marts.player_season_stats`, and `shared/db.query_df` rewrites the name
through `shared/segments.py` for the scope selected in the sidebar
(**Games included**). The rewrite happens before the cache decorator, so the
scope is part of the cache key and no call site can be missed.

Two deliberate fallbacks on a partially built warehouse:

- `all` with no `_all` table reads the regular table — before the postseason
  ingest ran, "regular + playoffs" *is* the regular season.
- `playoffs` with no `_playoffs` table returns no rows, not the regular table.
  Falling back would put regular-season numbers under a playoffs heading.

The Schedule, Data Guide and Data QA pages opt out (`scope=False` in
`init_page`); they cover every segment on their own terms.

## Checks

None of these need the API token, so all four run locally against the committed
warehouse:

| Command | What it proves |
| --- | --- |
| `python scripts/check_mart_functions.py` | every mart builder is called with the arguments it declares, and the refactor changed no number |
| `python scripts/check_segment_scopes.py` | the resolver picks the right table per scope, and in a rebuilt warehouse regular + playoffs reconciles with `all` |
| `python scripts/smoke_pages.py` | `app.py` and the analytics pages run top to bottom without an exception |
| `python scripts/smoke_pages.py scope=every` | the same, under each scope — including the empty-result path, which must render an empty state |

Run `check_segment_scopes.py` again after the first rebuild that writes segment
variants: until then it reports that there is nothing to reconcile, which is a
build that has not run rather than a failure.
