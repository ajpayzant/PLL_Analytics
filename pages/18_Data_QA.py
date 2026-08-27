"""
Data QA — is the warehouse complete, and can the numbers be trusted?

This is the only page whose subject is the data rather than the lacrosse. The
warehouse row counts that used to open the Overview page live here now: "6,756
player-game rows" answers a data-engineering question, and putting it above the
scoring leaders implied the reader should care about it before the analysis.

Coverage by season is the part that actually changes how you read other pages, so
it comes first, ahead of the check tables.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from shared import metrics as M
from shared import page as P
from shared import segments
from shared import ui
from shared.db import (ARTIFACT_INDEX_PATH, DB_PATH, _pll_get_table_columns,
                       query_df, schedule_display_table, startup_counts,
                       table_exists, table_index)

ctx = P.init_page(
    "Data QA",
    "Warehouse coverage, validation checks and artifact inventory.",
    # This page covers every segment on its own terms, so the sidebar's
    # regular/playoffs selector would be decorative here.
    scope=False,
)

# ============================================================
# QUALITY HEADLINE
# ============================================================

quality = query_df("""
    SELECT *
    FROM qc.quality_summary
    ORDER BY
        CASE status
            WHEN 'fail' THEN 1
            WHEN 'warning' THEN 2
            WHEN 'pass' THEN 3
            ELSE 4
        END,
        check_name
""")

statuses = quality["status"] if "status" in quality.columns else pd.Series(dtype=str)
fail_count = int((statuses == "fail").sum())
warning_count = int((statuses == "warning").sum())
pass_count = int((statuses == "pass").sum())

k = st.columns(4)
with k[0]:
    ui.stat_card("Failures", f"{fail_count:,}",
                 tone="bad" if fail_count else "good",
                 sub="needs attention" if fail_count else "none")
with k[1]:
    ui.stat_card("Warnings", f"{warning_count:,}",
                 sub="review before trusting affected pages" if warning_count else "none")
with k[2]:
    ui.stat_card("Passes", f"{pass_count:,}")
with k[3]:
    ui.stat_card("Total Checks", f"{len(quality):,}")

if fail_count:
    ui.note_box(
        "Failing checks present",
        "Numbers on the analysis pages that depend on a failing check may be "
        "wrong. The Quality Checks table below names the affected area.",
    )

# ============================================================
# WAREHOUSE CONTENTS
# ============================================================

ui.section(
    "Warehouse contents",
    "Row counts for the tables every other page reads. These moved here from the "
    "Overview page, which now leads with analysis instead.",
)

counts = startup_counts()
cards = st.columns(5)
# `players`/`teams` are spelled out rather than passed through the registry: those
# keys name a text column elsewhere in the warehouse (a comma-joined team list),
# and one key can't mean two things.
for col, (key, label) in zip(cards, [
    ("completed_games", None),
    ("player_game_rows", None),
    ("team_game_rows", None),
    ("players", "Players"),
    ("teams", "Teams"),
]):
    with col:
        ui.stat_card(label or M.label(key), f"{int(counts.get(key, 0)):,}")
ui.definition_caption(["completed_games", "player_game_rows", "team_game_rows"])

# Season coverage is the readout that changes how the rest of the app should be
# read: a season with few stat-available games gives thin per-game averages.
schedule = schedule_display_table()

# The widest manifest the build produced, because the schedule it is compared
# against holds every fixture — counting regular-season stat games against a
# playoff-inclusive schedule would read as missing data.
manifest_all = (segments.resolve_table("clean", "game_manifest", segments.ALL)
                or "clean.game_manifest")

stat_games = query_df(f"""
    SELECT season, COUNT(DISTINCT game_id) AS stat_available_games
    FROM {manifest_all}
    GROUP BY season
    ORDER BY season
""", scoped=False)

scheduled = (schedule.groupby("season", dropna=False).size()
             .reset_index(name="scheduled_games"))
coverage = scheduled.merge(stat_games, on="season", how="outer").sort_values("season")
coverage["stat_available_games"] = coverage["stat_available_games"].fillna(0)
coverage["coverage_pct"] = (
    pd.to_numeric(coverage["stat_available_games"], errors="coerce")
    / pd.to_numeric(coverage["scheduled_games"], errors="coerce").replace(0, pd.NA)
)

left, right = st.columns([1.0, 1.2])
with left:
    st.markdown("**Stat coverage by season**")
    ui.display_table(coverage, height=300, highlight="coverage_pct")
with right:
    st.markdown("**Schedule status by season**")
    status_counts = (
        schedule.groupby(["season", "status_display"], dropna=False)
        .size().reset_index(name="games")
        .sort_values(["season", "status_display"])
    )
    ui.safe_bar_chart(
        status_counts,
        x_col="season",
        y_col="games",
        color_col="status_display",
        title="Scheduled games by status",
    )

# ============================================================
# REGULAR SEASON AND PLAYOFFS
# ============================================================
#
# The warehouse keeps three copies of every game-grain table and every mart built
# from one: regular season (the unsuffixed name, unchanged), `_all` and
# `_playoffs`. This is where to look when a scope in the sidebar shows fewer
# numbers than expected — a variant that was not built is a variant the app
# cannot serve.

ui.section(
    "Regular season and playoffs",
    "What the postseason ingest added, and which scoped tables this build wrote.",
)

manifest_cols = _pll_get_table_columns(*manifest_all.split(".", 1))

if "competition_type" not in manifest_cols:
    st.info(
        "This warehouse build predates the postseason ingest: every game in it is "
        "a regular-season game. Run the **Update PLL Data Warehouse** workflow to "
        "add playoff games and the scoped tables that go with them."
    )
else:
    by_segment = query_df(f"""
        SELECT season,
               competition_type,
               COUNT(DISTINCT game_id) AS games_with_stats
        FROM {manifest_all}
        GROUP BY season, competition_type
        ORDER BY season, competition_type
    """, scoped=False)

    seg_left, seg_right = st.columns([1.0, 1.0])
    with seg_left:
        st.markdown("**Games with stats by segment**")
        ui.display_table(by_segment, height=280)
    with seg_right:
        st.markdown("**Scoped tables built**")
        expected = sum(len(names) for names in segments.SCOPED_TABLES.values())
        rows = []
        for scope in (segments.ALL, segments.PLAYOFFS):
            built = sum(
                1
                for schema, names in segments.SCOPED_TABLES.items()
                for table in names
                if segments.variant_exists(schema, table, scope)
            )
            rows.append({
                "scope": segments.SCOPE_LABEL[scope],
                "suffix": segments.SCOPE_SUFFIX[scope],
                "tables_built": built,
                "tables_expected": expected,
            })
        ui.display_table(pd.DataFrame(rows), height=140)
        st.caption(
            "A shortfall is not always a fault: a table with no playoff rows at "
            "all is not written, and the app shows no rows for it rather than "
            "regular-season rows under a playoff heading."
        )

# The in-progress season is always short of its scheduled games, so flagging it
# would make this warning permanent noise. Only completed seasons are a problem.
current = max(ctx.seasons) if ctx.seasons else None
thin = coverage[
    (pd.to_numeric(coverage["coverage_pct"], errors="coerce") < 0.9)
    & (coverage["season"] != current)
]
if len(thin):
    seasons_text = ", ".join(str(s) for s in thin["season"].tolist())
    st.warning(
        f"Under 90% stat coverage in {seasons_text}, which is finished. Season "
        "totals there are built on a partial set of games, so compare rates "
        "rather than totals."
    )
if current is not None:
    st.caption(
        f"{current} is in progress, so its coverage is expected to be below 100%."
    )

# ============================================================
# CHECKS
# ============================================================

ui.section("Quality checks", "Automated validation run when the warehouse is built.")
ui.display_table(quality, height=460)

with st.expander("2023 schedule status repair check", expanded=False):
    st.caption(
        "The 2023 feed reported statuses the rest of the app can't use directly. "
        "This maps the raw label to the repaired one, so the repair can be audited."
    )
    repair = schedule_display_table()
    repair = (
        repair[repair["season"] == 2023]
        .groupby(["event_status_label", "status_display"])
        .size().reset_index(name="games")
    )
    ui.display_table(repair, height=200)

with st.expander("Defensive / opponent build QC", expanded=False):
    if table_exists("qc", "defensive_opponent_build_quality"):
        ui.display_table(
            query_df("""
                SELECT * FROM qc.defensive_opponent_build_quality
                ORDER BY status, check_name
            """),
            height=320,
        )
    else:
        st.info("No defensive/opponent QC table in this warehouse build.")

with st.expander("Possession data QC", expanded=False):
    st.caption(
        "Possession time is provider-tracked and imperfect. Every per-100-possession "
        "figure in the app rests on these rows, so anything other than `normal` is "
        "worth knowing about."
    )
    if table_exists("qc", "game_possession_quality"):
        possession = query_df("""
            SELECT *
            FROM qc.game_possession_quality
            ORDER BY
                CASE possession_data_status
                    WHEN 'normal' THEN 4
                    WHEN 'extended_or_ot_clock' THEN 3
                    WHEN 'short_or_provider_clock' THEN 2
                    WHEN 'missing_possession_time' THEN 1
                    ELSE 0
                END,
                season,
                game_number
        """)
        if "possession_data_status" in possession.columns:
            summary = (possession["possession_data_status"].value_counts()
                       .rename_axis("possession_data_status")
                       .reset_index(name="games"))
            ui.display_table(summary, height=180)
        # Game-level rows read from qc.*, where the pct columns are 0–1.
        ui.display_table(possession, height=360, clean_schema=True)
    else:
        st.info("No game possession quality table in this warehouse build.")

with st.expander("Warehouse tables", expanded=False):
    ui.display_table(table_index(), height=460)

with st.expander("Artifact index", expanded=False):
    if os.path.exists(ARTIFACT_INDEX_PATH):
        ui.display_table(pd.read_csv(ARTIFACT_INDEX_PATH), height=460)
    else:
        st.caption(
            "`artifact_index.csv` is not present. It is written by the warehouse "
            "build and is not required for the app to run."
        )

st.caption(f"Warehouse: `{DB_PATH}`")

st.divider()
nav = st.columns(3)
with nav[0]:
    P.link_to("league", "League overview →")
with nav[1]:
    P.link_to("guide", "Data guide →")
with nav[2]:
    P.link_to("schedule", "Schedule →")
