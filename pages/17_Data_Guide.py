"""
Data Guide — what each stat means, how each score is built, and what not to trust.

The previous version was three hand-maintained tables of prose. Its ranking
weights ("Offense: 62% Base Impact + 20% Role Context + 10% Usage + 8% Goal
Value") matched neither the Player Rankings page nor the warehouse that computes
the number, and its metric definitions were a 30-row subset while the registry
carried definitions for far more.

So this page no longer stores its own copy of anything. Definitions come from
`shared/metrics.py` — the same source the tables and charts format from, meaning a
definition here is the definition in use. Score weights come from
`shared/scoring.py`, and the check at the bottom of the Rankings tab re-derives
the score from the mart's component columns to prove the two still agree.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shared import metrics as M
from shared import page as P
from shared import roles
from shared import scoring
from shared import ui
from shared.db import query_df, table_exists

ctx = P.init_page(
    "Data Guide",
    "Definitions, formulas and the data's known limits — read from the same "
    "registry the rest of the app formats with.",
)

# One representative value per unit, so the glossary can show how a metric renders
# rather than naming an internal format code the reader has no way to interpret.
EXAMPLE_VALUES = {
    M.UNIT_INT: 1234,
    M.UNIT_NUM1: 12.34,
    M.UNIT_NUM2: 12.3,
    M.UNIT_AUTO: 12.34,
    M.UNIT_PCT01: 0.283,
    M.UNIT_PCT100: 28.3,
    M.UNIT_SCORE: 84.2,
    M.UNIT_SEC: 1275,
    M.UNIT_SEC_TOTAL: 4830,
    M.UNIT_TEXT: "text",
}

tab_glossary, tab_rankings, tab_styles, tab_limits = st.tabs(
    ["Glossary", "Player Rankings", "Team Styles", "Known limits"]
)

# ============================================================
# GLOSSARY
# ============================================================

with tab_glossary:
    st.markdown(
        "Every metric the app knows about, grouped by what it measures. "
        "The direction column says which way is good, which is what drives sorting "
        "and the colour scales elsewhere."
    )

    defined = [k for k, m in M.METRICS.items() if m.definition]
    grouped = M.by_family(defined)

    search = st.text_input(
        "Filter", placeholder="save, faceoff, possession…", key="guide_search",
    ).strip().lower()

    family_options = [M.FAMILY_LABELS.get(f, f.title()) for f in grouped]
    chosen = st.multiselect(
        "Families", options=family_options, default=[], key="guide_families",
        help="Leave empty to show every family.",
    )

    shown_any = False
    for fam, keys in grouped.items():
        label = M.FAMILY_LABELS.get(fam, fam.title())
        if chosen and label not in chosen:
            continue
        rows = []
        for key in keys:
            metric = M.METRICS[key]
            haystack = f"{key} {metric.label} {metric.definition}".lower()
            if search and search not in haystack:
                continue
            rows.append({
                "metric": metric.label,
                "column": key,
                "definition": metric.definition,
                "direction": M.direction_note(key),
                "format": M.format_as(metric.unit,
                                      EXAMPLE_VALUES.get(metric.unit, 12.34)),
            })
        if not rows:
            continue
        shown_any = True
        ui.section(label)
        st.dataframe(
            pd.DataFrame(rows).rename(columns={
                "metric": "Metric", "column": "Warehouse Column",
                "definition": "Definition", "direction": "Direction",
                "format": "Shown As",
            }),
            width="stretch", hide_index=True,
            height=min(520, 40 + 35 * len(rows)),
        )

    if not shown_any:
        st.info("No metrics match that filter.")

    undefined = [k for k, m in M.METRICS.items() if not m.definition]
    st.caption(
        f"{len(defined):,} of {len(M.METRICS):,} registered metrics carry a written "
        f"definition. The remaining {len(undefined):,} are self-explanatory counts "
        "and identity columns (team name, games, goals) that are labelled and "
        "formatted but not described."
    )

# ============================================================
# PLAYER RANKINGS
# ============================================================

with tab_rankings:
    # Loaded once and reused: the compression note quotes the mart's own peer-pool
    # sizes, and the check at the bottom re-derives the score from the same rows.
    profiles = (query_df("SELECT * FROM marts.player_ranking_profiles")
                if table_exists("marts", "player_ranking_profiles")
                else pd.DataFrame())

    st.markdown(
        "**Overall Score** blends three components. Each is on a 0–100 scale where "
        "50 is league average for the ranking context."
    )

    for key, (name, blurb) in scoring.COMPONENTS.items():
        st.markdown(f"**{name}** — {blurb}")

    ui.section("Weights by role",
               "The weighted blend the warehouse applies. Defenders and specialists "
               "lean harder on their role score because there is less cross-role "
               "production to measure them by.")

    weights = scoring.weights_frame()
    st.dataframe(
        weights.rename(columns={
            "role_group": "Role",
            "role_performance": "Role Performance",
            "peer_standing": "Peer Standing",
            "cross_role_impact": "Cross-Role Impact",
            "role_performance_inputs": "Role Performance is built from",
        }).style.format({
            "Role Performance": "{:.0%}",
            "Peer Standing": "{:.0%}",
            "Cross-Role Impact": "{:.0%}",
        }),
        width="stretch", hide_index=True, height=190,
    )

    # Career is the widest pool, and the one the note quotes. Pools halve in a
    # single-season context, which the limits tab shows in full.
    career = (profiles[profiles["ranking_context"] == "Career"]
              if "ranking_context" in profiles.columns else profiles)
    ui.note_box("Comparing roles", scoring.RPS_NORMALIZATION_NOTE)
    ui.note_box(
        "Specialist compression",
        scoring.transfer_note(scoring.peer_sizes_from_mart(career), "the career view"),
    )
    ui.note_box("Scale calibration", scoring.CALIBRATION_NOTE)

    ui.section("Score tiers", "How to read a score at a glance.")
    st.dataframe(
        pd.DataFrame(scoring.SCORE_TIERS, columns=["Score", "Tier", "Meaning"]),
        width="stretch", hide_index=True, height=220,
    )

    # The weights above are asserted by this page; this check proves them.
    ui.section("Formula check",
               "The weights on this page are re-applied to the mart's own component "
               "columns and compared against its published score. Agreement means "
               "this page still describes what the warehouse actually does.")

    if len(profiles):
        results = []
        contexts = (profiles["ranking_context"].dropna().unique()
                    if "ranking_context" in profiles.columns else [None])
        for context in contexts:
            subset = (profiles if context is None
                      else profiles[profiles["ranking_context"] == context])
            check = scoring.verify_against_mart(subset)
            results.append({
                "context": context or "All",
                "players_checked": check["checked"],
                "agrees": "yes" if check["matches"] else "no",
                "spread": check["spread"],
                "calibration_shift": check.get("median_shift"),
                "clipped": check.get("clipped", 0),
            })
        frame = pd.DataFrame(results)
        st.dataframe(
            frame.rename(columns={
                "context": "Ranking Context", "players_checked": "Players Checked",
                "agrees": "Agrees", "spread": "Spread",
                "calibration_shift": "Calibration Shift",
                "clipped": "Excluded (at 0/100)",
            }).style.format({"Spread": "{:.4f}", "Calibration Shift": "{:+.3f}"},
                            na_rep="—"),
            width="stretch", hide_index=True, height=180,
        )
        st.caption(
            "Spread is the range of the difference between the rebuilt and published "
            "scores within a context. It is not zero because the warehouse shifts "
            "each context's median toward 50 after blending — a constant offset, "
            "shown in the calibration column. A spread near zero means only that "
            "constant separates them. The last column counts players whose score the "
            "warehouse clipped to 0 or 100; for them the difference is the clip "
            "rather than the shift, so they sit outside the comparison."
        )
        if any(r["agrees"] == "no" for r in results):
            st.error(
                "The weights on this page no longer reproduce the mart's score. "
                "`shared/scoring.py` needs updating against "
                "`scripts/build_warehouse.py`."
            )
    else:
        st.info("The ranking mart is not present in this warehouse build.")

# ============================================================
# TEAM STYLES
# ============================================================

with tab_styles:
    st.markdown(
        "Team style scores describe **how** a team plays rather than how well. "
        "There are six, each a 0–100 blend of per-game rates, and one overall score "
        "that blends the six."
    )

    ui.section("What each score measures",
               "And what it is built from, with each input's weight inside that "
               "score. The first column's weight is the score's share of Overall "
               "Style.")

    style_weights = scoring.style_weights_frame()
    st.dataframe(
        style_weights.rename(columns={
            "style_score": "Style Score",
            "weight_in_overall": "Share of Overall",
            "built_from": "Built From",
            "definition": "Measures",
        }).style.format({"Share of Overall": "{:.0%}"}),
        width="stretch", hide_index=True, height=250,
    )

    ui.note_box("Style is not quality", scoring.STYLE_QUALITY_NOTE)
    ui.note_box("Scores are relative to the teams shown", scoring.STYLE_SCALING_NOTE)

    ui.section("Label bands",
               "The text profiles (Pace, Offensive Profile, Defensive Profile, "
               "Possession Profile) are these bands applied to the matching score.")
    st.dataframe(
        pd.DataFrame(scoring.STYLE_LABEL_BANDS, columns=["Score", "Band"]),
        width="stretch", hide_index=True, height=220,
    )

    ui.section("Formula check",
               "The shares above are re-applied to the mart's component scores and "
               "compared against its published Overall Style.")
    if table_exists("marts", "team_style_profiles"):
        style_check = scoring.verify_style_overall(
            query_df("SELECT * FROM marts.team_style_profiles"))
        if style_check["checked"]:
            st.caption(
                f"{style_check['checked']:,} team-context rows checked — "
                f"largest difference {style_check['max_diff']:.4f}."
            )
            if not style_check["matches"]:
                st.error(
                    "The style weights on this page no longer reproduce the mart's "
                    "Overall Style. `shared/scoring.py` needs updating against "
                    "`scripts/build_warehouse.py`."
                )
        else:
            st.info("The style mart does not carry the component score columns.")
    else:
        st.info("The team style mart is not present in this warehouse build.")

# ============================================================
# KNOWN LIMITS
# ============================================================

with tab_limits:
    st.markdown(
        "What this data cannot tell you. Each of these affects how a number on "
        "another page should be read."
    )

    current = max(ctx.seasons) if ctx.seasons else None

    ui.note_box(
        "Completed games versus scheduled games",
        "Every average and total in the app is built from games whose stats have "
        "landed in the warehouse. A game that has been played but not yet scraped "
        "is still flagged `scheduled` and contributes nothing — see the Schedule "
        "page, which separates the two, and Data QA for coverage by season.",
    )

    if current is not None:
        ui.note_box(
            f"{current} is in progress",
            "Early-season ranks, trends and composite scores rest on few games. The "
            "ranking scores compress specialists harder in small samples for exactly "
            "this reason, but a ten-game sample is still a ten-game sample.",
        )

    ui.note_box(
        "Possession data is provider-tracked",
        "Possession time and the offensive-sequence proxy come from the provider "
        "feed and are not consistent across every historical game. Everything "
        "per-100-possessions depends on them, so check coverage on Data QA before "
        "leaning on a pace-adjusted figure.",
    )

    ui.note_box(
        "Offensive sequences are a proxy, not a possession count",
        "The league does not publish a possession count the way basketball does. "
        "`offensive_sequence_proxy` is estimated, so per-100 rates are good for "
        "comparing teams within a season and poor for precise claims about a rate's "
        "absolute level.",
    )

    ui.note_box(
        "Scores and goals are different things",
        "A two-point goal is one goal and two scores. Any comparison of scoring "
        "needs to pick one and stay with it — the registry's labels say which is "
        "which, and the Glossary tab spells out both.",
    )

    ui.note_box(
        "Composite scores are opinions",
        "Overall Score, the role scores and the style scores are weighted blends "
        "chosen by this project, not league-official figures. The Player Rankings "
        "tab shows every weight so the opinion is inspectable and arguable.",
    )

    ui.note_box(
        "A rank is only as wide as its pool",
        "Specialist ranks and all-player ranks answer different questions. The "
        "pools below are what a player is actually compared against in each "
        "ranking context — a season pool is roughly half a career pool, so "
        "\"third among goalies\" means less in one than the other.",
    )

    pools = scoring.peer_sizes_frame(profiles)
    if len(pools):
        ui.section("Peer-pool sizes by ranking context")
        ui.display_table(
            pools.rename(columns={"ranking_context": "Ranking Context"}), height=280)
    else:
        counted = roles.role_counts(ctx.players)
        if len(counted):
            ui.section("Players by role")
            ui.display_table(counted, height=200)

st.divider()
nav = st.columns(3)
with nav[0]:
    P.link_to("rankings", "Player rankings →")
with nav[1]:
    P.link_to("styles", "Team styles →")
with nav[2]:
    P.link_to("qa", "Data QA →")
