"""
shared/scoring.py — how the composite scores are built, in one place.

The Data Guide used to state the ranking weights as hardcoded prose ("Offense:
62% Base Impact + 20% Role Context + 10% Usage + 8% Goal Value") that matched
neither the Player Rankings page (60/25/15) nor the mart that actually computes
the number. Three descriptions of one formula, two of them wrong.

The weights below mirror `scripts/build_warehouse.py`'s `overall_score_raw`
calculation, which is the only place the score is really decided. Both pages read
them from here, so a page can no longer disagree with the warehouse silently, and
`verify_against_mart()` re-derives the score from the mart's own component columns
to check that this file still matches the build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared import roles

# ============================================================
# OVERALL SCORE
# ============================================================

# Component column names in marts.player_rankings, with the plain-English name
# the pages show. RPS differs per role; PSS and CIS are shared.
RPS_COLUMNS = {
    roles.ROLE_OFFENSE: "offense_rps",
    roles.ROLE_DEFENSE: "defense_rps",
    roles.ROLE_FACEOFF: "faceoff_rps",
    # Goalies enter the cross-role ranking through a compressed version of their
    # RPS, not the raw one — see transfer_note().
    roles.ROLE_GOALIE: "goalie_base_for_overall",
}

PSS_COLUMN = "peer_standing_score"
CIS_COLUMN = "cross_role_impact"
# Goalies' peer standing is compressed on the same principle as their RPS.
PSS_COLUMN_GOALIE = "goalie_role_context_for_overall"

COMPONENTS = {
    "rps": ("Role Performance", "How well the player does their specific job."),
    "pss": ("Peer Standing", "Where they rank among players in the same role."),
    "cis": ("Cross-Role Impact", "Contributions that count regardless of position "
                                 "— ground balls, usage, ball security."),
}

# (rps, pss, cis) weights per role. Mirrors build_warehouse.py's np.select.
OVERALL_WEIGHTS = {
    roles.ROLE_OFFENSE: (0.60, 0.25, 0.15),
    roles.ROLE_DEFENSE: (0.65, 0.25, 0.10),
    roles.ROLE_FACEOFF: (0.65, 0.25, 0.10),
    roles.ROLE_GOALIE: (0.70, 0.25, 0.05),
}

def transfer_note(peer_sizes: dict | None = None, context: str | None = None) -> str:
    """
    Why specialists are compressed before entering the cross-role ranking.

    Takes the peer-pool sizes rather than naming them inline: a previous version
    of this text said "249 offensive players" when the pool was 185, which is
    exactly the kind of drift this module exists to prevent. Pass `{role: size}`
    from `peer_sizes_from_mart` and the numbers are whatever the data says today.
    Pool sizes differ per ranking context, so `context` names the one being quoted.
    """
    sizes = peer_sizes or {}
    goalies = sizes.get(roles.ROLE_GOALIE)
    faceoff = sizes.get(roles.ROLE_FACEOFF)
    offense = sizes.get(roles.ROLE_OFFENSE)

    if goalies and faceoff and offense:
        where = f" in {context}" if context else ""
        scale = (f"{goalies} goalies and {faceoff} faceoff men against "
                 f"{offense} offensive players{where}. Being first of {goalies} is "
                 f"easier than being first of {offense}, so ")
    else:
        scale = ("much smaller than the offensive and defensive groups. Topping a "
                 "small pool is easier than topping a large one, so ")

    return (
        "Goalies and faceoff specialists are ranked inside peer pools that are "
        + scale
        + "their role scores are compressed toward the league average before "
        "entering the all-player ranking. The compression eases as the season "
        "lengthens (from 0.55 toward 0.70 of full value for goalie role "
        "performance at ten games). Their dedicated views use uncompressed scores, "
        "so a goalie's rank among goalies is unaffected."
    )


def peer_sizes_from_mart(df: pd.DataFrame) -> dict:
    """
    Map role -> peer-pool size using the mart's own `role_group_size`.

    Pass a single ranking context: a season's pool is roughly half the career pool,
    so mixing contexts would report a size no context actually used.
    """
    if df is None or len(df) == 0:
        return {}
    if "position" not in df.columns or "role_group_size" not in df.columns:
        return {}
    role = df["position"].map(roles.role_for_position)
    sizes = pd.to_numeric(df["role_group_size"], errors="coerce")
    grouped = sizes.groupby(role).max().dropna()
    return {str(k): int(v) for k, v in grouped.items()}


def peer_sizes_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Peer-pool size per role per ranking context, in ROLE_ORDER."""
    if df is None or len(df) == 0 or "ranking_context" not in df.columns:
        return pd.DataFrame()

    rows = []
    for context, group in df.groupby("ranking_context", sort=False):
        sizes = peer_sizes_from_mart(group)
        row = {"ranking_context": context}
        for role in roles.ROLE_ORDER:
            if role in sizes:
                row[roles.role_label(role)] = sizes[role]
        rows.append(row)
    return pd.DataFrame(rows)


def context_order(df: pd.DataFrame,
                  context_col: str = "ranking_context",
                  type_col: str | None = None,
                  sort_col: str | None = None) -> list:
    """
    Context labels in reading order: Career first, then newest season down.

    Pages 13 and 14 each carried a copy of this under the name
    `_pll_context_order`, differing only in which column prefix they passed
    (`ranking_*` vs `profile_*`), so the prefix is now derived from `context_col`.
    The mart ships a sort column; the fallback parses the label only when it does
    not, which is why "Last 5"/"Last 10" get negative keys — they sort after
    Career but before any real season.
    """
    if df is None or len(df) == 0 or context_col not in df.columns:
        return []

    prefix = context_col.rsplit("_", 1)[0]
    type_col = type_col or f"{prefix}_context_type"
    sort_col = sort_col or f"{prefix}_context_sort"

    work = df.copy()
    if type_col not in work.columns:
        work[type_col] = "Other"
    if sort_col not in work.columns:
        labels = work[context_col].astype(str)
        derived = pd.to_numeric(labels.str.extract(r"(20\d{2})", expand=False),
                                errors="coerce")
        derived = np.where(labels.str.contains("Career", case=False, na=False), 0, derived)
        derived = np.where(labels.str.contains("Last 10", case=False, na=False), -10, derived)
        derived = np.where(labels.str.contains("Last 5", case=False, na=False), -5, derived)
        work[sort_col] = derived

    out = work[[context_col, type_col, sort_col]].drop_duplicates().copy()
    out["_type_order"] = np.where(out[type_col].astype(str).eq("Career"), 0, 1)
    out["_sort"] = pd.to_numeric(out[sort_col], errors="coerce")
    out = out.sort_values(["_type_order", "_sort", context_col],
                          ascending=[True, False, True], na_position="last")
    return out[context_col].tolist()


def default_context(options: list, prefer: str = "Season") -> str | None:
    """
    The context a page should open on — the newest single season.

    Career is first in `context_order` because it reads as the headline, but a
    reader arriving cold wants the current season, not a five-year aggregate.
    """
    if not options:
        return None
    matches = [c for c in options if prefer in str(c)]
    return matches[0] if matches else options[0]


CALIBRATION_NOTE = (
    "After the weighted blend, each ranking context is shifted so its eligible "
    "median sits near 50. That makes 50 mean \"league average in this context\" "
    "across seasons of different lengths. The shift is uniform, so it never "
    "changes the order."
)

# What Role Performance is built from, per role. Sourced from the RPS blocks in
# build_warehouse.py; shown so a reader can tell why a player scores as they do.
RPS_INPUTS = {
    roles.ROLE_OFFENSE: [
        "Points production", "Creation efficiency", "Assist conversion rate",
        "Shot quality", "2PT conversion",
    ],
    roles.ROLE_DEFENSE: ["Caused turnovers", "Ground balls", "Ball security"],
    roles.ROLE_FACEOFF: ["Faceoff win %", "Total wins", "Volume"],
    roles.ROLE_GOALIE: [
        "Clean save rate (skill-based stops)", "Overall save %", "Volume",
        "Goals-against outcomes",
    ],
}

SCORE_TIERS = [
    ("85+", "Elite", "Top ~10% of the role — a clear difference-maker."),
    ("70–84", "High-End", "Top ~25% — reliably above the line."),
    ("55–69", "Solid Starter", "Above average, performing the role well."),
    ("45–54", "Average", "Close to league average for the role."),
    ("Below 45", "Developmental", "Below average, or too few games to tell."),
]

# How Peer Standing is derived, which is not obvious from the weights alone.
PSS_METHOD_NOTE = (
    "Peer Standing is a player's role-performance rank within their role group, "
    "passed through a sigmoid so 50 is the role average and 85+ is roughly the top "
    "10%. The underlying z-score uses the interquartile range rather than the "
    "standard deviation, so one outlier season cannot stretch the scale for "
    "everyone else."
)


def method_markdown(peer_sizes: dict | None = None,
                    context: str | None = None) -> str:
    """
    The full ranking method as markdown, for the pages that explain the score.

    Both the Data Guide and Player Rankings used to carry their own copy of this
    text and they disagreed — the Guide's weights were fiction. Generating it from
    the same constants the score is verified against means there is one description.
    """
    lines = [
        "Each player's **Overall Score** blends three components, each on a 0–100 "
        "scale where 50 is league average for the ranking context.",
        "",
    ]

    rps_name = COMPONENTS["rps"][0]
    lines.append(f"**1. {rps_name}** — {COMPONENTS['rps'][1]}")
    for role in roles.ROLE_ORDER:
        inputs = RPS_INPUTS.get(role)
        if inputs:
            lines.append(f"- *{roles.role_label(role)}:* " + ", ".join(inputs))
    lines.append("")

    for slot in ("pss", "cis"):
        number = 2 if slot == "pss" else 3
        name, blurb = COMPONENTS[slot]
        lines.append(f"**{number}. {name}** — {blurb}")
        if slot == "pss":
            lines.append(f"- {PSS_METHOD_NOTE}")
        lines.append("")

    lines.append("**Weights by role:**")
    for role, (rps, pss, cis) in OVERALL_WEIGHTS.items():
        lines.append(
            f"- *{roles.role_label(role)}:* {rps:.0%} {COMPONENTS['rps'][0]}"
            f" + {pss:.0%} {COMPONENTS['pss'][0]}"
            f" + {cis:.0%} {COMPONENTS['cis'][0]}"
        )
    lines += [
        "",
        f"**Specialist compression.** {transfer_note(peer_sizes, context)}",
        "",
        f"**Scale calibration.** {CALIBRATION_NOTE}",
    ]
    return "\n".join(lines)


def tiers_frame() -> pd.DataFrame:
    """Score tiers as a display table."""
    return pd.DataFrame(SCORE_TIERS, columns=["Score", "Tier", "Meaning"])


# ============================================================
# TEAM STYLE SCORES
# ============================================================
#
# Mirrors the team-style block in build_warehouse.py. Each component score is a
# weighted blend of min-max-scaled per-game rates; the overall style score is then
# a weighted blend of the six. Listed here for the same reason as the ranking
# weights: so the page describing them cannot drift from the build.

STYLE_SCORE_COLUMN = "team_style_overall_score"

# score column -> (input metric key, weight, higher_is_better)
STYLE_INPUTS = {
    "offensive_volume_score": [
        ("scores_per_game", 0.35, True),
        ("shots_per_game", 0.25, True),
        ("touches_per_game", 0.20, True),
        ("offensive_sequence_proxy_per_game", 0.20, True),
    ],
    "offensive_efficiency_score": [
        ("scores_per_game", 0.45, True),
        ("shot_pct_calc", 0.25, True),
        ("turnovers_per_game", 0.20, False),
        ("score_margin_per_game", 0.10, True),
    ],
    "ball_movement_score": [
        ("assists_per_game", 0.55, True),
        ("total_passes_per_game", 0.25, True),
        ("touches_per_game", 0.20, True),
    ],
    "possession_control_score": [
        ("touches_per_game", 0.45, True),
        ("time_in_possession_per_game", 0.35, True),
        ("faceoff_pct_calc", 0.20, True),
    ],
    "defensive_suppression_score": [
        ("scores_allowed_per_game", 0.40, False),
        ("opponent_shots_per_game", 0.25, False),
        ("opponent_goal_pct", 0.20, False),
        ("save_pct_proxy", 0.15, True),
    ],
    "pace_tempo_score": [
        ("shots_per_game", 0.35, True),
        ("touches_per_game", 0.30, True),
        ("offensive_sequence_proxy_per_game", 0.20, True),
        ("time_in_possession_per_game", 0.15, True),
    ],
}

# Weight each component carries in team_style_overall_score.
STYLE_OVERALL_WEIGHTS = {
    "offensive_volume_score": 0.22,
    "offensive_efficiency_score": 0.20,
    "ball_movement_score": 0.16,
    "possession_control_score": 0.18,
    "defensive_suppression_score": 0.18,
    "pace_tempo_score": 0.06,
}

STYLE_SCALING_NOTE = (
    "Every style input is min-max scaled inside its own context before blending, so "
    "0 is the worst team in that context and 100 the best. That makes the scores a "
    "comparison between the teams shown and nothing else — a 90 in a season where "
    "every team shot well is not a 90 in a season where nobody did, and the scores "
    "should not be read across seasons."
)

# Shown on both the Team Styles page and the Data Guide, so it lives here rather
# than in either of them.
STYLE_QUALITY_NOTE = (
    "A high Pace score means a team plays fast, not that it plays well — it carries "
    "no direction, which is why the app never colour-scales it. For quality, read "
    "the pace-adjusted efficiency columns on the League Overview."
)

# Score bands that produce the *_label text columns (build_warehouse._label_from_score).
STYLE_LABEL_BANDS = [
    ("80–100", "Strongest band"),
    ("65–79", "Above average"),
    ("45–64", "Middle tier"),
    ("30–44", "Below average"),
    ("Below 30", "Weakest band"),
]


def style_weights_frame() -> pd.DataFrame:
    """The team-style weights as a display table, one row per component score."""
    from shared import metrics as M

    rows = []
    for column, weight in STYLE_OVERALL_WEIGHTS.items():
        inputs = ", ".join(
            f"{M.label(key)} {pct:.0%}" + ("" if higher else " (lower is better)")
            for key, pct, higher in STYLE_INPUTS.get(column, [])
        )
        rows.append({
            "style_score": M.label(column),
            "weight_in_overall": weight,
            "built_from": inputs,
            "definition": M.definition(column),
        })
    return pd.DataFrame(rows)


def verify_style_overall(df: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """
    Re-blend the six component scores and compare against the mart's overall style.

    Unlike the player score there is no post-hoc calibration shift here, so these
    should agree to rounding.
    """
    if df is None or len(df) == 0 or STYLE_SCORE_COLUMN not in df.columns:
        return {"checked": 0, "matches": False, "max_diff": float("nan")}

    rebuilt = pd.Series(0.0, index=df.index)
    for column, weight in STYLE_OVERALL_WEIGHTS.items():
        if column not in df.columns:
            return {"checked": 0, "matches": False, "max_diff": float("nan")}
        rebuilt += weight * pd.to_numeric(df[column], errors="coerce")

    actual = pd.to_numeric(df[STYLE_SCORE_COLUMN], errors="coerce")
    diff = (actual - rebuilt.clip(0, 100)).abs().dropna()
    if len(diff) == 0:
        return {"checked": 0, "matches": False, "max_diff": float("nan")}

    worst = float(diff.max())
    return {"checked": int(len(diff)), "matches": worst <= tolerance,
            "max_diff": worst}


# ============================================================
# DISPLAY HELPERS
# ============================================================

def weights_frame() -> pd.DataFrame:
    """The overall-score weights as a display table."""
    rows = []
    for role, (rps, pss, cis) in OVERALL_WEIGHTS.items():
        rows.append({
            "role_group": roles.role_label(role),
            "role_performance": rps,
            "peer_standing": pss,
            "cross_role_impact": cis,
            "role_performance_inputs": ", ".join(RPS_INPUTS.get(role, [])),
        })
    return pd.DataFrame(rows)


def rebuild_overall_score(df: pd.DataFrame) -> pd.Series:
    """
    Re-derive the pre-calibration overall score from the mart's component columns.

    Used by `verify_against_mart` and by the Data Guide's own check, so the weights
    in this file are testable rather than merely asserted.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype="float64")

    # Role comes from `position` via the shared taxonomy, not from the mart's own
    # `role_group`, so this check exercises the same classification the pages use.
    if "position" in df.columns:
        role = df["position"].map(roles.role_for_position)
    else:
        role = pd.Series(roles.ROLE_UNKNOWN, index=df.index)

    def numeric(col):
        if col not in df.columns:
            return pd.Series(50.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(50.0)

    cis = numeric(CIS_COLUMN)
    out = pd.Series(np.nan, index=df.index)
    for role_key, (w_rps, w_pss, w_cis) in OVERALL_WEIGHTS.items():
        mask = role == role_key
        if not mask.any():
            continue
        rps = numeric(RPS_COLUMNS[role_key])
        pss = numeric(PSS_COLUMN_GOALIE if role_key == roles.ROLE_GOALIE else PSS_COLUMN)
        out.loc[mask] = (w_rps * rps + w_pss * pss + w_cis * cis).loc[mask]
    return out.clip(0, 100)


def verify_against_mart(df: pd.DataFrame, tolerance: float = 0.5) -> dict:
    """
    Compare the rebuilt score against the mart's own `overall_score`.

    The mart applies a uniform context shift after the blend, so the two differ by
    a constant per context rather than matching exactly. This reports the spread of
    that difference: a tight spread means the weights here still describe the
    build, a wide one means they have drifted.
    """
    if df is None or len(df) == 0 or "overall_score" not in df.columns:
        return {"checked": 0, "matches": False, "spread": float("nan")}

    rebuilt = rebuild_overall_score(df)
    actual = pd.to_numeric(df["overall_score"], errors="coerce")
    delta = (actual - rebuilt).dropna()
    if len(delta) == 0:
        return {"checked": 0, "matches": False, "spread": float("nan")}

    spread = float(delta.max() - delta.min())
    return {
        "checked": int(len(delta)),
        "matches": spread <= tolerance,
        "spread": spread,
        "median_shift": float(delta.median()),
    }
