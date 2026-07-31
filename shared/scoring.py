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
#
# The Overall blend reads `role_primary_score_normalized` for every role, not the
# per-role `offense_rps` / `defense_rps` / `faceoff_rps` columns. Those raw columns
# are on incomparable scales — a role's composite ceiling is set by how correlated
# its inputs are, so faceoff RPS peaked at 98.9 against offence's 91.9 for reasons
# that have nothing to do with the players. The normalized column puts every role
# on one median/IQR scale for the cross-role comparison; the raw columns still
# drive the role-specific views, where the raw scale is the meaningful one.
RPS_COLUMNS = {
    roles.ROLE_OFFENSE: "role_primary_score_normalized",
    roles.ROLE_DEFENSE: "role_primary_score_normalized",
    roles.ROLE_FACEOFF: "role_primary_score_normalized",
    # Goalies enter the cross-role ranking through a compressed version of the
    # normalized RPS, not the raw one — see transfer_note().
    roles.ROLE_GOALIE: "goalie_base_for_overall",
}

# The raw per-role RPS columns, for the views that rank inside a single role.
RPS_COLUMNS_ROLE_VIEW = {
    roles.ROLE_OFFENSE: "offense_rps",
    roles.ROLE_DEFENSE: "defense_rps",
    roles.ROLE_FACEOFF: "faceoff_rps",
    roles.ROLE_GOALIE: "goalie_rps",
}

PSS_COLUMN = "peer_standing_score"
CIS_COLUMN = "cross_role_impact"
# Goalies' peer standing is compressed on the same principle as their RPS.
PSS_COLUMN_GOALIE = "goalie_role_context_for_overall"

COMPONENTS = {
    "rps": ("Role Performance", "How well the player does their specific job."),
    "cis": ("Cross-Role Impact", "Contributions that count regardless of position "
                                 "— possession volume, ground balls, passing, ball "
                                 "security — measured against the player's own role, "
                                 "so it adds to the role score instead of restating "
                                 "the position."),
}

# (rps, cis) weights per role. Mirrors build_warehouse.py's np.select.
#
# Peer Standing used to be the second component at a flat 25%. It was dropped
# because it was not independent information: it is Role Performance's own rank,
# passed through a sigmoid, so it correlated with Role Performance at 0.99 and the
# blend was in effect 85-90% one number wearing two hats. Removing it improved the
# board against the only external check available — team-mean player score against
# team wins rose from +0.577 to +0.625, and against score margin from +0.678 to
# +0.743 over 40 team-seasons. `peer_standing_score` is still published and still
# shown on the player pages, as a readable "where do they rank in their role"
# figure; it no longer feeds the Overall Score.
OVERALL_WEIGHTS = {
    roles.ROLE_OFFENSE: (0.85, 0.15),
    roles.ROLE_DEFENSE: (0.90, 0.10),
    roles.ROLE_FACEOFF: (0.90, 0.10),
    roles.ROLE_GOALIE: (0.95, 0.05),
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
        "lengthens (from 0.62 toward 0.88 of full value for goalie role "
        "performance at ten games). Their dedicated views use uncompressed scores, "
        "so a goalie's rank among goalies is unaffected. Goalies were previously "
        "compressed hard enough that the best goalie in the league could not place "
        "higher than 43rd overall — an artefact of the compression, not a judgement "
        "about goalkeeping, since goalie play tracks team wins about as strongly as "
        "any role. At the current setting an elite goalie can top the board."
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

# Why a short-sample player's score sits closer to average than their raw rates do.
SHRINKAGE_NOTE = (
    "A player's Role Performance is pulled toward their own role's median in "
    "proportion to how few games they have played, reaching full value at eight "
    "games and never dropping below 35% of the distance from the median. This is not "
    "a penalty for missing time — it is what a short sample is actually worth as "
    "evidence. Measured out of sample, a player's performance over their first few "
    "games predicts the rest of their season at a slope of about 0.56, meaning a "
    "two-game hot streak is roughly half signal and half noise, and the honest "
    "estimate sits nearer average than the raw rate does. Rate stats are damped the "
    "same way before they are scored: a goalie's save percentage on 20 shots is "
    "blended toward the league rate, so 3-for-3 shooting no longer outranks a full "
    "season of good finishing. The displayed rates are the real ones — only the "
    "scoring uses the damped versions."
)

# Contexts are separately calibrated, which makes cross-context score arithmetic wrong.
CONTEXT_COMPARABILITY_NOTE = (
    "Each ranking context — Career, Last 10, Last 5, and each season — is scored and "
    "calibrated independently, against its own pool and its own median. A score is "
    "therefore a statement about a player's standing *inside that context*, and "
    "differences across contexts are not a trend: a player at 71 in 2025 and 68 in "
    "2026 has not necessarily declined, because the two 50s are different 50s and the "
    "sample-size damping is looser in a completed season than a partial one. Compare "
    "ranks within one context, or read the same context across players. The season "
    "span each context covers is shown alongside it, and \"Career\" means 2022 "
    "onward — the seasons in this warehouse, not the whole of league history, which "
    "began in 2019."
)

# Why the cross-role board rescales Role Performance before comparing roles.
RPS_NORMALIZATION_NOTE = (
    "Role Performance is rescaled within each role before roles are compared. A "
    "composite's ceiling depends on how much its inputs overlap, not on how good "
    "its best player is: faceoff performance is built from three closely related "
    "stats (they correlate at about 0.65, so averaging them is close to averaging "
    "one), which let its leader reach 98.9, while offensive performance is built "
    "from eight looser ones (correlating at about 0.37) where a player who is "
    "merely average at any single input cannot get there — the best attackman "
    "topped out at 91.9. Left alone, that put specialists above attackmen for "
    "reasons unrelated to how they played. Each role is now placed on a common "
    "median-and-spread scale, so 50 is that role's average and a given score means "
    "the same distance above average whichever role you are reading. The rescaling "
    "is order-preserving inside a role, so no player moves relative to their own "
    "peers, and the role-specific views are unaffected."
)

# What Role Performance is built from, per role. Sourced from the RPS blocks in
# build_warehouse.py; shown so a reader can tell why a player scores as they do.
RPS_INPUTS = {
    roles.ROLE_OFFENSE: [
        "Points production", "Creation efficiency", "Assist conversion rate",
        "Shot quality", "2PT conversion",
    ],
    # Weighted by measured effect on winning rather than evenly: caused turnovers
    # carry roughly twice the weight of ground balls because they predict team score
    # margin about twice as strongly. Discipline is small but real.
    roles.ROLE_DEFENSE: [
        "Caused turnovers (58%)", "Ground balls (20%)", "Ball security (15%)",
        "Discipline — penalties against (7%)",
    ],
    roles.ROLE_FACEOFF: ["Faceoff win %", "Total wins", "Volume"],
    roles.ROLE_GOALIE: [
        "Clean save rate (skill-based stops)", "Overall save %", "Volume",
        "Goals-against outcomes",
    ],
}

# Bands with the share of the eligible pool each one actually holds. The previous
# version labelled 85+ as "top ~10%" and 70–84 as "top ~25%", which the
# distribution never supported: 85+ holds 0.6–1.6% across contexts and the 90th
# percentile lands at 78–79, inside the band below. Descriptions now quote the
# measured shares (2026, 2025 and Career all agree to about a point), so a reader
# comparing a score against a tier gets the right idea of how rare it is.
SCORE_TIERS = [
    ("85+", "Elite", "Top ~1% — the outright best in the league."),
    ("70–84", "High-End", "Roughly the top quarter, and the 90th percentile "
                          "sits in this band."),
    ("55–69", "Solid Starter", "Above average, performing the role well "
                               "(~20% of players)."),
    ("45–54", "Average", "Close to league average for the context (~12%)."),
    ("Below 45", "Developmental", "Below average, or too few games to tell "
                                  "(~43%)."),
]

# Peer Standing is still published and still shown; it is no longer an input.
PSS_METHOD_NOTE = (
    "Peer Standing is a player's role-performance rank within their role group, "
    "passed through a sigmoid so 50 is the role average and 85+ is roughly the top "
    "10%. The underlying z-score uses the interquartile range rather than the "
    "standard deviation, so one outlier season cannot stretch the scale for "
    "everyone else. **It is a reading of Role Performance, not a separate "
    "measurement**, and it no longer feeds the Overall Score: because it is that "
    "same number re-expressed as a rank, the two correlate at about 0.99, so "
    "weighting both counted one thing twice. It is kept on the player pages as a "
    "plain answer to \"where do they sit among their peers\"."
)

# Two-way credit: how a player earns points for the half of the field their role
# score ignores. Kept here so the pages describe the same rule the warehouse applies.
TWO_WAY_NOTE = (
    "A role score deliberately ignores half the field — Defense Role Performance "
    "counts no points, Offense Role Performance counts no caused turnovers — which "
    "left genuine two-way players unpaid for the part of their game that makes them "
    "valuable. Players who produce on their secondary side now receive up to 6 extra "
    "points of Role Performance. To qualify, a player must clear an absolute "
    "per-game bar on that secondary side (0.89 points per game for a defender, 0.50 "
    "caused turnovers per game for an attacker), stand above their own role's "
    "average on it, and have played at least 70% of the games available in the "
    "ranking context. The two bars differ because the production does: no "
    "short-stick defensive midfielder has ever reached the midfield scoring average, "
    "so one shared threshold made the credit unreachable in the defensive direction. "
    "The credit is added rather than blended in, so a two-way midfielder's offensive "
    "score is not diluted to make room for their defensive work — playing both ways "
    "is extra value, so it is scored as extra. Ground balls are excluded from the "
    "secondary axis because Cross-Role Impact already pays every role for them."
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
        "Each player's **Overall Score** blends two components, each on a 0–100 "
        "scale where 50 is league average for the ranking context — and both are "
        "measured *within the player's own role*, so 50 means \"average for this "
        "position\" in each and the two are comparable before they are weighted.",
        "",
    ]

    rps_name = COMPONENTS["rps"][0]
    lines.append(f"**1. {rps_name}** — {COMPONENTS['rps'][1]}")
    for role in roles.ROLE_ORDER:
        inputs = RPS_INPUTS.get(role)
        if inputs:
            lines.append(f"- *{roles.role_label(role)}:* " + ", ".join(inputs))
    lines.append("")

    cis_name, cis_blurb = COMPONENTS["cis"]
    lines += [f"**2. {cis_name}** — {cis_blurb}", ""]

    lines.append("**Weights by role:**")
    for role, (rps, cis) in OVERALL_WEIGHTS.items():
        lines.append(
            f"- *{roles.role_label(role)}:* {rps:.0%} {COMPONENTS['rps'][0]}"
            f" + {cis:.0%} {COMPONENTS['cis'][0]}"
        )
    lines += [
        "",
        f"**Two-way players.** {TWO_WAY_NOTE}",
        "",
        f"**Small samples.** {SHRINKAGE_NOTE}",
        "",
        f"**Comparing roles.** {RPS_NORMALIZATION_NOTE}",
        "",
        f"**Specialist compression.** {transfer_note(peer_sizes, context)}",
        "",
        f"**Scale calibration.** {CALIBRATION_NOTE}",
        "",
        f"**Comparing across contexts.** {CONTEXT_COMPARABILITY_NOTE}",
        "",
        f"**Peer Standing.** {PSS_METHOD_NOTE}",
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
    for role, (rps, cis) in OVERALL_WEIGHTS.items():
        rows.append({
            "role_group": roles.role_label(role),
            "role_performance": rps,
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
    for role_key, (w_rps, w_cis) in OVERALL_WEIGHTS.items():
        mask = role == role_key
        if not mask.any():
            continue
        rps = numeric(RPS_COLUMNS[role_key])
        out.loc[mask] = (w_rps * rps + w_cis * cis).loc[mask]
    return out.clip(0, 100)


def verify_against_mart(df: pd.DataFrame, tolerance: float = 0.5) -> dict:
    """
    Compare the rebuilt score against the mart's own `overall_score`.

    The mart applies a uniform context shift after the blend, so the two differ by
    a constant per context rather than matching exactly. This reports the spread of
    that difference: a tight spread means the weights here still describe the
    build, a wide one means they have drifted.

    Players the mart clipped to 0 or 100 are excluded from the spread, because for
    them the difference is no longer the shift. A one-game player whose blend comes
    to 4.8 in a context shifted by -5.7 would land at -0.9, so the mart publishes 0
    and the observed difference is -4.8 — an artefact of the floor, not of the
    weights. `clipped` reports how many were set aside so a real drift can never
    hide behind the exclusion.
    """
    if df is None or len(df) == 0 or "overall_score" not in df.columns:
        return {"checked": 0, "matches": False, "spread": float("nan"),
                "clipped": 0}

    rebuilt = rebuild_overall_score(df)
    actual = pd.to_numeric(df["overall_score"], errors="coerce")
    delta = (actual - rebuilt).dropna()

    at_bound = actual.reindex(delta.index).isin([0.0, 100.0])
    clipped = int(at_bound.sum())
    delta = delta[~at_bound]
    if len(delta) == 0:
        return {"checked": 0, "matches": False, "spread": float("nan"),
                "clipped": clipped}

    spread = float(delta.max() - delta.min())
    return {
        "checked": int(len(delta)),
        "matches": spread <= tolerance,
        "spread": spread,
        "median_shift": float(delta.median()),
        "clipped": clipped,
    }
