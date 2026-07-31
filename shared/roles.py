"""
shared/roles.py — one position/role taxonomy for the whole app.

The app previously classified positions inline, per page, and disagreed with
itself. Page 08 gated its offensive panels on
    {"A", "M", "AT", "MF", "SSDM"}
treating SSDM (short-stick defensive midfield) as an OFFENSIVE position, while
page 07 filtered its defensive box on
    ["D", "LSM", "SSDM", "G"]
treating the same position as defensive. SSDM is a defensive midfielder — it is
the second-largest position group in the league (64 of 400 players) — so page 08
was showing playmaking-efficiency panels to a cohort of defenders and page 13's
role groups didn't line up with either.

Positions actually present in clean.player_directory:
    M 110, A 75, SSDM 64, D 61, LSM 40, G 27, FO 23

Aliases (AT, MF, ATT, MID, …) are accepted because the warehouse's older seasons
and the ranking mart's `role_group` column use different spellings.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# ============================================================
# ROLE GROUPS
# ============================================================

ROLE_OFFENSE = "offense"
ROLE_DEFENSE = "defense"
ROLE_FACEOFF = "faceoff"
ROLE_GOALIE = "goalie"
ROLE_UNKNOWN = "unknown"

ROLE_ORDER = [ROLE_OFFENSE, ROLE_DEFENSE, ROLE_FACEOFF, ROLE_GOALIE, ROLE_UNKNOWN]

ROLE_LABELS = {
    ROLE_OFFENSE: "Offense",
    ROLE_DEFENSE: "Defense",
    ROLE_FACEOFF: "Faceoff",
    ROLE_GOALIE: "Goalie",
    ROLE_UNKNOWN: "Unclassified",
}

# Canonical position codes → role. Keys are upper-case and punctuation-free.
_POSITION_TO_ROLE = {
    # Offense
    "A": ROLE_OFFENSE,      # Attack
    "AT": ROLE_OFFENSE,
    "ATT": ROLE_OFFENSE,
    "ATTACK": ROLE_OFFENSE,
    "M": ROLE_OFFENSE,      # Midfield — offensive midfield in PLL usage
    "MF": ROLE_OFFENSE,
    "MID": ROLE_OFFENSE,
    "MIDFIELD": ROLE_OFFENSE,
    "OM": ROLE_OFFENSE,
    # Defense — SSDM belongs here, not with offense
    "D": ROLE_DEFENSE,      # Close defense
    "DEF": ROLE_DEFENSE,
    "DEFENSE": ROLE_DEFENSE,
    "LSM": ROLE_DEFENSE,    # Long-stick midfield
    "SSDM": ROLE_DEFENSE,   # Short-stick defensive midfield
    "DM": ROLE_DEFENSE,
    # Faceoff
    "FO": ROLE_FACEOFF,
    "FOGO": ROLE_FACEOFF,
    "F": ROLE_FACEOFF,
    "FACEOFF": ROLE_FACEOFF,
    # Goalie
    "G": ROLE_GOALIE,
    "GK": ROLE_GOALIE,
    "GOALIE": ROLE_GOALIE,
    "GOALKEEPER": ROLE_GOALIE,
}

# Position display names, for profile headers and pickers.
POSITION_LABELS = {
    "A": "Attack",
    "M": "Midfield",
    "SSDM": "Short-Stick D-Midfield",
    "D": "Defense",
    "LSM": "Long-Stick Midfield",
    "FO": "Faceoff Specialist",
    "G": "Goalie",
}

# Order positions the way a depth chart reads, not alphabetically.
POSITION_ORDER = ["A", "M", "SSDM", "LSM", "D", "FO", "G"]


def normalize_position(value) -> str:
    """Upper-case, strip punctuation/whitespace. '' for missing."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    for ch in " ._-/\\":
        text = text.replace(ch, "")
    return text


def role_for_position(value) -> str:
    """Role group for a raw position value."""
    return _POSITION_TO_ROLE.get(normalize_position(value), ROLE_UNKNOWN)


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, str(role).title())


def position_label(value) -> str:
    key = normalize_position(value)
    return POSITION_LABELS.get(key, str(value) if value else "Unknown")


def positions_for_role(role: str) -> list[str]:
    """Canonical positions in `role`, in depth-chart order."""
    return [p for p in POSITION_ORDER if _POSITION_TO_ROLE.get(p) == role]


def sort_positions(values: Iterable[str]) -> list[str]:
    """Sort positions by POSITION_ORDER, unknowns alphabetically at the end."""
    seen = list(dict.fromkeys(values))
    known = [p for p in POSITION_ORDER if p in seen]
    rest = sorted(p for p in seen if p not in POSITION_ORDER)
    return known + rest


# ============================================================
# PREDICATES
# ============================================================

def is_offense(value) -> bool:
    return role_for_position(value) == ROLE_OFFENSE


def is_defense(value) -> bool:
    return role_for_position(value) == ROLE_DEFENSE


def is_goalie(value) -> bool:
    return role_for_position(value) == ROLE_GOALIE


def is_faceoff(value) -> bool:
    return role_for_position(value) == ROLE_FACEOFF


def is_runner(value) -> bool:
    """Field player — anyone who is not a goalie."""
    return role_for_position(value) != ROLE_GOALIE


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def add_role_column(df: pd.DataFrame, position_col: str = "position",
                    out_col: str = "role_group") -> pd.DataFrame:
    """
    Add/overwrite `out_col` with the canonical role.

    The ranking mart ships its own `role_group`; this deliberately overwrites it
    so a single taxonomy governs display everywhere. The mart's scoring columns
    are untouched.
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if position_col not in out.columns:
        out[out_col] = ROLE_UNKNOWN
        return out
    out[out_col] = out[position_col].map(role_for_position)
    return out


def filter_by_role(df: pd.DataFrame, role: str, position_col: str = "position") -> pd.DataFrame:
    """Rows whose position maps to `role`."""
    if df is None or len(df) == 0 or position_col not in df.columns:
        return df
    mask = df[position_col].map(role_for_position) == role
    return df[mask]


def role_counts(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """Player counts per role, ordered by ROLE_ORDER."""
    if df is None or len(df) == 0 or position_col not in df.columns:
        return pd.DataFrame(columns=["role_group", "players"])
    roles = df[position_col].map(role_for_position)
    counts = roles.value_counts().to_dict()
    rows = [
        {"role_group": role_label(r), "players": int(counts[r])}
        for r in ROLE_ORDER
        if r in counts
    ]
    return pd.DataFrame(rows)


# ============================================================
# ROLE-APPROPRIATE STAT BLOCKS
# ============================================================
#
# Which metrics matter for which role. Pages use these instead of hardcoding
# column lists, so a defender's profile leads with defensive work rather than
# with the goals-and-assists block that suited attackers.

ROLE_HEADLINE_METRICS = {
    ROLE_OFFENSE: [
        "points_per_game", "goals_per_game", "assists_per_game",
        "shots_per_game", "shot_pct_calc", "touches_per_game",
    ],
    ROLE_DEFENSE: [
        "caused_turnovers_per_game", "ground_balls_per_game", "turnovers_per_game",
        "touches_per_game", "points_per_game", "shots_per_game",
    ],
    ROLE_FACEOFF: [
        "faceoff_pct_calc", "faceoffs_won_per_game", "faceoffs_per_game",
        "ground_balls_per_game", "turnovers_per_game", "points_per_game",
    ],
    ROLE_GOALIE: [
        "save_pct_calc", "saves_per_game", "goals_against_per_game",
        "clean_save_rate", "clean_save_pct", "saa",
    ],
    ROLE_UNKNOWN: [
        "points_per_game", "goals_per_game", "assists_per_game",
        "ground_balls_per_game", "caused_turnovers_per_game", "turnovers_per_game",
    ],
}

# Metrics that are meaningless for a role and should be hidden rather than shown
# as zero. A goalie's "Shot %" of 0.0 is noise, not information.
ROLE_SUPPRESSED_METRICS = {
    ROLE_OFFENSE: {"save_pct_calc", "save_pct_display", "saves", "saves_per_game",
                   "clean_saves", "messy_saves", "clean_save_pct", "clean_save_rate",
                   "goals_against", "goals_against_per_game", "saa", "saa_per_game"},
    ROLE_DEFENSE: {"save_pct_calc", "save_pct_display", "saves", "saves_per_game",
                   "clean_saves", "messy_saves", "clean_save_pct", "clean_save_rate",
                   "goals_against", "goals_against_per_game", "saa", "saa_per_game"},
    ROLE_FACEOFF: {"save_pct_calc", "save_pct_display", "saves", "saves_per_game",
                   "clean_saves", "messy_saves", "clean_save_pct", "clean_save_rate",
                   "goals_against", "goals_against_per_game", "saa", "saa_per_game"},
    ROLE_GOALIE: {"shot_pct_calc", "shots_on_goal_rate_calc", "assist_conv_rate",
                  "two_pt_conversion", "faceoff_pct_calc", "faceoffs",
                  "faceoffs_won", "faceoffs_lost", "points_per_touch"},
    ROLE_UNKNOWN: set(),
}


def headline_metrics(position) -> list[str]:
    """Headline stat-card metrics for a position's role."""
    return list(ROLE_HEADLINE_METRICS.get(role_for_position(position),
                                          ROLE_HEADLINE_METRICS[ROLE_UNKNOWN]))


def suppressed_metrics(position) -> set[str]:
    """Metrics to hide for a position's role."""
    return set(ROLE_SUPPRESSED_METRICS.get(role_for_position(position), set()))


def relevant_metrics(position, candidates: Iterable[str]) -> list[str]:
    """`candidates` minus the metrics that are meaningless for this role."""
    drop = suppressed_metrics(position)
    return [c for c in candidates if c not in drop]
