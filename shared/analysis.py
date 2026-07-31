"""
shared/analysis.py — derived analytical views the warehouse doesn't ship.

Everything here is computed from columns that already exist. Three additions:

1. PER-100-POSSESSION RATES. The app was entirely per-game, which conflates
   "this team is good" with "this team plays fast". `offensive_sequence_proxy`
   has been in the team marts all along and was only ever shown as a raw count.
   Scores/100 possessions is the single most useful pace-independent measure
   available from this data.

2. OPPONENT ADJUSTMENT (schedule strength). `team_game_opponent_context` has one
   row per team-game with full team_*/opponent_* symmetry, so a team's per-game
   average can be compared against the average its opponents conceded to
   everyone else. In an 8-team league playing an unbalanced schedule this is a
   real effect, and nothing in the app accounted for it.

3. LEAGUE CONTEXT. Percentile/rank/z of a value against its league-season
   cohort, so a stat card can say "24.1 — 2nd of 8" instead of a bare number.

Style notes: pace/possession fields are provider-tracked and imperfect (see
qc.game_possession_quality), so every helper degrades to NaN rather than
raising, and callers are expected to show a caveat when possession data is thin.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from shared import metrics as M

# ============================================================
# PER-100-POSSESSION
# ============================================================

# Preference order for the possession denominator. offensive_sequence_proxy is
# first because it is populated for every game; official_total_possessions is
# provider-supplied and patchy across historical seasons.
#
# `team_offensive_sequence_proxy` is the same quantity under the name used by the
# defensive marts (team_defense_season_stats, team_game_opponent_context), which
# prefix every own-team column with `team_` to keep the opponent symmetry
# readable. Without it, per-100 rates silently skip the defensive tables.
POSSESSION_DENOMINATORS = [
    "offensive_sequence_proxy",
    "team_offensive_sequence_proxy",
    "official_total_possessions",
]

OPPONENT_POSSESSION_DENOMINATORS = [
    "opponent_offensive_sequence_proxy",
    "opponent_official_total_possessions",
]

# Allowed/opponent counting stats normalize by the OPPONENT's possessions: a
# team concedes on the opponent's offensive sequences, not on its own.
#
# Two vocabularies for the same idea live in the warehouse — the offensive marts
# say `scores_against`, the defensive marts say `scores_allowed` — so both
# spellings are listed.
_OPPONENT_NUMERATORS = {
    # defensive-mart spelling
    "scores_allowed", "goals_allowed", "assists_allowed",
    "one_point_goals_allowed", "two_point_goals_allowed",
    # offensive-mart spelling
    "scores_against", "goals_against", "two_point_goals_against",
    "power_play_goals_against",
    # opponent_* counting stats
    "opponent_shots", "opponent_shots_on_goal", "opponent_turnovers",
    "opponent_ground_balls", "opponent_caused_turnovers", "opponent_saves",
    "opponent_two_point_shots", "opponent_two_point_shots_on_goal",
    "opponent_touches", "opponent_total_passes",
    "opponent_clears", "opponent_clear_attempts",
}


def possession_denominator(df: pd.DataFrame, opponent: bool = False) -> str | None:
    """First available possession-count column, or None."""
    candidates = OPPONENT_POSSESSION_DENOMINATORS if opponent else POSSESSION_DENOMINATORS
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def add_per_100_possessions(df: pd.DataFrame,
                            metrics: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Add `<metric>_per_100_poss` for each counting stat present.

    Allowed/opponent stats use the opponent possession count where available and
    fall back to the team's own (the two are close but not identical, since a
    game's possessions alternate).
    """
    if df is None or len(df) == 0:
        return df

    own = possession_denominator(df, opponent=False)
    opp = possession_denominator(df, opponent=True) or own
    if own is None:
        return df

    out = df.copy()
    own_poss = pd.to_numeric(out[own], errors="coerce").replace(0, np.nan)
    opp_poss = pd.to_numeric(out[opp], errors="coerce").replace(0, np.nan) if opp else own_poss

    wanted = list(metrics) if metrics is not None else M.PER_100_CANDIDATES
    for base in wanted:
        if base not in out.columns:
            continue
        numerator = pd.to_numeric(out[base], errors="coerce")
        denom = opp_poss if base in _OPPONENT_NUMERATORS else own_poss
        out[M.per_100_key(base)] = numerator / denom * 100.0
    return out


def per_100_columns(df: pd.DataFrame) -> list[str]:
    """Per-100-possession columns present in `df`."""
    return [c for c in df.columns if str(c).endswith(M.PER_100_SUFFIX)]


def possession_coverage(df: pd.DataFrame) -> float:
    """
    Share of rows with a usable possession denominator, so a page can warn
    ("possession data available for 78% of games") instead of silently
    presenting per-100 rates built on partial data.
    """
    if df is None or len(df) == 0:
        return 0.0
    col = possession_denominator(df)
    if col is None:
        return 0.0
    values = pd.to_numeric(df[col], errors="coerce")
    return float((values > 0).sum()) / float(len(df))


# ============================================================
# LEAGUE CONTEXT
# ============================================================

def add_league_context(df: pd.DataFrame, metric: str,
                       group_cols: Iterable[str] | None = None,
                       prefix: str | None = None) -> pd.DataFrame:
    """
    Add rank / percentile / z-score for `metric` within each group.

    Direction comes from the metric registry, so rank 1 is the best value —
    lowest for Scores Allowed/G, highest for Scores/G.
    """
    if df is None or len(df) == 0 or metric not in df.columns:
        return df

    out = df.copy()
    p = prefix or metric
    values = pd.to_numeric(out[metric], errors="coerce")
    ascending = M.is_lower_better(metric)

    group_cols = [c for c in (group_cols or []) if c in out.columns]
    if group_cols:
        grouped = values.groupby([out[c] for c in group_cols])
        out[f"{p}_rank"] = grouped.rank(ascending=ascending, method="min")
        out[f"{p}_percentile"] = grouped.rank(ascending=not ascending, pct=True) * 100.0
        mean = grouped.transform("mean")
        std = grouped.transform("std")
    else:
        out[f"{p}_rank"] = values.rank(ascending=ascending, method="min")
        out[f"{p}_percentile"] = values.rank(ascending=not ascending, pct=True) * 100.0
        mean = values.mean()
        std = values.std()

    std = pd.Series(std, index=out.index).replace(0, np.nan) if not np.isscalar(std) \
        else (np.nan if not std else std)
    z = (values - mean) / std
    out[f"{p}_z"] = -z if ascending else z
    return out


def rank_text(df: pd.DataFrame, metric: str, row_index) -> str:
    """'2nd of 8' style rank text for a stat card. Empty string if unavailable."""
    if df is None or metric not in df.columns or row_index not in df.index:
        return ""
    values = pd.to_numeric(df[metric], errors="coerce")
    total = int(values.notna().sum())
    if total == 0:
        return ""
    ranks = values.rank(ascending=M.is_lower_better(metric), method="min")
    r = ranks.get(row_index)
    if pd.isna(r):
        return ""
    return f"{ordinal(int(r))} of {total}"


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ============================================================
# OPPONENT ADJUSTMENT (SCHEDULE STRENGTH)
# ============================================================

def opponent_adjusted(game_df: pd.DataFrame, metric: str,
                      team_col: str = "team_id",
                      opponent_col: str = "opponent_team_id") -> pd.DataFrame:
    """
    Adjust a per-game metric for the quality of opposition faced.

    Method — a single pass of the standard "average opponent" correction:

      team_raw      = team's own mean of `metric`
      league_mean   = mean across all team-games
      opp_baseline  = for each opponent, its mean of `metric` against everyone
                      EXCEPT this team (leave-one-out, so a team's own results
                      can't inflate its own opponents' difficulty)
      strength      = mean(opp_baseline) - league_mean
      adjusted      = team_raw - strength

    The subtraction is correct for both "for" and "allowed" metrics, but the SIGN
    of `strength` reads differently between them, because what it measures flips:

      metric = team_scores      → opp_baseline is what opponents CONCEDE, so
                                  strength > 0 means weak defences faced
                                  (an easy schedule, inflating raw).
      metric = scores_allowed   → opp_baseline is what opponents SCORE, so
                                  strength > 0 means strong offences faced
                                  (a hard schedule, inflating raw).

    Either way a positive value means the raw figure was flattered by opposition
    and the adjusted figure is lower. Use `schedule_strength_note(metric)` for
    wording that matches the metric on screen.

    One pass, not iterated to convergence: with 8 teams and ~10 games each,
    iterating amplifies noise more than it removes bias. Returns raw, adjusted,
    and the strength term so the UI can show the size of the correction.
    """
    required = {team_col, opponent_col, metric}
    if game_df is None or len(game_df) == 0 or not required.issubset(game_df.columns):
        return pd.DataFrame()

    df = game_df[[team_col, opponent_col, metric]].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[team_col, opponent_col, metric])
    if len(df) == 0:
        return pd.DataFrame()

    league_mean = df[metric].mean()

    # What each team conceded, by opponent: from the opponent's perspective the
    # row's metric is what they allowed.
    conceded_sum = df.groupby(opponent_col)[metric].sum()
    conceded_n = df.groupby(opponent_col)[metric].size()

    rows = []
    for team, team_rows in df.groupby(team_col):
        raw = team_rows[metric].mean()
        # Leave-one-out: subtract this team's own contribution to each opponent.
        own_by_opp_sum = team_rows.groupby(opponent_col)[metric].sum()
        own_by_opp_n = team_rows.groupby(opponent_col)[metric].size()

        allowances = []
        for opp, n_vs in own_by_opp_n.items():
            total_sum = conceded_sum.get(opp, np.nan)
            total_n = conceded_n.get(opp, 0)
            rem_n = total_n - n_vs
            if rem_n <= 0 or pd.isna(total_sum):
                continue
            rem_sum = total_sum - own_by_opp_sum.get(opp, 0.0)
            # Weight by how often this team faced that opponent.
            allowances.extend([rem_sum / rem_n] * int(n_vs))

        if allowances:
            strength = float(np.mean(allowances)) - league_mean
        else:
            strength = np.nan

        rows.append({
            team_col: team,
            "games": int(len(team_rows)),
            f"{metric}_raw": raw,
            "schedule_strength": strength,
            f"{metric}_adjusted": raw - strength if pd.notna(strength) else raw,
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out["league_mean"] = league_mean
    return M.sort_df(out, f"{metric}_adjusted")


SCHEDULE_STRENGTH_NOTE = (
    "Schedule strength compares the opponents faced against the league average. "
    "A positive value means the raw figure was flattered by the opposition, so the "
    "adjusted figure is lower; a negative value means the opposite."
)


def schedule_strength_note(metric: str) -> str:
    """
    Wording for the schedule-strength column that matches the metric's side of
    the ball, since a positive value means an easy schedule for a "for" metric and
    a hard one for an "allowed" metric.
    """
    if M.is_lower_better(metric):
        return (
            "Schedule strength here is how much better than league average the "
            "offences faced were. Positive means a tougher schedule, so the "
            "adjusted figure is better (lower) than the raw one."
        )
    return (
        "Schedule strength here is how much more than league average the defences "
        "faced conceded. Positive means an easier schedule, so the adjusted figure "
        "is lower than the raw one."
    )


# ============================================================
# FOUR-FACTOR STYLE SUMMARY
# ============================================================
#
# A compact "why is this team good" breakdown, in the spirit of basketball's four
# factors, using the pace-independent quantities this data supports.

FOUR_FACTORS = [
    ("scores_per_100_poss", "Offensive Efficiency", "Scores per 100 offensive sequences"),
    ("scores_allowed_per_100_poss", "Defensive Efficiency", "Scores allowed per 100 opponent sequences"),
    ("turnovers_per_100_poss", "Ball Security", "Turnovers per 100 offensive sequences"),
    ("offensive_sequence_proxy_per_game", "Pace", "Offensive sequences per game"),
]


def four_factor_frame(df: pd.DataFrame, entity_col: str = "team_name") -> pd.DataFrame:
    """
    Long-format four-factor table with league rank per factor.
    Expects `df` to already carry per-100 columns (see add_per_100_possessions).
    """
    if df is None or len(df) == 0 or entity_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for key, label, definition in FOUR_FACTORS:
        if key not in df.columns:
            continue
        values = pd.to_numeric(df[key], errors="coerce")
        ranks = values.rank(ascending=M.is_lower_better(key), method="min")
        for idx in df.index:
            if pd.isna(values.get(idx)):
                continue
            rows.append({
                entity_col: df.at[idx, entity_col],
                "factor": label,
                "metric": key,
                "value": values.get(idx),
                "rank": ranks.get(idx),
                "definition": definition,
            })
    return pd.DataFrame(rows)


# ============================================================
# ROLLING FORM
# ============================================================

def add_rolling(df: pd.DataFrame, metric: str, window: int = 5,
                sort_cols: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Add a `<metric>_roll<window>` trailing mean over game rows.

    The Recent Form sections showed Last-5/Last-10 aggregates only, which hide
    the shape of a run: a player averaging 3.0 while trending from 5 down to 1
    reads identically to one holding steady.
    """
    if df is None or len(df) == 0 or metric not in df.columns:
        return df
    out = df.copy()
    sort_cols = [c for c in (sort_cols or ["season", "game_number"]) if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    values = pd.to_numeric(out[metric], errors="coerce")
    out[f"{metric}_roll{window}"] = values.rolling(window, min_periods=1).mean()
    return out


def form_delta(df: pd.DataFrame, metric: str, window: int = 5) -> float:
    """
    Recent-window mean minus the mean of everything before it. Positive means
    the player/team is trending up. NaN when there isn't enough history.
    """
    if df is None or metric not in df.columns or len(df) < window + 1:
        return float("nan")
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(values) < window + 1:
        return float("nan")
    recent = values.iloc[-window:].mean()
    prior = values.iloc[:-window].mean()
    return float(recent - prior)
