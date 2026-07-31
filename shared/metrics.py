"""
shared/metrics.py — the single source of truth for what every metric means.

Before this module the app re-declared the same knowledge on every page: display
labels lived in ui.COL_LABELS, sort direction lived in four different
`lower_is_better` sets (pages 06, 10, 15 ×2) that did not agree with each other,
and number formatting was decided by dtype alone — so a percentage stored as
0.283 rendered as "0.28" instead of "28.3%".

Everything about a metric is now declared once, here:

    label       what the user sees
    unit        how to format it (and, critically, what scale it is stored on)
    direction   "hi" = higher is better, "lo" = lower is better, None = neither
    definition  plain-English meaning, surfaced in the Data Guide and tooltips
    family      grouping used to build stat blocks and metric pickers

Only metrics the app actually surfaces are declared explicitly. The warehouse has
443 distinct column names, most of them intermediate scoring columns, so
`describe()` falls back to inference from the column name for anything not
registered. The inference rules are deliberately conservative and every known
exception is registered explicitly — see UNIT NOTES below.

UNIT NOTES — the traps this module exists to prevent:

* `pct01` vs `pct100`. Almost every rate in the warehouse is stored 0–1
  (shot_pct_calc, faceoff_pct_calc, win_pct, clear_pct, power_play_pct, …), but
  a handful are stored 0–100 and MUST NOT be multiplied again:
    - clean_save_pct           (marts: clean_saves / saves * 100)
    - every *_percentile column
    - every *_score column (0–100 composite scores)
    - role_reliability         (0–100; currently constant at 100.0)
  Note `clean.player_game_stats.clean_save_pct` is 0–1 while the marts version
  is 0–100. Game-level goalie display goes through the marts convention, so the
  registry declares the marts scale and `clean_pct_scale()` exists for the rare
  caller that reads the clean table directly.

* `goalie_save_pct_for_overall` is NOT a save percentage, despite the name, and
  must never be shown as one. Across marts.player_ranking_profiles it ranges
  17.2–85.0 (capped at 85), correlates -0.74 with the actual save_pct and +0.90
  with goalie_rps: it is a transformed, clamped scoring input, not a rate. The
  name reads as a percentage, so the inference rules would have formatted it as
  one — hence the explicit registration.

* `clean_save_pct` and `clean_save_rate` are DIFFERENT METRICS that were both
  labelled "Clean Save %":
    - clean_save_pct  = clean_saves / saves            → "Clean Save Share"
    - clean_save_rate = clean_saves / (saves + GA)      → "Clean Saves/Shot"
  Blaze Riorden's career: 38.4% share, 21.8% per shot faced. Showing both as
  "Clean Save %" made the goalie panel look broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# ============================================================
# UNITS
# ============================================================

UNIT_INT = "int"            # whole-number count            → 1,234
UNIT_NUM1 = "num1"          # one decimal                   → 12.3
UNIT_NUM2 = "num2"          # two decimals, always          → 1.23, 12.00
UNIT_AUTO = "auto"          # two decimals unless whole     → 1.23, 12
UNIT_PCT01 = "pct01"        # stored 0–1, shown as percent   → 28.3%
UNIT_PCT100 = "pct100"      # stored 0–100, shown as percent → 28.3%
UNIT_SCORE = "score"        # 0–100 composite, no % sign     → 84.2
UNIT_SEC = "sec"            # seconds → M:SS
UNIT_SEC_TOTAL = "sec_total"  # seconds → H:MM:SS
UNIT_TEXT = "text"

# UNIT_NUM2 vs UNIT_AUTO: a registered rate is NUM2 so a column of them lines up
# ("12.00" beside "11.10", not "12"). UNIT_AUTO is for columns the registry has
# never seen, where the value could be either a count or a rate — guessing "45.00"
# for a count reads worse than adapting per value.

_PCT_UNITS = {UNIT_PCT01, UNIT_PCT100}

HI = "hi"
LO = "lo"

# Families, in the order they should appear when a page renders "all families".
FAMILY_ORDER = [
    "identity",
    "volume",
    "scoring",
    "shooting",
    "playmaking",
    "possession",
    "groundball",
    "defense",
    "goalie",
    "faceoff",
    "special",
    "discipline",
    "opponent",
    "composite",
    "meta",
]

FAMILY_LABELS = {
    "identity": "Identity",
    "volume": "Volume",
    "scoring": "Scoring",
    "shooting": "Shooting",
    "playmaking": "Playmaking",
    "possession": "Possession & Pace",
    "groundball": "Ground Balls",
    "defense": "Defense",
    "goalie": "Goaltending",
    "faceoff": "Faceoffs",
    "special": "Special Situations",
    "discipline": "Discipline",
    "opponent": "Opponent Allowed",
    "results": "Results",
    "composite": "Composite Scores",
    "qc": "Data Quality",
    "meta": "Context",
}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str = UNIT_NUM2
    direction: str | None = None
    definition: str = ""
    family: str = "meta"
    short: str | None = None

    @property
    def display_short(self) -> str:
        return self.short or self.label

    @property
    def is_pct(self) -> bool:
        return self.unit in _PCT_UNITS


def _m(key, label, unit=UNIT_NUM2, direction=None, definition="", family="meta", short=None):
    return Metric(key, label, unit, direction, definition, family, short)


# ============================================================
# REGISTRY
# ============================================================
#
# Ordered by family so related metrics stay together when this file is edited.

_REGISTRY: list[Metric] = [
    # ---------- identity / context ----------
    _m("full_name", "Player", UNIT_TEXT, family="identity"),
    _m("team_name", "Team", UNIT_TEXT, family="identity"),
    _m("teams", "Teams", UNIT_TEXT, family="identity"),
    _m("team_names", "Teams", UNIT_TEXT, family="identity"),
    _m("opponent_team_name", "Opponent", UNIT_TEXT, family="identity"),
    _m("opponents", "Opponents", UNIT_TEXT, family="identity"),
    _m("position", "Pos", UNIT_TEXT, family="identity"),
    _m("position_name", "Position", UNIT_TEXT, family="identity"),
    _m("role_group", "Role", UNIT_TEXT, family="identity"),
    _m("season", "Season", UNIT_TEXT, family="identity"),
    _m("seasons", "Seasons", UNIT_TEXT, family="identity"),
    _m("game_number", "Game #", UNIT_INT, family="identity"),
    _m("game_date_utc", "Date", UNIT_TEXT, family="identity"),
    _m("game_date_guess", "Date", UNIT_TEXT, family="identity"),
    _m("game_label", "Game", UNIT_TEXT, family="identity"),
    _m("is_home", "Home/Away", UNIT_TEXT, family="identity"),
    _m("split_type", "Split", UNIT_TEXT, family="identity"),
    _m("result", "Result", UNIT_TEXT, family="identity"),
    _m("venue", "Venue", UNIT_TEXT, family="identity"),
    _m("week", "Week", UNIT_INT, family="identity"),

    # ---------- volume / record ----------
    _m("games", "Games", UNIT_INT, family="volume",
       definition="Games with stats recorded in the warehouse."),
    _m("wins", "Wins", UNIT_INT, HI, family="volume"),
    _m("losses", "Losses", UNIT_INT, LO, family="volume"),
    _m("ties", "Ties", UNIT_INT, family="volume"),
    _m("win_pct", "Win %", UNIT_PCT01, HI, family="volume",
       definition="Wins divided by games played."),
    _m("score_margin", "Score Margin", UNIT_NUM1, HI, family="volume",
       definition="Scores for minus scores against."),
    _m("score_margin_per_game", "Margin/G", UNIT_NUM2, HI, family="volume",
       definition="Average scoreboard margin per game. The cleanest single measure of team quality."),
    _m("net_scores_per_game", "Net Scores/G", UNIT_NUM2, HI, family="volume",
       definition="Scores per game minus scores allowed per game."),
    # Computed in the pages, not the warehouse — the pace-adjusted counterpart to
    # Margin/G, and the best single "who is good" column this data supports.
    _m("net_efficiency_per_100_poss", "Net Eff/100 Poss", UNIT_NUM1, HI, family="volume",
       definition="Scores per 100 offensive sequences minus scores allowed per 100 "
                  "opponent sequences. Margin with pace removed, so a fast team and "
                  "a slow team can be compared directly."),

    # ---------- scoring ----------
    _m("scores", "Scores", UNIT_INT, HI, family="scoring",
       definition="Scoreboard total. Differs from goals because a 2-point goal counts as two scores."),
    _m("scores_per_game", "Scores/G", UNIT_NUM2, HI, family="scoring"),
    _m("goals", "Goals", UNIT_INT, HI, family="scoring",
       definition="Made goals regardless of scoreboard value. A 2-point goal is one goal but two scores."),
    _m("goals_per_game", "Goals/G", UNIT_NUM2, HI, family="scoring"),
    _m("one_point_goals", "1PT Goals", UNIT_INT, HI, family="scoring",
       definition="Goals scored from inside the 2-point arc."),
    _m("one_point_goals_per_game", "1PT Goals/G", UNIT_NUM2, HI, family="scoring"),
    _m("two_point_goals", "2PT Goals", UNIT_INT, HI, family="scoring",
       definition="Goals scored from beyond the 2-point arc; worth two scores."),
    _m("two_point_goals_per_game", "2PT Goals/G", UNIT_NUM2, HI, family="scoring"),
    _m("points", "Points", UNIT_INT, HI, family="scoring",
       definition="Goals plus assists."),
    _m("points_per_game", "Points/G", UNIT_NUM2, HI, family="scoring"),
    _m("scoring_points", "Scoring Pts", UNIT_INT, HI, family="scoring",
       definition="Points weighted by scoreboard value, so a 2-point goal counts double."),
    _m("scoring_points_per_game", "Scoring Pts/G", UNIT_NUM2, HI, family="scoring"),

    # ---------- shooting ----------
    _m("shots", "Shots", UNIT_INT, HI, family="shooting"),
    _m("shots_per_game", "Shots/G", UNIT_NUM2, HI, family="shooting"),
    _m("shots_on_goal", "SOG", UNIT_INT, HI, family="shooting"),
    _m("shots_on_goal_per_game", "SOG/G", UNIT_NUM2, HI, family="shooting"),
    _m("shot_pct", "Shot %", UNIT_PCT01, HI, family="shooting",
       definition="Goals divided by shots."),
    _m("shot_pct_calc", "Shot %", UNIT_PCT01, HI, family="shooting",
       definition="Goals divided by shots, recomputed from totals."),
    _m("shots_on_goal_rate", "SOG Rate", UNIT_PCT01, HI, family="shooting",
       definition="Shots on goal divided by total shots — how often a shot is at least on frame."),
    _m("shots_on_goal_rate_calc", "SOG Rate", UNIT_PCT01, HI, family="shooting",
       definition="Shots on goal divided by total shots, recomputed from totals."),
    _m("shots_on_goal_pct", "SOG Rate", UNIT_PCT01, HI, family="shooting"),
    _m("two_point_shots", "2PT Shots", UNIT_INT, family="shooting"),
    _m("two_point_shots_per_game", "2PT Shots/G", UNIT_NUM2, family="shooting"),
    _m("two_point_shots_on_goal", "2PT SOG", UNIT_INT, family="shooting"),
    _m("two_point_shots_on_goal_per_game", "2PT SOG/G", UNIT_NUM2, family="shooting"),
    _m("two_point_shot_pct", "2PT Shot %", UNIT_PCT01, HI, family="shooting",
       definition="2-point goals divided by 2-point shots."),
    _m("two_point_goal_pct_calc", "2PT Shot %", UNIT_PCT01, HI, family="shooting"),
    _m("two_pt_conversion", "2PT Conversion", UNIT_PCT01, HI, family="shooting",
       definition="2-point goals divided by 2-point shot attempts."),
    _m("goals_per_shot", "Goals/Shot", UNIT_NUM2, HI, family="shooting"),

    # ---------- playmaking ----------
    _m("assists", "Assists", UNIT_INT, HI, family="playmaking"),
    _m("assists_per_game", "Assists/G", UNIT_NUM2, HI, family="playmaking"),
    _m("assist_opportunities", "Assist Opp", UNIT_INT, HI, family="playmaking",
       definition="Passes that created a shot attempt, whether or not it was converted."),
    _m("assist_opportunities_per_game", "Assist Opp/G", UNIT_NUM2, HI, family="playmaking"),
    _m("assist_opp_per_game", "Assist Opp/G", UNIT_NUM2, HI, family="playmaking"),
    _m("assist_conv_rate", "Assist Conv %", UNIT_PCT01, HI, family="playmaking",
       definition="Assists divided by assist opportunities — how often a created chance is converted by the shooter."),
    _m("total_passes", "Passes", UNIT_INT, family="playmaking"),
    _m("total_passes_per_game", "Passes/G", UNIT_NUM2, family="playmaking"),
    _m("passes_per_touch", "Passes/Touch", UNIT_NUM2, family="playmaking",
       definition="Passes divided by touches — how often possession is moved on rather than carried."),
    _m("assists_per_touch", "Assists/Touch", UNIT_NUM2, HI, family="playmaking"),
    _m("points_per_touch", "Points/Touch", UNIT_NUM2, HI, family="playmaking",
       definition="Points divided by touches — scoring output per unit of involvement."),

    # ---------- possession & pace ----------
    _m("touches", "Touches", UNIT_INT, family="possession",
       definition="Provider-tracked touches. A usage/involvement indicator, not an official possession count."),
    _m("touches_per_game", "Touches/G", UNIT_NUM2, family="possession"),
    _m("time_in_possession", "Possession Time", UNIT_SEC_TOTAL, family="possession"),
    _m("time_in_possession_per_game", "Possession/G", UNIT_SEC, family="possession"),
    _m("time_in_possession_pct", "Possession %", UNIT_PCT01, HI, family="possession",
       definition="Share of tracked possession time held by this team."),
    _m("official_total_possessions", "Official Possessions", UNIT_INT, family="possession",
       definition="Provider possession count. Not consistently populated across all historical games."),
    _m("official_total_possessions_per_game", "Possessions/G", UNIT_NUM2, family="possession"),
    _m("offensive_sequence_proxy", "Offensive Sequences", UNIT_INT, family="possession",
       definition="Estimated offensive possessions, used where official possession counts are missing or inconsistent."),
    _m("offensive_sequence_proxy_per_game", "Off Sequences/G", UNIT_NUM2, family="possession",
       definition="Estimated offensive possessions per game — the app's pace measure."),
    _m("seconds_possession_per_touch", "Seconds/Touch", UNIT_NUM2, family="possession"),
    _m("touches_per_offensive_sequence_proxy", "Touches/Sequence", UNIT_NUM2, family="possession"),
    _m("passes_per_offensive_sequence_proxy", "Passes/Sequence", UNIT_NUM2, family="possession"),
    _m("turnovers", "Turnovers", UNIT_INT, LO, family="possession"),
    _m("turnovers_per_game", "TO/G", UNIT_NUM2, LO, family="possession"),
    _m("turnovers_per_touch", "TO/Touch", UNIT_NUM2, LO, family="possession",
       definition="Turnovers divided by touches — ball security relative to involvement."),
    _m("shot_clock_expirations", "Shot Clock Expirations", UNIT_INT, LO, family="possession",
       definition="Possessions that ended with the shot clock running out — a failure to generate a look."),
    _m("shot_clock_expirations_per_game", "Shot Clock Exp/G", UNIT_NUM2, LO, family="possession"),

    # ---------- ground balls ----------
    _m("ground_balls", "Ground Balls", UNIT_INT, HI, family="groundball", short="GB"),
    _m("ground_balls_per_game", "GB/G", UNIT_NUM2, HI, family="groundball"),

    # ---------- defense ----------
    _m("caused_turnovers", "Caused TO", UNIT_INT, HI, family="defense", short="CT",
       definition="Turnovers forced by this player or team's defensive action."),
    _m("caused_turnovers_per_game", "CT/G", UNIT_NUM2, HI, family="defense"),
    _m("caused_turnovers_for", "Caused TO", UNIT_INT, HI, family="defense", short="CT"),
    _m("caused_turnovers_for_per_game", "CT/G", UNIT_NUM2, HI, family="defense"),
    _m("clears", "Clears", UNIT_INT, HI, family="defense",
       definition="Successful transitions of the ball from the defensive half to the offensive half."),
    _m("clears_per_game", "Clears/G", UNIT_NUM2, HI, family="defense"),
    _m("clear_attempts", "Clear Att", UNIT_INT, family="defense"),
    _m("clear_attempts_per_game", "Clear Att/G", UNIT_NUM2, family="defense"),
    _m("clear_pct", "Clear %", UNIT_PCT01, HI, family="defense",
       definition="Clears divided by clear attempts."),
    _m("clear_pct_calc", "Clear %", UNIT_PCT01, HI, family="defense",
       definition="Clears divided by clear attempts, recomputed from totals."),
    _m("ride_attempts", "Ride Attempts", UNIT_INT, HI, family="defense",
       definition="Attempts to pressure the opponent's clear — how aggressively a team contests the transition."),
    _m("ride_attempts_per_game", "Ride Att/G", UNIT_NUM2, HI, family="defense"),
    _m("ct_per_opponent_turnover", "CT/Opp TO", UNIT_NUM2, HI, family="defense",
       definition="Share of opponent turnovers the defense actively caused, rather than benefiting from unforced errors."),

    # ---------- goaltending ----------
    _m("saves", "Saves", UNIT_INT, HI, family="goalie"),
    _m("saves_per_game", "Saves/G", UNIT_NUM2, family="goalie"),
    _m("saves_for", "Saves", UNIT_INT, HI, family="goalie"),
    _m("saves_for_per_game", "Saves/G", UNIT_NUM2, family="goalie"),
    _m("save_pct", "Save %", UNIT_PCT01, HI, family="goalie",
       definition="Saves divided by shots faced (saves + goals against)."),
    _m("save_pct_calc", "Save %", UNIT_PCT01, HI, family="goalie",
       definition="Saves divided by saves plus goals against, recomputed from totals."),
    _m("save_pct_display", "Save %", UNIT_PCT01, HI, family="goalie"),
    _m("save_pct_proxy", "Save % (Team)", UNIT_PCT01, HI, family="goalie",
       definition="Team-level save rate from the opponent-context mart, not tied to one goalie."),
    _m("shots_faced_calc", "Shots Faced", UNIT_INT, family="goalie",
       definition="Saves plus goals against."),
    _m("shots_faced_per_game_calc", "Shots Faced/G", UNIT_NUM2, family="goalie"),
    _m("clean_saves", "Clean Saves", UNIT_INT, HI, family="goalie",
       definition="Saves controlled cleanly, with no rebound or scramble."),
    _m("clean_saves_per_game", "Clean Saves/G", UNIT_NUM2, HI, family="goalie"),
    _m("messy_saves", "Messy Saves", UNIT_INT, family="goalie",
       definition="Saves that were not controlled cleanly."),
    _m("messy_saves_per_game", "Messy Saves/G", UNIT_NUM2, family="goalie"),
    # The two metrics that were both labelled "Clean Save %". Distinct labels now.
    _m("clean_save_pct", "Clean Save Share", UNIT_PCT100, HI, family="goalie",
       definition="Share of a goalie's saves that were clean (clean saves / saves). Stored 0-100."),
    _m("clean_save_rate", "Clean Saves/Shot", UNIT_PCT01, HI, family="goalie",
       definition="Clean saves per shot faced (clean saves / (saves + goals against)). Rewards both volume and control."),
    _m("goals_against", "Goals Against", UNIT_INT, LO, family="goalie"),
    _m("goals_against_per_game", "GA/G", UNIT_NUM2, LO, family="goalie"),
    _m("scores_against", "Scores Against", UNIT_INT, LO, family="goalie",
       definition="Scoreboard scores allowed; differs from goals against when 2-point goals occur."),
    _m("scores_against_per_game", "Scores Against/G", UNIT_NUM2, LO, family="goalie"),
    _m("saa", "SAA", UNIT_NUM2, LO, family="goalie",
       definition="Scores against average — scores allowed per game of goalie work."),
    _m("saa_per_game", "SAA/G", UNIT_NUM2, LO, family="goalie"),
    _m("scores_against_average", "SAA", UNIT_NUM2, LO, family="goalie"),
    _m("two_point_goals_against", "2PT GA", UNIT_INT, LO, family="goalie"),
    _m("two_point_goals_against_per_game", "2PT GA/G", UNIT_NUM2, LO, family="goalie"),
    _m("two_point_gaa", "2PT GAA", UNIT_NUM2, LO, family="goalie"),

    # ---------- faceoffs ----------
    _m("faceoffs", "Faceoffs", UNIT_INT, family="faceoff", short="FO"),
    _m("faceoffs_per_game", "FO/G", UNIT_NUM2, family="faceoff"),
    _m("faceoffs_won", "FO Won", UNIT_INT, HI, family="faceoff"),
    _m("faceoffs_won_per_game", "FO Won/G", UNIT_NUM2, HI, family="faceoff"),
    _m("faceoffs_lost", "FO Lost", UNIT_INT, LO, family="faceoff"),
    _m("faceoffs_lost_per_game", "FO Lost/G", UNIT_NUM2, LO, family="faceoff"),
    _m("faceoff_pct", "FO Win %", UNIT_PCT01, HI, family="faceoff",
       definition="Faceoffs won divided by faceoffs taken."),
    _m("faceoff_pct_calc", "FO Win %", UNIT_PCT01, HI, family="faceoff",
       definition="Faceoffs won divided by faceoffs taken, recomputed from totals."),
    _m("faceoff_pct_for_ranking", "FO Win %", UNIT_PCT01, HI, family="faceoff"),
    _m("fo_record", "FO Record", UNIT_TEXT, family="faceoff"),

    # ---------- special situations ----------
    _m("power_play_goals", "PP Goals", UNIT_INT, HI, family="special"),
    _m("power_play_goals_per_game", "PP Goals/G", UNIT_NUM2, HI, family="special"),
    _m("power_play_shots", "PP Shots", UNIT_INT, family="special"),
    _m("power_play_shots_per_game", "PP Shots/G", UNIT_NUM2, family="special"),
    _m("power_play_pct", "PP Conversion", UNIT_PCT01, HI, family="special",
       definition="Share of man-up opportunities converted into a goal."),
    _m("times_man_up", "Man-Up Opportunities", UNIT_INT, family="special"),
    _m("times_man_up_per_game", "Man-Up/G", UNIT_NUM2, family="special"),
    _m("times_short_handed", "Man-Down Situations", UNIT_INT, family="special"),
    _m("times_short_handed_per_game", "Man-Down/G", UNIT_NUM2, family="special"),
    _m("power_play_goals_against", "PP Goals Allowed", UNIT_INT, LO, family="special"),
    _m("power_play_goals_against_per_game", "PP Goals Allowed/G", UNIT_NUM2, LO, family="special"),
    # These two are complements, verified against clean.team_game_stats:
    #   power_play_goals_against_pct =     PP GA / times_short_handed   → conceded
    #   man_down_pct                 = 1 - PP GA / times_short_handed   → killed
    # They sum to 1.0 row by row. The names invite exactly the wrong reading, so
    # the labels say which is which and the directions differ accordingly.
    _m("power_play_goals_against_pct", "Man-Down Concede Rate", UNIT_PCT01, LO,
       family="special",
       definition="Share of man-down situations that resulted in a goal against. "
                  "Lower is better. Complement of Man-Down Kill Rate."),
    _m("man_down_pct", "Man-Down Kill Rate", UNIT_PCT01, HI, family="special",
       definition="Share of man-down situations survived without conceding. "
                  "Higher is better. Complement of Man-Down Concede Rate."),

    # ---------- discipline ----------
    _m("num_penalties", "Penalties", UNIT_INT, LO, family="discipline"),
    _m("num_penalties_per_game", "Penalties/G", UNIT_NUM2, LO, family="discipline"),
    _m("pim", "PIM", UNIT_NUM1, LO, family="discipline",
       definition="Penalty minutes."),
    _m("pim_per_game", "PIM/G", UNIT_NUM2, LO, family="discipline"),

    # ---------- opponent / allowed ----------
    _m("team_scores", "Scores For", UNIT_INT, HI, family="opponent"),
    _m("team_scores_per_game", "Scores For/G", UNIT_NUM2, HI, family="opponent"),
    _m("scores_allowed", "Scores Allowed", UNIT_INT, LO, family="opponent"),
    _m("scores_allowed_per_game", "Scores Allowed/G", UNIT_NUM2, LO, family="opponent",
       definition="Scoreboard scores conceded per game."),
    _m("goals_allowed", "Goals Allowed", UNIT_INT, LO, family="opponent"),
    _m("goals_allowed_per_game", "Goals Allowed/G", UNIT_NUM2, LO, family="opponent"),
    _m("one_point_goals_allowed", "1PT Goals Allowed", UNIT_INT, LO, family="opponent"),
    _m("one_point_goals_allowed_per_game", "1PT Allowed/G", UNIT_NUM2, LO, family="opponent"),
    _m("two_point_goals_allowed", "2PT Goals Allowed", UNIT_INT, LO, family="opponent",
       definition="2-point goals conceded — measures how well a defense guards the arc."),
    _m("two_point_goals_allowed_per_game", "2PT Allowed/G", UNIT_NUM2, LO, family="opponent"),
    _m("assists_allowed", "Assists Allowed", UNIT_INT, LO, family="opponent",
       definition="Opponent assists conceded — a proxy for how much ball movement the defense permits."),
    _m("assists_allowed_per_game", "Assists Allowed/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_shots", "Opp Shots", UNIT_INT, LO, family="opponent"),
    _m("opponent_shots_per_game", "Opp Shots/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_shots_on_goal", "Opp SOG", UNIT_INT, LO, family="opponent"),
    _m("opponent_shots_on_goal_per_game", "Opp SOG/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_two_point_shots", "Opp 2PT Shots", UNIT_INT, LO, family="opponent",
       definition="Two-point attempts conceded. High volume can mean a defence "
                  "that concedes long looks, or simply an opponent that shoots from range."),
    _m("opponent_two_point_shots_per_game", "Opp 2PT Shots/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_two_point_shots_on_goal", "Opp 2PT SOG", UNIT_INT, LO, family="opponent"),
    _m("opponent_two_point_shots_on_goal_per_game", "Opp 2PT SOG/G", UNIT_NUM2, LO,
       family="opponent"),
    _m("opponent_goal_pct", "Opp Shot %", UNIT_PCT01, LO, family="opponent",
       definition="Opponent goals divided by opponent shots — shot quality the defense concedes."),
    _m("opponent_sog_rate", "Opp SOG Rate", UNIT_PCT01, LO, family="opponent"),
    _m("opponent_sog_goal_pct", "Opp Goals/SOG", UNIT_PCT01, LO, family="opponent"),
    _m("opponent_goals_per_shot", "Opp Goals/Shot", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_turnovers", "Opp Turnovers", UNIT_INT, HI, family="opponent"),
    _m("opponent_turnovers_per_game", "Opp TO/G", UNIT_NUM2, HI, family="opponent"),
    _m("opponent_touches", "Opp Touches", UNIT_INT, LO, family="opponent"),
    _m("opponent_touches_per_game", "Opp Touches/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_total_passes", "Opp Passes", UNIT_INT, LO, family="opponent"),
    _m("opponent_total_passes_per_game", "Opp Passes/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_ground_balls", "Opp GB", UNIT_INT, LO, family="opponent"),
    _m("opponent_ground_balls_per_game", "Opp GB/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_caused_turnovers", "Opp CT", UNIT_INT, LO, family="opponent"),
    _m("opponent_caused_turnovers_per_game", "Opp CT/G", UNIT_NUM2, LO, family="opponent"),
    _m("opponent_offensive_sequence_proxy", "Opp Off Sequences", UNIT_INT, family="opponent"),
    _m("opponent_offensive_sequence_proxy_per_game", "Opp Off Sequences/G", UNIT_NUM2, family="opponent"),
    _m("opponent_scores_per_offensive_sequence_proxy", "Scores Allowed/Sequence", UNIT_NUM2, LO, family="opponent",
       definition="Scores conceded per opponent offensive sequence — pace-independent defensive efficiency."),
    _m("opponent_time_in_possession", "Opp Possession Time", UNIT_SEC_TOTAL, family="opponent"),
    _m("opponent_time_in_possession_per_game", "Opp Possession/G", UNIT_SEC, family="opponent"),
    # `team_`-prefixed own-team stats in the defensive marts. Family is
    # "possession", not "opponent" — these are OUR possessions, sitting in a
    # defensive table for comparison against the opponent_* twin.
    _m("team_time_in_possession", "Possession Time", UNIT_SEC_TOTAL, family="possession"),
    _m("team_time_in_possession_per_game", "Possession/G", UNIT_SEC, family="possession"),

    # Per-game result flags from team_game_opponent_context. Summing them gives
    # W/L, which pages 09 and 12 each rebuilt with their own inline CTE.
    _m("win_flag", "Win", UNIT_INT, HI, family="results",
       definition="1 if the team won this game, else 0. Sum for a win total."),
    _m("loss_flag", "Loss", UNIT_INT, LO, family="results",
       definition="1 if the team lost this game, else 0."),
    _m("score_margin", "Margin", UNIT_NUM1, HI, family="results",
       definition="Team scores minus scores allowed. Positive is a win."),
    # Parity columns the mart carries to prove the opponent join is correct.
    # Registered so QA pages label them properly rather than as bare stats.
    _m("opponent_scores_check", "Opp Scores (Parity Check)", UNIT_INT, family="qc",
       definition="Opponent scores as reported on the opponent's own row. Should "
                  "equal Scores Allowed; a mismatch means the opponent join is broken."),
    _m("opponent_goals_check", "Opp Goals (Parity Check)", UNIT_INT, family="qc",
       definition="Opponent goals from the opponent's own row. Should equal Goals Allowed."),

    # ---------- composite scores (all 0–100) ----------
    _m("overall_score", "Overall Score", UNIT_SCORE, HI, family="composite",
       definition="Headline player ranking score, role-aware and peer-adjusted."),
    _m("overall_rank", "Overall Rank", UNIT_INT, LO, family="composite"),
    _m("overall_percentile", "Overall %ile", UNIT_PCT100, HI, family="composite"),
    _m("view_rank", "Rank", UNIT_INT, LO, family="composite"),
    _m("position_rank", "Position Rank", UNIT_INT, LO, family="composite"),
    _m("position_percentile", "Position %ile", UNIT_PCT100, HI, family="composite"),
    _m("base_impact_score", "Base Impact", UNIT_SCORE, HI, family="composite"),
    _m("offense_rps", "Offense Score", UNIT_SCORE, HI, family="composite",
       definition="Role Performance Score for offensive production."),
    _m("defense_rps", "Defense Score", UNIT_SCORE, HI, family="composite",
       definition="Role Performance Score for defensive production."),
    _m("faceoff_rps", "Faceoff Score", UNIT_SCORE, HI, family="composite"),
    _m("goalie_rps", "Goalie Score", UNIT_SCORE, HI, family="composite"),
    # Named like a percentage but is not one — a clamped scoring input (17–85,
    # capped at 85), correlating -0.74 with actual save_pct. See UNIT NOTES.
    _m("goalie_save_pct_for_overall", "Goalie Save Input", UNIT_NUM1, HI,
       family="composite",
       definition="Transformed save-percentage input to the goalie composite, "
                  "clamped to a 0-85 band. Not a save percentage — read Save % instead."),
    _m("peer_standing_score", "Peer Standing", UNIT_SCORE, HI, family="composite",
       definition="How far a player stands out from others in the same role group."),
    _m("pss", "Peer Standing", UNIT_SCORE, HI, family="composite"),
    _m("cross_role_impact", "Cross-Role Impact", UNIT_SCORE, HI, family="composite",
       definition="Contribution outside the player's primary role."),
    _m("role_primary_score", "Role Score", UNIT_SCORE, HI, family="composite"),
    _m("role_primary_score_normalized", "Role Score (Cross-Role)", UNIT_SCORE, HI,
       family="composite",
       definition="Role Performance rescaled within the player's role so roles are "
                  "comparable — 50 is that role's average. This is the version the "
                  "Overall Score blends; the per-role Offense/Defense/Faceoff/Goalie "
                  "scores are the unscaled originals."),
    _m("role_primary_percentile", "Role %ile", UNIT_PCT100, HI, family="composite"),
    _m("role_context_value_score", "Role Context Value", UNIT_SCORE, HI, family="composite"),
    _m("role_context_rank", "Role Rank", UNIT_INT, LO, family="composite"),
    _m("role_context_percentile", "Role Context %ile", UNIT_PCT100, HI, family="composite"),
    _m("role_separation_score", "Peer Separation", UNIT_SCORE, HI, family="composite"),
    _m("role_adjusted_z", "Peer Separation Z", UNIT_NUM2, HI, family="composite"),
    _m("role_robust_z", "Raw Peer Z", UNIT_NUM2, HI, family="composite"),
    _m("role_value_tier", "Role Tier", UNIT_TEXT, family="composite"),
    _m("role_group_size", "Peer Group Size", UNIT_INT, family="composite"),
    # Stored 0–100, and currently a constant 100.0 for every row in
    # marts.player_ranking_profiles — the mart computes it but never varies it.
    # Registered as a score so it at least renders honestly ("100.0", not
    # "10000.0%"); pages should not lean on it until the mart populates it.
    _m("role_reliability", "Peer Reliability", UNIT_SCORE, HI, family="composite",
       definition="Confidence in the peer-group comparison. Currently constant "
                  "across all players, so it does not yet discriminate."),
    _m("scoring_value_score", "Scoring Value", UNIT_SCORE, HI, family="composite"),
    _m("goal_value_score", "Scoring Value", UNIT_SCORE, HI, family="composite"),
    _m("playmaking_value_score", "Playmaking Value", UNIT_SCORE, HI, family="composite"),
    _m("usage_score", "Usage Value", UNIT_SCORE, HI, family="composite"),
    _m("ground_ball_score", "Ground Ball Value", UNIT_SCORE, HI, family="composite"),
    _m("turnover_security_score", "Ball Security", UNIT_SCORE, HI, family="composite"),
    # Team style. Each score is min-max scaled across the teams in its own context,
    # so 0 and 100 are that context's worst and best rather than fixed anchors —
    # which is why these compare teams within a season and not across seasons.
    _m("team_style_overall_score", "Overall Style", UNIT_SCORE, HI, family="composite",
       definition="Weighted blend of the six style scores. A summary of playing "
                  "identity, not a quality rating."),
    _m("offensive_volume_score", "Offensive Volume", UNIT_SCORE, HI, family="composite",
       definition="How much offence a team generates: scores, shots, touches and "
                  "offensive sequences per game. Volume, not efficiency."),
    _m("offensive_efficiency_score", "Offensive Efficiency", UNIT_SCORE, HI,
       family="composite",
       definition="How much a team gets from its chances: scoring, shooting "
                  "percentage, low turnovers and score margin."),
    _m("ball_movement_score", "Ball Movement", UNIT_SCORE, HI, family="composite",
       definition="How much the ball moves before a shot: assists, passes and "
                  "touches per game."),
    _m("possession_control_score", "Possession Control", UNIT_SCORE, HI,
       family="composite",
       definition="How well a team holds the ball: touches, possession time and "
                  "faceoff win rate."),
    _m("defensive_suppression_score", "Defensive Suppression", UNIT_SCORE, HI,
       family="composite",
       definition="How much a team denies: scores and shots allowed, opponent "
                  "shooting percentage and goaltending."),
    # No direction: fast is not better than slow, it is a different way to play.
    _m("pace_tempo_score", "Pace / Tempo", UNIT_SCORE, family="composite",
       definition="How fast a team plays — shots, touches, sequences and "
                  "possession time. Higher is faster, not better."),
    _m("profile_rank", "Style Rank", UNIT_INT, LO, family="composite"),
    _m("profile_percentile", "Style %ile", UNIT_PCT100, HI, family="composite"),
    _m("pace_label", "Pace", UNIT_TEXT, family="composite",
       definition="Plain-English band for Pace / Tempo, from High Tempo to Very Slow."),
    _m("offensive_profile_label", "Offensive Profile", UNIT_TEXT, family="composite",
       definition="Plain-English band for Offensive Efficiency, from Elite Offense "
                  "to Poor Offense."),
    _m("defensive_profile_label", "Defensive Profile", UNIT_TEXT, family="composite",
       definition="Plain-English band for Defensive Suppression, from Elite Defense "
                  "to Vulnerable Defense."),
    _m("possession_profile_label", "Possession Profile", UNIT_TEXT, family="composite",
       definition="Plain-English band for Possession Control, from Elite Possession "
                  "to Poor Possession."),
    _m("style_summary", "Style Summary", UNIT_TEXT, family="composite",
       definition="The four style labels in one line: pace, offence, defence, "
                  "possession."),

    # ---------- context / meta ----------
    _m("ranking_context", "Context", UNIT_TEXT, family="meta"),
    _m("ranking_context_type", "Context Type", UNIT_TEXT, family="meta"),
    _m("ranking_context_max_games", "Max GP", UNIT_INT, family="meta"),
    _m("ranking_formula_version", "Formula Version", UNIT_TEXT, family="meta"),
    _m("profile_context", "Context", UNIT_TEXT, family="meta"),
    _m("profile_context_type", "Context Type", UNIT_TEXT, family="meta"),
    _m("min_games_default", "Default Min GP", UNIT_INT, family="meta"),
    _m("eligible_for_default_ranking", "Eligible", UNIT_TEXT, family="meta"),
    _m("sample_size_note", "Sample Note", UNIT_TEXT, family="meta"),
    _m("possession_data_status", "Possession Data", UNIT_TEXT, family="meta"),
    _m("possession_data_note", "Possession Note", UNIT_TEXT, family="meta"),
    # Possession-time bookkeeping. These share the clock column's name but are
    # coverage counters and shares, not durations — see _is_clock_column.
    _m("time_in_possession_available_game", "TOP Available", UNIT_INT, family="meta",
       definition="Games in this row's sample that have possession-time data."),
    _m("time_in_possession_available_game_per_game", "TOP Coverage", UNIT_PCT01, HI,
       family="meta",
       definition="Share of the sample's games with possession-time data. Below "
                  "100% means the possession rates rest on partial data."),
    _m("time_in_possession_available_team_rows", "TOP Team Rows", UNIT_INT, family="qc",
       definition="Team-game rows carrying a possession-time value."),
    _m("time_in_possession_raw_nonzero", "TOP Non-Zero Rows", UNIT_INT, family="qc",
       definition="Team-game rows whose raw possession time is greater than zero."),
    # Warehouse row counts (shared/db.startup_counts), shown on Data QA.
    _m("completed_games", "Completed Games", UNIT_INT, family="qc",
       definition="Games with stats in the warehouse. Scheduled games that have "
                  "not been played, or whose stats have not landed, are excluded."),
    _m("player_game_rows", "Player-Game Rows", UNIT_INT, family="qc",
       definition="One row per player per game — the grain every player mart "
                  "aggregates from."),
    _m("team_game_rows", "Team-Game Rows", UNIT_INT, family="qc",
       definition="One row per team per game, so two per completed game."),
    _m("scheduled_games", "Scheduled Games", UNIT_INT, family="qc"),
    _m("stat_available_games", "Stat-Available Games", UNIT_INT, family="qc"),
    _m("coverage_pct", "Stat Coverage", UNIT_PCT01, HI, family="qc",
       definition="Share of a season's scheduled games that have stats loaded."),
    _m("status_display", "Status", UNIT_TEXT, family="meta"),
    _m("event_status_label", "Raw Status", UNIT_TEXT, family="meta"),
    _m("away_team_name", "Away", UNIT_TEXT, family="meta"),
    _m("home_team_name", "Home", UNIT_TEXT, family="meta"),
    _m("away_score", "Away Score", UNIT_INT, family="meta"),
    _m("home_score", "Home Score", UNIT_INT, family="meta"),
    _m("slug", "Slug", UNIT_TEXT, family="meta"),
    _m("check_name", "Check", UNIT_TEXT, family="meta"),
    _m("status", "Status", UNIT_TEXT, family="meta"),
    _m("notes", "Notes", UNIT_TEXT, family="meta"),
    _m("row_type", "Row", UNIT_TEXT, family="meta"),
]

METRICS: dict[str, Metric] = {m.key: m for m in _REGISTRY}


# ============================================================
# PER-100-POSSESSION SUPPORT
# ============================================================
#
# Everything in the app was per-GAME, which conflates "this team is good" with
# "this team plays fast". These are the counting stats worth expressing per 100
# offensive sequences; see analysis.add_per_100_possessions().

PER_100_CANDIDATES = [
    # own offense
    "scores", "goals", "assists", "shots", "shots_on_goal",
    "one_point_goals", "two_point_goals", "two_point_shots",
    "turnovers", "caused_turnovers", "ground_balls", "saves",
    "touches", "total_passes", "shot_clock_expirations", "power_play_goals",
    # conceded — defensive-mart spelling
    "scores_allowed", "goals_allowed", "assists_allowed",
    "two_point_goals_allowed",
    # conceded — offensive-mart spelling (team_season_stats et al.)
    "scores_against", "goals_against", "two_point_goals_against",
    # opponent activity
    "opponent_shots", "opponent_shots_on_goal", "opponent_turnovers",
    "opponent_ground_balls", "opponent_caused_turnovers",
    "opponent_two_point_shots",
]

PER_100_SUFFIX = "_per_100_poss"


def per_100_key(base: str) -> str:
    return f"{base}{PER_100_SUFFIX}"


def _register_per_100(base: str) -> None:
    """Derive a per-100-possession Metric from its base counting stat."""
    base_metric = METRICS.get(base)
    if base_metric is None:
        return
    key = per_100_key(base)
    label = f"{base_metric.display_short}/100 Poss"
    METRICS[key] = Metric(
        key=key,
        label=label,
        # One decimal: these land in the 5–35 range, where a second decimal is
        # noise the possession proxy can't support.
        unit=UNIT_NUM1,
        direction=base_metric.direction,
        definition=(
            f"{base_metric.display_short} per 100 offensive sequences. "
            "Removes pace, so fast and slow teams can be compared directly."
        ),
        family=base_metric.family,
    )


for _base in PER_100_CANDIDATES:
    _register_per_100(_base)


# ============================================================
# INFERENCE FOR UNREGISTERED COLUMNS
# ============================================================

# Suffix/substring rules applied in order. Kept narrow on purpose: a wrong unit
# guess silently misreports a number, which is worse than a plain label.
_LO_HINTS = (
    "_allowed", "turnovers", "goals_against", "scores_against",
    "penalt", "pim", "expirations", "losses", "_lost",
)

# ------------------------------------------------------------
# Alias resolution
# ------------------------------------------------------------
#
# The warehouse spells the same quantity three ways depending on which mart it
# sits in. team_game_opponent_context / team_defense_* prefix a team's OWN stats
# with `team_` so the opponent_* symmetry reads cleanly, and use `_for` where a
# `_against` twin exists:
#
#   marts.team_season_stats            → goals,      caused_turnovers, saves
#   marts.team_game_opponent_context   → team_goals, caused_turnovers_for, saves_for
#
# Rather than duplicate ~30 registry entries per spelling, resolve the alias back
# to its canonical metric and reuse its unit/direction/definition. Only the label
# changes, and only where the distinction matters to the reader.
# (prefix/suffix, label prefix, flip direction). `opponent_` flips direction
# because what is good for the opponent is bad for the team: their faceoff wins
# and clears are HI for them and therefore LO for us, while their turnovers are
# LO for them and HI for us. The opponent_* metrics already registered by hand
# (opponent_shots LO, opponent_turnovers HI) follow the same rule, so inferred
# and registered entries agree.
_ALIAS_PREFIXES = (
    ("team_", "", False),
    ("opponent_", "Opp ", True),
)
_ALIAS_SUFFIXES = (
    ("_for", "", False),
)

# Identifiers, not `team_`/`opponent_`-prefixed stats.
_ALIAS_EXCEPTIONS = {
    "team_id", "team_name", "team_id_raw", "team_name_raw", "team_abbr",
    "team_city", "team_slug",
    "opponent_team_id", "opponent_team_name", "opponent_team_id_raw",
    "opponent_team_name_raw",
}


def _resolve_alias(key: str):
    """
    (canonical_key, label_prefix, flip_direction) for an alias, else None.
    """
    if key in _ALIAS_EXCEPTIONS:
        return None
    for prefix, label_prefix, flip in _ALIAS_PREFIXES:
        if key.startswith(prefix):
            base = key[len(prefix):]
            if base in METRICS:
                return base, label_prefix, flip
    for suffix, label_prefix, flip in _ALIAS_SUFFIXES:
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in METRICS:
                return base, label_prefix, flip
    return None


def _flip(direction: str | None) -> str | None:
    if direction == HI:
        return LO
    if direction == LO:
        return HI
    return None


# Suffixes that turn a clock column into something that is no longer a clock.
# This is a whitelist of endings rather than a blacklist because the warehouse
# hangs a lot of bookkeeping off the possession-time name — `..._pct_raw` is a
# share (0.58), `..._available_team_rows` is a row count (80) — and a blacklist
# rendered both as M:SS ("0:00" and "1:20"). A column only reads as a clock if it
# ends in the base name or in `_raw`/`_total`/`_seconds`/`_per_game`.
_CLOCK_STEMS = ("time_in_possession", "time_on_field", "possession_time")
_CLOCK_TAILS = ("", "_raw", "_total", "_seconds", "_secs", "_per_game")


def _is_clock_column(kl: str) -> bool:
    """True when `kl` holds a number of seconds that should render as M:SS."""
    for stem in _CLOCK_STEMS:
        idx = kl.find(stem)
        if idx < 0:
            continue
        if kl[idx + len(stem):] in _CLOCK_TAILS:
            return True
    return False


def _infer(key: str) -> Metric:
    k = str(key)
    kl = k.lower()

    # Prefer an alias of a registered metric over guessing from the name.
    resolved = _resolve_alias(kl)
    if resolved is not None:
        canonical, label_prefix, flip = resolved
        base = METRICS[canonical]
        return Metric(
            key=k,
            label=f"{label_prefix}{base.label}" if label_prefix else base.label,
            unit=base.unit,
            direction=_flip(base.direction) if flip else base.direction,
            definition=base.definition,
            family="opponent" if label_prefix else base.family,
            short=f"{label_prefix}{base.display_short}" if label_prefix else base.short,
        )

    # Composite scores and percentiles from the ranking/style marts are 0–100.
    if kl.endswith("_score") or kl.endswith("_rps"):
        unit = UNIT_SCORE
    elif kl.endswith("_percentile"):
        unit = UNIT_PCT100
    elif kl.endswith("_rank"):
        unit = UNIT_INT
    elif _is_clock_column(kl) and kl.endswith("_per_game"):
        unit = UNIT_SEC
    elif _is_clock_column(kl):
        unit = UNIT_SEC_TOTAL
    elif kl.endswith(("_pct", "_rate", "_conversion")) or "_pct_" in kl:
        # Everything not explicitly registered as pct100 is stored 0–1.
        unit = UNIT_PCT01
    elif kl.endswith("_per_game") or kl.endswith(PER_100_SUFFIX):
        unit = UNIT_NUM2
    elif kl.endswith(("_label", "_name", "_note", "_summary", "_tier", "_status", "_type", "_display", "_id")):
        unit = UNIT_TEXT
    else:
        # Unknown shape: the name says nothing about whether this is a count or a
        # rate, so let the value decide rather than forcing ".00" onto integers.
        unit = UNIT_AUTO

    direction = None
    if unit not in {UNIT_TEXT}:
        if any(h in kl for h in _LO_HINTS):
            direction = LO

    label = METRIC_LABEL_OVERRIDES.get(k) or k.replace("_", " ").title()
    return Metric(key=k, label=label, unit=unit, direction=direction, family="meta")


# Small set of label-only fixes for columns not worth a full registry entry.
METRIC_LABEL_OVERRIDES = {
    "time_in_possession_per_game_mmss": "Possession/G",
    "time_in_possession_display": "Possession Time",
    "time_in_possession_pct_display": "Possession %",
    "time_in_possession_total_display": "Possession Time",
    "time_in_possession_per_game_display": "Possession/G",
    "save_pct_display_pct": "Save %",
    "possession_pg": "Possession/G",
}

_INFERRED_CACHE: dict[str, Metric] = {}


def describe(key: str) -> Metric:
    """Return the Metric for `key`, inferring (and caching) if unregistered."""
    m = METRICS.get(key)
    if m is not None:
        return m
    m = _INFERRED_CACHE.get(key)
    if m is None:
        m = _infer(key)
        _INFERRED_CACHE[key] = m
    return m


# ============================================================
# PUBLIC ACCESSORS
# ============================================================

def label(key: str) -> str:
    return describe(key).label


def short_label(key: str) -> str:
    return describe(key).display_short


def unit(key: str) -> str:
    return describe(key).unit


def definition(key: str) -> str:
    return describe(key).definition


def family(key: str) -> str:
    return describe(key).family


def direction(key: str) -> str | None:
    return describe(key).direction


def higher_is_better(key: str, default: bool = True) -> bool:
    """
    Sort direction for `key`. This replaces the four disagreeing
    `lower_is_better` sets that used to live in pages 06, 10 and 15.
    """
    d = describe(key).direction
    if d == LO:
        return False
    if d == HI:
        return True
    return default


def is_lower_better(key: str) -> bool:
    return describe(key).direction == LO


def direction_note(key: str) -> str:
    d = describe(key).direction
    if d == LO:
        return "Lower is better"
    if d == HI:
        return "Higher is better"
    return "Neither direction is inherently better"


def is_percent(key: str) -> bool:
    return describe(key).is_pct


def is_text(key: str) -> bool:
    return describe(key).unit == UNIT_TEXT


# Columns whose scale differs between the `clean` and `marts` schemas. The
# registry declares the marts convention because that is what the pages read
# almost everywhere; a page reading clean.* game rows passes clean_schema=True.
_CLEAN_SCHEMA_UNITS = {
    "clean_save_pct": UNIT_PCT01,   # clean.*: 0–1;  marts.*: 0–100
}


def clean_pct_scale(key: str) -> str:
    """
    Scale of `key` in the `clean` schema, which does not always match `marts`.
    `clean.player_game_stats.clean_save_pct` is 0–1 while the marts version is
    0–100; callers reading game-level rows directly need to know that.
    """
    return _CLEAN_SCHEMA_UNITS.get(key, unit(key))


def unit_for(key: str, clean_schema: bool = False) -> str:
    """Unit for `key`, honouring the clean-schema scale when reading clean.*."""
    return clean_pct_scale(key) if clean_schema else unit(key)


def format_in_schema(key: str, value, clean_schema: bool = False,
                     dash: str = "—") -> str:
    return format_as(unit_for(key, clean_schema), value, dash=dash)


# ============================================================
# FORMATTING
# ============================================================

def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    return False


def format_seconds(value, total: bool = False, dash: str = "—") -> str:
    if _is_missing(value):
        return dash
    try:
        seconds = int(round(float(value)))
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if total and hours > 0:
        return f"{sign}{hours}:{minutes:02d}:{secs:02d}"
    if total:
        return f"{sign}{minutes}:{secs:02d}"
    return f"{sign}{minutes}:{secs:02d}"


def format_value(key: str, value, dash: str = "—") -> str:
    """Format `value` according to the registered unit for `key`."""
    return format_as(unit(key), value, dash=dash)


def format_as(unit_code: str, value, dash: str = "—") -> str:
    if _is_missing(value):
        return dash
    if unit_code == UNIT_TEXT:
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return dash

    if unit_code == UNIT_PCT01:
        return f"{v * 100:.1f}%"
    if unit_code == UNIT_PCT100:
        return f"{v:.1f}%"
    if unit_code == UNIT_SCORE:
        return f"{v:.1f}"
    if unit_code == UNIT_SEC:
        return format_seconds(v, total=False, dash=dash)
    if unit_code == UNIT_SEC_TOTAL:
        return format_seconds(v, total=True, dash=dash)
    if unit_code == UNIT_INT:
        return f"{int(round(v)):,}"
    if unit_code == UNIT_NUM1:
        return f"{v:,.1f}"
    if unit_code == UNIT_AUTO:
        # Unregistered column — could be a count or a rate, so adapt per value.
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v)):,}"
        return f"{v:,.2f}"
    # UNIT_NUM2 — always two decimals, including for whole numbers. Collapsing
    # 12.0 to "12" makes a column of rates ragged ("12" beside "11.10") and
    # implies a precision difference that isn't there. Genuine counts are
    # UNIT_INT, so nothing that should be integral lands here.
    return f"{v:,.2f}"


def formatter_for(key: str, dash: str = "—", clean_schema: bool = False):
    """A one-arg formatter for `key`, for pandas Styler.format / Series.apply."""
    u = unit_for(key, clean_schema)
    return lambda v: format_as(u, v, dash=dash)


def plotly_tickformat(key: str) -> str | None:
    """
    Axis tickformat for `key`, or None to let Plotly decide.

    The old standardize_chart() forced ".2f" on every axis, so a goal count
    rendered as "12.00". Percentages stored 0–1 get Plotly's own percent format.
    """
    u = unit(key)
    if u == UNIT_PCT01:
        return ".1%"
    if u in {UNIT_PCT100, UNIT_SCORE, UNIT_NUM1}:
        return ".1f"
    if u == UNIT_INT:
        return ",d"
    return None


def plotly_texttemplate(key: str) -> str:
    """Bar-label template for `key`. Counts must not show decimals."""
    u = unit(key)
    if u == UNIT_PCT01:
        return "%{text:.1%}"
    if u in {UNIT_PCT100, UNIT_SCORE, UNIT_NUM1}:
        return "%{text:.1f}"
    if u == UNIT_INT:
        return "%{text:,d}"
    return "%{text:,.2f}"


def format_series(series: pd.Series, key: str | None = None, dash: str = "—") -> pd.Series:
    k = key if key is not None else series.name
    fn = formatter_for(str(k), dash=dash)
    return series.map(fn)


# ============================================================
# SELECTION HELPERS
# ============================================================

def existing(df: pd.DataFrame, keys: Iterable[str]) -> list[str]:
    """Subset of `keys` present in `df`, order preserved."""
    if df is None or len(df.columns) == 0:
        return []
    cols = set(df.columns)
    return [k for k in keys if k in cols]


def with_data(df: pd.DataFrame, keys: Iterable[str]) -> list[str]:
    """Subset of `keys` present in `df` AND not entirely null."""
    out = []
    for k in existing(df, keys):
        col = df[k]
        if col.notna().any():
            out.append(k)
    return out


def with_values(row, keys: Iterable[str]) -> list[str]:
    """
    Subset of `keys` the single row actually has a value for.

    The row-level counterpart of `with_data`. Profile pages hand a mart row to a
    stat grid and need to know whether a section is worth drawing at all — an
    empty list means every metric in it is absent or null for this row.
    """
    if row is None or not hasattr(row, "index"):
        return []
    present = set(row.index)
    return [k for k in keys
            if k in present and pd.notna(row.get(k))]


def by_family(keys: Iterable[str]) -> dict[str, list[str]]:
    """Group `keys` by family, ordered by FAMILY_ORDER."""
    grouped: dict[str, list[str]] = {}
    for k in keys:
        grouped.setdefault(family(k), []).append(k)
    return {f: grouped[f] for f in FAMILY_ORDER if f in grouped}


def sort_df(df: pd.DataFrame, key: str, ascending: bool | None = None) -> pd.DataFrame:
    """
    Sort `df` by `key` using the registry's direction unless overridden.
    Nulls always sort last, which is what a leaderboard wants.
    """
    if df is None or len(df) == 0 or key not in df.columns:
        return df
    out = df.copy()
    if not is_text(key):
        out[key] = pd.to_numeric(out[key], errors="coerce")
    if ascending is None:
        ascending = is_lower_better(key)
    return out.sort_values(key, ascending=ascending, na_position="last")


def rank_series(df: pd.DataFrame, key: str) -> pd.Series:
    """1-based competition rank of `key`, respecting the registry direction."""
    if df is None or key not in df.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(df[key], errors="coerce")
    return values.rank(ascending=is_lower_better(key), method="min")
