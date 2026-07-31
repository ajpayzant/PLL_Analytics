"""
Specialists — goalies and faceoff men, judged inside their own role.

The sidebar filters are not requested: each section carries its own season and
minimum-volume controls in the main panel.

Five things were wrong or missing, beyond the shared-helper cleanup.

1. Sort direction was a hardcoded four-item set:
       lower_goalie_metrics = {"scores_against", "scores_against_per_game",
                               "goals_against", "goals_against_per_game"}
   SAA — scores against average, the goalie metric where low is the whole point —
   was not in it, so picking SAA sorted descending and printed "Higher is better"
   above a board led by the goalie who conceded most: Colin Kirst first at 96.5
   SAA on a .481 save rate, ahead of Dillon Ward's 81.9 on .556. `saa_per_game`
   had the same problem. The registry already carries direction for every one of
   these keys, so `M.sort_df` and `M.direction_note` now decide.

2. The faceoff board never filtered to faceoff men. It selected on
   `faceoffs >= min_faceoffs` alone, and the default of 20 was the only thing
   keeping runners out. Drop it to catch a short season and the leaderboard is
   led by Chris Merle, an SSDM who took exactly one faceoff in 2026 and won it —
   100%, first in the league. Twenty-six non-FO players took faceoffs in 2026.
   The section now gates on `roles.ROLE_FACEOFF`, with an opt-in for everyone
   else who took a draw, and the minimum applies inside that pool.

3. The faceoff direction control was a manual "Best high / Best low" selectbox.
   Nothing about a faceoff metric is ambiguous — the registry knows FO Lost is
   bad and FO Win % is good — and leaving it to the reader let a mis-set control
   silently invert a leaderboard.

4. `save_pct_display` from `_pll_apply_goalie_save_pct` is `save_pct_calc`. The
   recompute existed because "the provider's own `save_pct` can exceed 100%", and
   that is no longer true anywhere in the warehouse: max raw `save_pct` in
   `clean.player_game_stats` is 1.0 across all 6,756 rows, and `save_pct_calc`
   equals saves ÷ (saves + GA) on all 68 goalie-seasons that faced a shot. The
   page was showing the same number three times in one table — as "Save %", as a
   pre-rendered "Save Pct Display Pct" text column, and as "Save Pct Calc". It
   now reads the mart column and formats it once through the registry.

5. `shots` for a goalie is shots he *took*, not shots he faced: Dillon Ward's
   2026 row is 0 shots against 180 faced. It sat unlabelled in the advanced table
   beside the save columns. Shots faced is now derived and labelled, and `shots`
   is dropped from the goalie views.

Also: the faceoff "advanced metrics" expander dumped all 75 columns of the mart,
including all 17 goalie columns, at a faceoff specialist; and the "Average Save %"
card averaged per-goalie rates, which weights a two-game backup equally with a
seven-game starter — in 2022 that reads .467 against a real league rate of .515.
Both advanced views are now opt-in metric pickers over role-relevant candidates,
and the league card is pooled.

Local logic removed in favour of the shared layer: `_pll_apply_goalie_save_pct`
(see 4 — one derived column, `shots_faced`, survives as `_add_shots_faced`),
`_pll_safe_sort` and both direction controls (`M.sort_df`), `_pll_pct_text` and
`pretty_col`/`fmt_value` formatting (`M.format_value`), the inline plotly import
and figure for clean-vs-messy saves (`ui.safe_bar_chart` over a melt), and
`_pll_select_existing` (`M.existing`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import analysis
from shared import metrics as M
from shared import page as P
from shared import roles
from shared import ui
from shared.db import query_df

ctx = P.init_page(
    "Specialists",
    "Goalies and faceoff men, ranked on the metrics their role is actually "
    "judged on.",
)


# ============================================================
# SHARED
# ============================================================

def _add_shots_faced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `shots_faced` and its per-game form to a goalie frame.

    The provider does not ship shots faced, and a goalie's `shots` column counts
    shots he took — Dillon Ward's 2026 row is 0 shots and 180 faced — so the
    denominator behind every save rate has to be derived. Save % itself is not
    recomputed: `save_pct_calc` in the marts and `save_pct` in `clean` are both
    already saves ÷ (saves + goals against).
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    saves = (pd.to_numeric(out["saves"], errors="coerce") if "saves" in out.columns
             else pd.Series(np.nan, index=out.index))
    if "goals_against" in out.columns:
        against = pd.to_numeric(out["goals_against"], errors="coerce")
    elif "scores_against" in out.columns:
        against = pd.to_numeric(out["scores_against"], errors="coerce")
    else:
        against = pd.Series(np.nan, index=out.index)

    out["shots_faced"] = saves + against
    if "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)
        out["shots_faced_per_game"] = out["shots_faced"] / games
    return out


def _game_label(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological frame with a 'YYYY GN' x-axis label."""
    out = df.sort_values(["season", "game_number"]).copy()
    out["game_label"] = (out["season"].astype(str) + " G"
                         + out["game_number"].astype("Int64").astype(str))
    return out


def _trend_chart(games: pd.DataFrame, metric: str, who: str) -> None:
    """
    Per-game line with a three-game trailing mean.

    The rolling line is what separates a cold run from one bad night; the raw
    per-game series alone cannot show it.
    """
    trend = _game_label(games)
    rolled = analysis.add_rolling(trend, metric, window=3)
    y_cols = [metric]
    roll_col = f"{metric}_roll3"
    if roll_col in rolled.columns:
        y_cols.append(roll_col)
    ui.safe_line_chart(rolled, x_col="game_label", y_cols=y_cols,
                       title=f"{who} — {M.label(metric)} by game")


section_choice = st.radio(
    "Specialist section",
    options=["Goalies", "Faceoff Specialists"],
    horizontal=True,
    key="specialist_section_select",
)

# ============================================================
# GOALIES
# ============================================================

if section_choice == "Goalies":
    controls = st.columns([1.0, 1.0, 1.6])

    season = controls[0].selectbox(
        "Season",
        options=ctx.seasons,
        index=P.default_index(ctx.seasons, P.selected_season(), fallback=-1),
        key="goalie_season",
    )
    if season is not None:
        P.select_season(season)

    min_games = int(controls[1].number_input(
        "Minimum games", min_value=1, max_value=20, value=1, step=1,
        key="goalie_min_games",
        help="A one-game goalie can top a rate leaderboard on three saves. Raise "
             "this to see the season's actual starters.",
    ))

    # Position via roles.py, not `position = 'G' OR lower(position_name) LIKE
    # '%goalie%'`. The two agree in this warehouse — all 73 goalie-seasons carry
    # both — but the taxonomy belongs where SSDM and LSM are also settled.
    goalie_positions = roles.positions_for_role(roles.ROLE_GOALIE)
    placeholders = ", ".join("?" for _ in goalie_positions)
    goalies = query_df(f"""
        SELECT * FROM marts.player_season_stats
        WHERE season = ?
          AND games >= ?
          AND position IN ({placeholders})
        ORDER BY games DESC
    """, [season, min_games, *goalie_positions])

    if len(goalies) == 0:
        st.info(f"No goalie seasons in {season} with at least {min_games} game(s).")
        st.stop()

    goalies = _add_shots_faced(goalies)

    metric = ui.metric_selectbox(
        "Rank goalies by",
        options=M.with_data(goalies, [
            "save_pct_calc", "saves", "saves_per_game", "shots_faced",
            "shots_faced_per_game", "goals_against", "goals_against_per_game",
            "saa", "saa_per_game", "clean_save_pct", "clean_save_rate",
            "clean_saves", "messy_saves", "scores_against",
            "scores_against_per_game", "two_point_goals_against",
        ]),
        key="goalie_metric",
        default="save_pct_calc",
        container=controls[2],
        help="Direction comes from the metric registry, so SAA and goals against "
             "sort low-first without the reader setting anything.",
    )

    # M.sort_df reads direction from the registry, so SAA sorts ascending and the
    # caption under the picker reads "Lower is better" — instead of the page
    # asserting "Higher is better" over every metric outside a four-item set.
    ranked = M.sort_df(goalies, metric) if metric else goalies

    # Pooled, not the mean of per-goalie rates: averaging rates weights a
    # two-game backup equally with a seven-game starter.
    saves_total = pd.to_numeric(goalies.get("saves"), errors="coerce").sum()
    faced_total = pd.to_numeric(goalies.get("shots_faced"), errors="coerce").sum()
    league_rate = saves_total / faced_total if faced_total else np.nan

    cards = st.columns(4)
    with cards[0]:
        ui.stat_card("Goalies", f"{len(goalies):,}",
                     sub=f"{min_games}+ games in {season}")
    with cards[1]:
        leader = ranked.iloc[0] if len(ranked) else None
        ui.stat_card(
            f"Leads {M.label(metric)}" if metric else "Leader",
            str(leader.get("full_name", "—")) if leader is not None else "—",
            sub=(M.format_value(metric, leader.get(metric))
                 if (leader is not None and metric) else None),
        )
    with cards[2]:
        ui.stat_card("League Save %", M.format_value("save_pct_calc", league_rate),
                     sub="pooled across all saves and goals against")
    with cards[3]:
        ui.stat_card("Shots Faced", M.format_value("shots_faced", faced_total),
                     sub="derived: saves + goals against")

    if metric:
        ui.safe_bar_chart(
            # Reversed so rank 1 sits at the top of the horizontal bar.
            ranked.head(15).iloc[::-1],
            x_col="full_name", y_col=metric, color_col="teams",
            title=f"{season} goalies — {M.label(metric)}",
            orientation="h",
        )

    ui.section("Goalie leaders",
               "Save rate, volume and save quality. Shots faced is derived from "
               "saves plus goals against; the provider does not ship it.")
    ui.display_table(
        ranked[M.existing(ranked, [
            "full_name", "position", "teams", "games", "save_pct_calc",
            "saves", "shots_faced", "goals_against", "saa",
            "saves_per_game", "goals_against_per_game", "shots_faced_per_game",
            "clean_saves", "messy_saves", "clean_save_pct", "clean_save_rate",
        ])],
        height=420,
        highlight=metric,
    )
    ui.definition_caption(["save_pct_calc", "saa", "clean_save_pct",
                           "clean_save_rate", "shots_faced"])
    ui.download_csv(ranked, f"pll_goalies_{season}.csv")

    if "clean_save_pct" in goalies.columns and goalies["clean_save_pct"].notna().any():
        ui.section("Save quality",
                   "Clean Save Share is the proportion of a goalie's saves that "
                   "were controlled stops rather than scramble or rebound saves.")
        ui.safe_bar_chart(
            M.sort_df(goalies, "clean_save_pct").head(15).iloc[::-1],
            x_col="full_name", y_col="clean_save_pct", color_col="teams",
            title=f"{season} goalies — {M.label('clean_save_pct')}",
            orientation="h",
        )

    with st.expander("More goalie metrics", expanded=False):
        # An opt-in picker over goalie-relevant keys, rather than the fixed
        # 25-column dump that repeated Save % three times and included `shots`,
        # which for a goalie means shots he took.
        goalie_candidates = [
            c for c in roles.relevant_metrics("G", list(goalies.columns))
            if c not in {"shots", "shots_on_goal", "shots_per_game",
                         "shots_on_goal_per_game", "player_id", "full_name",
                         "teams", "team_names", "games", "season", "position",
                         "position_name"}
        ]
        extra = ui.family_metric_picker(
            goalies, goalie_candidates,
            key="goalie_extra_metrics",
            label="Goalie metrics",
            default=["save_pct_calc", "saves", "shots_faced", "goals_against",
                     "saa", "clean_save_pct"],
        )
        if extra:
            ui.display_table(
                goalies[M.existing(goalies, ["full_name", "teams", "games"] + extra)],
                height=360,
            )

    # ------------------------------------------------------------------
    # ONE GOALIE
    # ------------------------------------------------------------------

    ui.section("Goalie explorer",
               "One goalie's game log, with league rank on the season metrics "
               "that define the position.")

    names = goalies["full_name"].dropna().unique().tolist()
    selected = st.selectbox(
        "Select goalie", options=names,
        index=P.default_index(names, P.selected_player()),
        key="selected_goalie",
    )

    if selected:
        P.select_player(selected)
        row = goalies[goalies["full_name"] == selected].iloc[0]
        player_id = row["player_id"]

        ui.profile_header(
            selected,
            f"{season} · {row.get('teams', '—')} · "
            f"{M.format_value('games', row.get('games'))} games",
        )
        # context=goalies gives each card its rank within the season's goalie
        # pool, which is what turns a save rate into a judgement.
        ui.metric_grid(
            row, roles.headline_metrics("G"), columns=3,
            context=goalies, row_index=row.name,
        )

        games = query_df("""
            SELECT season, game_number, game_date_utc, team_name,
                   opponent_team_name, is_home, saves, clean_saves, messy_saves,
                   goals_against, scores_against, saa, save_pct, clean_save_pct,
                   touches, total_passes
            FROM clean.player_game_stats
            WHERE player_id = ?
            ORDER BY season DESC, game_number DESC
        """, [player_id])
        games = _add_shots_faced(games)

        if len(games) == 0:
            st.info("No game rows for this goalie.")
        else:
            saves_sum = pd.to_numeric(games.get("saves"), errors="coerce").sum()
            clean_sum = pd.to_numeric(games.get("clean_saves"), errors="coerce").sum()
            messy_sum = pd.to_numeric(games.get("messy_saves"), errors="coerce").sum()

            totals = st.columns(4)
            with totals[0]:
                ui.stat_card("Career Saves", M.format_value("saves", saves_sum),
                             sub=f"{len(games)} games, all seasons")
            with totals[1]:
                ui.stat_card("Clean Saves", M.format_value("clean_saves", clean_sum))
            with totals[2]:
                ui.stat_card("Messy Saves", M.format_value("messy_saves", messy_sum))
            with totals[3]:
                share = clean_sum / saves_sum if saves_sum else np.nan
                ui.stat_card("Clean Save Share",
                             M.format_value("clean_save_rate", share),
                             sub="clean ÷ all saves")

            # clean_schema: clean.player_game_stats stores clean_save_pct on a
            # 0–1 scale where every mart stores it 0–100.
            ui.display_table(
                games[M.existing(games, [
                    "season", "game_number", "game_date_utc", "team_name",
                    "opponent_team_name", "is_home", "saves", "goals_against",
                    "scores_against", "shots_faced", "save_pct", "clean_saves",
                    "messy_saves", "clean_save_pct", "saa", "touches",
                    "total_passes",
                ])],
                height=360,
                clean_schema=True,
            )
            ui.download_csv(
                games, f"{str(selected).replace(' ', '_').lower()}_goalie_log.csv")

            trend_metric = ui.metric_selectbox(
                "Game trend metric",
                options=M.with_data(games, [
                    "save_pct", "saves", "shots_faced", "goals_against",
                    "scores_against", "clean_saves", "messy_saves", "saa",
                ]),
                key="goalie_game_metric",
                default="save_pct",
            )
            if trend_metric:
                _trend_chart(games, trend_metric, selected)

            if {"clean_saves", "messy_saves"}.issubset(games.columns) and \
                    games["clean_saves"].notna().any():
                stacked = _game_label(games)[["game_label", "clean_saves",
                                              "messy_saves"]]
                stacked = stacked.dropna(subset=["clean_saves", "messy_saves"],
                                         how="all")
                if len(stacked):
                    melted = stacked.melt(
                        id_vars="game_label",
                        value_vars=["clean_saves", "messy_saves"],
                        var_name="save_type", value_name="saves")
                    melted["save_type"] = melted["save_type"].map(
                        {"clean_saves": M.label("clean_saves"),
                         "messy_saves": M.label("messy_saves")})
                    ui.safe_bar_chart(
                        melted, x_col="game_label", y_col="saves",
                        color_col="save_type",
                        title=f"{selected} — save quality by game",
                        show_labels=False,
                    )

# ============================================================
# FACEOFF SPECIALISTS
# ============================================================

else:
    controls = st.columns([1.0, 1.0, 1.6])

    season = controls[0].selectbox(
        "Season",
        options=ctx.seasons,
        index=P.default_index(ctx.seasons, P.selected_season(), fallback=-1),
        key="faceoff_season",
    )
    if season is not None:
        P.select_season(season)

    min_faceoffs = int(controls[1].number_input(
        "Minimum faceoffs", min_value=1, max_value=500, value=20, step=5,
        key="min_faceoffs",
    ))

    # The old query filtered on `faceoffs >= min_faceoffs` and nothing else, so
    # the default of 20 was the only thing keeping runners off a faceoff
    # leaderboard. 26 non-FO players took a draw in 2026; at a minimum of 1 the
    # board is led by an SSDM who went 1-for-1.
    include_runners = st.checkbox(
        "Include non-specialists who took faceoffs",
        value=False,
        key="faceoff_include_runners",
        help="Off by default: a midfielder who wins his only draw of the season "
             "is a 100% winner and not a faceoff leader.",
    )

    faceoff_positions = roles.positions_for_role(roles.ROLE_FACEOFF)
    placeholders = ", ".join("?" for _ in faceoff_positions)
    role_clause = "" if include_runners else f"AND position IN ({placeholders})"
    params = [season, min_faceoffs]
    if not include_runners:
        params.extend(faceoff_positions)

    faceoff = query_df(f"""
        SELECT * FROM marts.player_season_stats
        WHERE season = ?
          AND faceoffs >= ?
          {role_clause}
        ORDER BY faceoff_pct_calc DESC NULLS LAST
    """, params)

    if len(faceoff) == 0:
        st.info(f"No players in {season} with at least {min_faceoffs} faceoffs.")
        st.stop()

    faceoff = roles.add_role_column(faceoff)

    metric = ui.metric_selectbox(
        "Rank by",
        options=M.with_data(faceoff, [
            "faceoff_pct_calc", "faceoffs_won", "faceoffs", "faceoffs_lost",
            "faceoffs_won_per_game", "faceoffs_per_game", "ground_balls",
            "ground_balls_per_game", "turnovers_per_game", "points",
            "points_per_game", "touches", "touches_per_game",
        ]),
        key="faceoff_metric",
        default="faceoff_pct_calc",
        container=controls[2],
        help="Direction comes from the registry: FO Lost sorts low-first, FO "
             "Win % high-first. The old page asked the reader to set this.",
    )

    # Registry direction, replacing the "Best high / Best low" selectbox that let
    # a mis-set control invert the board with nothing on screen to flag it.
    ranked = M.sort_df(faceoff, metric) if metric else faceoff

    won_total = pd.to_numeric(faceoff.get("faceoffs_won"), errors="coerce").sum()
    taken_total = pd.to_numeric(faceoff.get("faceoffs"), errors="coerce").sum()

    cards = st.columns(4)
    with cards[0]:
        ui.stat_card("Players", f"{len(faceoff):,}",
                     sub=f"{min_faceoffs}+ faceoffs in {season}")
    with cards[1]:
        leader = ranked.iloc[0] if len(ranked) else None
        ui.stat_card(
            f"Leads {M.label(metric)}" if metric else "Leader",
            str(leader.get("full_name", "—")) if leader is not None else "—",
            sub=(M.format_value(metric, leader.get(metric))
                 if (leader is not None and metric) else None),
        )
    with cards[2]:
        ui.stat_card("Faceoffs Taken", M.format_value("faceoffs", taken_total))
    with cards[3]:
        ui.stat_card("Pool Win %",
                     M.format_value("faceoff_pct_calc",
                                    won_total / taken_total if taken_total else np.nan),
                     sub="pooled across the players shown")

    if include_runners and "role_group" in faceoff.columns:
        non_specialists = int((faceoff["role_group"] != roles.ROLE_FACEOFF).sum())
        if non_specialists:
            ui.note_box(
                "Non-specialists included",
                f"{non_specialists} of {len(faceoff)} players shown are not "
                "faceoff specialists. On low volume their win rate says more "
                "about which draws they were sent out for than about their work "
                "at the X.",
            )

    if metric:
        ui.safe_bar_chart(
            ranked.head(15).iloc[::-1],
            x_col="full_name", y_col=metric, color_col="teams",
            title=f"{season} faceoffs — {M.label(metric)}",
            orientation="h",
        )

    ui.section("Faceoff leaders",
               "Win rate and volume at the X, with the ground balls and "
               "possession work that follow from it.")
    ui.display_table(
        ranked[M.existing(ranked, [
            "full_name", "position", "teams", "games", "faceoff_pct_calc",
            "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoffs_per_game",
            "faceoffs_won_per_game", "ground_balls", "ground_balls_per_game",
            "turnovers", "caused_turnovers", "points", "touches",
        ])],
        height=420,
        highlight=metric,
    )
    ui.definition_caption(["faceoff_pct_calc", "faceoffs_won_per_game",
                           "ground_balls_per_game"])
    ui.download_csv(ranked, f"pll_faceoffs_{season}.csv")

    if {"faceoff_pct_calc", "faceoffs"}.issubset(faceoff.columns):
        ui.section("Win rate against volume",
                   "Quadrants split at the median. The top-right corner is the "
                   "faceoff man a team can lean on: a high rate on real volume.")
        ui.safe_scatter(
            faceoff, x_col="faceoffs", y_col="faceoff_pct_calc",
            color_col="teams", hover_col="full_name",
            title=f"{season} — faceoff win rate against draws taken",
            quadrants=True,
        )

    with st.expander("More faceoff metrics", expanded=False):
        # Opt-in, replacing a dump of all 75 mart columns — which put all 17
        # goalie columns in front of a faceoff specialist.
        faceoff_candidates = [
            c for c in roles.relevant_metrics("FO", list(faceoff.columns))
            if c not in {"player_id", "full_name", "teams", "team_names",
                         "games", "season", "position", "position_name",
                         "role_group"}
        ]
        extra = ui.family_metric_picker(
            faceoff, faceoff_candidates,
            key="faceoff_extra_metrics",
            label="Faceoff metrics",
            default=["faceoff_pct_calc", "faceoffs_won", "faceoffs",
                     "ground_balls", "turnovers", "points"],
        )
        if extra:
            ui.display_table(
                faceoff[M.existing(faceoff, ["full_name", "teams", "games"] + extra)],
                height=360,
            )

    # ------------------------------------------------------------------
    # ONE FACEOFF MAN
    # ------------------------------------------------------------------

    ui.section("Faceoff explorer",
               "One player's game log, with league rank on the season metrics "
               "that define the role.")

    names = faceoff["full_name"].dropna().unique().tolist()
    selected = st.selectbox(
        "Select player", options=names,
        index=P.default_index(names, P.selected_player()),
        key="selected_faceoff_player",
    )

    if selected:
        P.select_player(selected)
        row = faceoff[faceoff["full_name"] == selected].iloc[0]
        player_id = row["player_id"]

        ui.profile_header(
            selected,
            f"{season} · {row.get('teams', '—')} · "
            f"{M.format_value('games', row.get('games'))} games",
        )
        ui.metric_grid(
            row, roles.headline_metrics(row.get("position")), columns=3,
            context=faceoff, row_index=row.name,
        )

        games = query_df("""
            SELECT season, game_number, game_date_utc, team_name,
                   opponent_team_name, is_home, points, goals, assists,
                   ground_balls, faceoffs_won, faceoffs_lost, faceoffs,
                   faceoff_pct, turnovers, caused_turnovers, touches,
                   total_passes
            FROM clean.player_game_stats
            WHERE player_id = ?
            ORDER BY season DESC, game_number DESC
        """, [player_id])

        if len(games) == 0:
            st.info("No game rows for this player.")
        else:
            ui.display_table(
                games[M.existing(games, [
                    "season", "game_number", "game_date_utc", "team_name",
                    "opponent_team_name", "is_home", "faceoffs_won",
                    "faceoffs_lost", "faceoffs", "faceoff_pct", "ground_balls",
                    "points", "turnovers", "caused_turnovers", "touches",
                ])],
                height=360,
                clean_schema=True,
            )
            ui.download_csv(
                games, f"{str(selected).replace(' ', '_').lower()}_faceoff_log.csv")

            trend_metric = ui.metric_selectbox(
                "Game trend metric",
                options=M.with_data(games, [
                    "faceoff_pct", "faceoffs_won", "faceoffs", "faceoffs_lost",
                    "ground_balls", "points", "turnovers", "touches",
                ]),
                key="faceoff_game_metric",
                default="faceoff_pct",
            )
            if trend_metric:
                _trend_chart(games, trend_metric, selected)

st.divider()
nav = st.columns(4)
with nav[0]:
    P.link_to("players", "Player profiles →")
with nav[1]:
    P.link_to("compare_players", "Compare players →")
with nav[2]:
    P.link_to("rankings", "Player rankings →")
with nav[3]:
    P.link_to("league", "League overview →")
