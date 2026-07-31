"""
Team Profiles — one team, offence and defence, across every context.

The sidebar filters are not requested: the team picker, the context radio and the
game-log controls in the main panel are what drive the queries here.

Four things were wrong or missing.

1. Recent Form read `marts.team_last5_stats`, the league-wide "last five games
   played anywhere" mart, with no season filter. Every row in it is 2026, so
   Atlas's 2022 profile reported 10.6 scores/game from games played in May–July
   2026 under the heading "Last 5"; their actual last five of 2022 was 12.4. The
   game list under it had the same problem — `ORDER BY game_date_utc DESC LIMIT
   5` over the whole game log — so a 2022 profile listed five 2026 opponents.
   Page 07 hit this and was fixed; `team_season_last5_stats` is the scoped mart.

2. Nothing on the page was pace-adjusted. A team's 13.6 scores/game says nothing
   about efficiency until it is divided by the possessions it took to get there,
   and the page had no rank for anything either: every number sat on its own with
   no indication of whether it was first in the league or last. `add_league_context`
   and `metric_grid`'s `context=` hook both existed for this and had no callers.

3. The four-factor summary — offensive efficiency, defensive efficiency, ball
   security, pace — was written in `shared/analysis.py` and called from nowhere.
   It is the one view that answers "is this team good, and at what" in a glance.

4. The defensive section showed eight hardcoded `stat_grid` tuples with labels and
   digit counts set here, disagreeing with the registry. `opponent_goal_pct` was
   labelled "Opp Goal %" with 2 decimals where the registry knows it as a
   percentage; the 2PT-allowed, assists-allowed and ride/clear columns the marts
   ship were not displayed at all.

`stat_grid`'s (label, key, digits, pct) tuples are replaced throughout by
`ui.metric_grid`, which reads all four from the registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import analysis
from shared import metrics as M
from shared import page as P
from shared import ui
from shared.db import query_df, table_exists, _pll_get_table_columns
from shared.ui import (
    stat_card, safe_bar_chart, safe_line_chart, display_table,
    fmt_value, pretty_col, profile_header, add_window_summary_rows, download_csv,
)

ctx = P.init_page(
    "Team Profiles",
    "Team-level career, season, recent-form, game-log and opponent splits.",
)

team_options = ctx.team_names

selected_team = st.selectbox(
    "Select team",
    options=team_options,
    # Honours a team picked on another page.
    index=P.default_index(team_options, P.selected_team()),
    key="team_explorer_select"
)

if selected_team:
    P.select_team(selected_team)
    team_id = ctx.team_id_for(selected_team)

    career = query_df("""
        WITH record AS (
            SELECT
                team_id,
                SUM(CASE WHEN scores > scores_against THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN scores < scores_against THEN 1 ELSE 0 END) AS losses,
                CASE
                    WHEN COUNT(*) > 0
                    THEN SUM(CASE WHEN scores > scores_against THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
                    ELSE NULL
                END AS win_pct
            FROM clean.team_game_stats
            GROUP BY team_id
        )
        SELECT c.*, r.wins, r.losses, r.win_pct
        FROM marts.team_career_stats c
        LEFT JOIN record r ON c.team_id = r.team_id
        WHERE c.team_id = ?
    """, [team_id])

    team_seasons = query_df("""
        SELECT * FROM marts.team_season_stats WHERE team_id = ? ORDER BY season
    """, [team_id])

    available_contexts = ["Career"] + [str(int(x)) for x in team_seasons["season"].dropna().unique().tolist()]

    selected_context = st.radio(
        "Summary context",
        options=available_contexts,
        horizontal=True,
        key=f"team_context_{team_id}"
    )

    if selected_context == "Career":
        summary = career.iloc[0] if len(career) else pd.Series(dtype="object")
        subtitle = f"Multi-year summary | Games: {fmt_value(summary.get('games', np.nan), 0)} | Record: {fmt_value(summary.get('wins', np.nan), 0)}-{fmt_value(summary.get('losses', np.nan), 0)}"
    else:
        season_int = int(selected_context)
        season_df = team_seasons[team_seasons["season"] == season_int]
        summary = season_df.iloc[0] if len(season_df) else pd.Series(dtype="object")
        subtitle = f"{selected_context} Season | Games: {fmt_value(summary.get('games', np.nan), 0)} | Record: {fmt_value(summary.get('wins', np.nan), 0)}-{fmt_value(summary.get('losses', np.nan), 0)}"

    profile_header(selected_team, subtitle)

    # ------------------------------------------------------------------
    # LEAGUE CONTEXT
    #
    # The whole league in the same context as the profile, so every figure below
    # can carry a rank. Without it a page of numbers gives the reader no way to
    # tell 13.6 scores/game apart from a league-leading 13.6 scores/game.
    # ------------------------------------------------------------------

    if selected_context == "Career":
        league = query_df("SELECT * FROM marts.team_career_stats")
        league_defense = (query_df("SELECT * FROM marts.team_defense_career_stats")
                          if table_exists("marts", "team_defense_career_stats")
                          else pd.DataFrame())
        context_label = "career"
    else:
        league = query_df("SELECT * FROM marts.team_season_stats WHERE season = ?",
                          [int(selected_context)])
        league_defense = (
            query_df("SELECT * FROM marts.team_defense_season_stats WHERE season = ?",
                     [int(selected_context)])
            if table_exists("marts", "team_defense_season_stats") else pd.DataFrame()
        )
        context_label = f"{selected_context}"

    if len(league_defense) and "team_id" in league_defense.columns:
        # Only the columns the offensive mart lacks, and none of the defensive
        # mart's `team_*` mirrors of own-team stats: those restate columns already
        # present under their plain names.
        keep = ["team_id"] + [c for c in league_defense.columns
                              if c not in league.columns
                              and not c.startswith("team_") and c != "team_id"]
        league = league.merge(league_defense[keep], on="team_id", how="left")

    league = analysis.add_per_100_possessions(league)
    if {"scores_per_100_poss", "scores_allowed_per_100_poss"}.issubset(league.columns):
        league["net_efficiency_per_100_poss"] = (
            pd.to_numeric(league["scores_per_100_poss"], errors="coerce")
            - pd.to_numeric(league["scores_allowed_per_100_poss"], errors="coerce")
        )

    # The profile's own row inside the league frame — this is what carries the
    # per-100 and allowed columns that `summary` (from the offensive mart alone)
    # does not have.
    team_rows = league[league["team_id"].astype(str) == str(team_id)]
    team_row = team_rows.iloc[0] if len(team_rows) else None

    coverage = analysis.possession_coverage(league)
    if coverage < 0.98:
        st.warning(
            f"Possession data covers {coverage:.0%} of teams in this context, so "
            "the per-100-possession figures below are based on partial data."
        )

    # ------------------------------------------------------------------
    # EFFICIENCY
    # ------------------------------------------------------------------

    if team_row is not None:
        eff_keys = M.with_values(team_row, [
            "scores_per_100_poss", "scores_allowed_per_100_poss",
            "net_efficiency_per_100_poss", "offensive_sequence_proxy_per_game",
            "turnovers_per_100_poss", "shot_pct_calc",
        ])
        if eff_keys:
            ui.section(
                "Efficiency",
                "Pace-adjusted production on both sides of the ball, with this "
                f"team's rank among the {len(league)} teams in the {context_label} "
                "table. Per-game figures flatter a fast team; per-100 does not.",
            )
            # context= turns each card's sub-line into "2nd of 8" — the hook on
            # metric_grid that nothing in the app was using.
            ui.metric_grid(team_row, eff_keys, columns=3,
                           context=league, row_index=team_row.name)
            ui.definition_caption(eff_keys)

        # --------------------------------------------------------------
        # FOUR FACTORS
        # --------------------------------------------------------------
        factors = analysis.four_factor_frame(league)
        if len(factors) and "team_name" in factors.columns:
            mine = factors[factors["team_name"] == selected_team]
            if len(mine):
                ui.section(
                    "Four factors",
                    "The four things that decide a lacrosse game, each with this "
                    "team's league rank. Written for this app and displayed "
                    "nowhere until now.",
                )
                cards = st.columns(len(mine))
                for col, (_, fr) in zip(cards, mine.iterrows()):
                    with col:
                        ui.stat_card(
                            fr["factor"],
                            M.format_value(fr["metric"], fr["value"]),
                            sub=f"{analysis.ordinal(int(fr['rank']))} of {len(league)}",
                            # A rank in the top third is good news, the bottom
                            # third bad, whichever direction the metric runs.
                            tone=("good" if fr["rank"] <= len(league) / 3 else
                                  "bad" if fr["rank"] > 2 * len(league) / 3 else None),
                        )
                with st.expander("What are the four factors?", expanded=False):
                    for _, fr in mine.iterrows():
                        st.markdown(f"**{fr['factor']}** — {fr['definition']}  \n"
                                    f"<span class='section-note'>"
                                    f"{M.direction_note(fr['metric'])}</span>",
                                    unsafe_allow_html=True)

    ui.section("Team Totals")
    ui.metric_grid(
        summary,
        ["games", "wins", "losses", "win_pct", "scores", "goals", "assists",
         "shots", "turnovers", "caused_turnovers", "touches", "total_passes"],
        columns=4,
    )

    ui.section(
        "Per-Game Rates",
        "Per-game figures, which are comparable across seasons of different "
        "lengths — 2026 is a shorter season than the ones before it. Each card "
        "carries its league rank in this context.",
    )
    ui.metric_grid(
        summary,
        ["scores_per_game", "score_margin_per_game", "shots_per_game",
         "shot_pct_calc", "assists_per_game", "turnovers_per_game",
         "faceoff_pct_calc", "clear_pct_calc", "touches_per_game",
         "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
         "saves_per_game"],
        columns=4,
        context=league,
        row_index=team_row.name if team_row is not None else None,
    )

    # ---- Team Player Totals section ----
    st.markdown("### Team Player Totals")
    st.caption("Player production for the selected team profile.")

    _tp_team_id = team_id
    _tp_selected_team = selected_team
    _tp_selected_context = selected_context if selected_context is not None else "Career"

    if not table_exists("marts", "player_season_stats_by_team"):
        st.info("Player season totals by team are not available yet.")
    else:
        try:
            _pst_cols = _pll_get_table_columns("marts", "player_season_stats_by_team")
        except Exception:
            _pst_cols_df = query_df("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'marts' AND table_name = 'player_season_stats_by_team'
                ORDER BY ordinal_position
            """)
            _pst_cols = _pst_cols_df["column_name"].astype(str).tolist()

        _has_team_id = "team_id" in _pst_cols
        _has_team_name = "team_name" in _pst_cols
        _has_season = "season" in _pst_cols

        if not _has_season:
            st.warning("Player season totals by team table is missing the season column.")
        elif not _has_team_id and not _has_team_name:
            st.warning("Player season totals by team table is missing team_id/team_name columns.")
        else:
            _team_filter_parts = []
            _team_filter_params = []
            if _has_team_id and _tp_team_id is not None:
                _team_filter_parts.append("CAST(p.team_id AS VARCHAR) = ?")
                _team_filter_params.append(str(_tp_team_id))
            if _has_team_name and _tp_selected_team is not None:
                _team_filter_parts.append("LOWER(CAST(p.team_name AS VARCHAR)) = LOWER(?)")
                _team_filter_params.append(str(_tp_selected_team))

            if not _team_filter_parts:
                st.info("Could not resolve the selected team for player totals.")
            else:
                _team_filter_sql = "(" + " OR ".join(_team_filter_parts) + ")"
                _team_player_all_rows = query_df(f"""
                    SELECT p.* FROM marts.player_season_stats_by_team p WHERE {_team_filter_sql}
                """, _team_filter_params)

                if len(_team_player_all_rows) == 0:
                    st.info(f"No player totals are available for {_tp_selected_team or _tp_team_id}.")
                else:
                    _team_player_all_rows = _team_player_all_rows.copy()
                    _available_seasons = (
                        _team_player_all_rows["season"].dropna().astype(int).sort_values().unique().tolist()
                    )

                    if not _available_seasons:
                        st.info("No seasons are available for the selected team.")
                    else:
                        _time_frame_options = ["Team Profile Context", "All Time", "Specific Season"]
                        _filter_cols = st.columns([1.15, 1.0, 1.4, 0.85, 1.15])

                        with _filter_cols[0]:
                            _team_player_time_frame = st.selectbox(
                                "Time Frame",
                                options=_time_frame_options,
                                index=0,
                                key=f"team_profile_player_time_frame_{_tp_team_id}"
                            )

                        _default_specific_season = _available_seasons[-1]
                        try:
                            if str(_tp_selected_context).lower() != "career":
                                _context_season = int(_tp_selected_context)
                                if _context_season in _available_seasons:
                                    _default_specific_season = _context_season
                        except Exception:
                            pass

                        _season_selector_disabled = _team_player_time_frame != "Specific Season"

                        with _filter_cols[1]:
                            _team_player_specific_season = st.selectbox(
                                "Season",
                                options=_available_seasons,
                                index=_available_seasons.index(_default_specific_season),
                                disabled=_season_selector_disabled,
                                key=f"team_profile_player_specific_season_{_tp_team_id}"
                            )

                        if _team_player_time_frame == "Team Profile Context":
                            if str(_tp_selected_context).lower() == "career":
                                _team_player_base = _team_player_all_rows.copy()
                                _team_player_context_label = "All Time"
                            else:
                                try:
                                    _context_season_int = int(_tp_selected_context)
                                    _team_player_base = _team_player_all_rows[
                                        pd.to_numeric(_team_player_all_rows["season"], errors="coerce") == _context_season_int
                                    ].copy()
                                    _team_player_context_label = f"{_context_season_int} Season"
                                except Exception:
                                    _team_player_base = _team_player_all_rows.copy()
                                    _team_player_context_label = "All Time"
                        elif _team_player_time_frame == "All Time":
                            _team_player_base = _team_player_all_rows.copy()
                            _team_player_context_label = "All Time"
                        else:
                            _team_player_base = _team_player_all_rows[
                                pd.to_numeric(_team_player_all_rows["season"], errors="coerce") == int(_team_player_specific_season)
                            ].copy()
                            _team_player_context_label = f"{int(_team_player_specific_season)} Season"

                        _multi_season = (
                            len(_team_player_base) > 0
                            and pd.to_numeric(_team_player_base["season"], errors="coerce").nunique() > 1
                        )

                        if _multi_season:
                            _id_cols = [c for c in ["player_id", "full_name"] if c in _team_player_base.columns]
                            if not _id_cols and "full_name" in _team_player_base.columns:
                                _id_cols = ["full_name"]
                            _sum_cols = [c for c in [
                                "games", "points", "scoring_points", "one_point_goals", "two_point_goals",
                                "goals", "assists", "shots", "shots_on_goal", "two_point_shots",
                                "ground_balls", "turnovers", "caused_turnovers", "faceoffs_won", "faceoffs_lost",
                                "faceoffs", "saves", "clean_saves", "messy_saves", "scores_against",
                                "goals_against", "touches", "total_passes", "penalties", "penalty_time",
                            ] if c in _team_player_base.columns]
                            _agg_dict = {c: "sum" for c in _sum_cols}
                            if "position" in _team_player_base.columns:
                                _agg_dict["position"] = lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) else None
                            if "position_name" in _team_player_base.columns:
                                _agg_dict["position_name"] = lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) else None
                            if "team_id" in _team_player_base.columns:
                                _agg_dict["team_id"] = "last"
                            if "team_name" in _team_player_base.columns:
                                _agg_dict["team_name"] = "last"
                            _team_player_display_base = _team_player_base.groupby(_id_cols, dropna=False).agg(_agg_dict).reset_index()
                            _team_player_display_base["season"] = "All Time"
                        else:
                            _team_player_display_base = _team_player_base.copy()

                        if len(_team_player_display_base) > 0:
                            if "games" in _team_player_display_base.columns:
                                _games = pd.to_numeric(_team_player_display_base["games"], errors="coerce").replace(0, np.nan)
                            else:
                                _games = pd.Series(np.nan, index=_team_player_display_base.index)

                            _rate_pairs = {
                                "points": "points_per_game", "scoring_points": "scoring_points_per_game",
                                "one_point_goals": "one_point_goals_per_game", "two_point_goals": "two_point_goals_per_game",
                                "goals": "goals_per_game", "assists": "assists_per_game",
                                "shots": "shots_per_game", "shots_on_goal": "shots_on_goal_per_game",
                                "ground_balls": "ground_balls_per_game", "turnovers": "turnovers_per_game",
                                "caused_turnovers": "caused_turnovers_per_game", "faceoffs_won": "faceoffs_won_per_game",
                                "faceoffs": "faceoffs_per_game", "saves": "saves_per_game",
                                "scores_against": "scores_against_per_game", "goals_against": "goals_against_per_game",
                                "touches": "touches_per_game", "total_passes": "total_passes_per_game",
                            }
                            for _total_col, _rate_col in _rate_pairs.items():
                                if _total_col in _team_player_display_base.columns:
                                    _team_player_display_base[_rate_col] = (
                                        pd.to_numeric(_team_player_display_base[_total_col], errors="coerce") / _games
                                    )

                            if "faceoffs_won" in _team_player_display_base.columns and "faceoffs" in _team_player_display_base.columns:
                                _team_player_display_base["faceoff_pct_calc"] = (
                                    pd.to_numeric(_team_player_display_base["faceoffs_won"], errors="coerce")
                                    / pd.to_numeric(_team_player_display_base["faceoffs"], errors="coerce").replace(0, np.nan)
                                )

                            if "saves" in _team_player_display_base.columns:
                                _saves = pd.to_numeric(_team_player_display_base["saves"], errors="coerce")
                                if "goals_against" in _team_player_display_base.columns:
                                    _ga = pd.to_numeric(_team_player_display_base["goals_against"], errors="coerce")
                                elif "scores_against" in _team_player_display_base.columns:
                                    _ga = pd.to_numeric(_team_player_display_base["scores_against"], errors="coerce")
                                else:
                                    _ga = pd.Series(np.nan, index=_team_player_display_base.index)
                                _team_player_display_base["save_pct_calc"] = (
                                    _saves / (_saves + _ga).replace(0, np.nan)
                                ).clip(lower=0, upper=1)

                        if len(_team_player_display_base) == 0:
                            st.info("No players match the selected time frame.")
                        else:
                            _position_options = (
                                sorted(_team_player_display_base["position"].dropna().astype(str).unique().tolist())
                                if "position" in _team_player_display_base.columns else []
                            )

                            with _filter_cols[2]:
                                _selected_positions = st.multiselect(
                                    "Positions", options=_position_options, default=[],
                                    key=f"team_profile_player_positions_{_tp_team_id}"
                                )

                            with _filter_cols[3]:
                                _min_games_team_players = st.number_input(
                                    "Min Games", min_value=0, max_value=100, value=0, step=1,
                                    key=f"team_profile_player_min_games_{_tp_team_id}"
                                )

                            _sort_options = [
                                c for c in [
                                    "points", "points_per_game", "scoring_points", "scoring_points_per_game",
                                    "one_point_goals", "one_point_goals_per_game", "two_point_goals", "two_point_goals_per_game",
                                    "goals", "goals_per_game", "assists", "assists_per_game",
                                    "shots", "shots_per_game", "shots_on_goal", "shots_on_goal_per_game",
                                    "touches", "touches_per_game", "ground_balls", "ground_balls_per_game",
                                    "caused_turnovers", "caused_turnovers_per_game", "turnovers", "turnovers_per_game",
                                    "faceoff_pct_calc", "save_pct_calc",
                                ]
                                if c in _team_player_display_base.columns
                            ]

                            with _filter_cols[4]:
                                _team_player_sort_metric = st.selectbox(
                                    "Sort By", options=_sort_options,
                                    index=0 if _sort_options else None,
                                    format_func=pretty_col,
                                    key=f"team_profile_player_sort_{_tp_team_id}"
                                )

                            _table_view = st.radio(
                                "Player Table View",
                                options=["Summary", "Per Game", "Specialists"],
                                horizontal=True,
                                key=f"team_profile_player_table_view_{_tp_team_id}"
                            )

                            _team_player_filtered = _team_player_display_base.copy()
                            if _selected_positions and "position" in _team_player_filtered.columns:
                                _team_player_filtered = _team_player_filtered[
                                    _team_player_filtered["position"].astype(str).isin(_selected_positions)
                                ]
                            if "games" in _team_player_filtered.columns:
                                _team_player_filtered = _team_player_filtered[
                                    pd.to_numeric(_team_player_filtered["games"], errors="coerce").fillna(0) >= _min_games_team_players
                                ]
                            if _team_player_sort_metric in _team_player_filtered.columns:
                                _team_player_filtered[_team_player_sort_metric] = pd.to_numeric(
                                    _team_player_filtered[_team_player_sort_metric], errors="coerce"
                                )
                                _sort_ascending = _team_player_sort_metric in {"turnovers", "turnovers_per_game", "goals_against", "goals_against_per_game", "scores_against", "scores_against_per_game"}
                                _team_player_filtered = _team_player_filtered.sort_values(
                                    _team_player_sort_metric, ascending=_sort_ascending, na_position="last"
                                )

                            _cards = st.columns(4)
                            with _cards[0]:
                                stat_card("Players", fmt_value(len(_team_player_filtered), 0))
                            with _cards[1]:
                                stat_card("Team", str(_tp_selected_team or _tp_team_id))
                            with _cards[2]:
                                stat_card("Time Frame", _team_player_context_label)
                            with _cards[3]:
                                _top_player_name = (
                                    _team_player_filtered["full_name"].iloc[0]
                                    if len(_team_player_filtered) and "full_name" in _team_player_filtered.columns
                                    else "—"
                                )
                                stat_card("Top Player", _top_player_name)

                            if _table_view == "Summary":
                                _display_cols = ["season", "full_name", "position", "games", "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists", "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers", "touches"]
                            elif _table_view == "Per Game":
                                _display_cols = ["season", "full_name", "position", "games", "points_per_game", "scoring_points_per_game", "one_point_goals_per_game", "two_point_goals_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "shots_on_goal_per_game", "ground_balls_per_game", "turnovers_per_game", "caused_turnovers_per_game", "touches_per_game", "total_passes_per_game"]
                            else:
                                _display_cols = ["season", "full_name", "position", "position_name", "games", "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists", "shots", "shots_on_goal", "two_point_shots", "shot_pct_calc", "shots_on_goal_rate_calc", "ground_balls", "turnovers", "caused_turnovers", "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct_calc", "saves", "scores_against", "goals_against", "save_pct_calc", "touches", "total_passes"]

                            _display_cols = [c for c in _display_cols if c in _team_player_filtered.columns]

                            if len(_team_player_filtered) > 0 and _team_player_sort_metric in _team_player_filtered.columns and "full_name" in _team_player_filtered.columns:
                                _chart_df = _team_player_filtered.head(15).copy()
                                safe_bar_chart(
                                    _chart_df.sort_values(_team_player_sort_metric, ascending=True),
                                    x_col="full_name", y_col=_team_player_sort_metric,
                                    color_col="position" if "position" in _chart_df.columns else None,
                                    title=f"{_tp_selected_team or _tp_team_id} — {_team_player_context_label} Player Leaders by {pretty_col(_team_player_sort_metric)}",
                                    orientation="h"
                                )

                            display_table(_team_player_filtered[_display_cols], height=430, hide_cols=[], max_cols=None)

                            with st.expander("Full player table", expanded=False):
                                display_table(_team_player_filtered, height=430, hide_cols=[], max_cols=None)

                            download_csv(
                                _team_player_filtered,
                                f"{str(_tp_selected_team or _tp_team_id).replace(' ', '_').lower()}_{str(_team_player_context_label).replace(' ', '_').lower()}_player_totals.csv",
                                label="Download team player totals CSV"
                            )

    # ---- Defensive Profile ----
    #
    # The old page had a second "Per-Game / Rate Stats" stat_grid here that
    # restated Scores/G, Shots/G, TO/G, Shot %, FO %, Clear % and Off. Seq./G —
    # every one already shown in the Per-Game Rates grid above, with labels and
    # digit counts hardcoded a second time. Only Win % was unique to it, and it
    # is in Team Totals.
    ui.section(
        "Defence and opponent profile",
        "What this team allowed. Rates are per game, with league rank; the "
        "efficiency figures in the Efficiency section above are the pace-adjusted "
        "view of the same thing.",
    )

    if table_exists("marts", "team_defense_season_stats"):
        if selected_context == "Career":
            defense_summary_df = query_df("""
                SELECT * FROM marts.team_defense_career_stats WHERE team_id = ?
            """, [team_id])
        else:
            defense_summary_df = query_df("""
                SELECT * FROM marts.team_defense_season_stats WHERE team_id = ? AND season = ?
            """, [team_id, int(selected_context)])

        if len(defense_summary_df) > 0:
            defense_summary = defense_summary_df.iloc[0]
            # Registry labels and units, and the allowed columns the marts ship
            # that the old eight hardcoded tuples left off: 2PT allowed, assists
            # allowed, opponent shooting accuracy and ride/clear pressure.
            ui.metric_grid(
                defense_summary,
                M.with_values(defense_summary, [
                    "scores_allowed_per_game", "goals_allowed_per_game",
                    "two_point_goals_allowed_per_game", "assists_allowed_per_game",
                    "opponent_shots_per_game", "opponent_goal_pct",
                    "opponent_sog_rate", "opponent_sog_goal_pct",
                    "save_pct_proxy", "caused_turnovers_for_per_game",
                    "opponent_turnovers_per_game", "ct_per_opponent_turnover",
                ]),
                columns=4,
                context=(league_defense if len(league_defense) else None),
                row_index=(
                    league_defense.index[
                        league_defense["team_id"].astype(str) == str(team_id)][0]
                    if len(league_defense) and "team_id" in league_defense.columns
                    and (league_defense["team_id"].astype(str) == str(team_id)).any()
                    else None
                ),
            )
            ui.definition_caption(["opponent_goal_pct", "opponent_sog_goal_pct",
                                   "save_pct_proxy", "ct_per_opponent_turnover"])

            defense_cols = [
                "team_name", "games", "wins", "losses", "win_pct", "team_scores_per_game",
                "scores_allowed_per_game", "goals_allowed_per_game", "opponent_shots_per_game",
                "opponent_goal_pct", "opponent_sog_rate", "save_pct_proxy",
                "caused_turnovers_for_per_game", "opponent_turnovers_per_game",
                "ct_per_opponent_turnover", "score_margin_per_game",
            ]
            display_table(defense_summary_df[[c for c in defense_cols if c in defense_summary_df.columns]], height=220)
        else:
            st.info("No defensive summary found for this team/context.")
    else:
        st.info("Defensive/opponent marts are not available in the warehouse yet.")

    # ---- Season Trend ----
    st.markdown("### Team Season Trend")

    trend_options = [c for c in ["scores_per_game", "shots_per_game", "turnovers_per_game", "saves_per_game", "offensive_sequence_proxy_per_game"] if c in team_seasons.columns]

    trend_selection = st.multiselect(
        "Trend metrics",
        options=trend_options,
        default=[c for c in ["scores_per_game", "shots_per_game", "turnovers_per_game"] if c in trend_options],
        format_func=pretty_col,
        key=f"team_trend_metrics_{team_id}"
    )

    season_trend_df = team_seasons[["season"] + trend_options].copy() if len(team_seasons) else pd.DataFrame()
    safe_line_chart(season_trend_df, x_col="season", y_cols=trend_selection, title=f"{selected_team} — Season Trend")

    # ---- Recent Form ----
    ui.section(
        "Recent form",
        "The team's last games in the context being viewed — not its last games "
        "overall.",
    )

    split_choice = st.radio(
        "Recent form window",
        options=["Last 5", "Last 10"],
        horizontal=True,
        key=f"team_recent_split_{team_id}"
    )

    window_n = 5 if split_choice == "Last 5" else 10

    # Season-scoped marts, not the league-wide last5/last10 ones. Those hold only
    # the most recent games played anywhere — every row in them is 2026 — so a
    # 2022 profile reported 10.6 scores/game from games played in mid-2026 under
    # the heading "Last 5". Atlas's real last five of 2022 is 12.4.
    if selected_context == "Career":
        split_table_name = ("marts.team_last5_stats" if split_choice == "Last 5"
                            else "marts.team_last10_stats")
        split_df = query_df(f"SELECT * FROM {split_table_name} WHERE team_id = ?",
                            [team_id])
        form_window_note = "most recent games in any season"
    else:
        split_table_name = ("marts.team_season_last5_stats" if split_choice == "Last 5"
                            else "marts.team_season_last10_stats")
        split_df = query_df(
            f"SELECT * FROM {split_table_name} WHERE team_id = ? AND season = ?",
            [team_id, int(selected_context)])
        form_window_note = f"last games of {selected_context}"

    if len(split_df) > 0:
        split_summary = split_df.iloc[0]
        profile_header(
            f"{selected_team} — {split_choice}",
            f"{form_window_note} | Games: "
            f"{fmt_value(split_summary.get('games', np.nan), 0)} | Opponents: "
            f"{split_summary.get('opponents', '—')}"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Window Totals")
            ui.metric_grid(
                split_summary,
                ["scores", "goals", "assists", "shots", "saves", "turnovers",
                 "touches", "offensive_sequence_proxy"],
                columns=4,
            )
        with c2:
            st.markdown("#### Window Averages")
            ui.metric_grid(
                split_summary,
                ["scores_per_game", "shots_per_game", "saves_per_game",
                 "turnovers_per_game", "touches_per_game", "total_passes_per_game",
                 "time_in_possession_per_game", "offensive_sequence_proxy_per_game"],
                columns=4,
            )

        # Scoped the same way as the aggregate above: an unscoped LIMIT 5 over the
        # whole game log listed five 2026 opponents on a 2022 profile.
        season_clause = "" if selected_context == "Career" else "AND season = ?"
        recent_params = [team_id]
        if selected_context != "Career":
            recent_params.append(int(selected_context))
        recent_games = query_df(f"""
            SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
                   scores, scores_against, goals, one_point_goals, two_point_goals, assists,
                   shots, shots_on_goal, saves, ground_balls, turnovers, caused_turnovers,
                   faceoffs_won, faceoffs_lost, clears, clear_attempts, touches, total_passes,
                   time_in_possession, official_total_possessions, offensive_sequence_proxy
            FROM clean.team_game_stats
            WHERE team_id = ? {season_clause}
            ORDER BY game_date_utc DESC, season DESC, game_number DESC
            LIMIT {window_n}
        """, recent_params)

        st.markdown(f"#### {split_choice} Individual Games")
        st.caption("The bottom two rows summarize the selected window across the individual games shown above.")
        recent_with_summary = add_window_summary_rows(recent_games)
        display_table(recent_with_summary, height=360)

        # A trailing mean over the window, which the aggregate cannot show: a team
        # averaging 12 while falling from 16 to 8 reads the same as one holding
        # steady at 12.
        if len(recent_games) > 2:
            form_metric = ui.metric_selectbox(
                "Form trend metric",
                options=M.with_data(recent_games, [
                    "scores", "scores_against", "shots", "shots_on_goal", "saves",
                    "turnovers", "caused_turnovers", "touches",
                    "offensive_sequence_proxy",
                ]),
                key=f"team_form_trend_{team_id}",
                default="scores",
            )
            if form_metric:
                trend = recent_games.sort_values(["season", "game_number"]).copy()
                trend["game_label"] = (trend["season"].astype(str) + " G"
                                       + trend["game_number"].astype("Int64").astype(str))
                rolled = analysis.add_rolling(trend, form_metric, window=3)
                y_cols = [form_metric]
                roll_col = f"{form_metric}_roll3"
                if roll_col in rolled.columns:
                    y_cols.append(roll_col)
                safe_line_chart(
                    rolled, x_col="game_label", y_cols=y_cols,
                    title=f"{selected_team} — {M.label(form_metric)} over the window",
                )
                delta = analysis.form_delta(trend, form_metric, window=3)
                if pd.notna(delta):
                    st.caption(
                        f"Last three games average {M.format_value(form_metric, abs(delta))} "
                        f"{'above' if delta > 0 else 'below'} the earlier games in this "
                        f"window."
                    )

    # ---- Team Game Log ----
    st.markdown("### Team Game Log")

    game_log = query_df("""
        SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
               scores, scores_against, goals, one_point_goals, two_point_goals, assists,
               shots, shots_on_goal, saves, ground_balls, turnovers, caused_turnovers,
               faceoffs_won, faceoffs_lost, clears, clear_attempts, touches, total_passes,
               time_in_possession, official_total_possessions, offensive_sequence_proxy
        FROM clean.team_game_stats
        WHERE team_id = ?
        ORDER BY season DESC, game_number DESC
    """, [team_id])

    tg_filters = st.columns(4)

    team_game_seasons = sorted(game_log["season"].dropna().astype(int).unique().tolist()) if len(game_log) else []
    team_game_opps = sorted(game_log["opponent_team_name"].dropna().unique().tolist()) if len(game_log) else []

    selected_tg_seasons = tg_filters[0].multiselect(
        "Game log seasons", team_game_seasons, default=team_game_seasons,
        key=f"team_gl_seasons_{team_id}"
    )
    selected_tg_opps = tg_filters[1].multiselect(
        "Opponents", team_game_opps, default=[],
        key=f"team_gl_opps_{team_id}"
    )
    selected_home = tg_filters[2].selectbox(
        "Home/Away", ["All", "Home", "Away"],
        key=f"team_home_filter_{team_id}"
    )
    min_score_filter = tg_filters[3].number_input(
        "Minimum scores", min_value=0, max_value=40, value=0, step=1,
        key=f"team_min_score_{team_id}"
    )

    filtered_game_log = game_log.copy()
    if selected_tg_seasons:
        filtered_game_log = filtered_game_log[filtered_game_log["season"].isin(selected_tg_seasons)]
    if selected_tg_opps:
        filtered_game_log = filtered_game_log[filtered_game_log["opponent_team_name"].isin(selected_tg_opps)]
    if selected_home == "Home":
        filtered_game_log = filtered_game_log[filtered_game_log["is_home"] == 1]
    elif selected_home == "Away":
        filtered_game_log = filtered_game_log[filtered_game_log["is_home"] == 0]
    if "scores" in filtered_game_log.columns:
        filtered_game_log = filtered_game_log[filtered_game_log["scores"] >= min_score_filter]

    display_table(filtered_game_log, height=430)
    download_csv(filtered_game_log, f"{selected_team.replace(' ', '_').lower()}_team_game_log.csv")
