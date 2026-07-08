import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from shared.db import query_df, table_exists, filter_values, _pll_get_table_columns
from shared.ui import (
    apply_css, stat_card, stat_grid, safe_bar_chart, safe_line_chart, display_table,
    fmt_value, pretty_col, profile_header, add_window_summary_rows, download_csv,
    _pll_select_existing
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Team Profiles · PLL Analytics", page_icon="🥍", layout="wide")
apply_css()

import os
from shared.db import DB_PATH
if not os.path.exists(DB_PATH):
    st.error(f"DuckDB warehouse not found: {DB_PATH}")
    st.stop()

try:
    seasons, teams_df, players_df, positions, selected_seasons, selected_teams, selected_positions, min_games = render_sidebar_filters()
except Exception as e:
    st.error("Failed to load PLL warehouse.")
    st.exception(e)
    st.stop()

st.subheader("Team Profiles")
st.markdown('<div class="section-note">Analyze team-level career, season, recent-form, game-log, and opponent splits.</div>', unsafe_allow_html=True)

team_options = teams_df["team_name"].dropna().tolist()

selected_team = st.selectbox(
    "Select team",
    options=team_options,
    index=0 if team_options else None,
    key="team_explorer_select"
)

if selected_team:
    team_id = teams_df[teams_df["team_name"] == selected_team]["team_id"].iloc[0]

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

    st.markdown("### Team Totals")
    stat_grid(
        summary,
        [
            ("Games", "games", 0), ("Wins", "wins", 0), ("Losses", "losses", 0),
            ("Scores", "scores", 0), ("Goals", "goals", 0), ("Assists", "assists", 0),
            ("Shots", "shots", 0), ("Turnovers", "turnovers", 0),
        ],
        columns=4
    )

    # ---- Team Player Totals section ----
    st.markdown("### Team Player Totals")
    st.caption("Player production for the selected team profile.")

    _tp_team_id = team_id
    _tp_selected_team = selected_team
    _tp_selected_context = selected_context if selected_context is not None else "Career"

    _pst_table_check = query_df("""
        SELECT COUNT(*) AS n FROM information_schema.tables
        WHERE table_schema = 'marts' AND table_name = 'player_season_stats_by_team'
    """)

    if int(_pst_table_check["n"].iloc[0]) == 0:
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

    # ---- Per-Game / Rate Stats ----
    st.markdown("### Per-Game / Rate Stats")
    stat_grid(
        summary,
        [
            ("Win %", "win_pct", 2, True), ("Scores/G", "scores_per_game", 2),
            ("Shots/G", "shots_per_game", 2), ("TO/G", "turnovers_per_game", 2),
            ("Shot %", "shot_pct_calc", 2, True), ("FO %", "faceoff_pct_calc", 2, True),
            ("Clear %", "clear_pct_calc", 2, True), ("Off. Seq./G", "offensive_sequence_proxy_per_game", 2),
        ],
        columns=4
    )

    # ---- Defensive Profile ----
    st.markdown("### Defensive / Opponent Profile")

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
            st.markdown("#### Defensive Summary")
            stat_grid(
                defense_summary,
                [
                    ("Scores Allowed/G", "scores_allowed_per_game", 2),
                    ("Goals Allowed/G", "goals_allowed_per_game", 2),
                    ("Opp Shots/G", "opponent_shots_per_game", 2),
                    ("Opp Goal %", "opponent_goal_pct", 2, True),
                    ("Save % Proxy", "save_pct_proxy", 2, True),
                    ("CT/G", "caused_turnovers_for_per_game", 2),
                    ("Opp TO/G", "opponent_turnovers_per_game", 2),
                    ("Margin/G", "score_margin_per_game", 2),
                ],
                columns=4
            )
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
    st.markdown("### Recent Form")

    split_choice = st.radio(
        "Recent form window",
        options=["Last 5", "Last 10"],
        horizontal=True,
        key=f"team_recent_split_{team_id}"
    )

    window_n = 5 if split_choice == "Last 5" else 10
    split_table_name = "marts.team_last5_stats" if split_choice == "Last 5" else "marts.team_last10_stats"

    split_df = query_df(f"SELECT * FROM {split_table_name} WHERE team_id = ?", [team_id])

    if len(split_df) > 0:
        split_summary = split_df.iloc[0]
        profile_header(
            f"{selected_team} — {split_choice}",
            f"Games: {fmt_value(split_summary.get('games', np.nan), 0)} | Opponents: {split_summary.get('opponents', '—')}"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Window Totals")
            stat_grid(
                split_summary,
                [
                    ("Scores", "scores", 0), ("Goals", "goals", 0), ("Assists", "assists", 0),
                    ("Shots", "shots", 0), ("Saves", "saves", 0), ("Turnovers", "turnovers", 0),
                    ("Touches", "touches", 0), ("Off. Seq.", "offensive_sequence_proxy", 0),
                ],
                columns=4
            )
        with c2:
            st.markdown("#### Window Averages")
            stat_grid(
                split_summary,
                [
                    ("Scores/G", "scores_per_game", 2), ("Shots/G", "shots_per_game", 2),
                    ("Saves/G", "saves_per_game", 2), ("TO/G", "turnovers_per_game", 2),
                    ("Touches/G", "touches_per_game", 2), ("Passes/G", "total_passes_per_game", 2),
                    ("Poss. Time/G", "time_in_possession_per_game", 2), ("Off. Seq./G", "offensive_sequence_proxy_per_game", 2),
                ],
                columns=4
            )

        recent_games = query_df(f"""
            SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
                   scores, scores_against, goals, one_point_goals, two_point_goals, assists,
                   shots, shots_on_goal, saves, ground_balls, turnovers, caused_turnovers,
                   faceoffs_won, faceoffs_lost, clears, clear_attempts, touches, total_passes,
                   time_in_possession, official_total_possessions, offensive_sequence_proxy
            FROM clean.team_game_stats
            WHERE team_id = ?
            ORDER BY game_date_utc DESC, season DESC, game_number DESC
            LIMIT {window_n}
        """, [team_id])

        st.markdown(f"#### {split_choice} Individual Games")
        st.caption("The bottom two rows summarize the selected window across the individual games shown above.")
        recent_with_summary = add_window_summary_rows(recent_games)
        display_table(recent_with_summary, height=360)

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
