import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from shared.db import query_df, filter_values
from shared.ui import (
    apply_css, stat_card, stat_grid, safe_bar_chart, safe_line_chart, display_table,
    fmt_value, pretty_col, profile_header, add_window_summary_rows, download_csv
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Player Profiles · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Player Profiles")
st.markdown('<div class="section-note">Review one player through career, season, recent-form, game-log, and opponent-split lenses.</div>', unsafe_allow_html=True)

player_names = players_df["full_name"].dropna().unique().tolist()

selected_player = st.selectbox(
    "Select player",
    options=player_names,
    index=0 if player_names else None,
    key="player_explorer_select"
)

if selected_player:
    player_row = players_df[players_df["full_name"] == selected_player].iloc[0]
    player_id = player_row["player_id"]

    career = query_df("""
        SELECT * FROM marts.player_career_stats WHERE player_id = ?
    """, [player_id])

    player_seasons = query_df("""
        SELECT * FROM marts.player_season_stats WHERE player_id = ? ORDER BY season
    """, [player_id])

    available_contexts = ["Career"] + [str(int(x)) for x in player_seasons["season"].dropna().unique().tolist()]

    selected_context = st.radio(
        "Summary context",
        options=available_contexts,
        horizontal=True,
        key=f"player_context_{player_id}"
    )

    if selected_context == "Career":
        summary = career.iloc[0] if len(career) else pd.Series(dtype="object")
        subtitle = f"{summary.get('position_name', player_row.get('position_name', ''))} | Teams: {summary.get('teams', '—')} | Games: {fmt_value(summary.get('games', np.nan), 0)}"
    else:
        season_int = int(selected_context)
        season_df = player_seasons[player_seasons["season"] == season_int]
        summary = season_df.iloc[0] if len(season_df) else pd.Series(dtype="object")
        subtitle = f"{selected_context} Season | {summary.get('position_name', player_row.get('position_name', ''))} | Team(s): {summary.get('teams', '—')} | Games: {fmt_value(summary.get('games', np.nan), 0)}"

    profile_header(selected_player, subtitle)

    st.markdown("### Totals")
    stat_grid(
        summary,
        [
            ("Games", "games", 0), ("Points", "points", 0), ("Goals", "goals", 0),
            ("Assists", "assists", 0), ("Shots", "shots", 0), ("SOG", "shots_on_goal", 0),
            ("GB", "ground_balls", 0), ("CT", "caused_turnovers", 0),
        ],
        columns=4
    )

    st.markdown("### Per-Game / Rate Stats")
    stat_grid(
        summary,
        [
            ("Points/G", "points_per_game", 2), ("Goals/G", "goals_per_game", 2),
            ("Assists/G", "assists_per_game", 2), ("Shots/G", "shots_per_game", 2),
            ("GB/G", "ground_balls_per_game", 2), ("TO/G", "turnovers_per_game", 2),
            ("CT/G", "caused_turnovers_per_game", 2), ("Shot %", "shot_pct_calc", 2, True),
        ],
        columns=4
    )

    # ------------------------------------------------------------------
    # Playmaking Efficiency — shown for offensive players (A, M)
    # ------------------------------------------------------------------
    _pp_position = str(summary.get("position", player_row.get("position", ""))).upper().strip()
    _pp_is_offense = _pp_position in {"A", "M", "AT", "MF", "SSDM"}
    _pp_is_goalie = _pp_position == "G"

    if _pp_is_offense:
        _pp_assist_conv = summary.get("assist_conv_rate", np.nan)
        _pp_assist_opp_pg = summary.get("assist_opp_per_game", np.nan)
        _pp_two_pt_conv = summary.get("two_pt_conversion", np.nan)

        # Only show section if at least one value is available
        _pp_has_playmaking = any(
            pd.notna(v) for v in [_pp_assist_conv, _pp_assist_opp_pg, _pp_two_pt_conv]
        )
        if _pp_has_playmaking:
            st.markdown("### Playmaking Efficiency")
            st.caption(
                "Assist conversion rate = Assists / Assist Opportunities. "
                "2PT conversion = 2PT Goals / 2PT Shots attempted."
            )
            _pm_cards = st.columns(4)
            with _pm_cards[0]:
                stat_card(
                    "Assist Conv %",
                    f"{_pp_assist_conv * 100:.1f}%" if pd.notna(_pp_assist_conv) else "—"
                )
            with _pm_cards[1]:
                stat_card(
                    "Assist Opp/G",
                    f"{_pp_assist_opp_pg:.2f}" if pd.notna(_pp_assist_opp_pg) else "—"
                )
            with _pm_cards[2]:
                stat_card(
                    "2PT Conv %",
                    f"{_pp_two_pt_conv * 100:.1f}%" if pd.notna(_pp_two_pt_conv) else "—"
                )
            with _pm_cards[3]:
                _pp_assist_opp_total = summary.get("assist_opportunities", np.nan)
                stat_card(
                    "Assist Opp",
                    f"{int(_pp_assist_opp_total)}" if pd.notna(_pp_assist_opp_total) else "—"
                )

    # ------------------------------------------------------------------
    # Save Quality — shown for goalies
    # ------------------------------------------------------------------
    if _pp_is_goalie:
        _pp_clean_saves = summary.get("clean_saves", np.nan)
        _pp_messy_saves = summary.get("messy_saves", np.nan)
        _pp_clean_save_pct = summary.get("clean_save_pct", np.nan)
        _pp_clean_save_rate = summary.get("clean_save_rate", np.nan)

        _pp_has_save_quality = any(
            pd.notna(v) for v in [_pp_clean_saves, _pp_messy_saves, _pp_clean_save_pct]
        )
        if _pp_has_save_quality:
            st.markdown("### Save Quality")
            st.caption(
                "Clean saves are skill-based stops; messy saves include scramble and rebound saves. "
                "Clean Save % = Clean Saves / Total Saves."
            )
            _sq_cards = st.columns(4)
            with _sq_cards[0]:
                stat_card(
                    "Clean Saves",
                    f"{int(_pp_clean_saves)}" if pd.notna(_pp_clean_saves) else "—"
                )
            with _sq_cards[1]:
                stat_card(
                    "Messy Saves",
                    f"{int(_pp_messy_saves)}" if pd.notna(_pp_messy_saves) else "—"
                )
            with _sq_cards[2]:
                stat_card(
                    "Clean Save%",
                    f"{_pp_clean_save_pct:.1f}%" if pd.notna(_pp_clean_save_pct) else "—"
                )
            with _sq_cards[3]:
                stat_card(
                    "Clean Save Rate",
                    f"{_pp_clean_save_rate * 100:.1f}%" if pd.notna(_pp_clean_save_rate) else "—"
                )

    st.markdown("### Season Totals and Averages")
    st.caption("Season-by-season totals and per-game averages for the selected player.")

    _pp_player_id = player_id
    _pp_selected_player = selected_player

    _pp_season_rows = query_df("""
        SELECT * FROM marts.player_season_stats WHERE player_id = ? ORDER BY season
    """, [_pp_player_id])

    if len(_pp_season_rows) == 0:
        st.info("No season-level player totals are available for this player.")
    else:
        _pp_season_rows = _pp_season_rows.copy()

        _pp_view = st.radio(
            "Season table view",
            options=["Summary", "Per Game", "Full Detail"],
            horizontal=True,
            key=f"player_profile_season_totals_view_{_pp_player_id}"
        )

        if _pp_view == "Summary":
            _pp_cols = ["season", "teams", "position", "games", "points", "scoring_points",
                        "one_point_goals", "two_point_goals", "goals", "assists", "shots",
                        "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers", "touches", "total_passes"]
        elif _pp_view == "Per Game":
            _pp_cols = ["season", "teams", "position", "games", "points_per_game", "scoring_points_per_game",
                        "one_point_goals_per_game", "two_point_goals_per_game", "goals_per_game", "assists_per_game",
                        "shots_per_game", "shots_on_goal_per_game", "ground_balls_per_game", "turnovers_per_game",
                        "caused_turnovers_per_game", "touches_per_game", "total_passes_per_game"]
        else:
            _pp_cols = list(_pp_season_rows.columns)

        _pp_cols = [c for c in _pp_cols if c in _pp_season_rows.columns]
        display_table(_pp_season_rows[_pp_cols], height=330, hide_cols=[], max_cols=None)

        _pp_download_name = (
            str(_pp_selected_player).replace(" ", "_").lower()
            if _pp_selected_player is not None else str(_pp_player_id)
        )
        download_csv(_pp_season_rows, f"{_pp_download_name}_season_totals.csv", label="Download player season totals CSV")

    st.markdown("### Season Trend")

    trend_cols = ["points_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game"]
    trend_options = [c for c in trend_cols if c in player_seasons.columns]

    trend_selection = st.multiselect(
        "Trend metrics",
        options=trend_options,
        default=[c for c in ["points_per_game", "goals_per_game", "assists_per_game"] if c in trend_options],
        format_func=pretty_col,
        key=f"player_trend_metrics_{player_id}"
    )

    season_trend_df = player_seasons[["season"] + trend_options].copy() if len(player_seasons) else pd.DataFrame()
    safe_line_chart(season_trend_df, x_col="season", y_cols=trend_selection, title=f"{selected_player} — Season Trend")

    st.markdown("### Recent Form")

    split_choice = st.radio(
        "Recent form window",
        options=["Last 5", "Last 10"],
        horizontal=True,
        key=f"player_recent_split_{player_id}"
    )

    window_n = 5 if split_choice == "Last 5" else 10
    split_table = "marts.player_last5_stats" if split_choice == "Last 5" else "marts.player_last10_stats"

    split_df = query_df(f"""
        SELECT * FROM {split_table} WHERE player_id = ?
    """, [player_id])

    if len(split_df) > 0:
        split_summary = split_df.iloc[0]
        profile_header(
            f"{selected_player} — {split_choice}",
            f"Games: {fmt_value(split_summary.get('games', np.nan), 0)} | Opponents: {split_summary.get('opponents', '—')} | Teams: {split_summary.get('teams', '—')}"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Window Totals")
            stat_grid(
                split_summary,
                [
                    ("Points", "points", 0), ("Goals", "goals", 0), ("Assists", "assists", 0),
                    ("Shots", "shots", 0), ("GB", "ground_balls", 0), ("TO", "turnovers", 0),
                    ("CT", "caused_turnovers", 0), ("Touches", "touches", 0),
                ],
                columns=4
            )

        with c2:
            st.markdown("#### Window Averages")
            stat_grid(
                split_summary,
                [
                    ("Points/G", "points_per_game", 2), ("Goals/G", "goals_per_game", 2),
                    ("Assists/G", "assists_per_game", 2), ("Shots/G", "shots_per_game", 2),
                    ("GB/G", "ground_balls_per_game", 2), ("TO/G", "turnovers_per_game", 2),
                    ("CT/G", "caused_turnovers_per_game", 2), ("Touches/G", "touches_per_game", 2),
                ],
                columns=4
            )

        recent_games = query_df(f"""
            SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
                   points, goals, assists, assist_opportunities, shots, shots_on_goal,
                   ground_balls, turnovers, caused_turnovers,
                   saves, clean_saves, messy_saves, saa, faceoffs_won, faceoffs_lost,
                   touches, total_passes
            FROM clean.player_game_stats
            WHERE player_id = ?
            ORDER BY game_date_utc DESC, season DESC, game_number DESC
            LIMIT {window_n}
        """, [player_id])

        st.markdown(f"#### {split_choice} Individual Games")
        st.caption("The bottom two rows summarize the selected window across the individual games shown above.")
        recent_with_summary = add_window_summary_rows(recent_games)
        display_table(recent_with_summary, height=360)

        recent_metric = st.selectbox(
            f"{split_choice} game-by-game chart metric",
            options=[c for c in ["points", "goals", "assists", "shots", "ground_balls", "turnovers", "caused_turnovers", "touches"] if c in recent_games.columns],
            index=0,
            format_func=pretty_col,
            key=f"player_recent_game_metric_{player_id}_{window_n}"
        )

        if len(recent_games) > 0 and recent_metric:
            recent_chart = recent_games.sort_values(["season", "game_number"]).copy()
            recent_chart["game_label"] = recent_chart["season"].astype(str) + " G" + recent_chart["game_number"].astype(str)
            safe_bar_chart(recent_chart, x_col="game_label", y_col=recent_metric, title=f"{selected_player} — {split_choice} {pretty_col(recent_metric)} by Game")

    st.markdown("### Game Log")

    game_log = query_df("""
        SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
               points, goals, assists, assist_opportunities, shots, shots_on_goal,
               ground_balls, turnovers, caused_turnovers,
               saves, clean_saves, messy_saves, saa,
               faceoffs_won, faceoffs_lost, touches, total_passes
        FROM clean.player_game_stats
        WHERE player_id = ?
        ORDER BY season DESC, game_number DESC
    """, [player_id])

    gl_filters = st.columns(4)

    player_game_seasons = sorted(game_log["season"].dropna().astype(int).unique().tolist()) if len(game_log) else []
    player_game_opps = sorted(game_log["opponent_team_name"].dropna().unique().tolist()) if len(game_log) else []

    selected_gl_seasons = gl_filters[0].multiselect(
        "Game log seasons", player_game_seasons, default=player_game_seasons,
        key=f"player_gl_seasons_{player_id}"
    )

    selected_gl_opps = gl_filters[1].multiselect(
        "Opponents", player_game_opps, default=[],
        key=f"player_gl_opps_{player_id}"
    )

    selected_home = gl_filters[2].selectbox(
        "Home/Away", ["All", "Home", "Away"],
        key=f"player_home_filter_{player_id}"
    )

    min_points_filter = gl_filters[3].number_input(
        "Minimum points", min_value=0, max_value=20, value=0, step=1,
        key=f"player_min_points_{player_id}"
    )

    filtered_game_log = game_log.copy()
    if selected_gl_seasons:
        filtered_game_log = filtered_game_log[filtered_game_log["season"].isin(selected_gl_seasons)]
    if selected_gl_opps:
        filtered_game_log = filtered_game_log[filtered_game_log["opponent_team_name"].isin(selected_gl_opps)]
    if selected_home == "Home":
        filtered_game_log = filtered_game_log[filtered_game_log["is_home"] == 1]
    elif selected_home == "Away":
        filtered_game_log = filtered_game_log[filtered_game_log["is_home"] == 0]
    if "points" in filtered_game_log.columns:
        filtered_game_log = filtered_game_log[filtered_game_log["points"] >= min_points_filter]

    display_table(filtered_game_log, height=430)
    download_csv(filtered_game_log, f"{selected_player.replace(' ', '_').lower()}_game_log.csv")

    game_chart_metrics = st.multiselect(
        "Game log chart metrics",
        options=[c for c in ["points", "goals", "assists", "shots", "ground_balls", "turnovers", "caused_turnovers", "touches"] if c in filtered_game_log.columns],
        default=[c for c in ["points", "goals", "assists", "shots"] if c in filtered_game_log.columns],
        format_func=pretty_col,
        key=f"player_game_chart_metrics_{player_id}"
    )

    if len(filtered_game_log) > 0:
        trend_df = filtered_game_log.sort_values(["season", "game_number"]).copy()
        trend_df["game_label"] = trend_df["season"].astype(str) + " G" + trend_df["game_number"].astype(str)
        safe_line_chart(trend_df, x_col="game_label", y_cols=game_chart_metrics, title=f"{selected_player} — Filtered Game Log")

    st.markdown("### Vs Opponent Splits")

    vs_opp = query_df("""
        SELECT opponent_team_name, games, points, goals, assists, shots, ground_balls,
               caused_turnovers, points_per_game, goals_per_game, assists_per_game,
               shots_per_game, ground_balls_per_game, caused_turnovers_per_game
        FROM marts.player_vs_opponent_stats
        WHERE player_id = ?
        ORDER BY points_per_game DESC NULLS LAST
    """, [player_id])

    opp_cols = st.columns(2)

    vs_metric = opp_cols[0].selectbox(
        "Opponent split metric",
        options=["points_per_game", "goals_per_game", "assists_per_game", "shots_per_game",
                 "ground_balls_per_game", "caused_turnovers_per_game", "points", "goals", "assists", "shots"],
        index=0,
        format_func=pretty_col,
        key=f"player_vs_metric_{player_id}"
    )

    min_vs_games = opp_cols[1].number_input(
        "Minimum games vs opponent", min_value=1, max_value=20, value=1, step=1,
        key=f"player_vs_min_games_{player_id}"
    )

    vs_opp_filtered = vs_opp[vs_opp["games"] >= min_vs_games].copy()

    safe_bar_chart(
        vs_opp_filtered.sort_values(vs_metric).tail(12),
        x_col="opponent_team_name", y_col=vs_metric,
        title=f"{selected_player} — {pretty_col(vs_metric)} by Opponent",
        orientation="h"
    )

    display_table(vs_opp_filtered, height=330)
