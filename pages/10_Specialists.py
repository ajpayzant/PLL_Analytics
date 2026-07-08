import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from shared.db import query_df, filter_values
from shared.ui import (
    apply_css, stat_card, safe_bar_chart, safe_line_chart, display_table,
    fmt_value, pretty_col, profile_header,
    _pll_select_existing, _pll_safe_sort, _pll_apply_goalie_save_pct, _pll_pct_text
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Specialists · PLL Analytics", page_icon="🥍", layout="wide")
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

st.subheader("Specialists")
st.markdown(
    '<div class="section-note">Dedicated evaluation pages for goalies and faceoff specialists. Goalie save percentage is standardized as Saves ÷ (Saves + Goals Against).</div>',
    unsafe_allow_html=True
)

specialist_section = st.radio(
    "Specialist section",
    options=["Goalies", "Faceoff Specialists"],
    horizontal=True,
    key="specialist_section_select"
)

# ============================================================
# GOALIES
# ============================================================

if specialist_section == "Goalies":
    st.markdown("### Goalie Leaders")
    st.caption("Goalie results use completed player stat rows only. Save Percentage is recalculated to prevent invalid values above 100%.")

    g_cols = st.columns([1.0, 1.0, 1.2, 1.0])

    goalie_season = g_cols[0].selectbox(
        "Season",
        options=seasons,
        index=len(seasons) - 1 if seasons else 0,
        key="goalie_season"
    )

    goalie_min_games = g_cols[1].number_input(
        "Minimum games",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="goalie_min_games"
    )

    goalie_df = query_df("""
        SELECT season, player_id, full_name, position, position_name, teams, games,
               saves, clean_saves, messy_saves, scores_against, saa, goals_against,
               shots, save_pct_calc, clean_save_pct, saves_per_game, clean_saves_per_game,
               messy_saves_per_game, scores_against_per_game, saa_per_game, goals_against_per_game
        FROM marts.player_season_stats
        WHERE season = ?
          AND games >= ?
          AND (position = 'G' OR lower(position_name) LIKE '%goalie%')
        ORDER BY games DESC
    """, [goalie_season, goalie_min_games])

    goalie_df = _pll_apply_goalie_save_pct(goalie_df)

    goalie_metric_options = [
        c for c in [
            "save_pct_display", "clean_save_pct", "saves", "saves_per_game",
            "shots_faced_calc", "shots_faced_per_game_calc",
            "goals_against", "goals_against_per_game",
            "scores_against", "scores_against_per_game", "saa", "saa_per_game",
            "clean_saves", "messy_saves",
        ]
        if c in goalie_df.columns
    ]

    goalie_metric = g_cols[2].selectbox(
        "Goalie metric",
        options=goalie_metric_options,
        index=0,
        format_func=pretty_col,
        key="goalie_metric"
    )

    lower_goalie_metrics = {"scores_against", "scores_against_per_game", "goals_against", "goals_against_per_game"}
    goalie_sort_ascending = goalie_metric in lower_goalie_metrics

    g_cols[3].caption("Sort logic")
    g_cols[3].markdown("**Lower is better**" if goalie_sort_ascending else "**Higher is better**")

    goalie_df = _pll_safe_sort(goalie_df, goalie_metric, lower_is_better=goalie_sort_ascending)

    top_goalie = goalie_df["full_name"].iloc[0] if len(goalie_df) else "—"
    best_save_pct = goalie_df["save_pct_display"].max() if "save_pct_display" in goalie_df.columns and len(goalie_df) else float("nan")
    avg_save_pct = goalie_df["save_pct_display"].mean() if "save_pct_display" in goalie_df.columns and len(goalie_df) else float("nan")

    gk1, gk2, gk3, gk4 = st.columns(4)
    with gk1:
        stat_card("Goalies", fmt_value(len(goalie_df), 0))
    with gk2:
        stat_card("Top Goalie", top_goalie)
    with gk3:
        stat_card("Best Save %", _pll_pct_text(best_save_pct))
    with gk4:
        stat_card("Average Save %", _pll_pct_text(avg_save_pct))

    safe_bar_chart(
        goalie_df.head(15).sort_values(goalie_metric, ascending=not goalie_sort_ascending),
        x_col="full_name", y_col=goalie_metric, color_col="teams",
        title=f"{goalie_season} Goalie Leaders — {pretty_col(goalie_metric)}",
        orientation="h"
    )

    goalie_display = goalie_df.copy()
    goalie_summary_cols = _pll_select_existing(
        goalie_display,
        ["season", "full_name", "position", "teams", "games",
         "saves", "clean_saves", "messy_saves", "goals_against", "scores_against",
         "shots_faced_calc", "save_pct_display_pct", "clean_save_pct",
         "saves_per_game", "goals_against_per_game",
         "scores_against_per_game", "shots_faced_per_game_calc"]
    )
    display_table(goalie_display[goalie_summary_cols], height=420)

    # Save Quality leaderboard — sorted by clean_save_pct
    _sq_has_data = "clean_save_pct" in goalie_display.columns and goalie_display["clean_save_pct"].notna().any()
    if _sq_has_data:
        st.markdown("#### Save Quality Leaders")
        st.caption("Sorted by Clean Save % — the proportion of saves that were clean (skill-based) vs messy (scramble).")
        _sq_df = goalie_display.sort_values("clean_save_pct", ascending=False).head(15)
        safe_bar_chart(
            _sq_df.sort_values("clean_save_pct"),
            x_col="full_name", y_col="clean_save_pct", color_col="teams",
            title=f"{goalie_season} Goalies — Clean Save %",
            orientation="h"
        )

    with st.expander("Advanced goalie metrics", expanded=False):
        goalie_advanced_cols = _pll_select_existing(
            goalie_display,
            ["season", "full_name", "position", "position_name", "teams", "games",
             "saves", "clean_saves", "messy_saves", "goals_against", "scores_against",
             "shots_faced_calc", "save_pct_display", "save_pct_display_pct",
             "saa", "shots", "save_pct_calc", "saves_per_game", "clean_saves_per_game",
             "messy_saves_per_game", "goals_against_per_game", "scores_against_per_game",
             "shots_faced_per_game_calc", "saa_per_game"]
        )
        display_table(goalie_display[goalie_advanced_cols], height=360)

    st.markdown("### Goalie Explorer")

    goalie_names = goalie_df["full_name"].dropna().unique().tolist()

    if goalie_names:
        selected_goalie = st.selectbox(
            "Select goalie", options=goalie_names, index=0,
            key="selected_goalie"
        )

        selected_goalie_id = goalie_df[goalie_df["full_name"] == selected_goalie]["player_id"].iloc[0]

        goalie_games = query_df("""
            SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
                   saves, clean_saves, messy_saves, scores_against, saa, goals_against,
                   shots, save_pct, touches, total_passes
            FROM clean.player_game_stats
            WHERE player_id = ?
            ORDER BY season DESC, game_number DESC
        """, [selected_goalie_id])

        goalie_games = _pll_apply_goalie_save_pct(goalie_games)
        profile_header(selected_goalie, "Goalie game log and trend view")

        # Save Quality breakdown stat cards
        _gx_clean = goalie_games["clean_saves"].sum() if "clean_saves" in goalie_games.columns else np.nan
        _gx_messy = goalie_games["messy_saves"].sum() if "messy_saves" in goalie_games.columns else np.nan
        _gx_total = goalie_games["saves"].sum() if "saves" in goalie_games.columns else np.nan
        _gx_clean_pct = (_gx_clean / _gx_total * 100) if (pd.notna(_gx_clean) and pd.notna(_gx_total) and _gx_total > 0) else np.nan

        _gx_has_quality = any(pd.notna(v) for v in [_gx_clean, _gx_messy, _gx_clean_pct])
        if _gx_has_quality:
            st.markdown("#### Save Quality Breakdown")
            _gx_cols = st.columns(4)
            with _gx_cols[0]:
                stat_card("Total Saves", f"{int(_gx_total)}" if pd.notna(_gx_total) else "—")
            with _gx_cols[1]:
                stat_card("Clean Saves", f"{int(_gx_clean)}" if pd.notna(_gx_clean) else "—")
            with _gx_cols[2]:
                stat_card("Messy Saves", f"{int(_gx_messy)}" if pd.notna(_gx_messy) else "—")
            with _gx_cols[3]:
                stat_card("Clean Save%", f"{_gx_clean_pct:.1f}%" if pd.notna(_gx_clean_pct) else "—")

        goalie_game_cols = _pll_select_existing(
            goalie_games,
            ["season", "game_number", "game_date_utc", "team_name", "opponent_team_name",
             "is_home", "saves", "goals_against", "scores_against", "shots_faced_calc",
             "save_pct_display_pct", "clean_saves", "messy_saves", "touches", "total_passes"]
        )
        display_table(goalie_games[goalie_game_cols], height=360)

        goalie_game_metric_options = [
            c for c in ["saves", "clean_saves", "messy_saves", "goals_against", "scores_against",
                        "shots_faced_calc", "save_pct_display"]
            if c in goalie_games.columns
        ]

        goalie_game_metric = st.selectbox(
            "Goalie game trend metric",
            options=goalie_game_metric_options, index=0,
            format_func=pretty_col,
            key="goalie_game_metric"
        )

        if len(goalie_games) > 0:
            goalie_trend = goalie_games.sort_values(["season", "game_number"]).copy()
            goalie_trend["game_label"] = goalie_trend["season"].astype(str) + " G" + goalie_trend["game_number"].astype(str)
            safe_line_chart(
                goalie_trend, x_col="game_label", y_cols=[goalie_game_metric],
                title=f"{selected_goalie} — {pretty_col(goalie_game_metric)} by Game"
            )

            # Clean vs Messy Saves bar chart (game-by-game)
            _cv_has = (
                "clean_saves" in goalie_trend.columns
                and "messy_saves" in goalie_trend.columns
                and goalie_trend["clean_saves"].notna().any()
            )
            if _cv_has:
                import pandas as _pd_cv
                import plotly.express as _px_cv
                _cv_long = goalie_trend[["game_label", "clean_saves", "messy_saves"]].copy()
                _cv_long = _cv_long.dropna(subset=["clean_saves", "messy_saves"], how="all")
                if len(_cv_long) > 0:
                    _cv_melted = _cv_long.melt(
                        id_vars="game_label",
                        value_vars=["clean_saves", "messy_saves"],
                        var_name="save_type",
                        value_name="saves"
                    )
                    _cv_melted["save_type"] = _cv_melted["save_type"].map(
                        {"clean_saves": "Clean Saves", "messy_saves": "Messy Saves"}
                    )
                    _cv_fig = _px_cv.bar(
                        _cv_melted,
                        x="game_label",
                        y="saves",
                        color="save_type",
                        barmode="stack",
                        title=f"{selected_goalie} — Clean vs Messy Saves by Game",
                        labels={"game_label": "Game", "saves": "Saves", "save_type": "Save Type"}
                    )
                    _cv_fig.update_layout(margin=dict(l=10, r=20, t=45, b=10))
                    st.plotly_chart(_cv_fig, use_container_width=True)

# ============================================================
# FACEOFF SPECIALISTS
# ============================================================

else:
    st.markdown("### Faceoff Leaders")
    st.caption("Faceoff leaders are filtered by minimum total faceoffs to avoid small-sample noise.")

    f_cols = st.columns([1.0, 1.0, 1.2, 1.0])

    faceoff_season = f_cols[0].selectbox(
        "Season",
        options=seasons,
        index=len(seasons) - 1 if seasons else 0,
        key="faceoff_season"
    )

    min_faceoffs = f_cols[1].number_input(
        "Minimum faceoffs", min_value=1, max_value=500, value=20, step=5,
        key="min_faceoffs"
    )

    faceoff_metric_options = [
        "faceoff_pct_calc", "faceoffs_won", "faceoffs",
        "faceoffs_won_per_game", "faceoffs_per_game",
        "ground_balls", "ground_balls_per_game", "points", "touches"
    ]

    faceoff_metric = f_cols[2].selectbox(
        "Faceoff metric",
        options=faceoff_metric_options, index=0,
        format_func=pretty_col,
        key="faceoff_metric"
    )

    faceoff_sort_ascending = f_cols[3].selectbox(
        "Sort direction",
        options=["Best high", "Best low"],
        index=0,
        key="faceoff_sort_direction"
    ) == "Best low"

    faceoff_df = query_df("""
        SELECT season, player_id, full_name, position, position_name, teams, games,
               points, goals, assists, ground_balls, faceoffs_won, faceoffs_lost,
               faceoffs, faceoff_pct_calc, faceoffs_won_per_game, faceoffs_per_game,
               ground_balls_per_game, touches, touches_per_game
        FROM marts.player_season_stats
        WHERE season = ?
          AND faceoffs >= ?
        ORDER BY faceoff_pct_calc DESC NULLS LAST
    """, [faceoff_season, min_faceoffs])

    if faceoff_metric in faceoff_df.columns:
        faceoff_df = _pll_safe_sort(faceoff_df, faceoff_metric, lower_is_better=faceoff_sort_ascending)

    safe_bar_chart(
        faceoff_df.head(15).sort_values(faceoff_metric, ascending=not faceoff_sort_ascending),
        x_col="full_name", y_col=faceoff_metric, color_col="teams",
        title=f"{faceoff_season} Faceoff Leaders — {pretty_col(faceoff_metric)}",
        orientation="h"
    )

    faceoff_summary_cols = _pll_select_existing(
        faceoff_df,
        ["season", "full_name", "position", "teams", "games",
         "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct_calc",
         "faceoffs_per_game", "faceoffs_won_per_game",
         "ground_balls", "ground_balls_per_game", "points", "touches"]
    )
    display_table(faceoff_df[faceoff_summary_cols], height=420)

    with st.expander("Advanced faceoff metrics", expanded=False):
        display_table(faceoff_df, height=360)

    st.markdown("### Faceoff Explorer")

    faceoff_names = faceoff_df["full_name"].dropna().unique().tolist()

    if faceoff_names:
        selected_faceoff_player = st.selectbox(
            "Select faceoff player", options=faceoff_names, index=0,
            key="selected_faceoff_player"
        )

        selected_faceoff_id = faceoff_df[faceoff_df["full_name"] == selected_faceoff_player]["player_id"].iloc[0]

        faceoff_games = query_df("""
            SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
                   points, goals, assists, ground_balls, faceoffs_won, faceoffs_lost,
                   faceoffs, faceoff_pct, turnovers, caused_turnovers, touches, total_passes
            FROM clean.player_game_stats
            WHERE player_id = ?
            ORDER BY season DESC, game_number DESC
        """, [selected_faceoff_id])

        profile_header(selected_faceoff_player, "Faceoff game log and trend view")

        faceoff_game_cols = _pll_select_existing(
            faceoff_games,
            ["season", "game_number", "game_date_utc", "team_name", "opponent_team_name",
             "is_home", "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct",
             "ground_balls", "points", "turnovers", "caused_turnovers", "touches"]
        )
        display_table(faceoff_games[faceoff_game_cols], height=360)

        faceoff_game_metric = st.selectbox(
            "Faceoff game trend metric",
            options=[c for c in ["faceoff_pct", "faceoffs_won", "faceoffs", "ground_balls", "points"] if c in faceoff_games.columns],
            index=0,
            format_func=pretty_col,
            key="faceoff_game_metric"
        )

        if len(faceoff_games) > 0:
            faceoff_trend = faceoff_games.sort_values(["season", "game_number"]).copy()
            faceoff_trend["game_label"] = faceoff_trend["season"].astype(str) + " G" + faceoff_trend["game_number"].astype(str)
            safe_line_chart(
                faceoff_trend, x_col="game_label", y_cols=[faceoff_game_metric],
                title=f"{selected_faceoff_player} — {pretty_col(faceoff_game_metric)} by Game"
            )
