import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px

from shared.db import query_df, DB_PATH
from shared.ui import (
    apply_css, stat_card, display_table, download_csv,
    fmt_value, pretty_col, profile_header, _pll_select_existing
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Team Styles · PLL Analytics", page_icon="🥍", layout="wide")
apply_css()

import os
if not os.path.exists(DB_PATH):
    st.error(f"DuckDB warehouse not found: {DB_PATH}")
    st.stop()

try:
    seasons, teams_df, players_df, positions, selected_seasons, selected_teams, selected_positions, min_games = render_sidebar_filters()
except Exception as e:
    st.error("Failed to load PLL warehouse.")
    st.exception(e)
    st.stop()


# ============================================================
# LOCAL HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def _pll_extra_table_exists(schema_name, table_name):
    df = query_df("""
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
    """, [schema_name, table_name])
    return bool(len(df) > 0 and int(df["n"].iloc[0]) > 0)


@st.cache_data(ttl=600, show_spinner=False)
def _pll_load_team_style_profiles():
    if not _pll_extra_table_exists("marts", "team_style_profiles"):
        return pd.DataFrame()
    return query_df("SELECT * FROM marts.team_style_profiles")


def _pll_context_order(df, context_col, type_col, sort_col):
    if df is None or len(df) == 0:
        return []
    work = df.copy()
    if context_col not in work.columns:
        return []
    if type_col not in work.columns:
        work[type_col] = "Other"
    if sort_col not in work.columns:
        labels = work[context_col].astype(str)
        extracted_year = labels.str.extract(r"(20\d{2})", expand=False)
        derived = pd.to_numeric(extracted_year, errors="coerce")
        derived = np.where(labels.str.contains("Career", case=False, na=False), 0, derived)
        derived = np.where(labels.str.contains("Last 10", case=False, na=False), -10, derived)
        derived = np.where(labels.str.contains("Last 5", case=False, na=False), -5, derived)
        work[sort_col] = derived
    out = work[[context_col, type_col, sort_col]].drop_duplicates().copy()
    out["_type_order"] = np.where(out[type_col].astype(str).eq("Career"), 0, 1)
    out["_sort"] = pd.to_numeric(out[sort_col], errors="coerce")
    out = out.sort_values(["_type_order", "_sort", context_col], ascending=[True, False, True], na_position="last")
    return out[context_col].tolist()


def _pll_prepare_team_profiles(team_profiles):
    if team_profiles is None or len(team_profiles) == 0:
        return team_profiles
    df = team_profiles.copy()
    if "scores_per_game" in df.columns and "def_scores_allowed_per_game" in df.columns:
        df["net_scores_per_game"] = (
            pd.to_numeric(df["scores_per_game"], errors="coerce")
            - pd.to_numeric(df["def_scores_allowed_per_game"], errors="coerce")
        )
    else:
        df["net_scores_per_game"] = np.nan
    df["team_identity_label"] = df.get("style_summary", pd.Series("", index=df.index)).fillna("").astype(str)
    return df


def _pll_metric_bar(df, metric, label_col, color_col=None, title=None, n=20):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return
    if metric not in df.columns or label_col not in df.columns:
        st.info("Required chart columns are not available.")
        return
    chart_df = df.copy()
    chart_df[metric] = pd.to_numeric(chart_df[metric], errors="coerce")
    chart_df = chart_df.dropna(subset=[metric]).head(n)
    if len(chart_df) == 0:
        st.info("No chart data available.")
        return
    chart_df = chart_df.sort_values(metric, ascending=True)
    fig = px.bar(
        chart_df,
        x=metric,
        y=label_col,
        color=color_col if color_col in chart_df.columns else None,
        orientation="h",
        text=metric,
        title=title or pretty_col(metric),
        labels={c: pretty_col(c) for c in chart_df.columns}
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="", xaxis_tickformat=".2f", margin=dict(l=10, r=20, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE CONTENT
# ============================================================

st.subheader("Team Styles")
st.markdown(
    '<div class="section-note">Compare team identity using offense, defense, possession, ball movement, pace, and scoring margin.</div>',
    unsafe_allow_html=True
)

team_profiles = _pll_load_team_style_profiles()
team_profiles = _pll_prepare_team_profiles(team_profiles)

if len(team_profiles) == 0:
    st.info(
        "Team style profiles are not available yet. "
        "Rebuild the warehouse to refresh team style profile data."
    )
else:
    profile_context_options = _pll_context_order(
        team_profiles,
        "profile_context",
        "profile_context_type",
        "profile_context_sort"
    )

    season_profile_contexts = [c for c in profile_context_options if "Season" in str(c)]
    default_profile_context = season_profile_contexts[0] if season_profile_contexts else (profile_context_options[0] if profile_context_options else None)

    if not profile_context_options:
        st.info("No team style profile contexts found in the data.")
        st.stop()

    profile_controls = st.columns([1.2, 1.5, 1.1])

    with profile_controls[0]:
        selected_profile_context = st.selectbox(
            "Profile context",
            options=profile_context_options,
            index=profile_context_options.index(default_profile_context) if default_profile_context else 0,
            key="team_style_profile_context"
        )

    profile_context_df = team_profiles[
        team_profiles["profile_context"] == selected_profile_context
    ].copy()

    team_profile_options = sorted(profile_context_df["team_name"].dropna().astype(str).unique().tolist())

    with profile_controls[1]:
        selected_profile_teams = st.multiselect(
            "Teams",
            options=team_profile_options,
            default=team_profile_options,
            key="team_style_profile_teams"
        )

    profile_metric_options = [
        c for c in [
            "team_style_overall_score", "net_scores_per_game",
            "offensive_volume_score", "offensive_efficiency_score",
            "ball_movement_score", "possession_control_score",
            "defensive_suppression_score", "pace_tempo_score",
            "scores_per_game", "def_scores_allowed_per_game", "touches_per_game",
        ]
        if c in profile_context_df.columns
    ]

    with profile_controls[2]:
        selected_profile_metric = st.selectbox(
            "Primary metric",
            options=profile_metric_options,
            index=0,
            format_func=pretty_col,
            key="team_style_profile_metric"
        )

    filtered_profiles = profile_context_df.copy()

    if selected_profile_teams:
        filtered_profiles = filtered_profiles[
            filtered_profiles["team_name"].isin(selected_profile_teams)
        ]

    filtered_profiles = filtered_profiles.sort_values("profile_rank", ascending=True, na_position="last")

    top_team = filtered_profiles["team_name"].iloc[0] if len(filtered_profiles) else "—"
    best_net = (
        filtered_profiles.sort_values("net_scores_per_game", ascending=False)["team_name"].iloc[0]
        if len(filtered_profiles) and "net_scores_per_game" in filtered_profiles.columns
        else "—"
    )
    best_offense = (
        filtered_profiles.sort_values("offensive_efficiency_score", ascending=False)["team_name"].iloc[0]
        if len(filtered_profiles) and "offensive_efficiency_score" in filtered_profiles.columns
        else "—"
    )
    best_defense = (
        filtered_profiles.sort_values("defensive_suppression_score", ascending=False)["team_name"].iloc[0]
        if len(filtered_profiles) and "defensive_suppression_score" in filtered_profiles.columns
        else "—"
    )
    fastest_team = (
        filtered_profiles.sort_values("pace_tempo_score", ascending=False)["team_name"].iloc[0]
        if len(filtered_profiles) and "pace_tempo_score" in filtered_profiles.columns
        else "—"
    )

    profile_cards = st.columns(5)

    with profile_cards[0]:
        stat_card("Top Overall", top_team)

    with profile_cards[1]:
        stat_card("Best Net Margin", best_net)

    with profile_cards[2]:
        stat_card("Best Offense", best_offense)

    with profile_cards[3]:
        stat_card("Best Defense", best_defense)

    with profile_cards[4]:
        stat_card("Fastest Tempo", fastest_team)

    st.markdown("### Team Style Table")
    st.caption("Use Summary View for quick review, Metrics View for component scores, or Full Detail for the full exportable table.")

    team_style_view = st.radio(
        "Team style table view",
        options=["Summary View", "Metrics View", "Full Detail"],
        horizontal=True,
        key="team_style_table_view"
    )

    summary_cols = [
        "profile_rank", "team_name", "games", "wins", "losses", "win_pct",
        "team_style_overall_score", "net_scores_per_game",
        "offensive_profile_label", "defensive_profile_label",
        "possession_profile_label", "pace_label", "style_summary",
    ]

    metrics_cols = [
        "profile_rank", "team_name", "team_style_overall_score",
        "offensive_volume_score", "offensive_efficiency_score",
        "ball_movement_score", "possession_control_score",
        "defensive_suppression_score", "pace_tempo_score",
        "scores_per_game", "def_scores_allowed_per_game",
        "shots_per_game", "def_opponent_shots_per_game",
        "touches_per_game", "time_in_possession_per_game_mmss", "def_save_pct_proxy",
    ]

    full_cols = [
        "profile_rank", "team_name", "games", "wins", "losses", "win_pct",
        "team_style_overall_score", "net_scores_per_game",
        "offensive_volume_score", "offensive_efficiency_score",
        "ball_movement_score", "possession_control_score",
        "defensive_suppression_score", "pace_tempo_score",
        "scores_per_game", "def_scores_allowed_per_game",
        "shots_per_game", "def_opponent_shots_per_game",
        "touches_per_game", "time_in_possession_per_game_mmss", "def_save_pct_proxy",
        "pace_label", "offensive_profile_label", "defensive_profile_label",
        "possession_profile_label", "style_summary",
    ]

    if team_style_view == "Summary View":
        team_style_display_cols = _pll_select_existing(filtered_profiles, summary_cols)
    elif team_style_view == "Metrics View":
        team_style_display_cols = _pll_select_existing(filtered_profiles, metrics_cols)
    else:
        team_style_display_cols = _pll_select_existing(filtered_profiles, full_cols)

    display_table(filtered_profiles[team_style_display_cols], height=420, hide_cols=[], max_cols=None)

    with st.expander("How to Read Team Styles", expanded=False):
        st.markdown(
            """
            - **Overall Style** is the composite team identity score.
            - **Net Scores/G** is scoring margin per completed, stat-available game.
            - **Offensive Efficiency** captures how well the team converts chances into scoring.
            - **Defensive Suppression** captures how well the team limits opponent scoring and shot quality.
            - **Possession Control** uses possession time, touches, and possession-oriented signals.
            - **Pace / Tempo** captures volume and speed of play rather than quality alone.
            """
        )

    download_csv(
        filtered_profiles[team_style_display_cols],
        f"pll_team_style_profiles_{selected_profile_context.replace(' ', '_').lower()}.csv",
        label="Download visible team style table CSV"
    )

    chart_cols = st.columns([1.05, 1.0])

    with chart_cols[0]:
        st.markdown(f"### Team Comparison — {pretty_col(selected_profile_metric)}")
        _pll_metric_bar(
            filtered_profiles,
            metric=selected_profile_metric,
            label_col="team_name",
            color_col="team_name",
            title=f"{pretty_col(selected_profile_metric)} — {selected_profile_context}",
            n=12
        )

    with chart_cols[1]:
        st.markdown("### Offense vs Defense")

        if len(filtered_profiles) > 0 and "offensive_efficiency_score" in filtered_profiles.columns and "defensive_suppression_score" in filtered_profiles.columns:
            hover_data_cols = [
                c for c in [
                    "profile_rank", "games", "wins", "losses", "style_summary",
                    "scores_per_game", "def_scores_allowed_per_game",
                    "net_scores_per_game", "touches_per_game",
                ]
                if c in filtered_profiles.columns
            ]
            fig = px.scatter(
                filtered_profiles,
                x="offensive_efficiency_score",
                y="defensive_suppression_score",
                size="team_style_overall_score" if "team_style_overall_score" in filtered_profiles.columns else None,
                color="net_scores_per_game" if "net_scores_per_game" in filtered_profiles.columns else None,
                text="team_name",
                hover_name="team_name",
                hover_data=hover_data_cols,
                labels={c: pretty_col(c) for c in filtered_profiles.columns},
                title=f"Offensive Efficiency vs Defensive Suppression — {selected_profile_context}"
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_tickformat=".2f",
                yaxis_tickformat=".2f",
                margin=dict(l=10, r=20, t=45, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Style Score Breakdown")

    style_score_cols = [
        c for c in [
            "team_style_overall_score", "offensive_volume_score",
            "offensive_efficiency_score", "ball_movement_score",
            "possession_control_score", "defensive_suppression_score", "pace_tempo_score",
        ]
        if c in filtered_profiles.columns
    ]

    if len(filtered_profiles) > 0 and style_score_cols:
        style_long = filtered_profiles[["team_name"] + style_score_cols].melt(
            id_vars=["team_name"],
            value_vars=style_score_cols,
            var_name="style_metric",
            value_name="score"
        )
        style_long["style_metric_label"] = style_long["style_metric"].apply(pretty_col)

        fig = px.bar(
            style_long,
            x="team_name",
            y="score",
            color="style_metric_label",
            barmode="group",
            title=f"Team Style Component Breakdown — {selected_profile_context}",
            labels={"team_name": "Team", "score": "Score", "style_metric_label": "Metric"}
        )
        fig.update_layout(
            yaxis=dict(range=[0, 100], tickformat=".0f"),
            xaxis_title="",
            margin=dict(l=10, r=20, t=45, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Team Detail Profile")

    if len(filtered_profiles) > 0:
        selected_detail_team = st.selectbox(
            "Select team detail",
            options=filtered_profiles["team_name"].dropna().astype(str).tolist(),
            key="team_style_detail_team"
        )

        team_detail = filtered_profiles[filtered_profiles["team_name"] == selected_detail_team].head(1)

        if len(team_detail):
            row = team_detail.iloc[0]

            detail_cols = st.columns(5)

            with detail_cols[0]:
                stat_card("Style Rank", fmt_value(row.get("profile_rank", np.nan), 0))

            with detail_cols[1]:
                stat_card("Overall", fmt_value(row.get("team_style_overall_score", np.nan), 2))

            with detail_cols[2]:
                stat_card("Net Scores/G", fmt_value(row.get("net_scores_per_game", np.nan), 2))

            with detail_cols[3]:
                stat_card("Scores/G", fmt_value(row.get("scores_per_game", np.nan), 2))

            with detail_cols[4]:
                stat_card("Allowed/G", fmt_value(row.get("def_scores_allowed_per_game", np.nan), 2))

            profile_header(
                selected_detail_team,
                row.get("style_summary", "Team style profile")
            )

            team_detail_cols = [
                c for c in [
                    "team_name", "games", "wins", "losses", "win_pct",
                    "team_style_overall_score", "net_scores_per_game",
                    "offensive_volume_score", "offensive_efficiency_score",
                    "ball_movement_score", "possession_control_score",
                    "defensive_suppression_score", "pace_tempo_score",
                    "scores_per_game", "shots_per_game", "touches_per_game",
                    "total_passes_per_game", "time_in_possession_per_game_mmss",
                    "def_scores_allowed_per_game", "def_goals_allowed_per_game",
                    "def_opponent_shots_per_game", "def_opponent_goal_pct",
                    "def_save_pct_proxy", "pace_label", "offensive_profile_label",
                    "defensive_profile_label", "possession_profile_label", "style_summary",
                ]
                if c in team_detail.columns
            ]

            display_table(team_detail[team_detail_cols], height=240, hide_cols=[], max_cols=None)
