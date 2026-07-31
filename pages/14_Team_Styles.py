"""
Team Styles — what kind of team this is, rather than how good it is.

The sidebar filters are not requested: the context and team pickers live in the
main panel, and the six style scores are min-max scaled inside their own context,
so restricting the teams shown would change what a score means.

Three helpers came out of here. `_pll_extra_table_exists` re-implemented
`db.table_exists`; `_pll_context_order` was a byte-for-byte copy of page 13's;
`_pll_metric_bar` is `ui.metric_bar`. The style weights and the pace caveat now
come from `shared/scoring.py`, which the Data Guide verifies against the mart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared import metrics as M
from shared import page as P
from shared import scoring
from shared import ui
from shared.db import query_df, table_exists
from shared.ui import (
    stat_card, display_table, download_csv,
    fmt_value, pretty_col, profile_header, _pll_select_existing
)

ctx = P.init_page(
    "Team Styles",
    "Compare team identity using offense, defense, possession, ball movement, "
    "pace, and scoring margin.",
)


@st.cache_data(ttl=600, show_spinner=False)
def _pll_load_team_style_profiles():
    if not table_exists("marts", "team_style_profiles"):
        return pd.DataFrame()
    return query_df("SELECT * FROM marts.team_style_profiles")


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


# ============================================================
# PAGE CONTENT
# ============================================================

team_profiles = _pll_load_team_style_profiles()
team_profiles = _pll_prepare_team_profiles(team_profiles)

if len(team_profiles) == 0:
    st.info(
        "Team style profiles are not available yet. "
        "Rebuild the warehouse to refresh team style profile data."
    )
else:
    profile_context_options = scoring.context_order(team_profiles, "profile_context")

    if not profile_context_options:
        st.info("No team style profile contexts found in the data.")
        st.stop()

    default_profile_context = scoring.default_context(profile_context_options)

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
        # Built from shared/scoring.py: the weights, the inputs and the definitions
        # are the ones the Data Guide re-derives against the mart, so this can no
        # longer describe a formula the warehouse does not use.
        st.markdown(
            "**Overall Style** is a weighted blend of the six components below — a "
            "description of how a team plays, not a rating of how well.\n\n"
            "**Net Scores/G** is scoring margin per completed, stat-available game, "
            "and is the one column here that *is* a quality measure."
        )
        ui.display_table(scoring.style_weights_frame(), height=260)
        st.caption(scoring.STYLE_SCALING_NOTE)
        ui.note_box("On reading Pace", scoring.STYLE_QUALITY_NOTE)

    download_csv(
        filtered_profiles[team_style_display_cols],
        f"pll_team_style_profiles_{selected_profile_context.replace(' ', '_').lower()}.csv",
        label="Download visible team style table CSV"
    )

    chart_cols = st.columns([1.05, 1.0])

    with chart_cols[0]:
        st.markdown(f"### Team Comparison — {pretty_col(selected_profile_metric)}")
        # safe_bar_chart formats the axis and the value labels from the metric
        # registry, so a percentage renders as a percentage and a score as a score
        # — the old local helper hardcoded two decimals for everything.
        ui.safe_bar_chart(
            filtered_profiles.head(12),
            x_col="team_name",
            y_col=selected_profile_metric,
            color_col="team_name",
            title=f"{pretty_col(selected_profile_metric)} — {selected_profile_context}",
            orientation="h",
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
            st.plotly_chart(fig, width="stretch")

    ui.section(
        "Style Score Breakdown",
        "The six components behind Overall Style, per team.",
    )

    # The component list comes from the weights, so a component added to the
    # warehouse appears here without editing this page. Overall Style is prepended
    # because it is the summary of the other six, not one of them.
    style_score_cols = M.existing(
        filtered_profiles,
        [scoring.STYLE_SCORE_COLUMN] + list(scoring.STYLE_OVERALL_WEIGHTS),
    )

    if len(filtered_profiles) and style_score_cols:
        ui.metric_bar(
            filtered_profiles, "team_name", style_score_cols,
            title=f"Team Style Component Breakdown — {selected_profile_context}",
            height=460,
        )

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
