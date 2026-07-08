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
    fmt_value, pretty_col, _pll_select_existing
)
from shared.filters import render_sidebar_filters

st.set_page_config(page_title="Player Rankings · PLL Analytics", page_icon="🥍", layout="wide")
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
def _pll_load_player_rankings():
    if not _pll_extra_table_exists("marts", "player_ranking_profiles"):
        return pd.DataFrame()
    return query_df("SELECT * FROM marts.player_ranking_profiles")


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


def _pll_pct_rank(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    return s.rank(pct=True, ascending=higher_is_better, method="average") * 100


def _pll_clip_score(series):
    return pd.to_numeric(series, errors="coerce").clip(lower=0, upper=100)


def _pll_safe_mean(df, cols):
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(np.nan, index=df.index)
    temp = pd.DataFrame(index=df.index)
    for c in valid_cols:
        temp[c] = pd.to_numeric(df[c], errors="coerce")
    return temp.mean(axis=1, skipna=True)


def _pll_weighted_score(df, weights, fallback_col="overall_impact_score"):
    score = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        valid = vals.notna()
        score.loc[valid] += vals.loc[valid] * float(weight)
        weight_sum.loc[valid] += float(weight)
    out = score / weight_sum.replace(0, np.nan)
    if fallback_col in df.columns:
        fallback = pd.to_numeric(df[fallback_col], errors="coerce")
        out = out.fillna(fallback)
    return _pll_clip_score(out)


def _pll_rank_score_by_context(df, score_col, rank_col, percentile_col, eligible_mask):
    df[rank_col] = np.nan
    df[percentile_col] = np.nan
    if score_col not in df.columns:
        return df
    for context_value, context_idx in df.groupby("ranking_context").groups.items():
        context_idx = list(context_idx)
        context_mask = df.index.isin(context_idx)
        context_eligible = context_mask & eligible_mask & df[score_col].notna()
        if context_eligible.sum() > 0:
            df.loc[context_eligible, rank_col] = (
                df.loc[context_eligible, score_col].rank(ascending=False, method="min")
            )
            df.loc[context_eligible, percentile_col] = (
                df.loc[context_eligible, score_col].rank(ascending=True, pct=True, method="average") * 100
            )
    return df


def _pll_assign_role_tier(adjusted_z):
    if pd.isna(adjusted_z):
        return "Unrated"
    if adjusted_z >= 2.00:
        return "Outlier Elite"
    if adjusted_z >= 1.25:
        return "Elite"
    if adjusted_z >= 0.65:
        return "High-End"
    if adjusted_z >= -0.35:
        return "Average / Starter"
    if adjusted_z >= -1.00:
        return "Below Average"
    return "Low Impact"


def _pll_add_role_separation_metrics(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out["role_primary_score"] = pd.to_numeric(out["role_primary_score"], errors="coerce")
    group_cols = ["ranking_context", "role_group"]
    grp = out.groupby(group_cols)["role_primary_score"]
    out["role_group_size"] = grp.transform(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
    out["role_score_median"] = grp.transform(lambda s: pd.to_numeric(s, errors="coerce").median())
    out["role_score_iqr"] = grp.transform(
        lambda s: (
            pd.to_numeric(s, errors="coerce").quantile(0.75)
            - pd.to_numeric(s, errors="coerce").quantile(0.25)
        )
    )
    out["role_score_std"] = grp.transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0))
    robust_scale = pd.to_numeric(out["role_score_iqr"], errors="coerce") / 1.349
    std_scale = pd.to_numeric(out["role_score_std"], errors="coerce")
    scale = robust_scale.copy()
    scale = scale.where(scale.notna() & (scale > 0), std_scale)
    scale = scale.where(scale.notna() & (scale > 0), np.nan)
    out["role_robust_z"] = (
        (pd.to_numeric(out["role_primary_score"], errors="coerce") - pd.to_numeric(out["role_score_median"], errors="coerce"))
        / scale
    )
    out["role_robust_z"] = out["role_robust_z"].replace([np.inf, -np.inf], np.nan).clip(lower=-4, upper=4)
    out["role_separation_score_raw"] = _pll_clip_score(50 + 12.5 * out["role_robust_z"])
    out["role_reliability"] = (
        pd.to_numeric(out["role_group_size"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=8) / 8.0
    )
    out["role_separation_score"] = (
        50
        + out["role_reliability"]
        * (pd.to_numeric(out["role_separation_score_raw"], errors="coerce") - 50)
    )
    out["role_separation_score"] = _pll_clip_score(out["role_separation_score"])
    out["role_adjusted_z"] = (out["role_separation_score"] - 50) / 12.5
    out["role_value_tier"] = out["role_adjusted_z"].apply(_pll_assign_role_tier)
    out["role_context_value_score"] = _pll_weighted_score(
        out,
        {"role_primary_score": 0.50, "role_primary_percentile": 0.25, "role_separation_score": 0.25},
        fallback_col="role_primary_score"
    )
    out["role_context_value_score"] = _pll_clip_score(out["role_context_value_score"])
    out["role_context_rank"] = np.nan
    out["role_context_percentile"] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        idx = list(idx)
        valid = out.index.isin(idx) & out["role_context_value_score"].notna()
        if valid.sum() > 0:
            out.loc[valid, "role_context_rank"] = (
                out.loc[valid, "role_context_value_score"].rank(ascending=False, method="min")
            )
            out.loc[valid, "role_context_percentile"] = (
                out.loc[valid, "role_context_value_score"].rank(ascending=True, pct=True, method="average") * 100
            )
    return out


def _pll_prepare_player_rankings(rankings):
    if rankings is None or len(rankings) == 0:
        return rankings
    df = rankings.copy()
    if "eligible_for_default_ranking" not in df.columns and "is_ranking_eligible" in df.columns:
        df["eligible_for_default_ranking"] = pd.to_numeric(df["is_ranking_eligible"], errors="coerce").fillna(0).astype(int)
    if "is_ranking_eligible" not in df.columns and "eligible_for_default_ranking" in df.columns:
        df["is_ranking_eligible"] = pd.to_numeric(df["eligible_for_default_ranking"], errors="coerce").fillna(0).astype(int)
    if "min_games_default" not in df.columns and "default_min_games_used" in df.columns:
        df["min_games_default"] = df["default_min_games_used"]
    if "default_min_games_used" not in df.columns and "min_games_default" in df.columns:
        df["default_min_games_used"] = df["min_games_default"]
    if "ranking_context_max_games" not in df.columns and "max_games_in_context" in df.columns:
        df["ranking_context_max_games"] = df["max_games_in_context"]
    if "max_games_in_context" not in df.columns and "ranking_context_max_games" in df.columns:
        df["max_games_in_context"] = df["ranking_context_max_games"]
    if "ranking_context_sort" not in df.columns and "ranking_sort_order" in df.columns:
        df["ranking_context_sort"] = df["ranking_sort_order"]
    if "ranking_sort_order" not in df.columns and "ranking_context_sort" in df.columns:
        df["ranking_sort_order"] = df["ranking_context_sort"]
    if "overall_score" not in df.columns:
        for score_col in ["overall_score", "overall_impact_score", "base_impact_score"]:
            if score_col in df.columns:
                df["overall_score"] = pd.to_numeric(df[score_col], errors="coerce")
                break
    if "overall_impact_score" not in df.columns and "overall_score" in df.columns:
        df["overall_impact_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    if "usage_possession_score" not in df.columns and "usage_score" in df.columns:
        df["usage_possession_score"] = df["usage_score"]
    if "usage_score" not in df.columns and "usage_possession_score" in df.columns:
        df["usage_score"] = df["usage_possession_score"]
    if "scoring_value_score" not in df.columns and "goal_value_score" in df.columns:
        df["scoring_value_score"] = df["goal_value_score"]
    if "goal_value_score" not in df.columns and "scoring_value_score" in df.columns:
        df["goal_value_score"] = df["scoring_value_score"]
    if "playmaking_value_score" not in df.columns:
        df["playmaking_value_score"] = np.nan
    if "offensive_creation_score" not in df.columns:
        if "scoring_value_score" in df.columns and "playmaking_value_score" in df.columns:
            df["offensive_creation_score"] = (
                0.60 * pd.to_numeric(df["scoring_value_score"], errors="coerce").fillna(50)
                + 0.40 * pd.to_numeric(df["playmaking_value_score"], errors="coerce").fillna(50)
            )
        else:
            df["offensive_creation_score"] = np.nan
    for col in [
        "overall_rank", "overall_percentile", "position_rank", "position_percentile",
        "offensive_rank", "defensive_rank", "faceoff_rank", "goalie_rank",
        "role_context_value_score", "role_context_rank", "role_context_percentile",
        "role_primary_score", "role_primary_percentile", "role_separation_score",
        "scoring_value_score", "goal_value_score", "playmaking_value_score",
        "offensive_creation_score", "ground_ball_score", "base_impact_score",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    if "eligible_for_default_ranking" in df.columns:
        eligible_mask = pd.to_numeric(df["eligible_for_default_ranking"], errors="coerce").fillna(0).astype(bool)
    else:
        eligible_mask = df["overall_score"].notna() if "overall_score" in df.columns else pd.Series(False, index=df.index)
    if "overall_score" in df.columns:
        if df["overall_rank"].isna().all():
            df = _pll_rank_score_by_context(df, "overall_score", "overall_rank", "overall_percentile", eligible_mask)
        if df["position_rank"].isna().all() and "position" in df.columns:
            for _, idx in df.groupby(["ranking_context", "position"]).groups.items():
                idx = list(idx)
                valid = df.index.isin(idx) & eligible_mask & df["overall_score"].notna()
                if valid.sum() > 0:
                    df.loc[valid, "position_rank"] = df.loc[valid, "overall_score"].rank(ascending=False, method="min")
                    df.loc[valid, "position_percentile"] = df.loc[valid, "overall_score"].rank(ascending=True, pct=True, method="average") * 100
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


def _pll_tier_distribution_chart(df):
    if df is None or len(df) == 0 or "role_value_tier" not in df.columns:
        st.info("No tier distribution data available.")
        return
    tier_order = [
        "Outlier Elite", "Elite", "High-End",
        "Average / Starter", "Below Average", "Low Impact", "Unrated",
    ]
    tier_df = (
        df.groupby(["role_group", "role_value_tier"], dropna=False)
        .size()
        .reset_index(name="players")
    )
    tier_df["role_value_tier"] = pd.Categorical(tier_df["role_value_tier"], categories=tier_order, ordered=True)
    tier_df = tier_df.sort_values(["role_group", "role_value_tier"])
    fig = px.bar(
        tier_df,
        x="role_group",
        y="players",
        color="role_value_tier",
        barmode="stack",
        title="Role Value Tier Distribution",
        labels={"role_group": "Role", "players": "Players", "role_value_tier": "Role Tier"}
    )
    fig.update_layout(margin=dict(l=10, r=20, t=45, b=10), yaxis_tickformat=".0f")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE CONTENT
# ============================================================

st.subheader("Player Rankings")
st.markdown(
    '<div class="section-note">Rank players using the official overall score, which combines production, role value, usage, scoring value, and peer separation.</div>',
    unsafe_allow_html=True
)

rankings = _pll_load_player_rankings()

if len(rankings) == 0:
    st.info(
        "Player ranking profiles are not available yet. "
        "Rebuild the warehouse to refresh player ranking data."
    )
else:
    rankings = _pll_prepare_player_rankings(rankings)

    context_options = _pll_context_order(
        rankings,
        "ranking_context",
        "ranking_context_type",
        "ranking_context_sort"
    )

    season_contexts = [c for c in context_options if "Season" in str(c)]
    default_context = season_contexts[0] if season_contexts else (context_options[0] if context_options else None)

    if not context_options:
        st.info("No ranking contexts found in the data.")
        st.stop()

    controls = st.columns([1.35, 1.0, 0.75, 1.45])

    with controls[0]:
        selected_ranking_context = st.selectbox(
            "Ranking context",
            options=context_options,
            index=context_options.index(default_context) if default_context else 0,
            key="player_rankings_context"
        )

    with controls[1]:
        ranking_view = st.selectbox(
            "Ranking view",
            options=["Overall", "Offense", "Defense", "Faceoff", "Goalie"],
            key="player_rankings_view"
        )

    context_rankings = rankings[rankings["ranking_context"] == selected_ranking_context].copy()

    with controls[2]:
        default_min_games = 1
        if "min_games_default" in context_rankings.columns and context_rankings["min_games_default"].notna().any():
            default_min_games = int(max(1, pd.to_numeric(context_rankings["min_games_default"], errors="coerce").dropna().min()))
        min_rank_games = st.number_input(
            "Min GP",
            min_value=0,
            max_value=50,
            value=default_min_games,
            step=1,
            key="player_rankings_min_gp"
        )

    with controls[3]:
        ranking_player_search = st.text_input(
            "Search player",
            value="",
            key="player_rankings_search"
        )

    filter_cols = st.columns([1.2, 1.2, 1.0, 1.0, 1.0])

    available_positions = sorted(context_rankings["position"].dropna().astype(str).unique().tolist()) if len(context_rankings) else []
    available_teams = sorted(context_rankings["teams"].dropna().astype(str).unique().tolist()) if "teams" in context_rankings.columns and len(context_rankings) else []
    available_tiers = [
        t for t in [
            "Outlier Elite", "Elite", "High-End",
            "Average / Starter", "Below Average", "Low Impact", "Unrated",
        ]
        if t in (context_rankings.get("role_value_tier", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    ]

    with filter_cols[0]:
        selected_ranking_positions = st.multiselect(
            "Positions",
            options=available_positions,
            default=[],
            key="player_rankings_positions"
        )

    with filter_cols[1]:
        selected_ranking_teams = st.multiselect(
            "Teams",
            options=available_teams,
            default=[],
            key="player_rankings_teams"
        )

    with filter_cols[2]:
        selected_role_tiers = st.multiselect(
            "Role tiers",
            options=available_tiers,
            default=[],
            key="player_rankings_role_tiers"
        )

    with filter_cols[3]:
        max_available_rows = int(max(10, len(context_rankings))) if len(context_rankings) else 10
        default_ranking_rows = int(min(max_available_rows, max(100, max_available_rows)))
        ranking_rows = st.number_input(
            "Rows",
            min_value=10,
            max_value=max(500, max_available_rows),
            value=default_ranking_rows,
            step=25,
            key="player_rankings_rows",
            help="Increase this to review every matching player in the selected context. The default is all players in most season contexts."
        )

    with filter_cols[4]:
        show_detail_cols = st.checkbox(
            "Show advanced columns",
            value=False,
            key="player_rankings_show_extra_cols"
        )

    if ranking_view == "Overall":
        rank_col = "overall_rank"
        score_col = "overall_score"
        percentile_col = "overall_percentile"
        view_role = None
    elif ranking_view == "Offense":
        rank_col = "role_context_rank"
        score_col = "role_context_value_score"
        percentile_col = "role_context_percentile"
        view_role = "Offense"
    elif ranking_view == "Defense":
        rank_col = "role_context_rank"
        score_col = "role_context_value_score"
        percentile_col = "role_context_percentile"
        view_role = "Defense"
    elif ranking_view == "Faceoff":
        rank_col = "role_context_rank"
        score_col = "role_context_value_score"
        percentile_col = "role_context_percentile"
        view_role = "Faceoff"
    else:
        rank_col = "role_context_rank"
        score_col = "role_context_value_score"
        percentile_col = "role_context_percentile"
        view_role = "Goalie"

    filtered_rankings = context_rankings.copy()

    if view_role is not None and "role_group" in filtered_rankings.columns:
        filtered_rankings = filtered_rankings[filtered_rankings["role_group"] == view_role]

    if "games" in filtered_rankings.columns:
        filtered_rankings = filtered_rankings[
            pd.to_numeric(filtered_rankings["games"], errors="coerce").fillna(0) >= min_rank_games
        ]

    if selected_ranking_positions:
        filtered_rankings = filtered_rankings[filtered_rankings["position"].isin(selected_ranking_positions)]

    if selected_ranking_teams and "teams" in filtered_rankings.columns:
        filtered_rankings = filtered_rankings[filtered_rankings["teams"].isin(selected_ranking_teams)]

    if selected_role_tiers and "role_value_tier" in filtered_rankings.columns:
        filtered_rankings = filtered_rankings[filtered_rankings["role_value_tier"].isin(selected_role_tiers)]

    if ranking_player_search.strip():
        filtered_rankings = filtered_rankings[
            filtered_rankings["full_name"].astype(str).str.contains(
                ranking_player_search.strip(), case=False, na=False
            )
        ]

    if rank_col in filtered_rankings.columns:
        filtered_rankings["_sort_rank"] = pd.to_numeric(filtered_rankings[rank_col], errors="coerce")
    else:
        filtered_rankings["_sort_rank"] = np.nan

    if score_col in filtered_rankings.columns:
        filtered_rankings["_sort_score"] = pd.to_numeric(filtered_rankings[score_col], errors="coerce")
    else:
        filtered_rankings["_sort_score"] = np.nan

    filtered_rankings["_is_unranked"] = filtered_rankings["_sort_rank"].isna().astype(int)
    filtered_rankings = filtered_rankings.sort_values(
        ["_is_unranked", "_sort_rank", "_sort_score", "games", "full_name"],
        ascending=[True, True, False, False, True],
        na_position="last"
    )

    filtered_rankings["view_rank"] = np.arange(1, len(filtered_rankings) + 1)

    matching_player_count = len(filtered_rankings)
    filtered_rankings = filtered_rankings.head(int(ranking_rows)).copy()

    summary_cols = st.columns(6)

    with summary_cols[0]:
        stat_card("Players Shown", f"{fmt_value(len(filtered_rankings), 0)} / {fmt_value(matching_player_count, 0)}")

    with summary_cols[1]:
        top_name = filtered_rankings["full_name"].iloc[0] if len(filtered_rankings) else "—"
        stat_card("Top Player", top_name)

    with summary_cols[2]:
        avg_score = pd.to_numeric(filtered_rankings.get(score_col, pd.Series(dtype=float)), errors="coerce").mean()
        stat_card("Avg Score", fmt_value(avg_score, 2))

    with summary_cols[3]:
        elite_count = (
            filtered_rankings["role_value_tier"].isin(["Outlier Elite", "Elite"]).sum()
            if "role_value_tier" in filtered_rankings.columns
            else 0
        )
        stat_card("Elite Tier Players", fmt_value(elite_count, 0))

    with summary_cols[4]:
        avg_role_z = pd.to_numeric(filtered_rankings.get("role_adjusted_z", pd.Series(dtype=float)), errors="coerce").mean()
        stat_card("Avg Role Z", fmt_value(avg_role_z, 2))

    with summary_cols[5]:
        max_gp = pd.to_numeric(context_rankings.get("games", pd.Series(dtype=float)), errors="coerce").max()
        stat_card("Max GP", fmt_value(max_gp, 0))

    with st.expander("Ranking Method", expanded=False):
        st.markdown(
            """
            **Role Context Value** combines:
            - **50% Role Score**
            - **25% Role Percentile**
            - **25% Peer Separation Score**

            **Peer Separation Score** uses IQR-based robust z-score distance from the role-group median and shrinks toward average when the role group sample is small.

            **Usage Score** blends leaguewide usage value with role-peer usage value so primary offensive players are not understated and specialists are not artificially inflated.

            **Overall Score**
            - **Offense:** 60% Base Impact + 20% Role Context + 10% Usage + 10% Offensive Creation
            - **Offensive Creation:** 60% Scoring Value + 40% Playmaking Value
            - **Defense:** 58% Base Impact + 34% Role Context + 8% Usage, with defensive impact driven more heavily by caused turnovers
            - **Faceoff:** 74% Base Impact + 14% Role Context + 7% Ground-Ball Value + 5% Usage
            - **Goalie:** 72% transfer-adjusted Base Impact + 12% transfer-adjusted Role Context + 10% transfer-adjusted Save Percentage + 6% transfer-adjusted Save Volume

            **Goalie Transfer Adjustment** compresses goalie-only value toward the league-average baseline for the all-player Overall view. This is not a flat penalty: the dedicated Goalie view still uses full goalie-specific value, while Overall prevents a smaller goalie peer pool from dominating cross-position rankings.

            **Score Scale Calibration** shifts each context slightly toward an average-player baseline near 50 without changing rank order inside that context.
            """
        )

    st.caption(
        "Table guide: Overall Score is the official ranking output. "
        "Role Context Value blends role score, role percentile, and true role separation. "
        "Role Tier summarizes how meaningfully separated the player is from his role peers."
    )

    compact_cols_by_view = {
        "Overall": [
            "overall_rank", "position_rank", "full_name", "position", "role_group", "teams", "games",
            "overall_score", "base_impact_score", "role_context_value_score", "role_value_tier",
            "offensive_score", "usage_possession_score", "defensive_score", "faceoff_score", "goalie_score",
            "scoring_value_score", "playmaking_value_score", "ground_ball_score",
            "points_per_game", "scoring_points_per_game", "goals_per_game",
            "one_point_goals_per_game", "two_point_goals_per_game", "assists_per_game",
            "shots_per_game", "touches_per_game", "caused_turnovers_per_game",
            "ground_balls_per_game", "turnovers_per_game",
            "faceoff_pct_for_ranking", "faceoffs_per_game", "faceoffs_won_per_game",
            "save_pct_for_ranking", "saves_per_game", "scores_against_per_game", "goals_against_per_game",
        ],
        "Offense": [
            "role_context_rank", "overall_rank", "position_rank", "full_name", "position", "teams", "games",
            "role_context_value_score", "overall_score", "role_primary_score", "role_primary_percentile",
            "role_separation_score", "role_value_tier",
            "scoring_value_score", "playmaking_value_score",
            "points_per_game", "scoring_points_per_game", "one_point_goals_per_game",
            "two_point_goals_per_game", "goals_per_game", "assists_per_game", "shots_per_game", "points_per_touch",
        ],
        "Defense": [
            "role_context_rank", "overall_rank", "position_rank", "full_name", "position", "teams", "games",
            "role_context_value_score", "overall_score", "defensive_score", "ct_score", "ground_ball_score",
            "role_primary_score", "role_primary_percentile", "role_separation_score", "role_value_tier",
            "caused_turnovers_per_game", "ground_balls_per_game", "turnovers_per_game",
            "touches_per_game", "points_per_game",
        ],
        "Faceoff": [
            "role_context_rank", "overall_rank", "position_rank", "full_name", "position", "teams", "games",
            "role_context_value_score", "overall_score", "role_primary_score", "role_primary_percentile",
            "role_separation_score", "role_value_tier",
            "faceoff_pct_for_ranking", "faceoffs_per_game", "faceoffs_won_per_game",
            "ground_balls_per_game", "points_per_game",
        ],
        "Goalie": [
            "role_context_rank", "overall_rank", "position_rank", "full_name", "position", "teams", "games",
            "role_context_value_score", "overall_score", "goalie_score", "save_pct_score", "saves_score",
            "role_primary_score", "role_primary_percentile", "role_separation_score", "role_value_tier",
            "save_pct_for_ranking", "saves_per_game", "scores_against_per_game", "goals_against_per_game",
            "touches_per_game",
        ],
    }

    extra_cols = [
        "role_robust_z", "role_adjusted_z", "role_group_size", "role_reliability",
        "offensive_score", "usage_possession_score", "defensive_score", "faceoff_score", "goalie_score",
        "scoring_value_score", "playmaking_value_score", "ground_ball_score",
        "one_point_goal_score", "two_point_goal_score",
        "shot_pct_for_ranking", "sog_rate_for_ranking", "goals_per_shot",
    ]

    ranking_display_cols = compact_cols_by_view.get(ranking_view, compact_cols_by_view["Overall"])
    ranking_display_cols = ["view_rank"] + [c for c in ranking_display_cols if c != "view_rank"]

    if show_detail_cols:
        ranking_display_cols = ranking_display_cols + extra_cols

    ranking_display_cols = list(dict.fromkeys([
        c for c in ranking_display_cols
        if c and c in filtered_rankings.columns
    ]))

    ranking_table_df = filtered_rankings[ranking_display_cols].copy()

    if "role_group" in filtered_rankings.columns:
        role_series = filtered_rankings["role_group"].astype(str)
        role_score_blank_rules = {
            "offensive_score": "Offense",
            "defensive_score": "Defense",
            "faceoff_score": "Faceoff",
            "goalie_score": "Goalie",
        }
        for score_name, role_name in role_score_blank_rules.items():
            if score_name in ranking_table_df.columns:
                ranking_table_df.loc[~role_series.eq(role_name), score_name] = np.nan

        specialist_stat_blank_rules = {
            "Faceoff": ["faceoff_pct_for_ranking", "faceoffs_per_game", "faceoffs_won_per_game"],
            "Goalie": ["save_pct_for_ranking", "saves_per_game", "scores_against_per_game", "goals_against_per_game"],
        }
        for role_name, cols_to_blank in specialist_stat_blank_rules.items():
            not_role_mask = ~role_series.eq(role_name)
            for blank_col in cols_to_blank:
                if blank_col in ranking_table_df.columns:
                    ranking_table_df.loc[not_role_mask, blank_col] = np.nan

    protected_cols = {
        "view_rank", "overall_rank", "position_rank", "full_name", "position",
        "role_group", "teams", "games", "overall_score", "role_context_value_score", "role_value_tier"
    }
    drop_blank_cols = []
    for c in ranking_table_df.columns:
        if c in protected_cols:
            continue
        if ranking_table_df[c].isna().all():
            drop_blank_cols.append(c)
    if drop_blank_cols:
        ranking_table_df = ranking_table_df.drop(columns=drop_blank_cols)

    for c in ranking_table_df.columns:
        if c in {"view_rank", "overall_rank", "position_rank", "role_context_rank", "games"}:
            ranking_table_df[c] = pd.to_numeric(ranking_table_df[c], errors="coerce").round(0)
        elif pd.api.types.is_numeric_dtype(ranking_table_df[c]):
            ranking_table_df[c] = pd.to_numeric(ranking_table_df[c], errors="coerce").round(2)

    st.markdown("### Ranking Table")
    display_table(ranking_table_df, height=540, hide_cols=[], max_cols=None)

    download_csv(
        ranking_table_df,
        f"pll_player_rankings_{selected_ranking_context.replace(' ', '_').lower()}_{ranking_view.lower()}_official.csv",
        label="Download filtered rankings CSV"
    )

    visual_cols = st.columns([1.05, 1.0])

    with visual_cols[0]:
        st.markdown("### Top Scores")
        _pll_metric_bar(
            filtered_rankings,
            metric=score_col,
            label_col="full_name",
            color_col="role_group" if "role_group" in filtered_rankings.columns else "position",
            title=f"{ranking_view} Rankings — {selected_ranking_context}",
            n=min(25, int(ranking_rows))
        )

    with visual_cols[1]:
        st.markdown("### Role Tier Distribution")
        _pll_tier_distribution_chart(
            context_rankings[
                pd.to_numeric(context_rankings.get("games", pd.Series(dtype=float)), errors="coerce").fillna(0) >= min_rank_games
            ]
        )

    st.markdown("### Role Context Value vs Overall Score")

    scatter_df = context_rankings.copy()

    if len(scatter_df):
        if "games" in scatter_df.columns:
            scatter_df = scatter_df[
                pd.to_numeric(scatter_df["games"], errors="coerce").fillna(0) >= min_rank_games
            ]

        hover_cols = [
            c for c in [
                "position", "teams", "games", "overall_rank", "base_impact_score",
                "role_primary_score", "role_primary_percentile", "role_separation_score",
                "role_adjusted_z", "role_value_tier", "goal_value_score",
                "points_per_game", "one_point_goals_per_game", "two_point_goals_per_game",
                "touches_per_game",
            ]
            if c in scatter_df.columns
        ]

        if "role_context_value_score" in scatter_df.columns and "overall_score" in scatter_df.columns:
            fig = px.scatter(
                scatter_df,
                x="role_context_value_score",
                y="overall_score",
                color="role_group" if "role_group" in scatter_df.columns else None,
                size="games" if "games" in scatter_df.columns else None,
                hover_name="full_name" if "full_name" in scatter_df.columns else None,
                hover_data=hover_cols,
                labels={c: pretty_col(c) for c in scatter_df.columns},
                title=f"Role Context Value vs Overall Score — {selected_ranking_context}"
            )
            fig.update_layout(
                xaxis_tickformat=".2f",
                yaxis_tickformat=".2f",
                margin=dict(l=10, r=20, t=45, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Player Detail")

    player_detail_options = filtered_rankings["full_name"].dropna().astype(str).tolist()

    if player_detail_options:
        selected_detail_player = st.selectbox(
            "Select player",
            options=player_detail_options,
            key="player_rankings_detail_player"
        )

        player_detail = filtered_rankings[
            filtered_rankings["full_name"] == selected_detail_player
        ].head(1)

        if len(player_detail):
            row = player_detail.iloc[0]

            detail_cols = st.columns(5)

            with detail_cols[0]:
                stat_card("Overall Rank", fmt_value(row.get("overall_rank", np.nan), 0))

            with detail_cols[1]:
                stat_card("Overall Score", fmt_value(row.get("overall_score", np.nan), 2))

            with detail_cols[2]:
                stat_card("Role Context", fmt_value(row.get("role_context_value_score", np.nan), 2))

            with detail_cols[3]:
                stat_card("Role Z", fmt_value(row.get("role_adjusted_z", np.nan), 2))

            with detail_cols[4]:
                stat_card("Role Tier", row.get("role_value_tier", "—"))

            breakdown_df = pd.DataFrame({
                "metric": [
                    "Overall Score", "Base Impact", "Role Context", "Role Score",
                    "Role Percentile", "Peer Separation", "Usage", "Goal Value", "Ground Ball Value",
                ],
                "score": [
                    row.get("overall_score", np.nan),
                    row.get("base_impact_score", np.nan),
                    row.get("role_context_value_score", np.nan),
                    row.get("role_primary_score", np.nan),
                    row.get("role_primary_percentile", np.nan),
                    row.get("role_separation_score", np.nan),
                    row.get("usage_possession_score", np.nan),
                    row.get("goal_value_score", np.nan),
                    row.get("ground_ball_score", np.nan),
                ]
            }).dropna()

            if len(breakdown_df):
                fig = px.bar(
                    breakdown_df.sort_values("score"),
                    x="score",
                    y="metric",
                    orientation="h",
                    text="score",
                    title=f"{selected_detail_player} — Ranking Component Breakdown",
                    labels={"score": "Score", "metric": "Component"}
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
                fig.update_layout(
                    xaxis=dict(range=[0, 100], tickformat=".2f"),
                    yaxis_title="",
                    margin=dict(l=10, r=20, t=45, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            detail_display_cols = list(dict.fromkeys([
                c for c in [
                    "full_name", "position", "role_group", "teams", "games",
                    "overall_rank", "position_rank", "overall_score", "overall_percentile",
                    "base_impact_score", "role_context_value_score",
                    "role_primary_score", "role_primary_percentile", "role_separation_score",
                    "role_adjusted_z", "role_value_tier",
                    "goal_value_score", "offensive_score", "usage_possession_score",
                    "defensive_score", "faceoff_score", "goalie_score",
                    "points", "scoring_points", "one_point_goals", "two_point_goals",
                    "goals", "assists", "shots",
                    "points_per_game", "scoring_points_per_game",
                    "one_point_goals_per_game", "two_point_goals_per_game",
                    "goals_per_game", "assists_per_game", "shots_per_game", "touches_per_game",
                ]
                if c in player_detail.columns
            ]))

            display_table(player_detail[detail_display_cols], height=240, hide_cols=[], max_cols=None)
