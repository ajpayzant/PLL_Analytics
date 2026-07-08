"""
shared/ui.py — UI helpers, formatting utilities, and display constants for PLL Analytics.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from html import escape

# ============================================================
# CSS
# ============================================================

SHARED_CSS = """
<style>
    .main .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1700px;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .section-note {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: -0.35rem;
        margin-bottom: 0.6rem;
    }

    .stat-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        box-shadow: 0 8px 20px rgba(0,0,0,0.11);
        min-height: 92px;
        margin-bottom: 10px;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }

    .stat-value {
        font-size: 1.58rem;
        font-weight: 800;
        line-height: 1.15;
        color: #f8fafc;
    }

    .stat-sub {
        color: #cbd5e1;
        font-size: 0.80rem;
        margin-top: 4px;
    }

    .profile-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(135deg, rgba(30,41,59,0.88), rgba(15,23,42,0.75));
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        margin-bottom: 14px;
    }

    .profile-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .profile-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .mini-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(15, 23, 42, 0.54);
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
        margin-bottom: 10px;
        min-height: 126px;
    }

    .mini-title {
        font-size: 1.04rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .mini-line {
        color: #cbd5e1;
        font-size: 0.84rem;
        line-height: 1.45;
    }

    .mini-label {
        color: #94a3b8;
        font-weight: 700;
    }

    .note-box {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 16px 18px;
        background: rgba(15, 23, 42, 0.52);
        margin-bottom: 14px;
    }

    .note-title {
        font-size: 1.04rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
    }
</style>
"""


def apply_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ============================================================
# DISPLAY LABELS
# ============================================================

COL_LABELS = {
    "row_type": "Row",
    "game_label": "Game",
    "season": "Season",
    "game_number": "Game #",
    "game_date_utc": "Date",
    "game_date_guess": "Date",
    "team_name": "Team",
    "team_names": "Teams",
    "teams": "Teams",
    "opponent_team_name": "Opponent",
    "opponents": "Opponents",
    "is_home": "Home/Away",
    "full_name": "Player",
    "position": "Pos",
    "position_name": "Position",
    "games": "Games",
    "seasons": "Seasons",
    "wins": "Wins",
    "losses": "Losses",
    "win_pct": "Win %",
    "points": "Points",
    "scoring_points": "Scoring Pts",
    "scores": "Scores",
    "scores_against": "Scores Against",
    "goals": "Goals",
    "one_point_goals": "1PT Goals",
    "two_point_goals": "2PT Goals",
    "assists": "Assists",
    "shots": "Shots",
    "shots_on_goal": "SOG",
    "two_point_shots": "2PT Shots",
    "two_point_shots_on_goal": "2PT SOG",
    "ground_balls": "GB",
    "turnovers": "TO",
    "caused_turnovers": "CT",
    "saves": "Saves",
    "clean_saves": "Clean Saves",
    "messy_saves": "Messy Saves",
    "scores_against_average": "SAA Avg",
    "goals_against": "Goals Against",
    "two_point_goals_against": "2PT Goals Against",
    "saa": "Scores Against Avg",
    "faceoffs": "FO",
    "faceoffs_won": "FO Won",
    "faceoffs_lost": "FO Lost",
    "faceoff_pct": "FO %",
    "faceoff_pct_calc": "FO %",
    "save_pct": "Save %",
    "save_pct_calc": "Save %",
    "shot_pct": "Shot %",
    "shot_pct_calc": "Shot %",
    "shots_on_goal_rate": "SOG Rate",
    "shots_on_goal_rate_calc": "SOG Rate",
    "clear_pct": "Clear %",
    "clear_pct_calc": "Clear %",
    "clears": "Clears",
    "clear_attempts": "Clear Att",
    "num_penalties": "Penalties",
    "pim": "PIM",
    "touches": "Touches",
    "total_passes": "Passes",
    "time_in_possession": "Possession Time",
    "official_total_possessions": "Official Possessions",
    "offensive_sequence_proxy": "Offensive Sequences",
    "points_per_game": "Points/G",
    "scoring_points_per_game": "Scoring Pts/G",
    "scores_per_game": "Scores/G",
    "goals_per_game": "Goals/G",
    "one_point_goals_per_game": "1PT Goals/G",
    "two_point_goals_per_game": "2PT Goals/G",
    "assists_per_game": "Assists/G",
    "shots_per_game": "Shots/G",
    "shots_on_goal_per_game": "SOG/G",
    "ground_balls_per_game": "GB/G",
    "turnovers_per_game": "TO/G",
    "caused_turnovers_per_game": "CT/G",
    "saves_per_game": "Saves/G",
    "scores_against_per_game": "Scores Against/G",
    "saa_per_game": "Scores Against Avg/G",
    "faceoffs_per_game": "FO/G",
    "faceoffs_won_per_game": "FO Won/G",
    "faceoffs_lost_per_game": "FO Lost/G",
    "touches_per_game": "Touches/G",
    "total_passes_per_game": "Passes/G",
    "time_in_possession_per_game": "Possession Time/G",
    "official_total_possessions_per_game": "Official Possessions/G",
    "offensive_sequence_proxy_per_game": "Offensive Sequences/G",
    "team_scores": "Scores For",
    "team_scores_per_game": "Scores For/G",
    "scores_allowed": "Scores Allowed",
    "scores_allowed_per_game": "Scores Allowed/G",
    "goals_allowed": "Goals Allowed",
    "goals_allowed_per_game": "Goals Allowed/G",
    "one_point_goals_allowed": "1PT Goals Allowed",
    "two_point_goals_allowed": "2PT Goals Allowed",
    "assists_allowed": "Assists Allowed",
    "opponent_shots": "Opponent Shots",
    "opponent_shots_per_game": "Opponent Shots/G",
    "opponent_shots_on_goal": "Opponent SOG",
    "opponent_shots_on_goal_per_game": "Opponent SOG/G",
    "opponent_goal_pct": "Opponent Goal %",
    "opponent_sog_rate": "Opponent SOG %",
    "opponent_sog_goal_pct": "Opponent Goals/SOG",
    "opponent_turnovers": "Opponent TO",
    "opponent_turnovers_per_game": "Opponent TO/G",
    "opponent_touches": "Opponent Touches",
    "opponent_touches_per_game": "Opponent Touches/G",
    "opponent_total_passes": "Opponent Passes",
    "opponent_total_passes_per_game": "Opponent Passes/G",
    "caused_turnovers_for": "CT",
    "caused_turnovers_for_per_game": "CT/G",
    "saves_for": "Saves",
    "saves_for_per_game": "Saves/G",
    "save_pct_proxy": "Save % Proxy",
    "ct_per_opponent_turnover": "CT/Opp TO",
    "opponent_scores_per_offensive_sequence_proxy": "Scores Allowed/Seq",
    "team_time_in_possession": "Possession Time",
    "team_time_in_possession_per_game": "Possession Time/G",
    "opponent_time_in_possession": "Opp Possession Time",
    "opponent_time_in_possession_per_game": "Opp Possession Time/G",
    "time_in_possession_display": "Possession Time",
    "time_in_possession_pct_display": "Possession %",
    "time_in_possession_available_game": "TOP Available",
    "possession_data_status": "Possession Data Status",
    "possession_data_note": "Possession Data Note",
    "passes_per_touch": "Passes/Touch",
    "seconds_possession_per_touch": "Seconds/Touch",
    "touches_per_offensive_sequence_proxy": "Touches/Sequence",
    "passes_per_offensive_sequence_proxy": "Passes/Sequence",
    "team_score": "Team Score",
    "opponent_score": "Opponent Score",
    "team_a": "Team",
    "team_b": "Opponent",
    "team_a_score": "Team Score",
    "team_b_score": "Opponent Score",
    "team_a_shots": "Team Shots",
    "team_b_shots": "Opponent Shots",
    "team_a_turnovers": "Team TO",
    "team_b_turnovers": "Opponent TO",
    "team_a_ground_balls": "Team GB",
    "team_b_ground_balls": "Opponent GB",
    "team_a_caused_turnovers": "Team CT",
    "team_b_caused_turnovers": "Opponent CT",
    "team_a_possession": "Team Possession",
    "team_b_possession": "Opponent Possession",
    "time_in_possession_pct": "Possession %",
    "event_status_label": "Raw Status",
    "status_display": "Status",
    "away_team_name": "Away",
    "home_team_name": "Home",
    "away_score": "Away Score",
    "home_score": "Home Score",
    "slug": "Slug",
    "event_id": "Event",
    "check_name": "Check",
    "status": "Status",
    "actual": "Actual",
    "expected": "Expected",
    "notes": "Notes",
    "split_type": "Split",
    "stat_type": "Type",
    "definition": "Definition",
    "source_notes": "Source / Notes",
    # UI Polish additions
    "time_in_possession_per_game_mmss": "Possession Time/G",
    "faceoff_pct_for_ranking": "Faceoff Win %",
    "shot_pct_calc": "Shot %",
    "shots_on_goal_rate_calc": "Shots on Goal Rate",
    "faceoff_pct_calc": "Faceoff Win %",
    "save_pct_display": "Save Percentage",
    "save_pct_display_pct": "Save %",
    "save_pct_for_ranking": "Save Percentage",
    "shots_faced_calc": "Shots Faced",
    "shots_faced_per_game_calc": "Shots Faced/G",
    "def_scores_allowed_per_game": "Scores Allowed/G",
    "def_goals_allowed_per_game": "Goals Allowed/G",
    "def_opponent_shots_per_game": "Opp Shots/G",
    "def_opponent_goal_pct": "Opp Goal %",
    "def_save_pct_proxy": "Save % Proxy",
    "score_margin_per_game": "Score Margin/G",
    "overall_rank": "Overall Rank",
    "view_rank": "View Rank",
    "overall_score": "Overall Score",
    "overall_percentile": "Overall Percentile",
    "position_rank": "Position Rank",
    "position_percentile": "Position Percentile",
    "role_group": "Role",
    "base_impact_score": "Base Impact",
    "role_primary_score": "Role Score",
    "role_primary_percentile": "Role Percentile",
    "role_context_value_score": "Role Context Value",
    "role_context_rank": "Role Rank",
    "role_context_percentile": "Role Context Percentile",
    "role_separation_score": "Peer Separation Score",
    "role_adjusted_z": "Peer Separation Z",
    "role_robust_z": "Raw Peer Separation Z",
    "role_value_tier": "Role Tier",
    "role_group_size": "Peer Group Size",
    "role_reliability": "Peer Group Reliability",
    "goal_value_score": "Scoring Value",
    "scoring_value_score": "Scoring Value",
    "playmaking_value_score": "Playmaking Value",
    "one_point_goal_score": "1PT Goal Value",
    "two_point_goal_score": "2PT Goal Value",
    "scoring_points_score": "Scoring Points Value",
    "ground_ball_score": "Ground Ball Value",
    "usage_score": "Usage Value",
    "team_style_overall_score": "Overall Style Score",
    "offensive_volume_score": "Offensive Volume",
    "offensive_efficiency_score": "Offensive Efficiency",
    "ball_movement_score": "Ball Movement",
    "possession_control_score": "Possession Control",
    "defensive_suppression_score": "Defensive Suppression",
    "pace_tempo_score": "Pace / Tempo",
    "net_scores_per_game": "Net Scores/G",
    "ranking_context": "Context",
    "ranking_context_type": "Context Type",
    "ranking_context_max_games": "Max GP",
    "min_games_default": "Default Min GP",
    "eligible_for_default_ranking": "Eligible",
    "sample_size_note": "Sample Note",
    "profile_context": "Context",
    "profile_context_type": "Context Type",
    "profile_rank": "Style Rank",
    "profile_percentile": "Style %ile",
    "pace_label": "Pace",
    "offensive_profile_label": "Off Profile",
    "defensive_profile_label": "Def Profile",
    "possession_profile_label": "Poss Profile",
    "style_summary": "Style Summary",
    "points_per_touch": "Pts/Touch",
    "goals_per_shot": "Goals/Shot",
}


DEFAULT_HIDE_COLS = {
    "player_id", "team_id", "opponent_team_id", "game_id", "event_id", "event_numeric_id",
    "schedule_slug", "game_slug", "source_path", "profile_url", "player_slug",
    "player_name_key", "first_name", "last_name", "team_id_raw", "team_name_raw",
    "opponent_team_id_raw", "opponent_team_name_raw", "away_team_id_raw", "home_team_id_raw",
    "winner_team_id_raw", "loser_team_id_raw", "winner_team_id", "loser_team_id",
    "event_summary_path", "team_game_stats_path", "player_game_stats_path",
    "discovery_source", "source", "source_name", "raw_path"
}


# ============================================================
# FORMATTING HELPERS
# ============================================================

def pretty_col(col):
    return COL_LABELS.get(col, str(col).replace("_", " ").title())


def fmt_value(x, digits=2, pct=False):
    if x is None or pd.isna(x):
        return "—"
    try:
        value = float(x)
    except Exception:
        return str(x)
    if pct:
        return f"{value:.2%}"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def nice_num(x):
    if x is None or pd.isna(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return x
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.2f}"


def format_seconds_for_table(x, total=False):
    if x is None or pd.isna(x):
        return ""
    try:
        x = int(round(float(x)))
    except Exception:
        return x
    sign = "-" if x < 0 else ""
    x = abs(x)
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    if total and h > 0:
        return f"{sign}{h}:{m:02d}:{s:02d}"
    return f"{sign}{m}:{s:02d}"


def mmss_from_seconds(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        seconds = int(round(float(x)))
    except Exception:
        return "—"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{sign}{minutes}:{secs:02d}"


def format_pct_safe(x):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def _pll_seconds_to_mmss(value):
    if value is None or pd.isna(value):
        return "—"
    try:
        seconds = int(round(float(value)))
    except Exception:
        return "—"
    if seconds < 0:
        return "—"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _pll_pct_text(value, digits=1):
    if value is None or pd.isna(value):
        return "—"
    try:
        v = float(value)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:.{digits}f}%"


# ============================================================
# UI COMPONENTS
# ============================================================

def stat_card(label, value, sub=None):
    label = escape(str(label))
    value = escape(str(value))
    sub_html = f'<div class="stat-sub">{escape(str(sub))}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def stat_grid(row, specs, columns=4):
    if row is None or len(specs) == 0:
        return
    cols = st.columns(columns)
    for i, spec in enumerate(specs):
        label, key = spec[0], spec[1]
        digits = spec[2] if len(spec) > 2 else 2
        pct = spec[3] if len(spec) > 3 else False
        value = row.get(key, np.nan) if hasattr(row, "get") else np.nan
        with cols[i % columns]:
            stat_card(label, fmt_value(value, digits=digits, pct=pct))


def profile_header(title, subtitle):
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-title">{escape(str(title))}</div>
            <div class="profile-subtitle">{escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def note_box(title, body):
    st.markdown(
        f"""
        <div class="note-box">
            <div class="mini-title">{escape(str(title))}</div>
            <div class="mini-line">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def profile_summary_cards(df, title_col, specs, columns=3):
    if df is None or len(df) == 0:
        return
    n_cols = max(1, min(columns, len(df)))
    cols = st.columns(n_cols)
    for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        title = escape(str(row.get(title_col, "Unknown")))
        lines = []
        for spec in specs:
            label, key = spec[0], spec[1]
            pct = spec[2] if len(spec) > 2 else False
            raw = row.get(key, np.nan)
            if isinstance(raw, (int, float, np.integer, np.floating)) or pd.api.types.is_number(raw):
                val = fmt_value(raw, pct=pct)
            else:
                val = "—" if raw is None or pd.isna(raw) else str(raw)
            lines.append(
                f'<div class="mini-line"><span class="mini-label">{escape(str(label))}:</span> {escape(str(val))}</div>'
            )
        html = f"""
        <div class="mini-card">
            <div class="mini-title">{title}</div>
            {''.join(lines)}
        </div>
        """
        with cols[i % n_cols]:
            st.markdown(html, unsafe_allow_html=True)


# ============================================================
# TABLE DISPLAY
# ============================================================

def make_unique_columns(cols):
    seen = {}
    output = []
    for col in cols:
        base = str(col)
        if base not in seen:
            seen[base] = 0
            output.append(base)
        else:
            seen[base] += 1
            output.append(f"{base} {seen[base]}")
    return output


def prepare_display_df(df, hide_cols=None, date_cols=None, max_cols=None):
    if df is None:
        return pd.DataFrame()
    out = df.copy().reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    hide = set(DEFAULT_HIDE_COLS)
    if hide_cols:
        hide.update(hide_cols)
    keep_cols = [c for c in out.columns if c not in hide]
    out = out[keep_cols]
    if max_cols is not None and len(out.columns) > max_cols:
        out = out.iloc[:, :max_cols]
    if date_cols is None:
        date_cols = [c for c in out.columns if "date" in c.lower()]
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in list(out.columns):
        c_lower = str(c).lower()
        if c_lower in {"time_in_possession", "team_time_in_possession", "opponent_time_in_possession"}:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(lambda v: format_seconds_for_table(v, total=True))
        elif "time_in_possession_per_game" in c_lower:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(lambda v: format_seconds_for_table(v, total=False))
    for c in out.columns:
        if str(c).lower() == "season":
            numeric_season = pd.to_numeric(out[c], errors="coerce")
            out[c] = numeric_season.astype("Int64").astype(str)
            out[c] = out[c].replace("<NA>", "")
    for c in out.columns:
        if str(c).lower() == "is_home":
            out[c] = out[c].map({1: "Home", 0: "Away", True: "Home", False: "Away"}).fillna(out[c])
    out.columns = make_unique_columns([pretty_col(c) for c in out.columns])
    out = out.reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def display_table(df, height=420, hide_cols=None, date_cols=None, max_cols=None):
    out = prepare_display_df(df, hide_cols=hide_cols, date_cols=date_cols, max_cols=max_cols)
    if out is None or len(out) == 0:
        st.info("No rows available for the selected filters.")
        return
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    fmt_map = {c: nice_num for c in numeric_cols}
    try:
        styler = (
            out.style
            .format(fmt_map, na_rep="")
            .set_properties(**{"text-align": "center", "vertical-align": "middle"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"), ("font-weight", "700"), ("vertical-align", "middle")]},
                {"selector": "td", "props": [("text-align", "center"), ("vertical-align", "middle")]}
            ])
        )
        st.dataframe(styler, use_container_width=True, hide_index=True, height=height)
    except Exception:
        st.dataframe(out, use_container_width=True, hide_index=True, height=height)


def comparison_matrix(df, entity_col, metrics):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        row = {"Metric": pretty_col(metric)}
        for _, r in df.iterrows():
            entity = str(r.get(entity_col, "Unknown"))
            value = r.get(metric, np.nan)
            row[entity] = fmt_value(value)
        rows.append(row)
    return pd.DataFrame(rows)


def display_comparison_matrix(df, entity_col, metrics, height=420):
    matrix = comparison_matrix(df, entity_col, metrics)
    display_table(matrix, height=height)


def download_csv(df, filename, label="Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


def add_window_summary_rows(df, label_col="row_type"):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy().reset_index(drop=True)
    out.insert(0, label_col, [f"Game {i + 1}" for i in range(len(out))])
    excluded_numeric = {"season", "game_number", "is_home"}
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns
        if c not in excluded_numeric
    ]
    total_row = {c: "" for c in out.columns}
    avg_row = {c: "" for c in out.columns}
    total_row[label_col] = "Window Total"
    avg_row[label_col] = "Window Avg"
    for c in numeric_cols:
        total_row[c] = out[c].sum(skipna=True)
        avg_row[c] = out[c].mean(skipna=True)
    return pd.concat([out, pd.DataFrame([total_row, avg_row])], ignore_index=True)


# ============================================================
# CHART HELPERS
# ============================================================

def clean_chart_x(df, x_col):
    out = df.copy()
    if x_col in out.columns and str(x_col).lower() == "season":
        out[x_col] = pd.to_numeric(out[x_col], errors="coerce").astype("Int64").astype(str)
        out[x_col] = out[x_col].replace("<NA>", "")
    return out


def standardize_chart(fig, category_x=False):
    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=60, b=25),
        hovermode="x unified"
    )
    fig.update_yaxes(tickformat=".2f")
    if category_x:
        fig.update_xaxes(type="category")
    fig.update_traces(hovertemplate=None)
    return fig


def safe_line_chart(df, x_col, y_cols, title, color_col=None):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return
    if x_col not in df.columns:
        st.warning(f"Missing x-axis column: {x_col}")
        return
    available_y_cols = [c for c in y_cols if c in df.columns]
    if not available_y_cols:
        st.warning("No requested y-axis columns are available.")
        return
    use_cols = [x_col] + ([color_col] if color_col and color_col in df.columns else []) + available_y_cols
    chart_df = clean_chart_x(df[use_cols].copy(), x_col)
    fig = px.line(
        chart_df,
        x=x_col,
        y=available_y_cols,
        color=color_col if color_col and color_col in chart_df.columns else None,
        markers=True,
        title=title,
        labels={c: pretty_col(c) for c in chart_df.columns}
    )
    fig = standardize_chart(fig, category_x=(str(x_col).lower() == "season"))
    st.plotly_chart(fig, use_container_width=True)


def safe_bar_chart(df, x_col, y_col, title, color_col=None, orientation="v"):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return
    required = [x_col, y_col]
    if color_col:
        required.append(color_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing chart columns: {missing}")
        return
    chart_df = clean_chart_x(df.copy(), x_col)
    if orientation == "h":
        fig = px.bar(
            chart_df, x=y_col, y=x_col, color=color_col, text=y_col,
            title=title, orientation="h",
            labels={c: pretty_col(c) for c in chart_df.columns}
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            chart_df, x=x_col, y=y_col, color=color_col, text=y_col,
            title=title,
            labels={c: pretty_col(c) for c in chart_df.columns}
        )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    if color_col == x_col:
        fig.update_layout(showlegend=False)
    fig = standardize_chart(fig, category_x=(str(x_col).lower() == "season"))
    st.plotly_chart(fig, use_container_width=True)


def safe_scatter(df, x_col, y_col, size_col=None, color_col=None, title="Scatter"):
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return
    required = [x_col, y_col]
    if size_col:
        required.append(size_col)
    if color_col:
        required.append(color_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing chart columns: {missing}")
        return
    fig = px.scatter(
        df, x=x_col, y=y_col, size=size_col, color=color_col,
        hover_name="full_name" if "full_name" in df.columns else None,
        title=title,
        labels={c: pretty_col(c) for c in df.columns}
    )
    fig = standardize_chart(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DATA MANIPULATION HELPERS
# ============================================================

def _pll_select_existing(df, cols):
    if df is None or len(df) == 0:
        return []
    return [c for c in cols if c in df.columns]


def _pll_safe_sort(df, metric, lower_is_better=False):
    if df is None or len(df) == 0 or metric not in df.columns:
        return df
    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out.sort_values(metric, ascending=lower_is_better, na_position="last")


def _pll_apply_goalie_save_pct(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    saves = pd.to_numeric(out["saves"], errors="coerce") if "saves" in out.columns else pd.Series(np.nan, index=out.index)
    if "goals_against" in out.columns:
        goals_against = pd.to_numeric(out["goals_against"], errors="coerce")
    elif "scores_against" in out.columns:
        goals_against = pd.to_numeric(out["scores_against"], errors="coerce")
    else:
        goals_against = pd.Series(np.nan, index=out.index)
    shots_faced = saves + goals_against
    save_pct = saves / shots_faced.replace(0, np.nan)
    out["shots_faced_calc"] = shots_faced
    out["save_pct_display"] = save_pct.clip(lower=0, upper=1)
    out["save_pct_display_pct"] = out["save_pct_display"].apply(_pll_pct_text)
    if "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)
        out["shots_faced_per_game_calc"] = shots_faced / games
    return out


def _pll_add_possession_mmss(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "time_in_possession_per_game" in out.columns:
        out["time_in_possession_per_game_mmss"] = out["time_in_possession_per_game"].apply(_pll_seconds_to_mmss)
    elif "time_in_possession" in out.columns and "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)
        out["time_in_possession_per_game"] = pd.to_numeric(out["time_in_possession"], errors="coerce") / games
        out["time_in_possession_per_game_mmss"] = out["time_in_possession_per_game"].apply(_pll_seconds_to_mmss)
    return out


def _pll_page_note(title, body):
    st.markdown(
        f"""
        <div class="note-box">
            <div class="note-title">{escape(str(title))}</div>
            <div>{escape(str(body))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
