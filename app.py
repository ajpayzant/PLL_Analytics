
import os
from html import escape

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
DB_PATH = os.getenv("PLL_DB_PATH", os.path.join(DATA_DIR, "analytics_database", "pll_warehouse.duckdb"))
ARTIFACT_INDEX_PATH = os.getenv("PLL_ARTIFACT_INDEX_PATH", os.path.join(DATA_DIR, "curated_data", "all_requested_seasons", "artifact_index.csv"))

st.set_page_config(
    page_title="PLL Analytics",
    page_icon="🥍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
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
""", unsafe_allow_html=True)

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
    # >>> PLL_DEFENSE_EXTENSION_LABELS_START
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
    # <<< PLL_DEFENSE_EXTENSION_LABELS_END
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
# GENERAL HELPERS
# ============================================================

def pretty_col(col):
    return COL_LABELS.get(col, str(col).replace("_", " ").title())

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


    # >>> PLL_DEFENSE_EXTENSION_PREPARE_DISPLAY_START
    for c in list(out.columns):
        c_lower = str(c).lower()

        if c_lower in {
            "time_in_possession",
            "team_time_in_possession",
            "opponent_time_in_possession"
        }:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(lambda v: format_seconds_for_table(v, total=True))

        elif "time_in_possession_per_game" in c_lower:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(lambda v: format_seconds_for_table(v, total=False))
    # <<< PLL_DEFENSE_EXTENSION_PREPARE_DISPLAY_END


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
            .set_properties(**{
                "text-align": "center",
                "vertical-align": "middle"
            })
            .set_table_styles([
                {"selector": "th", "props": [
                    ("text-align", "center"),
                    ("font-weight", "700"),
                    ("vertical-align", "middle")
                ]},
                {"selector": "td", "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle")
                ]}
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
            chart_df,
            x=y_col,
            y=x_col,
            color=color_col,
            text=y_col,
            title=title,
            orientation="h",
            labels={c: pretty_col(c) for c in chart_df.columns}
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            chart_df,
            x=x_col,
            y=y_col,
            color=color_col,
            text=y_col,
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
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        hover_name="full_name" if "full_name" in df.columns else None,
        title=title,
        labels={c: pretty_col(c) for c in df.columns}
    )

    fig = standardize_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

@st.cache_data(ttl=600, show_spinner=False)
def query_df(sql, params=None):
    con = get_connection()

    try:
        if params is None:
            return con.execute(sql).df()

        return con.execute(sql, params).df()

    finally:
        con.close()


@st.cache_data(ttl=600, show_spinner=False)
def read_table(table_name: str) -> pd.DataFrame:
    """
    Read a full DuckDB table safely into pandas.

    This helper is required by filter_values() and a few schema-safe loaders.
    The previous GitHub app build referenced read_table() inside try/except
    blocks without defining it, which silently made teams_df and players_df
    empty and caused the Player Profile / Team Profile / Compare tabs to have
    no selectable options.
    """
    con = get_connection()

    try:
        return con.execute(f"SELECT * FROM {table_name}").df()

    finally:
        con.close()

@st.cache_data(ttl=600, show_spinner=False)
def startup_counts():
    df = query_df("""
        SELECT
            (SELECT COUNT(*) FROM clean.game_manifest) AS completed_games,
            (SELECT COUNT(*) FROM clean.player_game_stats) AS player_game_rows,
            (SELECT COUNT(*) FROM clean.team_game_stats) AS team_game_rows,
            (SELECT COUNT(*) FROM clean.player_directory) AS players,
            (SELECT COUNT(*) FROM clean.team_directory) AS teams
    """)
    row = df.iloc[0]
    return {
        "completed_games": int(row["completed_games"]),
        "player_game_rows": int(row["player_game_rows"]),
        "team_game_rows": int(row["team_game_rows"]),
        "players": int(row["players"]),
        "teams": int(row["teams"]),
    }

@st.cache_data(ttl=600, show_spinner=False)
def filter_values():
    """
    Load global season/team/player/position filter values.

    This is intentionally schema-safe across the Colab warehouse and the
    GitHub-produced warehouse. The profile/compare tabs depend on these values.
    If teams_df or players_df is empty, the app will render but the selectboxes
    will appear unusable with no obvious error. This function avoids that by
    reading the underlying directory tables directly and applying fallbacks.
    """

    # ------------------------------------------------------------
    # Seasons
    # ------------------------------------------------------------
    seasons = []

    for table_name in ["clean.game_schedule_all", "clean.team_game_stats", "clean.player_game_stats"]:
        try:
            df = read_table(table_name)

            if df is not None and len(df) > 0 and "season" in df.columns:
                seasons = (
                    pd.to_numeric(df["season"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .drop_duplicates()
                    .sort_values()
                    .tolist()
                )

                if seasons:
                    break
        except Exception:
            continue

    # ------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------
    try:
        teams = read_table("clean.team_directory")
    except Exception:
        teams = pd.DataFrame()

    if teams is None or len(teams) == 0:
        # Fallback to team_game_stats if team_directory is missing/empty.
        try:
            tgs = read_table("clean.team_game_stats")
            teams = tgs[[c for c in ["team_id", "team_name"] if c in tgs.columns]].copy()
        except Exception:
            teams = pd.DataFrame()

    if teams is None or len(teams) == 0:
        teams = pd.DataFrame(columns=["team_id", "team_name"])
    else:
        teams = teams.copy()

        if "team_id" not in teams.columns:
            for c in ["latest_team_id", "official_team_id", "current_team_id", "team_abbrev", "team"]:
                if c in teams.columns:
                    teams["team_id"] = teams[c]
                    break

        if "team_name" not in teams.columns:
            for c in ["latest_team_name", "official_team_name", "current_team_name", "team_display_name", "name"]:
                if c in teams.columns:
                    teams["team_name"] = teams[c]
                    break

        if "team_id" not in teams.columns:
            teams["team_id"] = pd.NA

        if "team_name" not in teams.columns:
            teams["team_name"] = teams["team_id"]

        teams["team_id"] = teams["team_id"].astype("string")
        teams["team_name"] = teams["team_name"].astype("string")

        teams = (
            teams[["team_id", "team_name"]]
            .dropna(subset=["team_id", "team_name"])
            .query("team_id != '' and team_name != ''")
            .drop_duplicates()
            .sort_values("team_name", na_position="last")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------
    # Players
    # ------------------------------------------------------------
    try:
        players = read_table("clean.player_directory")
    except Exception:
        players = pd.DataFrame()

    if players is None or len(players) == 0:
        # Fallback to player_game_stats if player_directory is missing/empty.
        try:
            pgs = read_table("clean.player_game_stats")
            candidate_cols = [
                c for c in [
                    "player_id", "full_name", "player_name", "name", "display_name",
                    "position", "position_name", "team_id", "team_name"
                ]
                if c in pgs.columns
            ]
            players = pgs[candidate_cols].copy() if candidate_cols else pd.DataFrame()
        except Exception:
            players = pd.DataFrame()

    if players is None or len(players) == 0:
        players = pd.DataFrame(columns=["player_id", "full_name", "position", "position_name", "team_id", "team_name"])
    else:
        players = players.copy()

        if "full_name" not in players.columns:
            for c in ["player_name", "name", "display_name", "latest_full_name"]:
                if c in players.columns:
                    players["full_name"] = players[c]
                    break

        if "full_name" not in players.columns:
            if "player_id" in players.columns:
                players["full_name"] = players["player_id"]
            else:
                players["full_name"] = pd.NA

        if "position" not in players.columns:
            for c in ["position_name", "primary_position", "latest_position"]:
                if c in players.columns:
                    players["position"] = players[c]
                    break

        if "position" not in players.columns:
            players["position"] = pd.NA

        if "position_name" not in players.columns:
            players["position_name"] = players["position"]

        if "team_id" not in players.columns:
            for c in ["latest_team_id", "official_team_id", "current_team_id", "team_abbrev", "team"]:
                if c in players.columns:
                    players["team_id"] = players[c]
                    break

        if "team_id" not in players.columns:
            players["team_id"] = pd.NA

        if "team_name" not in players.columns:
            for c in ["latest_team_name", "official_team_name", "current_team_name", "team_display_name"]:
                if c in players.columns:
                    players["team_name"] = players[c]
                    break

        if "team_name" not in players.columns:
            players["team_name"] = players["team_id"]

        if "player_id" not in players.columns:
            players["player_id"] = (
                players["full_name"]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9]+", "_", regex=True)
                .str.strip("_")
            )

        keep = ["player_id", "full_name", "position", "position_name", "team_id", "team_name"]

        for c in keep:
            if c not in players.columns:
                players[c] = pd.NA

        players["player_id"] = players["player_id"].astype("string")
        players["full_name"] = players["full_name"].astype("string")
        players["position"] = players["position"].astype("string")
        players["position_name"] = players["position_name"].astype("string")
        players["team_id"] = players["team_id"].astype("string")
        players["team_name"] = players["team_name"].astype("string")

        players = (
            players[keep]
            .dropna(subset=["player_id", "full_name"])
            .query("player_id != '' and full_name != ''")
            .drop_duplicates()
            .sort_values("full_name", na_position="last")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------
    if players is not None and len(players) > 0 and "position" in players.columns:
        positions = (
            players["position"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    else:
        positions = []

    return seasons, teams, players, positions


@st.cache_data(ttl=600, show_spinner=False)
def table_index():
    return query_df("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('clean', 'marts', 'qc')
        ORDER BY table_schema, table_name
    """)


# >>> PLL_DEFENSE_EXTENSION_HELPERS_START
@st.cache_data(ttl=600, show_spinner=False)
def table_exists(schema_name, table_name):
    df = query_df("""
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
    """, [schema_name, table_name])

    return bool(len(df) > 0 and int(df["n"].iloc[0]) > 0)


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


# <<< PLL_DEFENSE_EXTENSION_HELPERS_END

# >>> PLL_MATCHUP_UI_CLEANUP_HELPERS_START

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


def render_matchup_scoreboard(matchup, away_name, home_name):
    away_score_raw = matchup.get("away_score", np.nan)
    home_score_raw = matchup.get("home_score", np.nan)

    status = str(matchup.get("status_display", matchup.get("event_status_label", "—"))).title()
    game_num = fmt_value(matchup.get("game_number", np.nan), 0)
    game_date = matchup.get("game_date_display", "—")

    away_score = fmt_value(away_score_raw, 0)
    home_score = fmt_value(home_score_raw, 0)

    away_score_numeric = pd.to_numeric(pd.Series([away_score_raw]), errors="coerce").iloc[0]
    home_score_numeric = pd.to_numeric(pd.Series([home_score_raw]), errors="coerce").iloc[0]

    away_class = ""
    home_class = ""

    if pd.notna(away_score_numeric) and pd.notna(home_score_numeric):
        if away_score_numeric > home_score_numeric:
            away_class = "winner"
        elif home_score_numeric > away_score_numeric:
            home_class = "winner"

    st.markdown(
        f"""
        <style>
            .matchup-scoreboard {{
                display: grid;
                grid-template-columns: 1fr auto 1fr;
                gap: 14px;
                align-items: stretch;
                margin: 8px 0 18px 0;
            }}
            .matchup-team-card {{
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 18px;
                padding: 18px 20px;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            }}
            .matchup-team-card.winner {{
                border-color: rgba(22, 163, 74, 0.55);
                box-shadow: 0 10px 28px rgba(22, 163, 74, 0.14);
            }}
            .matchup-team-label {{
                font-size: 0.75rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                color: #64748b;
                margin-bottom: 6px;
            }}
            .matchup-team-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
            }}
            .matchup-team-name {{
                font-size: 1.35rem;
                font-weight: 850;
                color: #0f172a;
                line-height: 1.1;
            }}
            .matchup-score {{
                font-size: 2.1rem;
                font-weight: 950;
                color: #0f172a;
                min-width: 64px;
                text-align: right;
            }}
            .matchup-vs-card {{
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 76px;
                border-radius: 18px;
                background: #0f172a;
                color: white;
                font-weight: 900;
                letter-spacing: 0.08em;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16);
            }}
            .matchup-meta {{
                font-size: 0.82rem;
                color: #475569;
                margin-top: 8px;
                font-weight: 600;
            }}
        </style>
        <div class="matchup-scoreboard">
            <div class="matchup-team-card {away_class}">
                <div class="matchup-team-label">Away</div>
                <div class="matchup-team-row">
                    <div class="matchup-team-name">{escape(str(away_name))}</div>
                    <div class="matchup-score">{escape(str(away_score))}</div>
                </div>
                <div class="matchup-meta">Game {escape(str(game_num))} · {escape(str(game_date))}</div>
            </div>
            <div class="matchup-vs-card">AT</div>
            <div class="matchup-team-card {home_class}">
                <div class="matchup-team-label">Home</div>
                <div class="matchup-team-row">
                    <div class="matchup-team-name">{escape(str(home_name))}</div>
                    <div class="matchup-score">{escape(str(home_score))}</div>
                </div>
                <div class="matchup-meta">Status: {escape(str(status))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def build_team_boxscore_matrix(team_box_df, away_id, away_name, home_id, home_name):
    if team_box_df is None or len(team_box_df) == 0:
        return pd.DataFrame()

    away = team_box_df[team_box_df["team_id"] == away_id]
    home = team_box_df[team_box_df["team_id"] == home_id]

    if len(away) == 0 or len(home) == 0:
        return pd.DataFrame()

    away = away.iloc[0]
    home = home.iloc[0]

    stat_specs = [
        ("Score", "scores", "number"),
        ("Goals", "goals", "number"),
        ("1PT Goals", "one_point_goals", "number"),
        ("2PT Goals", "two_point_goals", "number"),
        ("Assists", "assists", "number"),
        ("Shots", "shots", "number"),
        ("Shots on Goal", "shots_on_goal", "number"),
        ("Shot %", "shot_pct", "pct"),
        ("Ground Balls", "ground_balls", "number"),
        ("Turnovers", "turnovers", "number"),
        ("Caused Turnovers", "caused_turnovers", "number"),
        ("Faceoffs Won", "faceoffs_won", "number"),
        ("Faceoffs Lost", "faceoffs_lost", "number"),
        ("Faceoff %", "faceoff_pct", "pct"),
        ("Saves", "saves", "number"),
        ("Save %", "save_pct", "pct"),
        ("Penalties", "num_penalties", "number"),
        ("PIM", "pim", "number"),
        ("Touches", "touches", "number"),
        ("Passes", "total_passes", "number"),
        ("Possession Time", "time_in_possession", "time"),
        ("Possession %", "time_in_possession_pct", "pct"),
        ("Official Possessions", "official_total_possessions", "number"),
        ("Offensive Sequences", "offensive_sequence_proxy", "number"),
    ]

    rows = []

    for label, col, kind in stat_specs:
        if col not in team_box_df.columns:
            continue

        away_val = away.get(col, np.nan)
        home_val = home.get(col, np.nan)

        if kind == "time":
            away_fmt = mmss_from_seconds(away_val)
            home_fmt = mmss_from_seconds(home_val)
        elif kind == "pct":
            away_fmt = format_pct_safe(away_val)
            home_fmt = format_pct_safe(home_val)
        else:
            away_fmt = fmt_value(away_val, 2)

        if kind == "time":
            pass
        elif kind == "pct":
            pass
        else:
            home_fmt = fmt_value(home_val, 2)

        rows.append({
            away_name: away_fmt,
            "Stat": label,
            home_name: home_fmt,
        })

    return pd.DataFrame(rows)


def render_completed_game_review(matchup, away_id, away_name, home_id, home_name):
    status = str(matchup.get("status_display", "")).lower()

    if status != "final":
        return

    selected_game_id = matchup.get("event_id", None)

    if selected_game_id is None or pd.isna(selected_game_id):
        st.info("No completed-game ID was found for this matchup.")
        return

    st.markdown("### Completed Game Review")

    team_box = query_df("""
        SELECT
            team_id,
            team_name,
            opponent_team_id,
            opponent_team_name,
            result,
            scores,
            scores_against,
            goals,
            one_point_goals,
            two_point_goals,
            assists,
            shots,
            shot_pct,
            shots_on_goal,
            ground_balls,
            turnovers,
            caused_turnovers,
            faceoffs,
            faceoffs_won,
            faceoffs_lost,
            faceoff_pct,
            saves,
            save_pct,
            goals_against,
            num_penalties,
            pim,
            touches,
            total_passes,
            time_in_possession,
            time_in_possession_pct,
            official_total_possessions,
            offensive_sequence_proxy,
            possession_data_status
        FROM clean.team_game_stats
        WHERE game_id = ?
        ORDER BY CASE WHEN team_id = ? THEN 0 WHEN team_id = ? THEN 1 ELSE 2 END
    """, [selected_game_id, away_id, home_id])

    box_matrix = build_team_boxscore_matrix(team_box, away_id, away_name, home_id, home_name)

    if len(box_matrix) > 0:
        st.markdown("#### Team Box Score")
        display_table(box_matrix, height=520)
    else:
        st.info("No team box score rows found for this completed game.")

    if len(team_box) > 0 and "possession_data_status" in team_box.columns:
        statuses = sorted(team_box["possession_data_status"].dropna().astype(str).unique().tolist())

        if statuses:
            st.caption(
                "Possession data status: "
                + ", ".join(statuses)
                + ". Possession time is shown as MM:SS."
            )

    player_box = query_df("""
        SELECT
            team_id,
            team_name,
            position,
            position_name,
            full_name,
            points,
            scoring_points,
            one_point_goals,
            two_point_goals,
            goals,
            assists,
            shots,
            shot_pct,
            shots_on_goal,
            shots_on_goal_rate,
            ground_balls,
            turnovers,
            caused_turnovers,
            num_penalties,
            pim,
            touches,
            total_passes,
            fo_record,
            faceoff_pct,
            faceoffs,
            faceoffs_won,
            faceoffs_lost,
            scores_against,
            goals_against,
            saves,
            save_pct,
            clean_saves,
            messy_saves
        FROM clean.player_game_stats
        WHERE game_id = ?
        ORDER BY team_name, points DESC NULLS LAST, goals DESC NULLS LAST, assists DESC NULLS LAST, shots DESC NULLS LAST
    """, [selected_game_id])

    if len(player_box) == 0:
        st.info("No player box score rows found for this completed game.")
        return

    st.markdown("#### Player Box Score")

    team_options = [
        (away_name, away_id),
        (home_name, home_id),
    ]

    selected_team_label = st.radio(
        "Player box score team",
        options=[x[0] for x in team_options],
        horizontal=True,
        key=f"completed_box_team_{selected_game_id}"
    )

    selected_team_id = dict(team_options)[selected_team_label]
    selected_players = player_box[player_box["team_id"] == selected_team_id].copy()

    if len(selected_players) == 0:
        st.info("No players found for selected team.")
        return

    offensive_cols = [
        "full_name",
        "position",
        "points",
        "scoring_points",
        "goals",
        "one_point_goals",
        "two_point_goals",
        "assists",
        "shots",
        "shot_pct",
        "shots_on_goal",
        "ground_balls",
        "turnovers",
        "touches",
        "total_passes",
    ]

    defensive_cols = [
        "full_name",
        "position",
        "caused_turnovers",
        "ground_balls",
        "points",
        "num_penalties",
        "pim",
        "shots",
        "touches",
        "total_passes",
    ]

    faceoff_cols = [
        "full_name",
        "position",
        "fo_record",
        "faceoff_pct",
        "faceoffs",
        "faceoffs_won",
        "faceoffs_lost",
        "points",
        "assists",
        "ground_balls",
        "shots",
        "touches",
    ]

    goalie_cols = [
        "full_name",
        "position",
        "scores_against",
        "goals_against",
        "save_pct",
        "saves",
        "clean_saves",
        "messy_saves",
        "touches",
        "total_passes",
    ]

    tab_off, tab_def, tab_fo, tab_goalie = st.tabs([
        "Offense",
        "Defense",
        "Faceoff",
        "Goalie"
    ])

    with tab_off:
        off_df = selected_players[
            [c for c in offensive_cols if c in selected_players.columns]
        ].sort_values(
            [c for c in ["points", "goals", "assists", "shots"] if c in selected_players.columns],
            ascending=False
        )

        display_table(off_df, height=420)

    with tab_def:
        def_df = selected_players.copy()

        if "caused_turnovers" in def_df.columns:
            def_df = def_df[
                (pd.to_numeric(def_df["caused_turnovers"], errors="coerce").fillna(0) > 0)
                | (def_df["position"].astype(str).isin(["D", "LSM", "SSDM", "G"]))
                | (pd.to_numeric(def_df.get("ground_balls", 0), errors="coerce").fillna(0) > 0)
            ]

        def_df = def_df[
            [c for c in defensive_cols if c in def_df.columns]
        ].sort_values(
            [c for c in ["caused_turnovers", "ground_balls", "touches"] if c in def_df.columns],
            ascending=False
        )

        display_table(def_df, height=420)

    with tab_fo:
        fo_df = selected_players.copy()

        if "faceoffs" in fo_df.columns:
            fo_df = fo_df[
                (fo_df["position"].astype(str).isin(["FO", "FOS"]))
                | (pd.to_numeric(fo_df["faceoffs"], errors="coerce").fillna(0) > 0)
            ]

        fo_df = fo_df[
            [c for c in faceoff_cols if c in fo_df.columns]
        ].sort_values(
            [c for c in ["faceoffs", "faceoffs_won", "ground_balls"] if c in fo_df.columns],
            ascending=False
        )

        display_table(fo_df, height=360)

    with tab_goalie:
        goalie_df = selected_players.copy()

        if "saves" in goalie_df.columns:
            goalie_df = goalie_df[
                (goalie_df["position"].astype(str) == "G")
                | (pd.to_numeric(goalie_df["saves"], errors="coerce").fillna(0) > 0)
                | (pd.to_numeric(goalie_df.get("scores_against", 0), errors="coerce").fillna(0) > 0)
            ]

        goalie_df = goalie_df[
            [c for c in goalie_cols if c in goalie_df.columns]
        ].sort_values(
            [c for c in ["saves", "save_pct"] if c in goalie_df.columns],
            ascending=False
        )

        display_table(goalie_df, height=320)


# <<< PLL_MATCHUP_UI_CLEANUP_HELPERS_END


# >>> PLL_MATCHUP_TEAM_PROFILE_CLEANUP_HELPERS_START

def pll_safe_number(x):
    if x is None or pd.isna(x):
        return np.nan

    try:
        return float(x)
    except Exception:
        return np.nan


def pll_clock_from_seconds(x, total=False):
    """
    Converts seconds into readable clock format.

    total=False: 21:51
    total=True and >= 1 hour: 3:42:18
    """
    val = pll_safe_number(x)

    if pd.isna(val):
        return "—"

    seconds = int(round(val))
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if total and hours > 0:
        return f"{sign}{hours}:{minutes:02d}:{secs:02d}"

    return f"{sign}{minutes}:{secs:02d}"


def pll_fmt_profile_value(row, col, kind="number", decimals=2):
    if row is None or col not in row.index:
        return "—"

    val = row.get(col, np.nan)

    if val is None or pd.isna(val):
        return "—"

    if kind == "time_pg":
        return pll_clock_from_seconds(val, total=False)

    if kind == "time_total":
        return pll_clock_from_seconds(val, total=True)

    if kind == "pct":
        try:
            return f"{float(val):.{decimals}%}"
        except Exception:
            return "—"

    if kind == "int":
        try:
            return f"{int(round(float(val))):,}"
        except Exception:
            return "—"

    if kind == "record":
        return str(val)

    try:
        num = float(val)

        if abs(num - round(num)) < 0.0000001:
            return f"{int(round(num)):,}"

        return f"{num:,.{decimals}f}"

    except Exception:
        return str(val)


def pll_add_team_profile_derived_cols(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()

    numeric_cols = [
        "games", "wins", "losses",
        "scores", "goals", "one_point_goals", "two_point_goals", "assists",
        "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers",
        "faceoffs", "faceoffs_won", "faceoffs_lost",
        "saves", "clean_saves", "messy_saves",
        "num_penalties", "pim",
        "touches", "total_passes", "time_in_possession",
        "official_total_possessions", "offensive_sequence_proxy"
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "games" in out.columns:
        games = out["games"].replace(0, np.nan)

        for total_col in [
            "scores", "goals", "one_point_goals", "two_point_goals", "assists",
            "shots", "shots_on_goal", "ground_balls", "turnovers", "caused_turnovers",
            "faceoffs", "faceoffs_won", "faceoffs_lost",
            "saves", "clean_saves", "messy_saves",
            "num_penalties", "pim",
            "touches", "total_passes",
            "time_in_possession",
            "official_total_possessions", "offensive_sequence_proxy"
        ]:
            pg_col = f"{total_col}_per_game"

            if total_col in out.columns and pg_col not in out.columns:
                out[pg_col] = out[total_col] / games

    if "wins" in out.columns and "losses" in out.columns:
        out["record_display"] = (
            out["wins"].fillna(0).round(0).astype(int).astype(str)
            + "-"
            + out["losses"].fillna(0).round(0).astype(int).astype(str)
        )

    if "win_pct" not in out.columns and {"wins", "games"}.issubset(out.columns):
        out["win_pct"] = out["wins"] / out["games"].replace(0, np.nan)

    if "shot_pct_calc" not in out.columns and {"goals", "shots"}.issubset(out.columns):
        out["shot_pct_calc"] = out["goals"] / out["shots"].replace(0, np.nan)

    if "shots_on_goal_rate_calc" not in out.columns and {"shots_on_goal", "shots"}.issubset(out.columns):
        out["shots_on_goal_rate_calc"] = out["shots_on_goal"] / out["shots"].replace(0, np.nan)

    if "faceoff_pct_calc" not in out.columns and {"faceoffs_won", "faceoffs"}.issubset(out.columns):
        out["faceoff_pct_calc"] = out["faceoffs_won"] / out["faceoffs"].replace(0, np.nan)

    if "save_pct_calc" not in out.columns and {"saves", "goals_against"}.issubset(out.columns):
        out["save_pct_calc"] = out["saves"] / (out["saves"] + out["goals_against"]).replace(0, np.nan)

    if "passes_per_touch" not in out.columns and {"total_passes", "touches"}.issubset(out.columns):
        out["passes_per_touch"] = out["total_passes"] / out["touches"].replace(0, np.nan)

    if "seconds_possession_per_touch" not in out.columns and {"time_in_possession", "touches"}.issubset(out.columns):
        out["seconds_possession_per_touch"] = out["time_in_possession"] / out["touches"].replace(0, np.nan)

    if "touches_per_offensive_sequence_proxy" not in out.columns and {"touches", "offensive_sequence_proxy"}.issubset(out.columns):
        out["touches_per_offensive_sequence_proxy"] = (
            out["touches"] / out["offensive_sequence_proxy"].replace(0, np.nan)
        )

    if "passes_per_offensive_sequence_proxy" not in out.columns and {"total_passes", "offensive_sequence_proxy"}.issubset(out.columns):
        out["passes_per_offensive_sequence_proxy"] = (
            out["total_passes"] / out["offensive_sequence_proxy"].replace(0, np.nan)
        )

    return out


def pll_profile_matrix(profile_df, away_id, away_name, home_id, home_name, specs):
    if profile_df is None or len(profile_df) == 0:
        return pd.DataFrame()

    away_row = profile_df[profile_df["team_id"].astype(str) == str(away_id)]
    home_row = profile_df[profile_df["team_id"].astype(str) == str(home_id)]

    if len(away_row) == 0 or len(home_row) == 0:
        return pd.DataFrame()

    away_row = away_row.iloc[0]
    home_row = home_row.iloc[0]

    rows = []

    for label, col, kind, decimals in specs:
        rows.append({
            away_name: pll_fmt_profile_value(away_row, col, kind=kind, decimals=decimals),
            "Stat": label,
            home_name: pll_fmt_profile_value(home_row, col, kind=kind, decimals=decimals),
        })

    return pd.DataFrame(rows)


def render_clean_matchup_team_profile(matchup_season, away_id, away_name, home_id, home_name):
    st.markdown("### Team Season Profile")

    profile_context = f"{matchup_season} Season"

    season_profiles = query_df("""
        SELECT *
        FROM marts.team_season_stats
        WHERE season = ?
          AND team_id IN (?, ?)
        ORDER BY team_name
    """, [matchup_season, away_id, home_id])

    if len(season_profiles) < 2:
        profile_context = "Career"
        season_profiles = query_df("""
            SELECT *
            FROM marts.team_career_stats
            WHERE team_id IN (?, ?)
            ORDER BY team_name
        """, [away_id, home_id])

    if len(season_profiles) == 0:
        st.info("No team profile data found for this matchup.")
        return

    season_profiles = pll_add_team_profile_derived_cols(season_profiles)

    away_profile = season_profiles[season_profiles["team_id"].astype(str) == str(away_id)]
    home_profile = season_profiles[season_profiles["team_id"].astype(str) == str(home_id)]

    if len(away_profile) == 0 or len(home_profile) == 0:
        st.info("Could not find both teams in the selected season profile. Try using career context.")
        return

    away_profile = away_profile.iloc[0]
    home_profile = home_profile.iloc[0]

    st.caption(
        f"Context: {profile_context}. Possession time is displayed as clock time, not raw seconds."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        stat_card(f"{away_name} Record", pll_fmt_profile_value(away_profile, "record_display", kind="record"))

    with c2:
        stat_card(f"{away_name} Scores/G", pll_fmt_profile_value(away_profile, "scores_per_game", decimals=2))

    with c3:
        stat_card(f"{home_name} Record", pll_fmt_profile_value(home_profile, "record_display", kind="record"))

    with c4:
        stat_card(f"{home_name} Scores/G", pll_fmt_profile_value(home_profile, "scores_per_game", decimals=2))

    per_game_specs = [
        ("Games", "games", "int", 0),
        ("Scores/G", "scores_per_game", "number", 2),
        ("Goals/G", "goals_per_game", "number", 2),
        ("1PT Goals/G", "one_point_goals_per_game", "number", 2),
        ("2PT Goals/G", "two_point_goals_per_game", "number", 2),
        ("Assists/G", "assists_per_game", "number", 2),
        ("Shots/G", "shots_per_game", "number", 2),
        ("Shots on Goal/G", "shots_on_goal_per_game", "number", 2),
        ("Ground Balls/G", "ground_balls_per_game", "number", 2),
        ("Turnovers/G", "turnovers_per_game", "number", 2),
        ("Caused Turnovers/G", "caused_turnovers_per_game", "number", 2),
        ("Saves/G", "saves_per_game", "number", 2),
        ("Touches/G", "touches_per_game", "number", 2),
        ("Passes/G", "total_passes_per_game", "number", 2),
        ("Possession/G", "time_in_possession_per_game", "time_pg", 0),
        ("Official Possessions/G", "official_total_possessions_per_game", "number", 2),
        ("Offensive Sequences/G", "offensive_sequence_proxy_per_game", "number", 2),
    ]

    total_specs = [
        ("Games", "games", "int", 0),
        ("Wins", "wins", "int", 0),
        ("Losses", "losses", "int", 0),
        ("Scores", "scores", "int", 0),
        ("Goals", "goals", "int", 0),
        ("1PT Goals", "one_point_goals", "int", 0),
        ("2PT Goals", "two_point_goals", "int", 0),
        ("Assists", "assists", "int", 0),
        ("Shots", "shots", "int", 0),
        ("Shots on Goal", "shots_on_goal", "int", 0),
        ("Ground Balls", "ground_balls", "int", 0),
        ("Turnovers", "turnovers", "int", 0),
        ("Caused Turnovers", "caused_turnovers", "int", 0),
        ("Faceoffs", "faceoffs", "int", 0),
        ("Faceoffs Won", "faceoffs_won", "int", 0),
        ("Faceoffs Lost", "faceoffs_lost", "int", 0),
        ("Saves", "saves", "int", 0),
        ("Touches", "touches", "int", 0),
        ("Passes", "total_passes", "int", 0),
        ("Total Possession Time", "time_in_possession", "time_total", 0),
        ("Official Possessions", "official_total_possessions", "number", 0),
        ("Offensive Sequences", "offensive_sequence_proxy", "int", 0),
    ]

    rate_specs = [
        ("Win %", "win_pct", "pct", 1),
        ("Shot %", "shot_pct_calc", "pct", 1),
        ("SOG Rate", "shots_on_goal_rate_calc", "pct", 1),
        ("Faceoff %", "faceoff_pct_calc", "pct", 1),
        ("Clear %", "clear_pct_calc", "pct", 1),
        ("Save %", "save_pct_calc", "pct", 1),
        ("Passes/Touch", "passes_per_touch", "number", 2),
        ("Touches/Sequence", "touches_per_offensive_sequence_proxy", "number", 2),
        ("Passes/Sequence", "passes_per_offensive_sequence_proxy", "number", 2),
        ("Seconds/Touch", "seconds_possession_per_touch", "number", 2),
    ]

    possession_specs = [
        ("Touches/G", "touches_per_game", "number", 2),
        ("Total Touches", "touches", "int", 0),
        ("Passes/G", "total_passes_per_game", "number", 2),
        ("Total Passes", "total_passes", "int", 0),
        ("Possession/G", "time_in_possession_per_game", "time_pg", 0),
        ("Total Possession Time", "time_in_possession", "time_total", 0),
        ("Official Possessions/G", "official_total_possessions_per_game", "number", 2),
        ("Official Possessions", "official_total_possessions", "number", 0),
        ("Offensive Sequences/G", "offensive_sequence_proxy_per_game", "number", 2),
        ("Offensive Sequences", "offensive_sequence_proxy", "int", 0),
        ("Passes/Touch", "passes_per_touch", "number", 2),
        ("Touches/Sequence", "touches_per_offensive_sequence_proxy", "number", 2),
        ("Seconds/Touch", "seconds_possession_per_touch", "number", 2),
    ]

    tab_pg, tab_totals, tab_rates, tab_poss = st.tabs([
        "Per Game",
        "Totals",
        "Rates",
        "Possession / Touches"
    ])

    with tab_pg:
        pg_matrix = pll_profile_matrix(
            season_profiles,
            away_id,
            away_name,
            home_id,
            home_name,
            per_game_specs
        )
        display_table(pg_matrix, height=520)

    with tab_totals:
        total_matrix = pll_profile_matrix(
            season_profiles,
            away_id,
            away_name,
            home_id,
            home_name,
            total_specs
        )
        display_table(total_matrix, height=560)

    with tab_rates:
        rate_matrix = pll_profile_matrix(
            season_profiles,
            away_id,
            away_name,
            home_id,
            home_name,
            rate_specs
        )
        display_table(rate_matrix, height=420)

    with tab_poss:
        poss_matrix = pll_profile_matrix(
            season_profiles,
            away_id,
            away_name,
            home_id,
            home_name,
            possession_specs
        )
        display_table(poss_matrix, height=520)

        st.caption(
            "Possession time is based on the provider possession-time field converted from seconds. "
            "For games with missing or unusual provider possession timing, review the Possession Data QC section."
        )


# <<< PLL_MATCHUP_TEAM_PROFILE_CLEANUP_HELPERS_END

@st.cache_data(ttl=600, show_spinner=False)
def schedule_display_table():
    return query_df("""
        WITH stat_games AS (
            SELECT DISTINCT season, game_id
            FROM clean.game_manifest
        )
        SELECT
            s.*,
            CASE
                WHEN sg.game_id IS NOT NULL THEN 'final'
                WHEN s.event_status_label = 'unknown' AND s.season <= 2025 THEN 'final'
                ELSE s.event_status_label
            END AS status_display
        FROM clean.game_schedule_all s
        LEFT JOIN stat_games sg
            ON s.season = sg.season
           AND s.event_id = sg.game_id
    """)

def sql_in_filter(column, values):
    if not values:
        return "1=1", []

    placeholders = ", ".join(["?"] * len(values))

    return f"{column} IN ({placeholders})", list(values)

def filter_team_string(df, team_id_col_value):
    if df is None or len(df) == 0:
        return df

    if "teams" not in df.columns:
        return df

    return df[df["teams"].fillna("").astype(str).str.contains(str(team_id_col_value), regex=False)].copy()

# ============================================================
# STARTUP
# ============================================================

if not os.path.exists(DB_PATH):
    st.error(f"DuckDB warehouse not found: {DB_PATH}")
    st.stop()

try:
    counts = startup_counts()
    seasons, teams_df, players_df, positions = filter_values()

except Exception as e:
    st.error("Failed to load PLL warehouse.")
    st.exception(e)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("PLL Analytics")
st.sidebar.caption("Interactive PLL Data Dashboard")

st.sidebar.divider()

default_seasons = [max(seasons)] if seasons else []
selected_seasons = st.sidebar.multiselect(
    "Global Season Filter",
    options=seasons,
    default=default_seasons,
    key="sidebar_seasons"
)

team_options = teams_df["team_name"].dropna().tolist()

selected_teams = st.sidebar.multiselect(
    "Global Team Filter",
    options=team_options,
    default=[],
    key="sidebar_teams"
)

selected_positions = st.sidebar.multiselect(
    "Global Position Filter",
    options=positions,
    default=[],
    key="sidebar_positions"
)

min_games = st.sidebar.number_input(
    "Minimum Games",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    key="sidebar_min_games"
)

st.sidebar.caption("Global filters primarily affect Overview and Leaderboards. Explorer pages have their own filters.")


# >>> PLL_UI_POLISH_HELPERS_START

# ============================================================
# UI POLISH / METRIC NORMALIZATION HELPERS
# ============================================================

COL_LABELS.update({
    "season": "Season",
    "game_date_utc": "Date",
    "game_date_guess": "Date",
    "full_name": "Player",
    "team_name": "Team",
    "opponent_team_name": "Opponent",
    "teams": "Team(s)",
    "position": "Pos",
    "position_name": "Position",
    "games": "Games",
    "wins": "Wins",
    "losses": "Losses",
    "win_pct": "Win %",
    "scores": "Scores",
    "scores_per_game": "Scores/G",
    "scores_against": "Scores Against",
    "scores_against_per_game": "Scores Against/G",
    "goals": "Goals",
    "goals_per_game": "Goals/G",
    "one_point_goals": "1PT Goals",
    "two_point_goals": "2PT Goals",
    "one_point_goals_per_game": "1PT Goals/G",
    "two_point_goals_per_game": "2PT Goals/G",
    "scoring_points": "Scoring Points",
    "scoring_points_per_game": "Scoring Points/G",
    "points": "Points",
    "points_per_game": "Points/G",
    "assists": "Assists",
    "assists_per_game": "Assists/G",
    "shots": "Shots",
    "shots_per_game": "Shots/G",
    "shots_on_goal": "Shots on Goal",
    "shots_on_goal_per_game": "Shots on Goal/G",
    "shot_pct_calc": "Shot %",
    "shots_on_goal_rate_calc": "Shots on Goal Rate",
    "ground_balls": "Ground Balls",
    "ground_balls_per_game": "Ground Balls/G",
    "turnovers": "Turnovers",
    "turnovers_per_game": "Turnovers/G",
    "caused_turnovers": "Caused Turnovers",
    "caused_turnovers_per_game": "Caused Turnovers/G",
    "time_in_possession_per_game_mmss": "Possession Time/G",
    "faceoffs": "Faceoffs",
    "faceoffs_per_game": "Faceoffs/G",
    "faceoffs_won": "Faceoffs Won",
    "faceoffs_won_per_game": "Faceoffs Won/G",
    "faceoffs_lost": "Faceoffs Lost",
    "faceoff_pct_calc": "Faceoff Win %",
    "faceoff_pct_for_ranking": "Faceoff Win %",
    "clear_pct_calc": "Clear %",
    "saves": "Saves",
    "saves_per_game": "Saves/G",
    "clean_saves": "Clean Saves",
    "clean_saves_per_game": "Clean Saves/G",
    "messy_saves": "Messy Saves",
    "messy_saves_per_game": "Messy Saves/G",
    "goals_against": "Goals Against",
    "goals_against_per_game": "Goals Against/G",
    "shots_faced_calc": "Shots Faced",
    "shots_faced_per_game_calc": "Shots Faced/G",
    "save_pct": "Save Percentage",
    "save_pct_calc": "Save Percentage",
    "save_pct_display": "Save Percentage",
    "save_pct_display_pct": "Save %",
    "save_pct_for_ranking": "Save Percentage",
    "save_pct_proxy": "Save % Proxy",
    "def_scores_allowed_per_game": "Scores Allowed/G",
    "scores_allowed_per_game": "Scores Allowed/G",
    "goals_allowed_per_game": "Goals Allowed/G",
    "opponent_shots_per_game": "Opponent Shots/G",
    "def_opponent_shots_per_game": "Opponent Shots/G",
    "opponent_goal_pct": "Opponent Goal %",
    "def_opponent_goal_pct": "Opponent Goal %",
    "opponent_sog_rate": "Opponent SOG Rate",
    "caused_turnovers_for_per_game": "Caused Turnovers/G",
    "opponent_turnovers_per_game": "Opponent Turnovers/G",
    "score_margin_per_game": "Score Margin/G",
    "ct_per_opponent_turnover": "CT per Opponent TO",
    "overall_rank": "Overall Rank",
    "view_rank": "View Rank",
    "overall_score": "Overall Score",
    "position_rank": "Position Rank",
    "base_impact_score": "Base Impact",
    "role_context_value_score": "Role Context Value",
    "role_primary_score": "Role Score",
    "role_primary_percentile": "Role Percentile",
    "role_separation_score": "Peer Separation",
    "role_adjusted_z": "Peer Separation Z",
    "role_value_tier": "Role Tier",
    "goal_value_score": "Scoring Value",
    "scoring_value_score": "Scoring Value",
    "playmaking_value_score": "Playmaking Value",
    "team_style_overall_score": "Overall Style",
    "offensive_volume_score": "Offensive Volume",
    "offensive_efficiency_score": "Offensive Efficiency",
    "ball_movement_score": "Ball Movement",
    "possession_control_score": "Possession Control",
    "defensive_suppression_score": "Defensive Suppression",
    "pace_tempo_score": "Pace / Tempo",
    "net_scores_per_game": "Net Scores/G",
})

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

def _pll_select_existing(df, cols):
    if df is None or len(df) == 0:
        return []

    return [c for c in cols if c in df.columns]

def _pll_get_table_columns(schema_name, table_name):
    try:
        cols_df = query_df("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            ORDER BY ordinal_position
        """, [schema_name, table_name])
        return cols_df["column_name"].astype(str).tolist()
    except Exception:
        return []

def _pll_safe_sort(df, metric, lower_is_better=False):
    if df is None or len(df) == 0 or metric not in df.columns:
        return df

    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out.sort_values(metric, ascending=lower_is_better, na_position="last")

def _pll_apply_goalie_save_pct(df):
    """
    Standardizes goalie save percentage for display.

    Preferred formula:
        Save Percentage = Saves / (Saves + Goals Against)

    This prevents invalid values above 1.000 from being displayed when source
    fields are miscoded or use a nonstandard denominator.
    """
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

# <<< PLL_UI_POLISH_HELPERS_END


# >>> PLL_FINAL_COPY_CLEANUP_START

# ============================================================
# FINAL LABEL / COPY OVERRIDES
# ============================================================

COL_LABELS.update({
    # Official ranking labels
    "overall_rank": "Overall Rank",
    "view_rank": "View Rank",
    "overall_score": "Overall Score",
    "overall_percentile": "Overall Percentile",
    "position_rank": "Position Rank",
    "position_percentile": "Position Percentile",

    # Make role-context terminology more user-friendly
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

    # Scoring / specialist labels
    "goal_value_score": "Scoring Value",
    "one_point_goal_score": "1PT Goal Value",
    "two_point_goal_score": "2PT Goal Value",
    "scoring_points_score": "Scoring Points Value",
    "ground_ball_score": "Ground Ball Value",
    "usage_score": "Usage Value",

    # Team style labels
    "team_style_overall_score": "Overall Style Score",
    "offensive_volume_score": "Offensive Volume",
    "offensive_efficiency_score": "Offensive Efficiency",
    "ball_movement_score": "Ball Movement",
    "possession_control_score": "Possession Control",
    "defensive_suppression_score": "Defensive Suppression",
    "pace_tempo_score": "Pace / Tempo",
})


# Cleaner explanatory text helper for final build
def _pll_final_ranking_explanation():
    st.markdown(
        """
        The official player ranking combines overall production, role-specific value,
        peer ranking, and peer separation. A player is rewarded not only for ranking highly
        within his role, but also for being meaningfully separated from comparable players.
        """
    )

# <<< PLL_FINAL_COPY_CLEANUP_END


# ============================================================
# APP HEADER
# ============================================================

st.title("PLL Analytics")
st.caption("Player Rankings, Team Profiles, Matchup Research, and Trading Intelligence")

tabs = st.tabs([
    "Overview",
    "Season Dashboard",
    "Matchup Preview",
    "Player Profiles",
    "Team Profiles",
    "Specialists",
    "Compare Players",
    "Compare Teams",
    "Player Rankings",
    "Team Styles",
    "Leaderboards",
    "Schedule",
    "Data Guide",
    "Data QA"
])

(
    tab_overview,
    tab_season,
    tab_matchup,
    tab_players,
    tab_teams,
    tab_specialists,
    tab_player_compare,
    tab_team_compare,
    tab_player_rankings,
    tab_team_profiles,
    tab_leaders,
    tab_schedule,
    tab_dictionary,
    tab_quality
) = tabs

# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("Overview")
    st.markdown('<div class="section-note">High-level warehouse status and leaguewide summary views.</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    overview_cards = [
        ("Completed Games", counts["completed_games"]),
        ("Player-Game Rows", counts["player_game_rows"]),
        ("Team-Game Rows", counts["team_game_rows"]),
        ("Players", counts["players"]),
        ("Teams", counts["teams"]),
    ]

    for i, (label, value) in enumerate(overview_cards):
        with cols[i]:
            stat_card(label, fmt_value(value, digits=0))

    schedule_fixed = schedule_display_table()

    season_counts = query_df("""
        SELECT
            season,
            COUNT(DISTINCT game_id) AS completed_stat_games
        FROM clean.game_manifest
        GROUP BY season
        ORDER BY season
    """)

    schedule_counts = (
        schedule_fixed
        .groupby(["season", "status_display"], dropna=False)
        .size()
        .reset_index(name="games")
        .sort_values(["season", "status_display"])
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Completed Games by Season")
        safe_bar_chart(
            season_counts,
            x_col="season",
            y_col="completed_stat_games",
            title="Completed / Stat-Available Games"
        )

    with c2:
        st.markdown("### Schedule Status")
        safe_bar_chart(
            schedule_counts,
            x_col="season",
            y_col="games",
            color_col="status_display",
            title="Schedule Status by Season"
        )

    season_sql, season_params = sql_in_filter("season", selected_seasons)
    team_sql, team_params = sql_in_filter("team_name", selected_teams)
    pos_sql, pos_params = sql_in_filter("position", selected_positions)

    st.markdown("### Top Player Seasons")

    player_sort_metric = st.selectbox(
        "Player season chart metric",
        options=[
            "points", "points_per_game", "goals", "goals_per_game",
            "assists", "assists_per_game", "shots", "shots_per_game",
            "ground_balls", "caused_turnovers"
        ],
        index=0,
        format_func=pretty_col,
        key="overview_player_metric"
    )

    top_players = query_df(f"""
        SELECT
            season,
            full_name,
            position,
            games,
            teams,
            points,
            goals,
            assists,
            shots,
            ground_balls,
            caused_turnovers,
            points_per_game,
            goals_per_game,
            assists_per_game,
            shots_per_game
        FROM marts.player_season_stats
        WHERE games >= ?
          AND {season_sql}
          AND {pos_sql}
        ORDER BY {player_sort_metric} DESC NULLS LAST
        LIMIT 25
    """, [min_games] + season_params + pos_params)

    safe_bar_chart(
        top_players.head(15).sort_values(player_sort_metric),
        x_col="full_name",
        y_col=player_sort_metric,
        color_col="position",
        title=f"Top Player Seasons by {pretty_col(player_sort_metric)}",
        orientation="h"
    )

    display_table(top_players, height=360)

    st.markdown("### Top Team Seasons")

    team_sort_metric = st.selectbox(
        "Team season chart metric",
        options=[
            "scores", "scores_per_game", "goals", "shots", "shots_per_game",
            "turnovers", "turnovers_per_game", "saves", "touches",
            "time_in_possession", "offensive_sequence_proxy"
        ],
        index=1,
        format_func=pretty_col,
        key="overview_team_metric"
    )

    top_teams = query_df(f"""
        SELECT
            season,
            team_name,
            games,
            wins,
            losses,
            win_pct,
            scores,
            scores_per_game,
            goals,
            two_point_goals,
            assists,
            shots,
            shots_per_game,
            saves,
            turnovers,
            turnovers_per_game,
            touches,
            total_passes,
            time_in_possession,
            offensive_sequence_proxy
        FROM marts.team_season_stats
        WHERE games >= ?
          AND {season_sql}
          AND {team_sql}
        ORDER BY {team_sort_metric} DESC NULLS LAST
        LIMIT 25
    """, [min_games] + season_params + team_params)

    safe_bar_chart(
        top_teams.head(15).sort_values(team_sort_metric),
        x_col="team_name",
        y_col=team_sort_metric,
        color_col="season",
        title=f"Top Team Seasons by {pretty_col(team_sort_metric)}",
        orientation="h"
    )

    display_table(top_teams, height=360)


# ============================================================
# SEASON PAGE
# ============================================================

with tab_season:
    st.subheader("Season Dashboard")
    st.markdown(
        '<div class="section-note">Review season-level team performance, player production, defensive context, and schedule status. Current-season views only include completed, stat-available games.</div>',
        unsafe_allow_html=True
    )

    selected_season_page = st.selectbox(
        "Select season",
        options=seasons,
        index=len(seasons) - 1 if seasons else 0,
        key="season_page_season"
    )

    schedule_fixed = schedule_display_table()
    season_schedule = schedule_fixed[schedule_fixed["season"] == selected_season_page].copy()

    season_completed = query_df("""
        SELECT COUNT(DISTINCT game_id) AS games
        FROM clean.game_manifest
        WHERE season = ?
    """, [selected_season_page])["games"].iloc[0]

    season_team_stats = query_df("""
        SELECT *
        FROM marts.team_season_stats
        WHERE season = ?
        ORDER BY scores_per_game DESC NULLS LAST
    """, [selected_season_page])

    season_team_stats = _pll_add_possession_mmss(season_team_stats)

    season_player_stats = query_df("""
        SELECT *
        FROM marts.player_season_stats
        WHERE season = ?
        ORDER BY points DESC NULLS LAST
    """, [selected_season_page])

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        stat_card("Completed Games", fmt_value(season_completed, 0))

    with k2:
        stat_card("Scheduled Games", fmt_value(len(season_schedule), 0))

    with k3:
        stat_card("Teams", fmt_value(season_team_stats["team_name"].nunique() if len(season_team_stats) else 0, 0))

    with k4:
        stat_card("Players", fmt_value(season_player_stats["full_name"].nunique() if len(season_player_stats) else 0, 0))

    if selected_season_page == max(seasons):
        _pll_page_note(
            "Current Season",
            "This season is in progress. Records, ranks, rates, and leaderboards update when completed games become available in the warehouse."
        )

    st.markdown("### Team Rankings")
    st.caption("Team-level production and efficiency using completed, stat-available games only.")

    team_metric_options = [
        c for c in [
            "scores_per_game",
            "score_margin_per_game",
            "shots_per_game",
            "touches_per_game",
            "time_in_possession_per_game",
            "offensive_sequence_proxy_per_game",
            "turnovers_per_game",
            "saves_per_game",
            "faceoff_pct_calc",
            "clear_pct_calc",
        ]
        if c in season_team_stats.columns
    ]

    team_metric = st.selectbox(
        "Team ranking metric",
        options=team_metric_options,
        index=0,
        format_func=pretty_col,
        key="season_team_metric"
    )

    ranked_teams = _pll_safe_sort(season_team_stats, team_metric, lower_is_better=False)

    safe_bar_chart(
        ranked_teams.head(12).sort_values(team_metric),
        x_col="team_name",
        y_col=team_metric,
        color_col="team_name",
        title=f"{selected_season_page} Team Rankings — {pretty_col(team_metric)}",
        orientation="h"
    )

    team_summary_cols = _pll_select_existing(
        ranked_teams,
        [
            "team_name", "games", "wins", "losses", "win_pct",
            "scores_per_game", "score_margin_per_game", "shots_per_game",
            "touches_per_game", "time_in_possession_per_game_mmss",
            "turnovers_per_game", "saves_per_game", "faceoff_pct_calc", "clear_pct_calc",
        ]
    )

    display_table(ranked_teams[team_summary_cols], height=360)

    with st.expander("Advanced team season table", expanded=False):
        team_advanced_cols = _pll_select_existing(
            ranked_teams,
            [
                "season", "team_name", "games", "wins", "losses", "win_pct",
                "scores", "scores_per_game", "goals", "assists",
                "shots", "shots_per_game", "shots_on_goal", "shots_on_goal_per_game",
                "ground_balls", "turnovers", "turnovers_per_game", "caused_turnovers",
                "saves", "saves_per_game", "faceoffs_won", "faceoffs", "faceoff_pct_calc",
                "clear_pct_calc", "touches", "touches_per_game", "total_passes", "total_passes_per_game",
                "time_in_possession", "time_in_possession_per_game_mmss", "offensive_sequence_proxy",
                "offensive_sequence_proxy_per_game"
            ]
        )
        display_table(ranked_teams[team_advanced_cols], height=420)

    st.markdown("### Team Defensive / Opponent Rankings")
    st.caption("Opponent-allowed metrics. For allowed/suppression metrics, lower is generally better unless noted.")

    if table_exists("marts", "team_defense_season_stats"):
        season_defense_df = query_df("""
            SELECT *
            FROM marts.team_defense_season_stats
            WHERE season = ?
            ORDER BY scores_allowed_per_game ASC NULLS LAST
        """, [selected_season_page])

        defense_metric_options = [
            c for c in [
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "opponent_goal_pct",
                "opponent_sog_rate",
                "save_pct_proxy",
                "caused_turnovers_for_per_game",
                "opponent_turnovers_per_game",
                "ct_per_opponent_turnover",
                "score_margin_per_game",
            ]
            if c in season_defense_df.columns
        ]

        if defense_metric_options:
            defense_metric = st.selectbox(
                "Defensive ranking metric",
                options=defense_metric_options,
                index=0,
                format_func=pretty_col,
                key="season_defense_metric"
            )

            lower_is_better = {
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "opponent_goal_pct",
                "opponent_sog_rate",
                "opponent_scores_per_offensive_sequence_proxy",
            }

            ranked_defense = _pll_safe_sort(
                season_defense_df,
                defense_metric,
                lower_is_better=defense_metric in lower_is_better
            )

            safe_bar_chart(
                ranked_defense.head(12).sort_values(
                    defense_metric,
                    ascending=defense_metric not in lower_is_better
                ),
                x_col="team_name",
                y_col=defense_metric,
                color_col="team_name",
                title=f"{selected_season_page} Defensive Rankings — {pretty_col(defense_metric)}",
                orientation="h"
            )

            defense_summary_cols = _pll_select_existing(
                ranked_defense,
                [
                    "team_name", "games",
                    "scores_allowed_per_game", "goals_allowed_per_game",
                    "opponent_shots_per_game", "opponent_goal_pct", "opponent_sog_rate",
                    "save_pct_proxy", "caused_turnovers_for_per_game",
                    "opponent_turnovers_per_game", "score_margin_per_game",
                ]
            )
            display_table(ranked_defense[defense_summary_cols], height=360)

            with st.expander("Advanced defensive / opponent table", expanded=False):
                display_table(ranked_defense, height=420)
        else:
            st.info("No defensive metric columns are available for this season.")
    else:
        st.info("Defensive/opponent marts are not available in the warehouse yet.")

    st.markdown("### Player Leaders")
    st.caption("Player production leaderboard for the selected season. Use minimum games to manage early-season samples.")

    player_filter_cols = st.columns([1.2, 1.0, 1.2, 0.8])

    season_positions = sorted(season_player_stats["position"].dropna().unique().tolist()) if len(season_player_stats) else []

    season_position_filter = player_filter_cols[0].multiselect(
        "Position filter",
        options=season_positions,
        default=[],
        key="season_page_position_filter"
    )

    season_min_games = player_filter_cols[1].number_input(
        "Minimum games",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="season_page_min_games"
    )

    player_metric_options = [
        c for c in [
            "points", "points_per_game", "scoring_points", "scoring_points_per_game",
            "goals", "goals_per_game", "one_point_goals", "two_point_goals",
            "assists", "assists_per_game", "shots", "shots_per_game",
            "ground_balls", "caused_turnovers", "turnovers", "touches"
        ]
        if c in season_player_stats.columns
    ]

    player_metric = player_filter_cols[2].selectbox(
        "Player ranking metric",
        options=player_metric_options,
        index=0,
        format_func=pretty_col,
        key="season_player_metric"
    )

    player_rows = player_filter_cols[3].number_input(
        "Rows",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key="season_player_rows"
    )

    filtered_season_players = season_player_stats.copy()

    if "games" in filtered_season_players.columns:
        filtered_season_players = filtered_season_players[
            pd.to_numeric(filtered_season_players["games"], errors="coerce").fillna(0) >= season_min_games
        ]

    if season_position_filter:
        filtered_season_players = filtered_season_players[
            filtered_season_players["position"].isin(season_position_filter)
        ]

    filtered_season_players = _pll_safe_sort(filtered_season_players, player_metric, lower_is_better=False).head(int(player_rows))

    safe_bar_chart(
        filtered_season_players.head(20).sort_values(player_metric),
        x_col="full_name",
        y_col=player_metric,
        color_col="position",
        title=f"{selected_season_page} Player Leaders — {pretty_col(player_metric)}",
        orientation="h"
    )

    player_summary_cols = _pll_select_existing(
        filtered_season_players,
        [
            "full_name", "position", "teams", "games",
            "points", "scoring_points", "one_point_goals", "two_point_goals",
            "goals", "assists", "shots", "ground_balls", "turnovers", "caused_turnovers",
            "touches", "points_per_game", "goals_per_game", "assists_per_game", "shots_per_game",
        ]
    )
    display_table(filtered_season_players[player_summary_cols], height=420)

    with st.expander("Advanced player leader table", expanded=False):
        player_advanced_cols = _pll_select_existing(
            filtered_season_players,
            [
                "season", "full_name", "position", "teams", "games",
                "points", "points_per_game", "scoring_points", "scoring_points_per_game",
                "one_point_goals", "one_point_goals_per_game", "two_point_goals", "two_point_goals_per_game",
                "goals", "goals_per_game", "assists", "assists_per_game",
                "shots", "shots_per_game", "shots_on_goal", "shot_pct_calc",
                "ground_balls", "ground_balls_per_game", "turnovers", "turnovers_per_game",
                "caused_turnovers", "caused_turnovers_per_game", "touches", "touches_per_game"
            ]
        )
        display_table(filtered_season_players[player_advanced_cols], height=420)

    st.markdown("### Season Schedule")

    schedule_display = season_schedule.copy()

    if len(schedule_display):
        if "away_team_name" in schedule_display.columns and "home_team_name" in schedule_display.columns:
            schedule_display["matchup"] = (
                schedule_display["away_team_name"].astype(str)
                + " at "
                + schedule_display["home_team_name"].astype(str)
            )

        if "away_score" in schedule_display.columns and "home_score" in schedule_display.columns:
            schedule_display["result"] = np.where(
                pd.to_numeric(schedule_display["away_score"], errors="coerce").notna()
                & pd.to_numeric(schedule_display["home_score"], errors="coerce").notna(),
                schedule_display["away_score"].astype(str)
                + " - "
                + schedule_display["home_score"].astype(str),
                "—"
            )

    schedule_cols = _pll_select_existing(
        schedule_display,
        ["season", "game_number", "game_date_guess", "matchup", "result", "status_display", "slug"]
    )

    display_table(schedule_display[schedule_cols].sort_values("game_number"), height=360)

# ============================================================
# UPCOMING MATCHUP PREVIEW
# ============================================================

with tab_matchup:
    st.subheader("Matchup Preview")
    st.markdown('<div class="section-note">Select a scheduled or completed game and compare team form, season profile, head-to-head history, and key players.</div>', unsafe_allow_html=True)

    schedule_fixed = schedule_display_table().copy()
    schedule_fixed["game_date_display"] = pd.to_datetime(schedule_fixed["game_date_guess"], errors="coerce").dt.strftime("%Y-%m-%d")

    matchup_status_filter = st.radio(
        "Game group",
        options=["Upcoming / Scheduled", "Completed / Final", "All Games"],
        horizontal=True,
        key="matchup_status_filter"
    )

    matchup_season = st.selectbox(
        "Matchup season",
        options=seasons,
        index=len(seasons) - 1 if seasons else 0,
        key="matchup_season"
    )

    matchup_pool = schedule_fixed[schedule_fixed["season"] == matchup_season].copy()

    if matchup_status_filter == "Upcoming / Scheduled":
        matchup_pool = matchup_pool[matchup_pool["status_display"] != "final"].copy()
    elif matchup_status_filter == "Completed / Final":
        matchup_pool = matchup_pool[matchup_pool["status_display"] == "final"].copy()

    if len(matchup_pool) == 0:
        st.info("No games available for this filter.")
    else:
        matchup_pool = matchup_pool.sort_values(["game_number", "game_date_guess"]).reset_index(drop=True)
        matchup_pool["matchup_label"] = (
            matchup_pool["season"].astype(str)
            + " G"
            + matchup_pool["game_number"].astype(str)
            + ": "
            + matchup_pool["away_team_name"].astype(str)
            + " at "
            + matchup_pool["home_team_name"].astype(str)
            + " — "
            + matchup_pool["game_date_display"].astype(str)
            + " — "
            + matchup_pool["status_display"].astype(str)
        )

        selected_matchup_label = st.selectbox(
            "Select game",
            options=matchup_pool["matchup_label"].tolist(),
            index=0,
            key="selected_matchup_label"
        )

        matchup = matchup_pool[matchup_pool["matchup_label"] == selected_matchup_label].iloc[0]

        away_id = matchup["away_team_id"]
        home_id = matchup["home_team_id"]
        away_name = matchup["away_team_name"]
        home_name = matchup["home_team_name"]

        profile_header(
            f"{away_name} at {home_name}",
            f"{matchup.get('game_date_display', '—')} | Season {matchup_season} | Game {fmt_value(matchup.get('game_number', np.nan), 0)} | Status: {matchup.get('status_display', '—')}"
        )

        # >>> PLL_MATCHUP_SCOREBOARD_REPLACEMENT_START
        render_matchup_scoreboard(matchup, away_name, home_name)
        # <<< PLL_MATCHUP_SCOREBOARD_REPLACEMENT_END


        


        # >>> PLL_COMPLETED_GAME_REVIEW_START
        render_completed_game_review(matchup, away_id, away_name, home_id, home_name)

        # <<< PLL_COMPLETED_GAME_REVIEW_END


        # >>> PLL_MATCHUP_TEAM_PROFILE_CLEANUP_CALL_START
        render_clean_matchup_team_profile(matchup_season, away_id, away_name, home_id, home_name)

        # <<< PLL_MATCHUP_TEAM_PROFILE_CLEANUP_CALL_END

        # >>> PLL_MATCHUP_DEFENSE_PROFILE_START
        st.markdown("### Defense / Opponent Allowance Profile")

        if table_exists("marts", "team_defense_season_stats"):
            defense_profiles = query_df("""
                SELECT *
                FROM marts.team_defense_season_stats
                WHERE season = ?
                  AND team_id IN (?, ?)
                ORDER BY team_name
            """, [matchup_season, away_id, home_id])

            if len(defense_profiles) < 2:
                defense_profiles = query_df("""
                    SELECT *
                    FROM marts.team_defense_career_stats
                    WHERE team_id IN (?, ?)
                    ORDER BY team_name
                """, [away_id, home_id])

            defense_matchup_metrics = [
                "games",
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "opponent_goal_pct",
                "opponent_sog_rate",
                "save_pct_proxy",
                "caused_turnovers_for_per_game",
                "opponent_turnovers_per_game",
                "ct_per_opponent_turnover",
                "score_margin_per_game",
            ]

            profile_summary_cards(
                defense_profiles,
                title_col="team_name",
                specs=[
                    ("Scores Allowed/G", "scores_allowed_per_game"),
                    ("Opp Shots/G", "opponent_shots_per_game"),
                    ("Opp Goal %", "opponent_goal_pct", True),
                    ("Save % Proxy", "save_pct_proxy", True),
                    ("CT/G", "caused_turnovers_for_per_game"),
                ],
                columns=2
            )

            display_comparison_matrix(defense_profiles, "team_name", defense_matchup_metrics, height=420)

            matchup_defense_metric_options = [
                m for m in defense_matchup_metrics if m in defense_profiles.columns
            ]

            if matchup_defense_metric_options:
                _matchup_def_idx = matchup_defense_metric_options.index("scores_allowed_per_game") if "scores_allowed_per_game" in matchup_defense_metric_options else 0
                matchup_defense_metric = st.selectbox(
                    "Defense matchup chart metric",
                    options=matchup_defense_metric_options,
                    index=_matchup_def_idx,
                    format_func=pretty_col,
                    key="matchup_defense_metric"
                )

                safe_bar_chart(
                    defense_profiles.sort_values(matchup_defense_metric),
                    x_col="team_name",
                    y_col=matchup_defense_metric,
                    color_col="team_name",
                    title=f"Matchup Defensive Comparison — {pretty_col(matchup_defense_metric)}",
                    orientation="h"
                )

        else:
            st.info("Defensive/opponent marts are not available in the warehouse yet.")

        # <<< PLL_MATCHUP_DEFENSE_PROFILE_END

        st.markdown("### Current Form")

        last5 = query_df("""
            SELECT *
            FROM marts.team_last5_stats
            WHERE team_id IN (?, ?)
            ORDER BY team_name
        """, [away_id, home_id])

        last10 = query_df("""
            SELECT *
            FROM marts.team_last10_stats
            WHERE team_id IN (?, ?)
            ORDER BY team_name
        """, [away_id, home_id])

        form_context = st.radio(
            "Form window",
            options=["Last 5", "Last 10"],
            horizontal=True,
            key="matchup_form_window"
        )

        form_df = last5 if form_context == "Last 5" else last10

        form_metrics = [
            "games", "scores_per_game", "shots_per_game", "turnovers_per_game",
            "saves_per_game", "faceoff_pct_calc", "clear_pct_calc",
            "touches_per_game", "time_in_possession_per_game",
            "offensive_sequence_proxy_per_game"
        ]

        display_comparison_matrix(form_df, "team_name", form_metrics, height=420)

        _form_chart_options = [m for m in form_metrics if m in form_df.columns]
        _form_chart_idx = _form_chart_options.index("scores_per_game") if "scores_per_game" in _form_chart_options else 0
        form_chart_metric = st.selectbox(
            "Form chart metric",
            options=_form_chart_options,
            index=_form_chart_idx,
            format_func=pretty_col,
            key="matchup_form_metric"
        )

        safe_bar_chart(
            form_df.sort_values(form_chart_metric),
            x_col="team_name",
            y_col=form_chart_metric,
            color_col="team_name",
            title=f"{form_context} Matchup Comparison — {pretty_col(form_chart_metric)}",
            orientation="h"
        )

        st.markdown("### Head-to-Head History")

        h2h = query_df("""
            SELECT
                team_name,
                opponent_team_name,
                games,
                scores,
                scores_per_game,
                goals,
                assists,
                shots,
                shots_per_game,
                saves,
                turnovers,
                turnovers_per_game,
                ground_balls,
                caused_turnovers,
                faceoff_pct_calc,
                clear_pct_calc
            FROM marts.team_vs_opponent_stats
            WHERE (team_id = ? AND opponent_team_id = ?)
               OR (team_id = ? AND opponent_team_id = ?)
            ORDER BY team_name
        """, [away_id, home_id, home_id, away_id])

        h2h_metrics = [
            "games", "scores", "scores_per_game", "goals", "assists",
            "shots", "shots_per_game", "saves", "turnovers", "turnovers_per_game",
            "ground_balls", "caused_turnovers", "faceoff_pct_calc", "clear_pct_calc"
        ]

        display_comparison_matrix(h2h, "team_name", h2h_metrics, height=360)

        h2h_games = query_df("""
            WITH a AS (
                SELECT *
                FROM clean.team_game_stats
                WHERE team_id = ?
                  AND opponent_team_id = ?
            ),
            b AS (
                SELECT *
                FROM clean.team_game_stats
                WHERE team_id = ?
                  AND opponent_team_id = ?
            )
            SELECT
                a.season,
                a.game_number,
                a.game_date_utc,
                a.team_name AS team_a,
                a.scores AS team_a_score,
                b.team_name AS team_b,
                b.scores AS team_b_score,
                CASE
                    WHEN a.scores > b.scores THEN a.team_name
                    WHEN b.scores > a.scores THEN b.team_name
                    ELSE 'Tie'
                END AS winner,
                a.shots AS team_a_shots,
                b.shots AS team_b_shots,
                a.turnovers AS team_a_turnovers,
                b.turnovers AS team_b_turnovers,
                a.ground_balls AS team_a_ground_balls,
                b.ground_balls AS team_b_ground_balls,
                a.caused_turnovers AS team_a_caused_turnovers,
                b.caused_turnovers AS team_b_caused_turnovers,
                a.time_in_possession AS team_a_possession,
                b.time_in_possession AS team_b_possession
            FROM a
            INNER JOIN b
                ON a.game_id = b.game_id
            ORDER BY a.season DESC, a.game_number DESC
        """, [away_id, home_id, home_id, away_id])

        with st.expander("Show head-to-head game log", expanded=False):
            display_table(h2h_games, height=320)

        st.markdown("### Key Player Form")

        player_form_source = st.radio(
            "Player form source",
            options=["Season", "Last 5", "Last 10"],
            horizontal=True,
            key="matchup_key_player_source"
        )

        player_form_metric = st.selectbox(
            "Key player metric",
            options=[
                "points_per_game", "goals_per_game", "assists_per_game",
                "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game"
            ],
            index=0,
            format_func=pretty_col,
            key="matchup_key_player_metric"
        )

        if player_form_source == "Season":
            key_players = query_df("""
                SELECT
                    full_name,
                    position,
                    teams,
                    games,
                    points,
                    goals,
                    assists,
                    shots,
                    ground_balls,
                    caused_turnovers,
                    points_per_game,
                    goals_per_game,
                    assists_per_game,
                    shots_per_game,
                    ground_balls_per_game,
                    caused_turnovers_per_game
                FROM marts.player_season_stats
                WHERE season = ?
                  AND games >= 1
                ORDER BY points_per_game DESC NULLS LAST
            """, [matchup_season])
        elif player_form_source == "Last 5":
            key_players = query_df("""
                SELECT
                    full_name,
                    position,
                    teams,
                    games,
                    points,
                    goals,
                    assists,
                    shots,
                    ground_balls,
                    caused_turnovers,
                    points_per_game,
                    goals_per_game,
                    assists_per_game,
                    shots_per_game,
                    ground_balls_per_game,
                    caused_turnovers_per_game
                FROM marts.player_last5_stats
                WHERE games >= 1
                ORDER BY points_per_game DESC NULLS LAST
            """)
        else:
            key_players = query_df("""
                SELECT
                    full_name,
                    position,
                    teams,
                    games,
                    points,
                    goals,
                    assists,
                    shots,
                    ground_balls,
                    caused_turnovers,
                    points_per_game,
                    goals_per_game,
                    assists_per_game,
                    shots_per_game,
                    ground_balls_per_game,
                    caused_turnovers_per_game
                FROM marts.player_last10_stats
                WHERE games >= 1
                ORDER BY points_per_game DESC NULLS LAST
            """)

        away_players = filter_team_string(key_players, away_id).sort_values(player_form_metric, ascending=False).head(10)
        home_players = filter_team_string(key_players, home_id).sort_values(player_form_metric, ascending=False).head(10)

        kp1, kp2 = st.columns(2)

        with kp1:
            st.markdown(f"#### {away_name} Key Players")
            safe_bar_chart(
                away_players.sort_values(player_form_metric),
                x_col="full_name",
                y_col=player_form_metric,
                color_col="position",
                title=f"{away_name} — {pretty_col(player_form_metric)}",
                orientation="h"
            )
            display_table(away_players, height=320)

        with kp2:
            st.markdown(f"#### {home_name} Key Players")
            safe_bar_chart(
                home_players.sort_values(player_form_metric),
                x_col="full_name",
                y_col=player_form_metric,
                color_col="position",
                title=f"{home_name} — {pretty_col(player_form_metric)}",
                orientation="h"
            )
            display_table(home_players, height=320)

# ============================================================
# PLAYER EXPLORER
# ============================================================

with tab_players:
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
            SELECT *
            FROM marts.player_career_stats
            WHERE player_id = ?
        """, [player_id])

        player_seasons = query_df("""
            SELECT *
            FROM marts.player_season_stats
            WHERE player_id = ?
            ORDER BY season
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
                ("Games", "games", 0),
                ("Points", "points", 0),
                ("Goals", "goals", 0),
                ("Assists", "assists", 0),
                ("Shots", "shots", 0),
                ("SOG", "shots_on_goal", 0),
                ("GB", "ground_balls", 0),
                ("CT", "caused_turnovers", 0),
            ],
            columns=4
        )

        st.markdown("### Per-Game / Rate Stats")
        stat_grid(
            summary,
            [
                ("Points/G", "points_per_game", 2),
                ("Goals/G", "goals_per_game", 2),
                ("Assists/G", "assists_per_game", 2),
                ("Shots/G", "shots_per_game", 2),
                ("GB/G", "ground_balls_per_game", 2),
                ("TO/G", "turnovers_per_game", 2),
                ("CT/G", "caused_turnovers_per_game", 2),
                ("Shot %", "shot_pct_calc", 2, True),
            ],
            columns=4
        )

        # >>> PLL_PLAYER_PROFILE_SEASON_TOTALS_START

        st.markdown("### Season Totals and Averages")

        st.caption(

            "Season-by-season totals and per-game averages for the selected player."

        )


        _pp_player_id = globals().get("player_id", None)

        _pp_selected_player = globals().get("selected_player", None)


        if _pp_player_id is None and _pp_selected_player is not None:

            _pp_lookup = query_df("""

                SELECT player_id, full_name

                FROM clean.player_directory

                WHERE full_name = ?

                LIMIT 1

            """, [str(_pp_selected_player)])


            if len(_pp_lookup) > 0:

                _pp_player_id = _pp_lookup["player_id"].iloc[0]


        if _pp_player_id is None:

            st.info("Select a player to view season totals and averages.")

        else:

            _pp_season_rows = query_df("""

                SELECT *

                FROM marts.player_season_stats

                WHERE player_id = ?

                ORDER BY season

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

                    _pp_cols = [

                        "season",

                        "teams",

                        "position",

                        "games",

                        "points",

                        "scoring_points",

                        "one_point_goals",

                        "two_point_goals",

                        "goals",

                        "assists",

                        "shots",

                        "shots_on_goal",

                        "ground_balls",

                        "turnovers",

                        "caused_turnovers",

                        "touches",

                        "total_passes",

                    ]

                elif _pp_view == "Per Game":

                    _pp_cols = [

                        "season",

                        "teams",

                        "position",

                        "games",

                        "points_per_game",

                        "scoring_points_per_game",

                        "one_point_goals_per_game",

                        "two_point_goals_per_game",

                        "goals_per_game",

                        "assists_per_game",

                        "shots_per_game",

                        "shots_on_goal_per_game",

                        "ground_balls_per_game",

                        "turnovers_per_game",

                        "caused_turnovers_per_game",

                        "touches_per_game",

                        "total_passes_per_game",

                    ]

                else:

                    _pp_cols = list(_pp_season_rows.columns)


                _pp_cols = [

                    c for c in _pp_cols

                    if c in _pp_season_rows.columns

                ]


                display_table(

                    _pp_season_rows[_pp_cols],

                    height=330,

                    hide_cols=[],

                    max_cols=None

                )


                _pp_download_name = (

                    str(_pp_selected_player).replace(" ", "_").lower()

                    if _pp_selected_player is not None

                    else str(_pp_player_id)

                )


                download_csv(

                    _pp_season_rows,

                    f"{_pp_download_name}_season_totals.csv",

                    label="Download player season totals CSV"

                )

        # <<< PLL_PLAYER_PROFILE_SEASON_TOTALS_END


        st.markdown("### Season Trend")

        trend_cols = [
            "points_per_game", "goals_per_game", "assists_per_game",
            "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game"
        ]

        trend_options = [c for c in trend_cols if c in player_seasons.columns]

        trend_selection = st.multiselect(
            "Trend metrics",
            options=trend_options,
            default=[c for c in ["points_per_game", "goals_per_game", "assists_per_game"] if c in trend_options],
            format_func=pretty_col,
            key=f"player_trend_metrics_{player_id}"
        )

        season_trend_df = player_seasons[["season"] + trend_options].copy() if len(player_seasons) else pd.DataFrame()

        safe_line_chart(
            season_trend_df,
            x_col="season",
            y_cols=trend_selection,
            title=f"{selected_player} — Season Trend"
        )

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
            SELECT *
            FROM {split_table}
            WHERE player_id = ?
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
                        ("Points", "points", 0),
                        ("Goals", "goals", 0),
                        ("Assists", "assists", 0),
                        ("Shots", "shots", 0),
                        ("GB", "ground_balls", 0),
                        ("TO", "turnovers", 0),
                        ("CT", "caused_turnovers", 0),
                        ("Touches", "touches", 0),
                    ],
                    columns=4
                )

            with c2:
                st.markdown("#### Window Averages")
                stat_grid(
                    split_summary,
                    [
                        ("Points/G", "points_per_game", 2),
                        ("Goals/G", "goals_per_game", 2),
                        ("Assists/G", "assists_per_game", 2),
                        ("Shots/G", "shots_per_game", 2),
                        ("GB/G", "ground_balls_per_game", 2),
                        ("TO/G", "turnovers_per_game", 2),
                        ("CT/G", "caused_turnovers_per_game", 2),
                        ("Touches/G", "touches_per_game", 2),
                    ],
                    columns=4
                )

            recent_games = query_df(f"""
                SELECT
                    season,
                    game_number,
                    game_date_utc,
                    team_name,
                    opponent_team_name,
                    is_home,
                    points,
                    goals,
                    assists,
                    shots,
                    shots_on_goal,
                    ground_balls,
                    turnovers,
                    caused_turnovers,
                    saves,
                    saa,
                    faceoffs_won,
                    faceoffs_lost,
                    touches,
                    total_passes
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

                safe_bar_chart(
                    recent_chart,
                    x_col="game_label",
                    y_col=recent_metric,
                    title=f"{selected_player} — {split_choice} {pretty_col(recent_metric)} by Game"
                )

        st.markdown("### Game Log")

        game_log = query_df("""
            SELECT
                season,
                game_number,
                game_date_utc,
                team_name,
                opponent_team_name,
                is_home,
                points,
                goals,
                assists,
                shots,
                shots_on_goal,
                ground_balls,
                turnovers,
                caused_turnovers,
                saves,
                saa,
                faceoffs_won,
                faceoffs_lost,
                touches,
                total_passes
            FROM clean.player_game_stats
            WHERE player_id = ?
            ORDER BY season DESC, game_number DESC
        """, [player_id])

        gl_filters = st.columns(4)

        player_game_seasons = sorted(game_log["season"].dropna().astype(int).unique().tolist()) if len(game_log) else []
        player_game_opps = sorted(game_log["opponent_team_name"].dropna().unique().tolist()) if len(game_log) else []

        selected_gl_seasons = gl_filters[0].multiselect(
            "Game log seasons",
            player_game_seasons,
            default=player_game_seasons,
            key=f"player_gl_seasons_{player_id}"
        )

        selected_gl_opps = gl_filters[1].multiselect(
            "Opponents",
            player_game_opps,
            default=[],
            key=f"player_gl_opps_{player_id}"
        )

        selected_home = gl_filters[2].selectbox(
            "Home/Away",
            ["All", "Home", "Away"],
            key=f"player_home_filter_{player_id}"
        )

        min_points_filter = gl_filters[3].number_input(
            "Minimum points",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
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

            safe_line_chart(
                trend_df,
                x_col="game_label",
                y_cols=game_chart_metrics,
                title=f"{selected_player} — Filtered Game Log"
            )

        st.markdown("### Vs Opponent Splits")

        vs_opp = query_df("""
            SELECT
                opponent_team_name,
                games,
                points,
                goals,
                assists,
                shots,
                ground_balls,
                caused_turnovers,
                points_per_game,
                goals_per_game,
                assists_per_game,
                shots_per_game,
                ground_balls_per_game,
                caused_turnovers_per_game
            FROM marts.player_vs_opponent_stats
            WHERE player_id = ?
            ORDER BY points_per_game DESC NULLS LAST
        """, [player_id])

        opp_cols = st.columns(2)

        vs_metric = opp_cols[0].selectbox(
            "Opponent split metric",
            options=[
                "points_per_game", "goals_per_game", "assists_per_game",
                "shots_per_game", "ground_balls_per_game", "caused_turnovers_per_game",
                "points", "goals", "assists", "shots"
            ],
            index=0,
            format_func=pretty_col,
            key=f"player_vs_metric_{player_id}"
        )

        min_vs_games = opp_cols[1].number_input(
            "Minimum games vs opponent",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            key=f"player_vs_min_games_{player_id}"
        )

        vs_opp_filtered = vs_opp[vs_opp["games"] >= min_vs_games].copy()

        safe_bar_chart(
            vs_opp_filtered.sort_values(vs_metric).tail(12),
            x_col="opponent_team_name",
            y_col=vs_metric,
            title=f"{selected_player} — {pretty_col(vs_metric)} by Opponent",
            orientation="h"
        )

        display_table(vs_opp_filtered, height=330)

# ============================================================
# TEAM EXPLORER
# ============================================================

    


with tab_teams:
    st.subheader("Team Profiles")
    st.markdown('<div class="section-note">Analyze team-level career, season, recent-form, game-log, and opponent splits.</div>', unsafe_allow_html=True)

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
            SELECT
                c.*,
                r.wins,
                r.losses,
                r.win_pct
            FROM marts.team_career_stats c
            LEFT JOIN record r
                ON c.team_id = r.team_id
            WHERE c.team_id = ?
        """, [team_id])

        team_seasons = query_df("""
            SELECT *
            FROM marts.team_season_stats
            WHERE team_id = ?
            ORDER BY season
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
                ("Games", "games", 0),
                ("Wins", "wins", 0),
                ("Losses", "losses", 0),
                ("Scores", "scores", 0),
                ("Goals", "goals", 0),
                ("Assists", "assists", 0),
                ("Shots", "shots", 0),
                ("Turnovers", "turnovers", 0),
            ],
            columns=4
        )

        # >>> PLL_TEAM_PROFILE_ROSTER_TOTALS_START

        st.markdown("### Team Player Totals")

        st.caption(

            "Player production for the selected team profile. Use the filters below to review all-time team totals, a specific season, or the active team profile context."

        )


        _tp_team_id = globals().get("team_id", None)

        _tp_selected_team = globals().get("selected_team", None)

        _tp_selected_context = globals().get("selected_context", None)


        if _tp_team_id is None:

            for _candidate_name in ["selected_team_id", "team_profile_team_id", "current_team_id"]:

                _candidate_value = globals().get(_candidate_name, None)

                if _candidate_value is not None:

                    _tp_team_id = _candidate_value

                    break


        if _tp_selected_team is None:

            for _candidate_name in ["selected_team_name", "team_profile_team", "current_team_name"]:

                _candidate_value = globals().get(_candidate_name, None)

                if _candidate_value is not None:

                    _tp_selected_team = _candidate_value

                    break


        if _tp_selected_context is None:

            for _candidate_name in ["team_context", "selected_team_context", "team_profile_context"]:

                _candidate_value = globals().get(_candidate_name, None)

                if _candidate_value is not None:

                    _tp_selected_context = _candidate_value

                    break


        if _tp_team_id is None and _tp_selected_team is not None:

            _team_lookup = query_df("""

                SELECT team_id, team_name

                FROM clean.team_directory

                WHERE LOWER(team_name) = LOWER(?)

                   OR LOWER(team_id) = LOWER(?)

                LIMIT 1

            """, [str(_tp_selected_team), str(_tp_selected_team)])


            if len(_team_lookup) > 0:

                _tp_team_id = _team_lookup["team_id"].iloc[0]

                _tp_selected_team = _team_lookup["team_name"].iloc[0]


        if _tp_selected_team is None and _tp_team_id is not None:

            _team_lookup = query_df("""

                SELECT team_id, team_name

                FROM clean.team_directory

                WHERE team_id = ?

                LIMIT 1

            """, [str(_tp_team_id)])


            if len(_team_lookup) > 0:

                _tp_selected_team = _team_lookup["team_name"].iloc[0]


        if _tp_selected_context is None:

            _tp_selected_context = "Career"


        if _tp_team_id is None and _tp_selected_team is None:

            st.info("Select a team to view team player totals.")

        else:

            _pst_table_check = query_df("""

                SELECT COUNT(*) AS n

                FROM information_schema.tables

                WHERE table_schema = 'marts'

                  AND table_name = 'player_season_stats_by_team'

            """)


            if int(_pst_table_check["n"].iloc[0]) == 0:

                st.info("Player season totals by team are not available yet.")

            else:

                try:

                    _pst_cols = _pll_get_table_columns("marts", "player_season_stats_by_team")

                except Exception:

                    _pst_cols_df = query_df("""

                        SELECT column_name

                        FROM information_schema.columns

                        WHERE table_schema = 'marts'

                          AND table_name = 'player_season_stats_by_team'

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

                            SELECT p.*

                            FROM marts.player_season_stats_by_team p

                            WHERE {_team_filter_sql}

                        """, _team_filter_params)


                        if len(_team_player_all_rows) == 0:

                            st.info(f"No player totals are available for {_tp_selected_team or _tp_team_id}.")

                        else:

                            _team_player_all_rows = _team_player_all_rows.copy()


                            _available_seasons = (

                                _team_player_all_rows["season"]

                                .dropna()

                                .astype(int)

                                .sort_values()

                                .unique()

                                .tolist()

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

                                        key=f"team_profile_player_time_frame_{_tp_team_id}_{_tp_selected_context}"

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

                                        key=f"team_profile_player_specific_season_{_tp_team_id}_{_tp_selected_context}"

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

                                    _id_cols = [

                                        c for c in ["player_id", "full_name"]

                                        if c in _team_player_base.columns

                                    ]


                                    if not _id_cols and "full_name" in _team_player_base.columns:

                                        _id_cols = ["full_name"]


                                    _sum_cols = [

                                        c for c in [

                                            "games",

                                            "points",

                                            "scoring_points",

                                            "one_point_goals",

                                            "two_point_goals",

                                            "goals",

                                            "assists",

                                            "shots",

                                            "shots_on_goal",

                                            "two_point_shots",

                                            "ground_balls",

                                            "turnovers",

                                            "caused_turnovers",

                                            "faceoffs_won",

                                            "faceoffs_lost",

                                            "faceoffs",

                                            "saves",

                                            "clean_saves",

                                            "messy_saves",

                                            "scores_against",

                                            "goals_against",

                                            "touches",

                                            "total_passes",

                                            "penalties",

                                            "penalty_time",

                                        ]

                                        if c in _team_player_base.columns

                                    ]


                                    _agg_dict = {c: "sum" for c in _sum_cols}


                                    if "position" in _team_player_base.columns:

                                        _agg_dict["position"] = (

                                            lambda s: s.dropna().astype(str).mode().iloc[0]

                                            if len(s.dropna())

                                            else None

                                        )


                                    if "position_name" in _team_player_base.columns:

                                        _agg_dict["position_name"] = (

                                            lambda s: s.dropna().astype(str).mode().iloc[0]

                                            if len(s.dropna())

                                            else None

                                        )


                                    if "team_id" in _team_player_base.columns:

                                        _agg_dict["team_id"] = "last"


                                    if "team_name" in _team_player_base.columns:

                                        _agg_dict["team_name"] = "last"


                                    _team_player_display_base = (

                                        _team_player_base

                                        .groupby(_id_cols, dropna=False)

                                        .agg(_agg_dict)

                                        .reset_index()

                                    )


                                    _team_player_display_base["season"] = "All Time"

                                else:

                                    _team_player_display_base = _team_player_base.copy()


                                if len(_team_player_display_base) > 0:

                                    if "games" in _team_player_display_base.columns:

                                        _games = pd.to_numeric(

                                            _team_player_display_base["games"],

                                            errors="coerce"

                                        ).replace(0, np.nan)

                                    else:

                                        _games = pd.Series(np.nan, index=_team_player_display_base.index)


                                    _rate_pairs = {

                                        "points": "points_per_game",

                                        "scoring_points": "scoring_points_per_game",

                                        "one_point_goals": "one_point_goals_per_game",

                                        "two_point_goals": "two_point_goals_per_game",

                                        "goals": "goals_per_game",

                                        "assists": "assists_per_game",

                                        "shots": "shots_per_game",

                                        "shots_on_goal": "shots_on_goal_per_game",

                                        "ground_balls": "ground_balls_per_game",

                                        "turnovers": "turnovers_per_game",

                                        "caused_turnovers": "caused_turnovers_per_game",

                                        "faceoffs_won": "faceoffs_won_per_game",

                                        "faceoffs": "faceoffs_per_game",

                                        "saves": "saves_per_game",

                                        "scores_against": "scores_against_per_game",

                                        "goals_against": "goals_against_per_game",

                                        "touches": "touches_per_game",

                                        "total_passes": "total_passes_per_game",

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

                                        if "position" in _team_player_display_base.columns

                                        else []

                                    )


                                    with _filter_cols[2]:

                                        _selected_positions = st.multiselect(

                                            "Positions",

                                            options=_position_options,

                                            default=[],

                                            key=f"team_profile_player_positions_{_tp_team_id}_{_tp_selected_context}_{_team_player_context_label}"

                                        )


                                    with _filter_cols[3]:

                                        _min_games_team_players = st.number_input(

                                            "Min Games",

                                            min_value=0,

                                            max_value=100,

                                            value=0,

                                            step=1,

                                            key=f"team_profile_player_min_games_{_tp_team_id}_{_tp_selected_context}_{_team_player_context_label}"

                                        )


                                    _sort_options = [

                                        c for c in [

                                            "points",

                                            "points_per_game",

                                            "scoring_points",

                                            "scoring_points_per_game",

                                            "one_point_goals",

                                            "one_point_goals_per_game",

                                            "two_point_goals",

                                            "two_point_goals_per_game",

                                            "goals",

                                            "goals_per_game",

                                            "assists",

                                            "assists_per_game",

                                            "shots",

                                            "shots_per_game",

                                            "shots_on_goal",

                                            "shots_on_goal_per_game",

                                            "touches",

                                            "touches_per_game",

                                            "ground_balls",

                                            "ground_balls_per_game",

                                            "caused_turnovers",

                                            "caused_turnovers_per_game",

                                            "turnovers",

                                            "turnovers_per_game",

                                            "faceoff_pct_calc",

                                            "save_pct_calc",

                                        ]

                                        if c in _team_player_display_base.columns

                                    ]


                                    with _filter_cols[4]:

                                        _team_player_sort_metric = st.selectbox(

                                            "Sort By",

                                            options=_sort_options,

                                            index=0 if _sort_options else None,

                                            format_func=pretty_col,

                                            key=f"team_profile_player_sort_{_tp_team_id}_{_tp_selected_context}_{_team_player_context_label}"

                                        )


                                    _table_view = st.radio(

                                        "Player Table View",

                                        options=["Summary", "Per Game", "Specialists"],

                                        horizontal=True,

                                        key=f"team_profile_player_table_view_{_tp_team_id}_{_tp_selected_context}_{_team_player_context_label}"

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

                                            _team_player_filtered[_team_player_sort_metric],

                                            errors="coerce"

                                        )


                                        _sort_ascending = _team_player_sort_metric in {

                                            "turnovers",

                                            "turnovers_per_game",

                                            "goals_against",

                                            "goals_against_per_game",

                                            "scores_against",

                                            "scores_against_per_game",

                                        }


                                        _team_player_filtered = _team_player_filtered.sort_values(

                                            _team_player_sort_metric,

                                            ascending=_sort_ascending,

                                            na_position="last"

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

                                        _display_cols = [

                                            "season",

                                            "full_name",

                                            "position",

                                            "games",

                                            "points",

                                            "scoring_points",

                                            "one_point_goals",

                                            "two_point_goals",

                                            "goals",

                                            "assists",

                                            "shots",

                                            "shots_on_goal",

                                            "ground_balls",

                                            "turnovers",

                                            "caused_turnovers",

                                            "touches",

                                        ]

                                    elif _table_view == "Per Game":

                                        _display_cols = [

                                            "season",

                                            "full_name",

                                            "position",

                                            "games",

                                            "points_per_game",

                                            "scoring_points_per_game",

                                            "one_point_goals_per_game",

                                            "two_point_goals_per_game",

                                            "goals_per_game",

                                            "assists_per_game",

                                            "shots_per_game",

                                            "shots_on_goal_per_game",

                                            "ground_balls_per_game",

                                            "turnovers_per_game",

                                            "caused_turnovers_per_game",

                                            "touches_per_game",

                                            "total_passes_per_game",

                                        ]

                                    else:

                                        _display_cols = [

                                            "season",

                                            "full_name",

                                            "position",

                                            "position_name",

                                            "games",

                                            "points",

                                            "scoring_points",

                                            "one_point_goals",

                                            "two_point_goals",

                                            "goals",

                                            "assists",

                                            "shots",

                                            "shots_on_goal",

                                            "two_point_shots",

                                            "shot_pct_calc",

                                            "shots_on_goal_rate_calc",

                                            "ground_balls",

                                            "turnovers",

                                            "caused_turnovers",

                                            "faceoffs_won",

                                            "faceoffs_lost",

                                            "faceoffs",

                                            "faceoff_pct_calc",

                                            "saves",

                                            "scores_against",

                                            "goals_against",

                                            "save_pct_calc",

                                            "touches",

                                            "total_passes",

                                        ]


                                    _display_cols = [

                                        c for c in _display_cols

                                        if c in _team_player_filtered.columns

                                    ]


                                    if (

                                        len(_team_player_filtered) > 0

                                        and _team_player_sort_metric in _team_player_filtered.columns

                                        and "full_name" in _team_player_filtered.columns

                                    ):

                                        _chart_df = _team_player_filtered.head(15).copy()


                                        safe_bar_chart(

                                            _chart_df.sort_values(_team_player_sort_metric, ascending=True),

                                            x_col="full_name",

                                            y_col=_team_player_sort_metric,

                                            color_col="position" if "position" in _chart_df.columns else None,

                                            title=f"{_tp_selected_team or _tp_team_id} — {_team_player_context_label} Player Leaders by {pretty_col(_team_player_sort_metric)}",

                                            orientation="h"

                                        )


                                    display_table(

                                        _team_player_filtered[_display_cols],

                                        height=430,

                                        hide_cols=[],

                                        max_cols=None

                                    )


                                    with st.expander("Full player table", expanded=False):

                                        display_table(

                                            _team_player_filtered,

                                            height=430,

                                            hide_cols=[],

                                            max_cols=None

                                        )


                                    download_csv(

                                        _team_player_filtered,

                                        f"{str(_tp_selected_team or _tp_team_id).replace(' ', '_').lower()}_{str(_team_player_context_label).replace(' ', '_').lower()}_player_totals.csv",

                                        label="Download team player totals CSV"

                                    )

        # <<< PLL_TEAM_PROFILE_ROSTER_TOTALS_END


        st.markdown("### Per-Game / Rate Stats")
        stat_grid(
            summary,
            [
                ("Win %", "win_pct", 2, True),
                ("Scores/G", "scores_per_game", 2),
                ("Shots/G", "shots_per_game", 2),
                ("TO/G", "turnovers_per_game", 2),
                ("Shot %", "shot_pct_calc", 2, True),
                ("FO %", "faceoff_pct_calc", 2, True),
                ("Clear %", "clear_pct_calc", 2, True),
                ("Off. Seq./G", "offensive_sequence_proxy_per_game", 2),
            ],
            columns=4
        )


        # >>> PLL_TEAM_EXPLORER_DEFENSE_START
        st.markdown("### Defensive / Opponent Profile")

        if table_exists("marts", "team_defense_season_stats"):
            if selected_context == "Career":
                defense_summary_df = query_df("""
                    SELECT *
                    FROM marts.team_defense_career_stats
                    WHERE team_id = ?
                """, [team_id])
            else:
                defense_summary_df = query_df("""
                    SELECT *
                    FROM marts.team_defense_season_stats
                    WHERE team_id = ?
                      AND season = ?
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
                    "team_name",
                    "games",
                    "wins",
                    "losses",
                    "win_pct",
                    "team_scores_per_game",
                    "scores_allowed_per_game",
                    "goals_allowed_per_game",
                    "opponent_shots_per_game",
                    "opponent_goal_pct",
                    "opponent_sog_rate",
                    "save_pct_proxy",
                    "caused_turnovers_for_per_game",
                    "opponent_turnovers_per_game",
                    "ct_per_opponent_turnover",
                    "score_margin_per_game",
                ]

                display_table(
                    defense_summary_df[[c for c in defense_cols if c in defense_summary_df.columns]],
                    height=220
                )

            else:
                st.info("No defensive summary found for this team/context.")

        else:
            st.info("Defensive/opponent marts are not available in the warehouse yet.")

        # <<< PLL_TEAM_EXPLORER_DEFENSE_END

        st.markdown("### Team Season Trend")

        trend_options = [c for c in [
            "scores_per_game", "shots_per_game", "turnovers_per_game",
            "saves_per_game", "offensive_sequence_proxy_per_game"
        ] if c in team_seasons.columns]

        trend_selection = st.multiselect(
            "Trend metrics",
            options=trend_options,
            default=[c for c in ["scores_per_game", "shots_per_game", "turnovers_per_game"] if c in trend_options],
            format_func=pretty_col,
            key=f"team_trend_metrics_{team_id}"
        )

        season_trend_df = team_seasons[["season"] + trend_options].copy() if len(team_seasons) else pd.DataFrame()

        safe_line_chart(
            season_trend_df,
            x_col="season",
            y_cols=trend_selection,
            title=f"{selected_team} — Season Trend"
        )

        st.markdown("### Recent Form")

        split_choice = st.radio(
            "Recent form window",
            options=["Last 5", "Last 10"],
            horizontal=True,
            key=f"team_recent_split_{team_id}"
        )

        window_n = 5 if split_choice == "Last 5" else 10
        split_table = "marts.team_last5_stats" if split_choice == "Last 5" else "marts.team_last10_stats"

        split_df = query_df(f"""
            SELECT *
            FROM {split_table}
            WHERE team_id = ?
        """, [team_id])

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
                        ("Scores", "scores", 0),
                        ("Goals", "goals", 0),
                        ("Assists", "assists", 0),
                        ("Shots", "shots", 0),
                        ("Saves", "saves", 0),
                        ("Turnovers", "turnovers", 0),
                        ("Touches", "touches", 0),
                        ("Off. Seq.", "offensive_sequence_proxy", 0),
                    ],
                    columns=4
                )

            with c2:
                st.markdown("#### Window Averages")
                stat_grid(
                    split_summary,
                    [
                        ("Scores/G", "scores_per_game", 2),
                        ("Shots/G", "shots_per_game", 2),
                        ("Saves/G", "saves_per_game", 2),
                        ("TO/G", "turnovers_per_game", 2),
                        ("Touches/G", "touches_per_game", 2),
                        ("Passes/G", "total_passes_per_game", 2),
                        ("Poss. Time/G", "time_in_possession_per_game", 2),
                        ("Off. Seq./G", "offensive_sequence_proxy_per_game", 2),
                    ],
                    columns=4
                )

            recent_games = query_df(f"""
                SELECT
                    season,
                    game_number,
                    game_date_utc,
                    team_name,
                    opponent_team_name,
                    is_home,
                    scores,
                    scores_against,
                    goals,
                    one_point_goals,
                    two_point_goals,
                    assists,
                    shots,
                    shots_on_goal,
                    saves,
                    ground_balls,
                    turnovers,
                    caused_turnovers,
                    faceoffs_won,
                    faceoffs_lost,
                    clears,
                    clear_attempts,
                    touches,
                    total_passes,
                    time_in_possession,
                    official_total_possessions,
                    offensive_sequence_proxy
                FROM clean.team_game_stats
                WHERE team_id = ?
                ORDER BY game_date_utc DESC, season DESC, game_number DESC
                LIMIT {window_n}
            """, [team_id])

            st.markdown(f"#### {split_choice} Individual Games")
            st.caption("The bottom two rows summarize the selected window across the individual games shown above.")

            recent_with_summary = add_window_summary_rows(recent_games)
            display_table(recent_with_summary, height=360)

        st.markdown("### Team Game Log")

        game_log = query_df("""
            SELECT
                season,
                game_number,
                game_date_utc,
                team_name,
                opponent_team_name,
                is_home,
                scores,
                scores_against,
                goals,
                one_point_goals,
                two_point_goals,
                assists,
                shots,
                shots_on_goal,
                saves,
                ground_balls,
                turnovers,
                caused_turnovers,
                faceoffs_won,
                faceoffs_lost,
                clears,
                clear_attempts,
                touches,
                total_passes,
                time_in_possession,
                official_total_possessions,
                offensive_sequence_proxy
            FROM clean.team_game_stats
            WHERE team_id = ?
            ORDER BY season DESC, game_number DESC
        """, [team_id])

        tg_filters = st.columns(4)

        team_game_seasons = sorted(game_log["season"].dropna().astype(int).unique().tolist()) if len(game_log) else []
        team_game_opps = sorted(game_log["opponent_team_name"].dropna().unique().tolist()) if len(game_log) else []

        selected_tg_seasons = tg_filters[0].multiselect(
            "Game log seasons",
            team_game_seasons,
            default=team_game_seasons,
            key=f"team_gl_seasons_{team_id}"
        )

        selected_tg_opps = tg_filters[1].multiselect(
            "Opponents",
            team_game_opps,
            default=[],
            key=f"team_gl_opps_{team_id}"
        )

        selected_home = tg_filters[2].selectbox(
            "Home/Away",
            ["All", "Home", "Away"],
            key=f"team_home_filter_{team_id}"
        )

        min_score_filter = tg_filters[3].number_input(
            "Minimum scores",
            min_value=0,
            max_value=40,
            value=0,
            step=1,
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


# ============================================================
# GOALIE / FACEOFF SPECIALTY PAGES
# ============================================================

with tab_specialists:
    st.subheader("Specialists")
    st.markdown(
        '<div class="section-note">Dedicated evaluation pages for goalies and faceoff specialists. Goalie save percentage is standardized as Saves ÷ (Saves + Goals Against).</div>',
        unsafe_allow_html=True
    )

    goalie_tab, faceoff_tab = st.tabs(["Goalies", "Faceoff Specialists"])

    with goalie_tab:
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
            SELECT
                season,
                player_id,
                full_name,
                position,
                position_name,
                teams,
                games,
                saves,
                clean_saves,
                messy_saves,
                scores_against,
                saa,
                goals_against,
                shots,
                save_pct_calc,
                saves_per_game,
                clean_saves_per_game,
                messy_saves_per_game,
                scores_against_per_game,
                saa_per_game,
                goals_against_per_game
            FROM marts.player_season_stats
            WHERE season = ?
              AND games >= ?
              AND (position = 'G' OR lower(position_name) LIKE '%goalie%')
            ORDER BY games DESC
        """, [goalie_season, goalie_min_games])

        goalie_df = _pll_apply_goalie_save_pct(goalie_df)

        goalie_metric_options = [
            c for c in [
                "save_pct_display",
                "saves",
                "saves_per_game",
                "shots_faced_calc",
                "shots_faced_per_game_calc",
                "goals_against",
                "goals_against_per_game",
                "scores_against",
                "scores_against_per_game",
                "saa",
                "saa_per_game",
                "clean_saves",
                "messy_saves",
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

        lower_goalie_metrics = {
            "scores_against", "scores_against_per_game",
            "goals_against", "goals_against_per_game"
        }

        goalie_sort_ascending = goalie_metric in lower_goalie_metrics

        g_cols[3].caption("Sort logic")
        g_cols[3].markdown(
            "**Lower is better**" if goalie_sort_ascending else "**Higher is better**"
        )

        goalie_df = _pll_safe_sort(goalie_df, goalie_metric, lower_is_better=goalie_sort_ascending)

        top_goalie = goalie_df["full_name"].iloc[0] if len(goalie_df) else "—"
        best_save_pct = goalie_df["save_pct_display"].max() if "save_pct_display" in goalie_df.columns and len(goalie_df) else np.nan
        avg_save_pct = goalie_df["save_pct_display"].mean() if "save_pct_display" in goalie_df.columns and len(goalie_df) else np.nan

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
            x_col="full_name",
            y_col=goalie_metric,
            color_col="teams",
            title=f"{goalie_season} Goalie Leaders — {pretty_col(goalie_metric)}",
            orientation="h"
        )

        goalie_display = goalie_df.copy()

        goalie_summary_cols = _pll_select_existing(
            goalie_display,
            [
                "season", "full_name", "position", "teams", "games",
                "saves", "goals_against", "scores_against", "shots_faced_calc",
                "save_pct_display_pct", "saves_per_game", "goals_against_per_game",
                "scores_against_per_game", "shots_faced_per_game_calc",
            ]
        )

        display_table(goalie_display[goalie_summary_cols], height=420)

        with st.expander("Advanced goalie metrics", expanded=False):
            goalie_advanced_cols = _pll_select_existing(
                goalie_display,
                [
                    "season", "full_name", "position", "position_name", "teams", "games",
                    "saves", "clean_saves", "messy_saves", "goals_against", "scores_against",
                    "shots_faced_calc", "save_pct_display", "save_pct_display_pct",
                    "saa", "shots", "save_pct_calc",
                    "saves_per_game", "clean_saves_per_game", "messy_saves_per_game",
                    "goals_against_per_game", "scores_against_per_game",
                    "shots_faced_per_game_calc", "saa_per_game"
                ]
            )
            display_table(goalie_display[goalie_advanced_cols], height=360)

        st.markdown("### Goalie Explorer")

        goalie_names = goalie_df["full_name"].dropna().unique().tolist()

        if goalie_names:
            selected_goalie = st.selectbox(
                "Select goalie",
                options=goalie_names,
                index=0,
                key="selected_goalie"
            )

            selected_goalie_id = goalie_df[goalie_df["full_name"] == selected_goalie]["player_id"].iloc[0]

            goalie_games = query_df("""
                SELECT
                    season,
                    game_number,
                    game_date_utc,
                    team_name,
                    opponent_team_name,
                    is_home,
                    saves,
                    clean_saves,
                    messy_saves,
                    scores_against,
                    saa,
                    goals_against,
                    shots,
                    save_pct,
                    touches,
                    total_passes
                FROM clean.player_game_stats
                WHERE player_id = ?
                ORDER BY season DESC, game_number DESC
            """, [selected_goalie_id])

            goalie_games = _pll_apply_goalie_save_pct(goalie_games)

            profile_header(selected_goalie, "Goalie game log and trend view")

            goalie_game_cols = _pll_select_existing(
                goalie_games,
                [
                    "season", "game_number", "game_date_utc", "team_name", "opponent_team_name",
                    "is_home", "saves", "goals_against", "scores_against", "shots_faced_calc",
                    "save_pct_display_pct", "clean_saves", "messy_saves", "touches", "total_passes"
                ]
            )

            display_table(goalie_games[goalie_game_cols], height=360)

            goalie_game_metric_options = [
                c for c in ["saves", "goals_against", "scores_against", "shots_faced_calc", "save_pct_display"]
                if c in goalie_games.columns
            ]

            goalie_game_metric = st.selectbox(
                "Goalie game trend metric",
                options=goalie_game_metric_options,
                index=0,
                format_func=pretty_col,
                key="goalie_game_metric"
            )

            if len(goalie_games) > 0:
                goalie_trend = goalie_games.sort_values(["season", "game_number"]).copy()
                goalie_trend["game_label"] = goalie_trend["season"].astype(str) + " G" + goalie_trend["game_number"].astype(str)

                safe_line_chart(
                    goalie_trend,
                    x_col="game_label",
                    y_cols=[goalie_game_metric],
                    title=f"{selected_goalie} — {pretty_col(goalie_game_metric)} by Game"
                )

    with faceoff_tab:
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
            "Minimum faceoffs",
            min_value=1,
            max_value=500,
            value=20,
            step=5,
            key="min_faceoffs"
        )

        faceoff_metric_options = [
            "faceoff_pct_calc", "faceoffs_won", "faceoffs",
            "faceoffs_won_per_game", "faceoffs_per_game",
            "ground_balls", "ground_balls_per_game", "points", "touches"
        ]

        faceoff_metric = f_cols[2].selectbox(
            "Faceoff metric",
            options=faceoff_metric_options,
            index=0,
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
            SELECT
                season,
                player_id,
                full_name,
                position,
                position_name,
                teams,
                games,
                points,
                goals,
                assists,
                ground_balls,
                faceoffs_won,
                faceoffs_lost,
                faceoffs,
                faceoff_pct_calc,
                faceoffs_won_per_game,
                faceoffs_per_game,
                ground_balls_per_game,
                touches,
                touches_per_game
            FROM marts.player_season_stats
            WHERE season = ?
              AND faceoffs >= ?
            ORDER BY faceoff_pct_calc DESC NULLS LAST
        """, [faceoff_season, min_faceoffs])

        if faceoff_metric in faceoff_df.columns:
            faceoff_df = _pll_safe_sort(faceoff_df, faceoff_metric, lower_is_better=faceoff_sort_ascending)

        safe_bar_chart(
            faceoff_df.head(15).sort_values(faceoff_metric, ascending=not faceoff_sort_ascending),
            x_col="full_name",
            y_col=faceoff_metric,
            color_col="teams",
            title=f"{faceoff_season} Faceoff Leaders — {pretty_col(faceoff_metric)}",
            orientation="h"
        )

        faceoff_summary_cols = _pll_select_existing(
            faceoff_df,
            [
                "season", "full_name", "position", "teams", "games",
                "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct_calc",
                "faceoffs_per_game", "faceoffs_won_per_game",
                "ground_balls", "ground_balls_per_game", "points", "touches"
            ]
        )

        display_table(faceoff_df[faceoff_summary_cols], height=420)

        with st.expander("Advanced faceoff metrics", expanded=False):
            display_table(faceoff_df, height=360)

        st.markdown("### Faceoff Explorer")

        faceoff_names = faceoff_df["full_name"].dropna().unique().tolist()

        if faceoff_names:
            selected_faceoff_player = st.selectbox(
                "Select faceoff player",
                options=faceoff_names,
                index=0,
                key="selected_faceoff_player"
            )

            selected_faceoff_id = faceoff_df[faceoff_df["full_name"] == selected_faceoff_player]["player_id"].iloc[0]

            faceoff_games = query_df("""
                SELECT
                    season,
                    game_number,
                    game_date_utc,
                    team_name,
                    opponent_team_name,
                    is_home,
                    points,
                    goals,
                    assists,
                    ground_balls,
                    faceoffs_won,
                    faceoffs_lost,
                    faceoffs,
                    faceoff_pct,
                    turnovers,
                    caused_turnovers,
                    touches,
                    total_passes
                FROM clean.player_game_stats
                WHERE player_id = ?
                ORDER BY season DESC, game_number DESC
            """, [selected_faceoff_id])

            profile_header(selected_faceoff_player, "Faceoff game log and trend view")

            faceoff_game_cols = _pll_select_existing(
                faceoff_games,
                [
                    "season", "game_number", "game_date_utc", "team_name", "opponent_team_name",
                    "is_home", "faceoffs_won", "faceoffs_lost", "faceoffs", "faceoff_pct",
                    "ground_balls", "points", "turnovers", "caused_turnovers", "touches"
                ]
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
                    faceoff_trend,
                    x_col="game_label",
                    y_cols=[faceoff_game_metric],
                    title=f"{selected_faceoff_player} — {pretty_col(faceoff_game_metric)} by Game"
                )

# ============================================================
# PLAYER COMPARISON
# ============================================================

with tab_player_compare:
    st.subheader("Compare Players")
    st.markdown('<div class="section-note">Compare players with profile cards, matrix-style summaries, trends, and recent-form splits.</div>', unsafe_allow_html=True)

    player_names = players_df["full_name"].dropna().unique().tolist()

    selected_compare_players = st.multiselect(
        "Select 2–6 players",
        options=player_names,
        default=player_names[:2] if len(player_names) >= 2 else player_names,
        key="compare_players"
    )

    if len(selected_compare_players) < 2:
        st.info("Select at least two players to compare.")
    else:
        player_ids = players_df[players_df["full_name"].isin(selected_compare_players)]["player_id"].tolist()
        placeholders = ", ".join(["?"] * len(player_ids))

        compare_context = st.radio(
            "Comparison context",
            options=["Career", "Last 5", "Last 10", "Season"],
            horizontal=True,
            key="player_compare_context"
        )

        if compare_context == "Career":
            compare_df = query_df(f"""
                SELECT *
                FROM marts.player_career_stats
                WHERE player_id IN ({placeholders})
                ORDER BY points DESC NULLS LAST
            """, player_ids)

        elif compare_context == "Last 5":
            compare_df = query_df(f"""
                SELECT *
                FROM marts.player_last5_stats
                WHERE player_id IN ({placeholders})
                ORDER BY points_per_game DESC NULLS LAST
            """, player_ids)

        elif compare_context == "Last 10":
            compare_df = query_df(f"""
                SELECT *
                FROM marts.player_last10_stats
                WHERE player_id IN ({placeholders})
                ORDER BY points_per_game DESC NULLS LAST
            """, player_ids)

        else:
            selected_compare_season = st.selectbox(
                "Season",
                options=seasons,
                index=len(seasons) - 1,
                key="player_compare_season"
            )

            compare_df = query_df(f"""
                SELECT *
                FROM marts.player_season_stats
                WHERE player_id IN ({placeholders})
                  AND season = ?
                ORDER BY points DESC NULLS LAST
            """, player_ids + [selected_compare_season])

        st.markdown("### Selected Player Snapshot")

        profile_summary_cards(
            compare_df,
            title_col="full_name",
            specs=[
                ("Position", "position"),
                ("Teams", "teams"),
                ("Games", "games"),
                ("Points/G", "points_per_game"),
                ("Goals/G", "goals_per_game"),
                ("Assists/G", "assists_per_game"),
            ],
            columns=3
        )

        st.markdown("### Comparison Matrix")

        player_compare_metrics = [
            "games", "points", "goals", "assists", "shots", "ground_balls",
            "turnovers", "caused_turnovers", "touches", "total_passes",
            "points_per_game", "goals_per_game", "assists_per_game",
            "shots_per_game", "ground_balls_per_game", "turnovers_per_game",
            "caused_turnovers_per_game", "shot_pct_calc", "shots_on_goal_rate_calc"
        ]

        display_comparison_matrix(compare_df, "full_name", player_compare_metrics, height=500)

        st.markdown("### Visual Comparison")

        chart_metric = st.selectbox(
            "Chart metric",
            options=[m for m in player_compare_metrics if m in compare_df.columns],
            index=0,
            format_func=pretty_col,
            key="player_compare_chart_metric"
        )

        safe_bar_chart(
            compare_df.sort_values(chart_metric),
            x_col="full_name",
            y_col=chart_metric,
            color_col="full_name",
            title=f"{compare_context} Comparison — {pretty_col(chart_metric)}",
            orientation="h"
        )

        st.markdown("### Season Trend")

        compare_seasons = query_df(f"""
            SELECT
                season,
                full_name,
                position,
                games,
                points,
                goals,
                assists,
                shots,
                ground_balls,
                caused_turnovers,
                points_per_game,
                goals_per_game,
                assists_per_game,
                shots_per_game
            FROM marts.player_season_stats
            WHERE player_id IN ({placeholders})
            ORDER BY season, full_name
        """, player_ids)

        trend_metric = st.selectbox(
            "Season trend metric",
            options=[c for c in ["points_per_game", "goals_per_game", "assists_per_game", "shots_per_game"] if c in compare_seasons.columns],
            format_func=pretty_col,
            key="player_compare_trend_metric"
        )

        if len(compare_seasons) > 0 and trend_metric:
            plot_df = clean_chart_x(compare_seasons, "season")

            fig = px.line(
                plot_df,
                x="season",
                y=trend_metric,
                color="full_name",
                markers=True,
                title=f"Player Season Trend — {pretty_col(trend_metric)}",
                labels={c: pretty_col(c) for c in plot_df.columns}
            )

            fig = standardize_chart(fig, category_x=True)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TEAM COMPARISON
# ============================================================

with tab_team_compare:
    st.subheader("Compare Teams")
    st.markdown('<div class="section-note">Compare teams across multi-year profile, current form, season trends, and head-to-head splits.</div>', unsafe_allow_html=True)

    selected_compare_teams = st.multiselect(
        "Select 2–4 teams",
        options=team_options,
        default=team_options[:2] if len(team_options) >= 2 else team_options,
        key="compare_teams"
    )

    if len(selected_compare_teams) < 2:
        st.info("Select at least two teams to compare.")
    else:
        team_ids = teams_df[teams_df["team_name"].isin(selected_compare_teams)]["team_id"].tolist()
        placeholders = ", ".join(["?"] * len(team_ids))

        team_context = st.radio(
            "Comparison context",
            options=["Career", "Last 5", "Last 10", "Season"],
            horizontal=True,
            key="team_compare_context"
        )

        if team_context == "Career":
            compare_df = query_df(f"""
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
                SELECT
                    c.*,
                    r.wins,
                    r.losses,
                    r.win_pct
                FROM marts.team_career_stats c
                LEFT JOIN record r
                    ON c.team_id = r.team_id
                WHERE c.team_id IN ({placeholders})
                ORDER BY c.scores_per_game DESC NULLS LAST
            """, team_ids)

        elif team_context == "Last 5":
            compare_df = query_df(f"""
                SELECT *
                FROM marts.team_last5_stats
                WHERE team_id IN ({placeholders})
                ORDER BY scores_per_game DESC NULLS LAST
            """, team_ids)

        elif team_context == "Last 10":
            compare_df = query_df(f"""
                SELECT *
                FROM marts.team_last10_stats
                WHERE team_id IN ({placeholders})
                ORDER BY scores_per_game DESC NULLS LAST
            """, team_ids)

        else:
            selected_compare_season = st.selectbox(
                "Season",
                options=seasons,
                index=len(seasons) - 1,
                key="team_compare_season"
            )

            compare_df = query_df(f"""
                SELECT *
                FROM marts.team_season_stats
                WHERE team_id IN ({placeholders})
                  AND season = ?
                ORDER BY scores_per_game DESC NULLS LAST
            """, team_ids + [selected_compare_season])

        st.markdown("### Selected Team Snapshot")

        profile_summary_cards(
            compare_df,
            title_col="team_name",
            specs=[
                ("Games", "games"),
                ("Wins", "wins"),
                ("Losses", "losses"),
                ("Win %", "win_pct", True),
                ("Scores/G", "scores_per_game"),
                ("Shots/G", "shots_per_game"),
            ],
            columns=4
        )

        st.markdown("### Comparison Matrix")

        team_compare_metrics = [
            "games", "wins", "losses", "win_pct", "scores", "scores_per_game",
            "goals", "assists", "shots", "shots_per_game", "turnovers",
            "turnovers_per_game", "saves", "ground_balls", "caused_turnovers",
            "faceoff_pct_calc", "clear_pct_calc", "touches", "total_passes",
            "time_in_possession", "offensive_sequence_proxy",
            "offensive_sequence_proxy_per_game"
        ]

        display_comparison_matrix(compare_df, "team_name", team_compare_metrics, height=500)


        # >>> PLL_TEAM_COMPARISON_DEFENSE_START
        st.markdown("### Defensive Comparison Matrix")

        if table_exists("marts", "team_defense_season_stats"):
            if team_context == "Career":
                defense_compare_df = query_df(f"""
                    SELECT *
                    FROM marts.team_defense_career_stats
                    WHERE team_id IN ({placeholders})
                    ORDER BY scores_allowed_per_game ASC NULLS LAST
                """, team_ids)

            elif team_context == "Season":
                defense_compare_df = query_df(f"""
                    SELECT *
                    FROM marts.team_defense_season_stats
                    WHERE team_id IN ({placeholders})
                      AND season = ?
                    ORDER BY scores_allowed_per_game ASC NULLS LAST
                """, team_ids + [selected_compare_season])

            else:
                n_games = 5 if team_context == "Last 5" else 10

                defense_compare_df = query_df(f"""
                    WITH ranked AS (
                        SELECT
                            *,
                            ROW_NUMBER() OVER (
                                PARTITION BY team_id
                                ORDER BY game_date_utc DESC, season DESC, game_number DESC
                            ) AS rn
                        FROM marts.team_game_opponent_context
                        WHERE team_id IN ({placeholders})
                    ),
                    windowed AS (
                        SELECT *
                        FROM ranked
                        WHERE rn <= {n_games}
                    )
                    SELECT
                        team_id,
                        ANY_VALUE(team_name) AS team_name,
                        COUNT(DISTINCT game_id) AS games,
                        SUM(team_scores) AS team_scores,
                        SUM(scores_allowed) AS scores_allowed,
                        SUM(goals_allowed) AS goals_allowed,
                        SUM(opponent_shots) AS opponent_shots,
                        SUM(opponent_shots_on_goal) AS opponent_shots_on_goal,
                        SUM(saves_for) AS saves_for,
                        SUM(caused_turnovers_for) AS caused_turnovers_for,
                        SUM(opponent_turnovers) AS opponent_turnovers,
                        SUM(opponent_touches) AS opponent_touches,
                        SUM(opponent_offensive_sequence_proxy) AS opponent_offensive_sequence_proxy,
                        SUM(score_margin) AS score_margin,
                        SUM(scores_allowed)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS scores_allowed_per_game,
                        SUM(goals_allowed)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS goals_allowed_per_game,
                        SUM(opponent_shots)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS opponent_shots_per_game,
                        SUM(opponent_shots_on_goal)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS opponent_shots_on_goal_per_game,
                        SUM(caused_turnovers_for)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS caused_turnovers_for_per_game,
                        SUM(opponent_turnovers)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS opponent_turnovers_per_game,
                        SUM(score_margin)::DOUBLE / NULLIF(COUNT(DISTINCT game_id), 0) AS score_margin_per_game,
                        SUM(goals_allowed)::DOUBLE / NULLIF(SUM(opponent_shots), 0) AS opponent_goal_pct,
                        SUM(opponent_shots_on_goal)::DOUBLE / NULLIF(SUM(opponent_shots), 0) AS opponent_sog_rate,
                        SUM(saves_for)::DOUBLE / NULLIF(SUM(saves_for) + SUM(goals_allowed), 0) AS save_pct_proxy,
                        SUM(caused_turnovers_for)::DOUBLE / NULLIF(SUM(opponent_turnovers), 0) AS ct_per_opponent_turnover,
                        SUM(scores_allowed)::DOUBLE / NULLIF(SUM(opponent_offensive_sequence_proxy), 0) AS opponent_scores_per_offensive_sequence_proxy
                    FROM windowed
                    GROUP BY team_id
                    ORDER BY scores_allowed_per_game ASC NULLS LAST
                """, team_ids)

            defense_compare_metrics = [
                "games",
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "opponent_shots_on_goal_per_game",
                "opponent_goal_pct",
                "opponent_sog_rate",
                "save_pct_proxy",
                "caused_turnovers_for_per_game",
                "opponent_turnovers_per_game",
                "ct_per_opponent_turnover",
                "opponent_touches",
                "opponent_offensive_sequence_proxy",
                "opponent_scores_per_offensive_sequence_proxy",
                "score_margin_per_game",
            ]

            display_comparison_matrix(defense_compare_df, "team_name", defense_compare_metrics, height=500)

            defense_chart_options = [
                m for m in defense_compare_metrics if m in defense_compare_df.columns
            ]

            if defense_chart_options:
                _def_chart_default_idx = defense_chart_options.index("scores_allowed_per_game") if "scores_allowed_per_game" in defense_chart_options else 0
                defense_chart_metric = st.selectbox(
                    "Defensive chart metric",
                    options=defense_chart_options,
                    index=_def_chart_default_idx,
                    format_func=pretty_col,
                    key="team_compare_defense_chart_metric"
                )

                safe_bar_chart(
                    defense_compare_df.sort_values(defense_chart_metric),
                    x_col="team_name",
                    y_col=defense_chart_metric,
                    color_col="team_name",
                    title=f"{team_context} Defensive Comparison — {pretty_col(defense_chart_metric)}",
                    orientation="h"
                )

        else:
            st.info("Defensive/opponent marts are not available in the warehouse yet.")

        # <<< PLL_TEAM_COMPARISON_DEFENSE_END

        st.markdown("### Visual Comparison")

        _team_chart_options = [m for m in team_compare_metrics if m in compare_df.columns]
        _team_chart_default_idx = _team_chart_options.index("scores_per_game") if "scores_per_game" in _team_chart_options else 0
        chart_metric = st.selectbox(
            "Chart metric",
            options=_team_chart_options,
            index=_team_chart_default_idx,
            format_func=pretty_col,
            key="team_compare_chart_metric"
        )

        safe_bar_chart(
            compare_df.sort_values(chart_metric),
            x_col="team_name",
            y_col=chart_metric,
            color_col="team_name",
            title=f"{team_context} Comparison — {pretty_col(chart_metric)}",
            orientation="h"
        )


# >>> PLL_RANKINGS_TEAM_PROFILE_TABS_START

# ============================================================
# PLAYER RANKINGS + TEAM STYLE PROFILE TABS
# ============================================================

if "COL_LABELS" not in globals():
    COL_LABELS = {}

COL_LABELS.update({
    "ranking_context": "Context",
    "ranking_context_type": "Context Type",
    "ranking_context_max_games": "Max GP",
    "min_games_default": "Default Min GP",
    "eligible_for_default_ranking": "Eligible",
    "sample_size_note": "Sample Note",
    "role_group": "Role",
    "overall_rank": "Overall Rank",
    "view_rank": "View Rank",
    "overall_score": "Overall Score",
    "overall_percentile": "Overall %ile",
    "position_rank": "Pos Rank",
    "position_percentile": "Pos %ile",
    "base_impact_score": "Base Impact",
    "role_primary_score": "Role Score",
    "role_primary_percentile": "Role %ile",
    "role_robust_z": "Role Z",
    "role_adjusted_z": "Adjusted Role Z",
    "role_separation_score": "Peer Separation",
    "role_context_value_score": "Role Context Value",
    "role_context_rank": "Role Rank",
    "role_context_percentile": "Role Context %ile",
    "role_value_tier": "Role Tier",
    "role_group_size": "Role Group Size",
    "role_reliability": "Role Reliability",
    "goal_value_score": "Scoring Value",
    "scoring_value_score": "Scoring Value",
    "playmaking_value_score": "Playmaking Value",
    "one_point_goal_score": "1PT Value",
    "two_point_goal_score": "2PT Value",
    "overall_impact_score": "Base Impact",
    "offensive_score": "Offense",
    "usage_possession_score": "Usage",
    "defensive_score": "Defense",
    "faceoff_score": "Faceoff",
    "goalie_score": "Goalie",
    "ground_ball_score": "GB Value",
    "one_point_goals": "1PT Goals",
    "two_point_goals": "2PT Goals",
    "one_point_goals_per_game": "1PT G/G",
    "two_point_goals_per_game": "2PT G/G",
    "scoring_points": "Scoring Pts",
    "scoring_points_per_game": "Scoring Pts/G",
    "points_per_game": "Pts/G",
    "goals_per_game": "G/G",
    "assists_per_game": "A/G",
    "shots_per_game": "Shots/G",
    "touches_per_game": "Touches/G",
    "ground_balls_per_game": "GB/G",
    "turnovers_per_game": "TO/G",
    "caused_turnovers_per_game": "CT/G",
    "faceoff_pct_for_ranking": "FO %",
    "save_pct_for_ranking": "Save %",
    "points_per_touch": "Pts/Touch",
    "goals_per_shot": "Goals/Shot",
    "profile_context": "Context",
    "profile_context_type": "Context Type",
    "profile_rank": "Style Rank",
    "profile_percentile": "Style %ile",
    "team_style_overall_score": "Overall Style",
    "offensive_volume_score": "Off Volume",
    "offensive_efficiency_score": "Off Efficiency",
    "ball_movement_score": "Ball Movement",
    "possession_control_score": "Possession",
    "defensive_suppression_score": "Defense",
    "pace_tempo_score": "Tempo",
    "net_scores_per_game": "Net Scores/G",
    "time_in_possession_per_game_mmss": "Possession/G",
    "def_scores_allowed_per_game": "Scores Allowed/G",
    "def_goals_allowed_per_game": "Goals Allowed/G",
    "def_opponent_shots_per_game": "Opp Shots/G",
    "def_opponent_goal_pct": "Opp Goal %",
    "def_save_pct_proxy": "Save % Proxy",
    "pace_label": "Pace",
    "offensive_profile_label": "Off Profile",
    "defensive_profile_label": "Def Profile",
    "possession_profile_label": "Poss Profile",
    "style_summary": "Style Summary",
})


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

    return query_df("""
        SELECT *
        FROM marts.player_ranking_profiles
    """)


@st.cache_data(ttl=600, show_spinner=False)
def _pll_load_team_style_profiles():
    if not _pll_extra_table_exists("marts", "team_style_profiles"):
        return pd.DataFrame()

    return query_df("""
        SELECT *
        FROM marts.team_style_profiles
    """)


def _pll_context_order(df, context_col, type_col, sort_col):
    """
    Ordered context labels for rankings/team profiles.

    Final Colab output includes explicit sort columns. This fallback only exists
    so the deployed app does not crash if a warehouse is rebuilt from an older
    table contract; when the sort column exists, behavior is unchanged.
    """
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

        # Preserve Colab ordering intent: Career first; season contexts newest first;
        # rolling/recent contexts after season/career unless explicit sort exists.
        derived = np.where(labels.str.contains("Career", case=False, na=False), 0, derived)
        derived = np.where(labels.str.contains("Last 10", case=False, na=False), -10, derived)
        derived = np.where(labels.str.contains("Last 5", case=False, na=False), -5, derived)
        work[sort_col] = derived

    out = work[[context_col, type_col, sort_col]].drop_duplicates().copy()
    out["_type_order"] = np.where(out[type_col].astype(str).eq("Career"), 0, 1)
    out["_sort"] = pd.to_numeric(out[sort_col], errors="coerce")

    out = out.sort_values(
        ["_type_order", "_sort", context_col],
        ascending=[True, False, True],
        na_position="last"
    )

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
                df.loc[context_eligible, score_col]
                  .rank(ascending=False, method="min")
            )

            df.loc[context_eligible, percentile_col] = (
                df.loc[context_eligible, score_col]
                  .rank(ascending=True, pct=True, method="average") * 100
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
    """
    Adds magnitude-aware role context metrics.
    This solves the percentile-only issue by measuring actual separation
    from role peers inside each ranking context.
    """

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
        {
            "role_primary_score": 0.50,
            "role_primary_percentile": 0.25,
            "role_separation_score": 0.25,
        },
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
                out.loc[valid, "role_context_value_score"]
                .rank(ascending=False, method="min")
            )
            out.loc[valid, "role_context_percentile"] = (
                out.loc[valid, "role_context_value_score"]
                .rank(ascending=True, pct=True, method="average") * 100
            )

    return out


def _pll_prepare_player_rankings(rankings):
    """
    Prepare official player rankings for display.

    The warehouse is the source of truth for player ranking scores. This function
    only normalizes legacy column aliases and fills missing rank/percentile fields
    for display safety; it does not recalculate official scores.
    """
    if rankings is None or len(rankings) == 0:
        return rankings

    df = rankings.copy()

    # Compatibility aliases from older warehouse runs.
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

    # Playmaking Value should be created by the warehouse. If an older warehouse
    # is accidentally loaded, keep the column available but blank rather than
    # recalculating official rankings in Streamlit.
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

    # Ensure expected display columns exist.
    for col in [
        "overall_rank", "overall_percentile", "position_rank", "position_percentile",
        "offensive_rank", "defensive_rank", "faceoff_rank", "goalie_rank",
        "role_context_value_score", "role_context_rank", "role_context_percentile",
        "role_primary_score", "role_primary_percentile", "role_separation_score",
        "scoring_value_score", "goal_value_score", "playmaking_value_score", "offensive_creation_score", "ground_ball_score", "base_impact_score",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    if "eligible_for_default_ranking" in df.columns:
        eligible_mask = pd.to_numeric(df["eligible_for_default_ranking"], errors="coerce").fillna(0).astype(bool)
    else:
        eligible_mask = df["overall_score"].notna() if "overall_score" in df.columns else pd.Series(False, index=df.index)

    # Fill rank fields only when absent/missing. Do not alter warehouse ranks.
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
    fig.update_layout(
        yaxis_title="",
        xaxis_tickformat=".2f",
        margin=dict(l=10, r=20, t=45, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


def _pll_tier_distribution_chart(df):
    if df is None or len(df) == 0 or "role_value_tier" not in df.columns:
        st.info("No tier distribution data available.")
        return

    tier_order = [
        "Outlier Elite",
        "Elite",
        "High-End",
        "Average / Starter",
        "Below Average",
        "Low Impact",
        "Unrated",
    ]

    tier_df = (
        df.groupby(["role_group", "role_value_tier"], dropna=False)
        .size()
        .reset_index(name="players")
    )

    tier_df["role_value_tier"] = pd.Categorical(
        tier_df["role_value_tier"],
        categories=tier_order,
        ordered=True
    )

    tier_df = tier_df.sort_values(["role_group", "role_value_tier"])

    fig = px.bar(
        tier_df,
        x="role_group",
        y="players",
        color="role_value_tier",
        barmode="stack",
        title="Role Value Tier Distribution",
        labels={
            "role_group": "Role",
            "players": "Players",
            "role_value_tier": "Role Tier"
        }
    )

    fig.update_layout(
        margin=dict(l=10, r=20, t=45, b=10),
        yaxis_tickformat=".0f"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PLAYER RANKINGS TAB
# ============================================================

with tab_player_rankings:
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
        default_context = season_contexts[0] if season_contexts else context_options[0]

        controls = st.columns([1.35, 1.0, 0.75, 1.45])

        with controls[0]:
            selected_ranking_context = st.selectbox(
                "Ranking context",
                options=context_options,
                index=context_options.index(default_context),
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
        available_teams = sorted(context_rankings["teams"].dropna().astype(str).unique().tolist()) if len(context_rankings) else []
        available_tiers = [
            t for t in [
                "Outlier Elite",
                "Elite",
                "High-End",
                "Average / Starter",
                "Below Average",
                "Low Impact",
                "Unrated",
            ]
            if t in context_rankings.get("role_value_tier", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
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

        if selected_ranking_teams:
            filtered_rankings = filtered_rankings[filtered_rankings["teams"].isin(selected_ranking_teams)]

        if selected_role_tiers and "role_value_tier" in filtered_rankings.columns:
            filtered_rankings = filtered_rankings[filtered_rankings["role_value_tier"].isin(selected_role_tiers)]

        if ranking_player_search.strip():
            filtered_rankings = filtered_rankings[
                filtered_rankings["full_name"].astype(str).str.contains(
                    ranking_player_search.strip(),
                    case=False,
                    na=False
                )
            ]

        # Stable ranking sort. Ranked rows always appear before unranked rows;
        # rows without a persisted rank fall back to the selected score. This
        # prevents small-sample / previously unranked players from floating above
        # Rank 1 while still allowing the user to review every player.
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

        # Continuous rank for the currently selected table/filter. This avoids
        # confusion when Overall Rank skips numbers after filtering to a position,
        # team, or role view.
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
                "overall_rank",
                "position_rank",
                "full_name",
                "position",
                "role_group",
                "teams",
                "games",
                "overall_score",
                "base_impact_score",
                "role_context_value_score",
                "role_value_tier",
                "offensive_score",
                "usage_possession_score",
                "defensive_score",
                "faceoff_score",
                "goalie_score",
                "scoring_value_score",
                "playmaking_value_score",
                "ground_ball_score",
                "points_per_game",
                "scoring_points_per_game",
                "goals_per_game",
                "one_point_goals_per_game",
                "two_point_goals_per_game",
                "assists_per_game",
                "shots_per_game",
                "touches_per_game",
                "caused_turnovers_per_game",
                "ground_balls_per_game",
                "turnovers_per_game",
                "faceoff_pct_for_ranking",
                "faceoffs_per_game",
                "faceoffs_won_per_game",
                "save_pct_for_ranking",
                "saves_per_game",
                "scores_against_per_game",
                "goals_against_per_game",
            ],
            "Offense": [
                "role_context_rank",
                "overall_rank",
                "position_rank",
                "full_name",
                "position",
                "teams",
                "games",
                "role_context_value_score",
                "overall_score",
                "role_primary_score",
                "role_primary_percentile",
                "role_separation_score",
                "role_value_tier",
                "scoring_value_score",
                "playmaking_value_score",
                "points_per_game",
                "scoring_points_per_game",
                "one_point_goals_per_game",
                "two_point_goals_per_game",
                "goals_per_game",
                "assists_per_game",
                "shots_per_game",
                "points_per_touch",
            ],
            "Defense": [
                "role_context_rank",
                "overall_rank",
                "position_rank",
                "full_name",
                "position",
                "teams",
                "games",
                "role_context_value_score",
                "overall_score",
                "defensive_score",
                "ct_score",
                "ground_ball_score",
                "role_primary_score",
                "role_primary_percentile",
                "role_separation_score",
                "role_value_tier",
                "caused_turnovers_per_game",
                "ground_balls_per_game",
                "turnovers_per_game",
                "touches_per_game",
                "points_per_game",
            ],
            "Faceoff": [
                "role_context_rank",
                "overall_rank",
                "position_rank",
                "full_name",
                "position",
                "teams",
                "games",
                "role_context_value_score",
                "overall_score",
                "role_primary_score",
                "role_primary_percentile",
                "role_separation_score",
                "role_value_tier",
                "faceoff_pct_for_ranking",
                "faceoffs_per_game",
                "faceoffs_won_per_game",
                "ground_balls_per_game",
                "points_per_game",
            ],
            "Goalie": [
                "role_context_rank",
                "overall_rank",
                "position_rank",
                "full_name",
                "position",
                "teams",
                "games",
                "role_context_value_score",
                "overall_score",
                "goalie_score",
                "save_pct_score",
                "saves_score",
                "role_primary_score",
                "role_primary_percentile",
                "role_separation_score",
                "role_value_tier",
                "save_pct_for_ranking",
                "saves_per_game",
                "scores_against_per_game",
                "goals_against_per_game",
                "touches_per_game",
            ],
        }

        extra_cols = [
            "role_robust_z",
            "role_adjusted_z",
            "role_group_size",
            "role_reliability",
            "offensive_score",
            "usage_possession_score",
            "defensive_score",
            "faceoff_score",
            "goalie_score",
            "scoring_value_score",
            "playmaking_value_score",
            "ground_ball_score",
            "one_point_goal_score",
            "two_point_goal_score",
            "shot_pct_for_ranking",
            "sog_rate_for_ranking",
            "goals_per_shot",
        ]

        ranking_display_cols = compact_cols_by_view.get(ranking_view, compact_cols_by_view["Overall"])

        # Always show the continuous rank for the current filtered table first.
        ranking_display_cols = ["view_rank"] + [c for c in ranking_display_cols if c != "view_rank"]

        if show_detail_cols:
            ranking_display_cols = ranking_display_cols + extra_cols

        ranking_display_cols = list(dict.fromkeys([
            c for c in ranking_display_cols
            if c and c in filtered_rankings.columns
        ]))

        ranking_table_df = filtered_rankings[ranking_display_cols].copy()

        # Display-only masking: role-specific component scores should be blank
        # for players who do not play that role. Raw stat categories can still
        # remain visible in the Overall view where they are meaningful.
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

        # For position/ranking-specific views, remove columns that are entirely
        # blank after role masking so the table does not end with empty fields.
        protected_cols = {"view_rank", "overall_rank", "position_rank", "full_name", "position", "role_group", "teams", "games", "overall_score", "role_context_value_score", "role_value_tier"}
        drop_blank_cols = []
        for c in ranking_table_df.columns:
            if c in protected_cols:
                continue
            if ranking_table_df[c].isna().all():
                drop_blank_cols.append(c)
        if drop_blank_cols:
            ranking_table_df = ranking_table_df.drop(columns=drop_blank_cols)

        # Final numeric cleanup for readability. display_table will apply the
        # final renderer, but rounding here also makes downloaded CSVs cleaner.
        for c in ranking_table_df.columns:
            if c in {"view_rank", "overall_rank", "position_rank", "role_context_rank", "games"}:
                ranking_table_df[c] = pd.to_numeric(ranking_table_df[c], errors="coerce").round(0)
            elif pd.api.types.is_numeric_dtype(ranking_table_df[c]):
                ranking_table_df[c] = pd.to_numeric(ranking_table_df[c], errors="coerce").round(2)

        st.markdown("### Ranking Table")
        display_table(
            ranking_table_df,
            height=540,
            hide_cols=[],
            max_cols=None
        )

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

            fig = px.scatter(
                scatter_df,
                x="role_context_value_score",
                y="overall_score",
                color="role_group",
                size="games",
                hover_name="full_name",
                hover_data=[
                    "position",
                    "teams",
                    "games",
                    "overall_rank",
                    "base_impact_score",
                    "role_primary_score",
                    "role_primary_percentile",
                    "role_separation_score",
                    "role_adjusted_z",
                    "role_value_tier",
                    "goal_value_score",
                    "points_per_game",
                    "one_point_goals_per_game",
                    "two_point_goals_per_game",
                    "touches_per_game",
                ],
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
                        "Overall Score",
                        "Base Impact",
                        "Role Context",
                        "Role Score",
                        "Role Percentile",
                        "Peer Separation",
                        "Usage",
                        "Goal Value",
                        "Ground Ball Value",
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
                        "full_name",
                        "position",
                        "role_group",
                        "teams",
                        "games",
                        "overall_rank",
                        "position_rank",
                        "overall_score",
                        "overall_percentile",
                        "base_impact_score",
                        "role_context_value_score",
                        "role_primary_score",
                        "role_primary_percentile",
                        "role_separation_score",
                        "role_adjusted_z",
                        "role_value_tier",
                        "goal_value_score",
                        "offensive_score",
                        "usage_possession_score",
                        "defensive_score",
                        "faceoff_score",
                        "goalie_score",
                        "points",
                        "scoring_points",
                        "one_point_goals",
                        "two_point_goals",
                        "goals",
                        "assists",
                        "shots",
                        "points_per_game",
                        "scoring_points_per_game",
                        "one_point_goals_per_game",
                        "two_point_goals_per_game",
                        "goals_per_game",
                        "assists_per_game",
                        "shots_per_game",
                        "touches_per_game",
                    ]
                    if c in player_detail.columns
                ]))

                display_table(
                    player_detail[detail_display_cols],
                    height=240,
                    hide_cols=[],
                    max_cols=None
                )


# ============================================================
# TEAM STYLE PROFILES TAB
# ============================================================

with tab_team_profiles:
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
        default_profile_context = season_profile_contexts[0] if season_profile_contexts else profile_context_options[0]

        profile_controls = st.columns([1.2, 1.5, 1.1])

        with profile_controls[0]:
            selected_profile_context = st.selectbox(
                "Profile context",
                options=profile_context_options,
                index=profile_context_options.index(default_profile_context),
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
                "team_style_overall_score",
                "net_scores_per_game",
                "offensive_volume_score",
                "offensive_efficiency_score",
                "ball_movement_score",
                "possession_control_score",
                "defensive_suppression_score",
                "pace_tempo_score",
                "scores_per_game",
                "def_scores_allowed_per_game",
                "touches_per_game",
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
            "profile_rank",
            "team_name",
            "games",
            "wins",
            "losses",
            "win_pct",
            "team_style_overall_score",
            "net_scores_per_game",
            "offensive_profile_label",
            "defensive_profile_label",
            "possession_profile_label",
            "pace_label",
            "style_summary",
        ]

        metrics_cols = [
            "profile_rank",
            "team_name",
            "team_style_overall_score",
            "offensive_volume_score",
            "offensive_efficiency_score",
            "ball_movement_score",
            "possession_control_score",
            "defensive_suppression_score",
            "pace_tempo_score",
            "scores_per_game",
            "def_scores_allowed_per_game",
            "shots_per_game",
            "def_opponent_shots_per_game",
            "touches_per_game",
            "time_in_possession_per_game_mmss",
            "def_save_pct_proxy",
        ]

        full_cols = [
            "profile_rank",
            "team_name",
            "games",
            "wins",
            "losses",
            "win_pct",
            "team_style_overall_score",
            "net_scores_per_game",
            "offensive_volume_score",
            "offensive_efficiency_score",
            "ball_movement_score",
            "possession_control_score",
            "defensive_suppression_score",
            "pace_tempo_score",
            "scores_per_game",
            "def_scores_allowed_per_game",
            "shots_per_game",
            "def_opponent_shots_per_game",
            "touches_per_game",
            "time_in_possession_per_game_mmss",
            "def_save_pct_proxy",
            "pace_label",
            "offensive_profile_label",
            "defensive_profile_label",
            "possession_profile_label",
            "style_summary",
        ]

        if team_style_view == "Summary View":
            team_style_display_cols = _pll_select_existing(filtered_profiles, summary_cols)
        elif team_style_view == "Metrics View":
            team_style_display_cols = _pll_select_existing(filtered_profiles, metrics_cols)
        else:
            team_style_display_cols = _pll_select_existing(filtered_profiles, full_cols)

        display_table(
            filtered_profiles[team_style_display_cols],
            height=420,
            hide_cols=[],
            max_cols=None
        )

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

            if len(filtered_profiles) > 0:
                fig = px.scatter(
                    filtered_profiles,
                    x="offensive_efficiency_score",
                    y="defensive_suppression_score",
                    size="team_style_overall_score",
                    color="net_scores_per_game" if "net_scores_per_game" in filtered_profiles.columns else None,
                    text="team_name",
                    hover_name="team_name",
                    hover_data=[
                        c for c in [
                            "profile_rank",
                            "games",
                            "wins",
                            "losses",
                            "style_summary",
                            "scores_per_game",
                            "def_scores_allowed_per_game",
                            "net_scores_per_game",
                            "touches_per_game",
                        ]
                        if c in filtered_profiles.columns
                    ],
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
                "team_style_overall_score",
                "offensive_volume_score",
                "offensive_efficiency_score",
                "ball_movement_score",
                "possession_control_score",
                "defensive_suppression_score",
                "pace_tempo_score",
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
                labels={
                    "team_name": "Team",
                    "score": "Score",
                    "style_metric_label": "Metric"
                }
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
                        "team_name",
                        "games",
                        "wins",
                        "losses",
                        "win_pct",
                        "team_style_overall_score",
                        "net_scores_per_game",
                        "offensive_volume_score",
                        "offensive_efficiency_score",
                        "ball_movement_score",
                        "possession_control_score",
                        "defensive_suppression_score",
                        "pace_tempo_score",
                        "scores_per_game",
                        "shots_per_game",
                        "touches_per_game",
                        "total_passes_per_game",
                        "time_in_possession_per_game_mmss",
                        "def_scores_allowed_per_game",
                        "def_goals_allowed_per_game",
                        "def_opponent_shots_per_game",
                        "def_opponent_goal_pct",
                        "def_save_pct_proxy",
                        "pace_label",
                        "offensive_profile_label",
                        "defensive_profile_label",
                        "possession_profile_label",
                        "style_summary",
                    ]
                    if c in team_detail.columns
                ]

                display_table(
                    team_detail[team_detail_cols],
                    height=240,
                    hide_cols=[],
                    max_cols=None
                )

# <<< PLL_RANKINGS_TEAM_PROFILE_TABS_END



# ============================================================
# LEADERBOARDS
# ============================================================

with tab_leaders:
    st.subheader("Leaderboards")
    st.markdown(
        '<div class="section-note">Sortable league leaderboards with cleaner default tables and advanced raw-data views available in expanders.</div>',
        unsafe_allow_html=True
    )

    player_leader_tab, team_leader_tab, defense_leader_tab = st.tabs([
        "Player Leaders",
        "Team Leaders",
        "Defensive / Opponent Leaders"
    ])

    with player_leader_tab:
        lb_cols = st.columns([1.2, 1.2, 1.0, 1.0, 0.8])

        player_scope = lb_cols[0].selectbox(
            "Player leaderboard scope",
            ["Player Seasons", "Player Career", "Player Last 5", "Player Last 10"],
            key="leader_player_scope"
        )

        if player_scope == "Player Seasons":
            table = "marts.player_season_stats"
            season_sql, season_params = sql_in_filter("season", selected_seasons)
            pos_sql, pos_params = sql_in_filter("position", selected_positions)
            where_extra = f"AND {season_sql} AND {pos_sql}"
            params_base = season_params + pos_params
        elif player_scope == "Player Career":
            table = "marts.player_career_stats"
            pos_sql, pos_params = sql_in_filter("position", selected_positions)
            where_extra = f"AND {pos_sql}"
            params_base = pos_params
        elif player_scope == "Player Last 5":
            table = "marts.player_last5_stats"
            pos_sql, pos_params = sql_in_filter("position", selected_positions)
            where_extra = f"AND {pos_sql}"
            params_base = pos_params
        else:
            table = "marts.player_last10_stats"
            pos_sql, pos_params = sql_in_filter("position", selected_positions)
            where_extra = f"AND {pos_sql}"
            params_base = pos_params

        player_cols_available = _pll_get_table_columns("marts", table.split(".")[-1])

        player_sort_options = [
            c for c in [
                "points", "points_per_game",
                "scoring_points", "scoring_points_per_game",
                "goals", "goals_per_game",
                "one_point_goals", "two_point_goals",
                "assists", "assists_per_game",
                "shots", "shots_per_game",
                "ground_balls", "ground_balls_per_game",
                "caused_turnovers", "caused_turnovers_per_game",
                "turnovers", "turnovers_per_game",
                "saves", "save_pct_calc",
                "faceoffs_won", "faceoff_pct_calc",
                "touches", "touches_per_game",
            ]
            if c in player_cols_available
        ]

        selected_sort = lb_cols[1].selectbox(
            "Sort by",
            player_sort_options,
            index=0,
            format_func=pretty_col,
            key="leader_player_sort"
        )

        leader_min_games = lb_cols[2].number_input(
            "Minimum games",
            min_value=1,
            max_value=100,
            value=min_games,
            step=1,
            key="leader_player_min_games"
        )

        leader_rows = lb_cols[3].number_input(
            "Rows",
            min_value=10,
            max_value=100,
            value=25,
            step=5,
            key="leader_player_rows"
        )

        lower_player_metrics = {"turnovers", "turnovers_per_game", "goals_against", "goals_against_per_game"}
        lower_player = selected_sort in lower_player_metrics

        lb_cols[4].caption("Sort")
        lb_cols[4].markdown("**Low best**" if lower_player else "**High best**")

        player_select_cols = [
            c for c in [
                "season", "split_type", "full_name", "position", "teams", "games",
                "points", "points_per_game", "scoring_points", "scoring_points_per_game",
                "one_point_goals", "two_point_goals", "goals", "goals_per_game",
                "assists", "assists_per_game", "shots", "shots_per_game",
                "ground_balls", "ground_balls_per_game", "turnovers", "turnovers_per_game",
                "caused_turnovers", "caused_turnovers_per_game", "saves", "faceoffs_won",
                "faceoff_pct_calc", "touches", "touches_per_game", "total_passes"
            ]
            if c in player_cols_available
        ]

        leaderboard = query_df(f"""
            SELECT {", ".join(player_select_cols)}
            FROM {table}
            WHERE games >= ?
              {where_extra}
            ORDER BY {selected_sort} {"ASC" if lower_player else "DESC"} NULLS LAST
            LIMIT 200
        """, [leader_min_games] + params_base)

        leaderboard = leaderboard.head(int(leader_rows))

        safe_bar_chart(
            leaderboard.head(20).sort_values(selected_sort, ascending=not lower_player),
            x_col="full_name",
            y_col=selected_sort,
            color_col="position" if "position" in leaderboard.columns else None,
            title=f"{player_scope} — Top {min(20, len(leaderboard))} by {pretty_col(selected_sort)}",
            orientation="h"
        )

        player_summary_cols = _pll_select_existing(
            leaderboard,
            [
                "season", "split_type", "full_name", "position", "teams", "games",
                "points", "points_per_game", "scoring_points_per_game",
                "one_point_goals", "two_point_goals",
                "goals_per_game", "assists_per_game", "shots_per_game",
                "ground_balls_per_game", "caused_turnovers_per_game",
                "touches_per_game"
            ]
        )

        display_table(leaderboard[player_summary_cols], height=460)

        with st.expander("Advanced player leaderboard table", expanded=False):
            display_table(leaderboard, height=520)

        download_csv(leaderboard, "pll_player_leaderboard.csv")

    with team_leader_tab:
        lb_cols = st.columns([1.2, 1.2, 1.0, 1.0, 0.8])

        team_scope = lb_cols[0].selectbox(
            "Team leaderboard scope",
            ["Team Seasons", "Team Last 5", "Team Last 10"],
            key="leader_team_scope"
        )

        if team_scope == "Team Seasons":
            table = "marts.team_season_stats"
            season_sql, season_params = sql_in_filter("season", selected_seasons)
            team_sql, team_params = sql_in_filter("team_name", selected_teams)
            where_extra = f"AND {season_sql} AND {team_sql}"
            params_base = season_params + team_params
        elif team_scope == "Team Last 5":
            table = "marts.team_last5_stats"
            team_sql, team_params = sql_in_filter("team_name", selected_teams)
            where_extra = f"AND {team_sql}"
            params_base = team_params
        else:
            table = "marts.team_last10_stats"
            team_sql, team_params = sql_in_filter("team_name", selected_teams)
            where_extra = f"AND {team_sql}"
            params_base = team_params

        team_cols_available = _pll_get_table_columns("marts", table.split(".")[-1])

        team_sort_options = [
            c for c in [
                "scores_per_game", "scores", "score_margin_per_game",
                "goals", "assists", "shots_per_game", "shots",
                "touches_per_game", "touches",
                "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
                "turnovers_per_game", "saves_per_game", "faceoff_pct_calc", "clear_pct_calc"
            ]
            if c in team_cols_available
        ]

        selected_team_sort = lb_cols[1].selectbox(
            "Sort by",
            team_sort_options,
            index=0,
            format_func=pretty_col,
            key="leader_team_sort"
        )

        leader_team_min_games = lb_cols[2].number_input(
            "Minimum games",
            min_value=1,
            max_value=100,
            value=min_games,
            step=1,
            key="leader_team_min_games"
        )

        leader_team_rows = lb_cols[3].number_input(
            "Rows",
            min_value=8,
            max_value=100,
            value=25,
            step=5,
            key="leader_team_rows"
        )

        lower_team_metrics = {"turnovers", "turnovers_per_game"}
        lower_team = selected_team_sort in lower_team_metrics

        lb_cols[4].caption("Sort")
        lb_cols[4].markdown("**Low best**" if lower_team else "**High best**")

        team_select_cols = [
            c for c in [
                "season", "split_type", "team_name", "games", "wins", "losses", "win_pct",
                "scores", "scores_per_game", "goals", "assists",
                "shots", "shots_per_game", "saves", "saves_per_game",
                "turnovers", "turnovers_per_game",
                "ground_balls", "caused_turnovers", "touches", "touches_per_game",
                "total_passes", "total_passes_per_game", "time_in_possession",
                "time_in_possession_per_game", "offensive_sequence_proxy", "offensive_sequence_proxy_per_game"
            ]
            if c in team_cols_available
        ]

        team_leaderboard = query_df(f"""
            SELECT {", ".join(team_select_cols)}
            FROM {table}
            WHERE games >= ?
              {where_extra}
            ORDER BY {selected_team_sort} {"ASC" if lower_team else "DESC"} NULLS LAST
            LIMIT 200
        """, [leader_team_min_games] + params_base)

        team_leaderboard = _pll_add_possession_mmss(team_leaderboard).head(int(leader_team_rows))

        safe_bar_chart(
            team_leaderboard.head(20).sort_values(selected_team_sort, ascending=not lower_team),
            x_col="team_name",
            y_col=selected_team_sort,
            color_col="season" if "season" in team_leaderboard.columns else None,
            title=f"{team_scope} — Top {min(20, len(team_leaderboard))} by {pretty_col(selected_team_sort)}",
            orientation="h"
        )

        team_summary_cols = _pll_select_existing(
            team_leaderboard,
            [
                "season", "split_type", "team_name", "games", "wins", "losses", "win_pct",
                "scores_per_game", "shots_per_game", "touches_per_game",
                "time_in_possession_per_game_mmss", "turnovers_per_game",
                "saves_per_game", "offensive_sequence_proxy_per_game"
            ]
        )

        display_table(team_leaderboard[team_summary_cols], height=460)

        with st.expander("Advanced team leaderboard table", expanded=False):
            display_table(team_leaderboard, height=520)

        download_csv(team_leaderboard, "pll_team_leaderboard.csv")

    with defense_leader_tab:
        st.markdown("### Defensive / Opponent Team Leaderboard")
        st.caption("Opponent allowance and defensive suppression metrics. Lower is better for allowed metrics.")

        if table_exists("marts", "team_defense_season_stats"):
            defense_leader_cols = st.columns([1.0, 1.2, 1.0, 0.8])

            with defense_leader_cols[0]:
                defense_scope = st.radio(
                    "Scope",
                    options=["Season", "Career"],
                    horizontal=True,
                    key="defense_leader_scope"
                )

            with defense_leader_cols[1]:
                defense_leader_metric = st.selectbox(
                    "Defensive metric",
                    options=[
                        "scores_allowed_per_game",
                        "goals_allowed_per_game",
                        "opponent_shots_per_game",
                        "opponent_goal_pct",
                        "opponent_sog_rate",
                        "save_pct_proxy",
                        "caused_turnovers_for_per_game",
                        "opponent_turnovers_per_game",
                        "ct_per_opponent_turnover",
                        "score_margin_per_game"
                    ],
                    index=0,
                    format_func=pretty_col,
                    key="defense_leader_metric"
                )

            with defense_leader_cols[2]:
                defense_min_games = st.number_input(
                    "Minimum games",
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1,
                    key="defense_min_games"
                )

            lower_is_better = {
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "opponent_goal_pct",
                "opponent_sog_rate",
                "opponent_sog_goal_pct",
                "opponent_scores_per_offensive_sequence_proxy",
            }

            defense_lower = defense_leader_metric in lower_is_better

            with defense_leader_cols[3]:
                st.caption("Sort")
                st.markdown("**Low best**" if defense_lower else "**High best**")

            if defense_scope == "Season":
                defense_leader_df = query_df("""
                    SELECT *
                    FROM marts.team_defense_season_stats
                    WHERE games >= ?
                    ORDER BY season DESC, scores_allowed_per_game ASC NULLS LAST
                """, [defense_min_games])
            else:
                defense_leader_df = query_df("""
                    SELECT *
                    FROM marts.team_defense_career_stats
                    WHERE games >= ?
                    ORDER BY scores_allowed_per_game ASC NULLS LAST
                """, [defense_min_games])

            if defense_leader_metric in defense_leader_df.columns:
                defense_leader_df = _pll_safe_sort(
                    defense_leader_df,
                    defense_leader_metric,
                    lower_is_better=defense_lower
                )

                safe_bar_chart(
                    defense_leader_df.head(20).sort_values(
                        defense_leader_metric,
                        ascending=not defense_lower
                    ),
                    x_col="team_name",
                    y_col=defense_leader_metric,
                    color_col="season" if "season" in defense_leader_df.columns else "team_name",
                    title=f"Defensive Leaderboard — {pretty_col(defense_leader_metric)}",
                    orientation="h"
                )

            defense_summary_cols = _pll_select_existing(
                defense_leader_df,
                [
                    "season", "team_name", "games",
                    "scores_allowed_per_game", "goals_allowed_per_game",
                    "opponent_shots_per_game", "opponent_goal_pct",
                    "save_pct_proxy", "caused_turnovers_for_per_game",
                    "opponent_turnovers_per_game", "score_margin_per_game"
                ]
            )

            display_table(defense_leader_df[defense_summary_cols], height=460)

            with st.expander("Advanced defensive leaderboard table", expanded=False):
                display_table(defense_leader_df, height=520)

            download_csv(defense_leader_df, "pll_defensive_leaderboard.csv")
        else:
            st.info("Defensive/opponent marts are not available in the warehouse yet.")

# ============================================================
# SCHEDULE
# ============================================================

with tab_schedule:
    st.subheader("Schedule")
    st.markdown('<div class="section-note">Full schedule inventory including completed and future games.</div>', unsafe_allow_html=True)

    schedule_fixed = schedule_display_table()

    schedule_season = st.selectbox(
        "Schedule season",
        options=seasons,
        index=len(seasons) - 1 if seasons else 0,
        key="schedule_season"
    )

    status_options = ["all"] + sorted(schedule_fixed["status_display"].dropna().unique().tolist())
    selected_status = st.selectbox("Status", options=status_options, index=0, key="schedule_status_filter")

    sched = schedule_fixed[schedule_fixed["season"] == schedule_season].copy()

    if selected_status != "all":
        sched = sched[sched["status_display"] == selected_status]

    sched = sched.sort_values("game_number")

    display_cols = [
        "season",
        "game_number",
        "game_date_guess",
        "away_team_name",
        "home_team_name",
        "away_score",
        "home_score",
        "status_display",
        "slug"
    ]

    display_cols = [c for c in display_cols if c in sched.columns]

    display_table(sched[display_cols], height=650)
    download_csv(sched[display_cols], f"pll_schedule_{schedule_season}.csv")


# ============================================================
# DATA DICTIONARY / NOTES
# ============================================================

with tab_dictionary:
    st.subheader("Data Guide")
    st.markdown(
        '<div class="section-note">Definitions, formulas, interpretation notes, and known data caveats for the PLL data platform.</div>',
        unsafe_allow_html=True
    )

    guide_tabs = st.tabs([
        "Core Stats",
        "Goalie / Faceoff",
        "Rankings",
        "Team Style",
        "Data Notes"
    ])

    with guide_tabs[0]:
        st.markdown("### Core Scoring and Possession Terms")

        core_defs = pd.DataFrame([
            ["Scores", "Team scoreboard total. This can differ from goals because PLL 2-point goals count as two scores.", "Official / source field"],
            ["Goals", "Total made goals regardless of scoreboard value. A 2-point goal is still one goal but two scores.", "Official / source field"],
            ["Scoring Points", "Goal scoring value where 1PT goals and 2PT goals are valued by scoreboard impact when available.", "Calculated / source-dependent"],
            ["1PT Goals", "Goals scored from inside the 2-point arc.", "Official / source field"],
            ["2PT Goals", "Goals scored from beyond the 2-point arc.", "Official / source field"],
            ["Points", "Player points: goals plus assists unless otherwise specified by the source table.", "Official / source field"],
            ["Shots on Goal Rate", "Shots on goal divided by total shots.", "Calculated"],
            ["Shot %", "Goals divided by shots.", "Calculated"],
            ["Touches", "Provider-tracked player or team touches. Use as a possession/usage indicator, not as official possession count.", "Provider field"],
            ["Possession Time", "Provider-tracked time of possession. Displayed in MM:SS when used as a per-game value.", "Provider field"],
            ["Offensive Sequences", "Estimated offensive possessions/sequence proxy used when official possession counts are unavailable or inconsistent.", "Calculated proxy"],
        ], columns=["Metric", "Definition", "Source / Notes"])

        display_table(core_defs, height=420)

    with guide_tabs[1]:
        st.markdown("### Goalie and Faceoff Terms")

        specialist_defs = pd.DataFrame([
            ["Save Percentage", "Saves divided by saves plus goals against. The app recalculates this for goalie pages to prevent invalid values above 100%.", "Saves / (Saves + Goals Against)"],
            ["Shots Faced", "Estimated goalie shots faced based on saves plus goals against.", "Saves + Goals Against"],
            ["Scores Against", "Opponent scoreboard scores allowed while goalie/team is credited in the source.", "Source field"],
            ["Goals Against", "Opponent goals allowed. This can differ from scores against when 2-point goals occur.", "Source field"],
            ["Clean Saves", "Provider-tracked clean saves where available.", "Source field"],
            ["Messy Saves", "Provider-tracked non-clean saves where available.", "Source field"],
            ["Faceoff Win %", "Faceoffs won divided by total faceoffs.", "FO Won / Faceoffs"],
            ["Minimum Faceoffs", "Filter used to avoid small-sample faceoff leaderboard noise.", "User-selected filter"],
        ], columns=["Metric", "Definition", "Formula / Notes"])

        display_table(specialist_defs, height=420)

    with guide_tabs[2]:
        st.markdown("### Player Ranking Formula")

        st.markdown(
            """
            The official player ranking page uses **Overall Score**. The goal is to keep rankings grounded in production while also recognizing when a player is genuinely separated from comparable players in his role.

            **Role Context Value** combines three signals:

            - **50% Role Score**: the player’s main role score, such as offense, defense, faceoff, or goalie.
            - **25% Role Percentile**: where the player ranks among players in the same role group.
            - **25% Peer Separation**: a robust z-score style measure of how far above or below role peers the player is.

            **Peer Separation** is the key improvement over percentile alone. A player can rank first in a role group without being dramatically better than the field; the separation score helps identify whether the gap is actually meaningful.
            """
        )

        ranking_defs = pd.DataFrame([
            ["Base Impact", "General all-around player impact score before final role-context adjustment."],
            ["Role Score", "Primary score for the player’s role: offense, defense, faceoff, or goalie."],
            ["Role Percentile", "Rank-based position/role signal. Useful for order, but not enough by itself."],
            ["Peer Separation", "Magnitude-based score based on robust z-score distance from role peers."],
            ["Role Context Value", "Weighted blend of role score, role percentile, and role separation."],
            ["Role Tier", "Plain-English tier based on adjusted role separation, such as Elite or High-End."],
            ["Scoring Value", "Direct scoring value signal that includes scoring points, 1PT goals, 2PT goals, and scoring efficiency."],
            ["Playmaking Value", "Creation value signal that includes assists, assists per touch, points per touch, passing involvement, and turnover security."],
            ["Goalie Transfer Adjustment", "Overall-only adjustment that compresses goalie-specific scores toward average instead of subtracting fixed points; goalie-specific views keep full goalie value."],
        ], columns=["Ranking Term", "Definition"])

        display_table(ranking_defs, height=360)

        formula_df = pd.DataFrame([
            ["Offense", "62% Base Impact + 20% Role Context + 10% Usage + 8% Goal Value"],
            ["Defense", "58% Base Impact + 34% Role Context + 8% Usage"],
            ["Faceoff", "74% Base Impact + 14% Role Context + 7% Ground-Ball Value + 5% Usage"],
            ["Goalie", "72% transfer-adjusted Base Impact + 12% transfer-adjusted Role Context + 10% transfer-adjusted Save % + 6% transfer-adjusted Save Volume"],
        ], columns=["Role Group", "Overall Score Formula"])

        display_table(formula_df, height=240)

    with guide_tabs[3]:
        st.markdown("### Team Style Profile Formula")

        team_style_defs = pd.DataFrame([
            ["Overall Style", "Composite team identity score combining offense, defense, possession, ball movement, and tempo."],
            ["Offensive Volume", "How much offensive activity a team generates through scores, shots, touches, and sequences."],
            ["Offensive Efficiency", "How efficiently a team converts offensive chances into scores/goals."],
            ["Ball Movement", "Passing and assist-oriented style signal."],
            ["Possession Control", "Touches, possession time, and possession-oriented team indicators."],
            ["Defensive Suppression", "How well a team limits opponent scoring, shot quality, and efficiency."],
            ["Pace / Tempo", "How quickly or actively a team plays based on possession and volume signals."],
            ["Net Scores/G", "Scores per game minus scores allowed per game."],
        ], columns=["Team Style Metric", "Definition"])

        display_table(team_style_defs, height=420)

    with guide_tabs[4]:
        st.markdown("### Data Source and Interpretation Notes")

        note_box(
            "Completed vs Scheduled Games",
            "The app separates completed stat-available games from scheduled games. Current-season totals can be partial until new games are scraped and processed."
        )

        note_box(
            "2026 Current Season",
            "2026 is an in-progress season in the current warehouse. Early-season ranks, trends, and team profiles should be interpreted with sample size in mind."
        )

        note_box(
            "Possession Data Note",
            "PLL provider possession fields are not perfectly consistent across all historical games. The app displays possession time in MM:SS where appropriate and exposes data-quality warnings separately."
        )

        note_box(
            "Official vs Calculated Fields",
            "Some fields come directly from the source, while others are calculated to improve consistency, formatting, or interpretability."
        )

# ============================================================
# DATA QUALITY
# ============================================================

with tab_quality:
    st.subheader("Data QA")
    st.markdown('<div class="section-note">Warehouse validation, status repair checks, and artifact inventory.</div>', unsafe_allow_html=True)

    quality = query_df("""
        SELECT *
        FROM qc.quality_summary
        ORDER BY
            CASE status
                WHEN 'fail' THEN 1
                WHEN 'warning' THEN 2
                WHEN 'pass' THEN 3
                ELSE 4
            END,
            check_name
    """)

    fail_count = int((quality["status"] == "fail").sum()) if "status" in quality.columns else 0
    warning_count = int((quality["status"] == "warning").sum()) if "status" in quality.columns else 0
    pass_count = int((quality["status"] == "pass").sum()) if "status" in quality.columns else 0

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        stat_card("Failures", fmt_value(fail_count, 0))

    with q2:
        stat_card("Warnings", fmt_value(warning_count, 0))

    with q3:
        stat_card("Passes", fmt_value(pass_count, 0))

    with q4:
        stat_card("Total Checks", fmt_value(len(quality), 0))

    st.markdown("### Quality Checks")
    display_table(quality, height=520)

    st.markdown("### 2023 Schedule Status Repair Check")

    status_repair = schedule_display_table()
    status_repair = (
        status_repair[status_repair["season"] == 2023]
        .groupby(["event_status_label", "status_display"])
        .size()
        .reset_index(name="games")
    )

    display_table(status_repair, height=180)


    # >>> PLL_DATA_QUALITY_DEFENSE_START
    st.markdown("### Defensive / Opponent Build QC")

    if table_exists("qc", "defensive_opponent_build_quality"):
        defensive_qc_df = query_df("""
            SELECT *
            FROM qc.defensive_opponent_build_quality
            ORDER BY status, check_name
        """)

        display_table(defensive_qc_df, height=320)

    else:
        st.info("No defensive/opponent QC table found.")

    st.markdown("### Possession Data QC")

    if table_exists("qc", "game_possession_quality"):
        possession_qc_df = query_df("""
            SELECT *
            FROM qc.game_possession_quality
            ORDER BY
                CASE possession_data_status
                    WHEN 'normal' THEN 4
                    WHEN 'extended_or_ot_clock' THEN 3
                    WHEN 'short_or_provider_clock' THEN 2
                    WHEN 'missing_possession_time' THEN 1
                    ELSE 0
                END,
                season,
                game_number
        """)

        display_table(possession_qc_df, height=360)

    else:
        st.info("No game possession quality table found.")

    # <<< PLL_DATA_QUALITY_DEFENSE_END

    st.markdown("### Warehouse Tables")
    display_table(table_index(), height=520)

    st.markdown("### Artifact Index")

    if os.path.exists(ARTIFACT_INDEX_PATH):
        artifact_index = pd.read_csv(ARTIFACT_INDEX_PATH)
        display_table(artifact_index, height=520)
    else:
        st.warning("artifact_index.csv was not found.")
