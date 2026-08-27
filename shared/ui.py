"""
shared/ui.py — presentation layer for PLL Analytics.

What changed and why:

* FORMATTING IS METRIC-AWARE. `display_table` used to format every numeric column
  with a single `nice_num`, so Shot % rendered as "0.28" instead of "28.3%" and
  Possession Time showed a raw second count. Formatting now comes from
  shared.metrics, per column.

* CHARTS NO LONGER LIE ABOUT PRECISION. `standardize_chart` forced
  `tickformat=".2f"` on every y-axis (goal counts became "12.00") and
  `safe_bar_chart` forced `texttemplate="%{text:.2f}"` on every bar label. Both
  now ask the registry for the right format. `standardize_chart` also called
  `fig.update_traces(hovertemplate=None)`, which stripped hover labels; hover is
  now formatted rather than removed.

* CSS IS THEME-SAFE. SHARED_CSS hardcoded light-on-dark colours (#f8fafc text on
  translucent slate). With no [theme] block in config.toml the app inherited the
  viewer's browser preference, so light-mode users got white text on a white
  card. Colours are now derived from Streamlit's own theme variables, and
  config.toml pins a base theme. This also removes the need for page 07 to inject
  its own conflicting light-theme scoreboard CSS.

* LABELS COME FROM ONE PLACE. The ~250-entry COL_LABELS dict is gone; labels are
  in shared.metrics alongside each metric's unit, direction and definition.
  `pretty_col` is kept as a thin alias because it is referenced widely.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared import metrics as M

# ============================================================
# CSS
# ============================================================
#
# Colours are neutral greys at low alpha rather than fixed light/dark values, so
# a card reads correctly on either theme: borders and fills tint whatever is
# behind them, and body text simply inherits Streamlit's own text colour.
#
# Deliberately NOT using color-mix() against var(--text-color): an unsupported
# color-mix() inside a custom property is stored as-is and only becomes invalid
# when used, which drops the whole declaration instead of falling back. Grey at
# alpha has no such failure mode.

SHARED_CSS = """
<style>
    :root {
        --pll-muted: #6b7280;                    /* legible on light and dark */
        --pll-border: rgba(128, 128, 128, 0.28);
        --pll-surface: rgba(128, 128, 128, 0.07);
        --pll-surface-strong: rgba(128, 128, 128, 0.12);
        --pll-accent: rgba(59, 130, 246, 0.75);
        --pll-good: #16a34a;
        --pll-bad: #dc2626;
    }

    .main .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1700px;
    }

    h1, h2, h3 { letter-spacing: -0.03em; }

    .page-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin-bottom: 0.1rem;
    }

    .section-note, .page-subtitle {
        color: var(--pll-muted);
        font-size: 0.92rem;
        margin-top: -0.2rem;
        margin-bottom: 0.75rem;
    }

    .stat-card {
        border: 1px solid var(--pll-border);
        border-radius: 14px;
        padding: 12px 15px;
        background: var(--pll-surface);
        min-height: 88px;
        margin-bottom: 10px;
    }

    .stat-label {
        color: var(--pll-muted);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 5px;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.15;
    }

    .stat-sub { color: var(--pll-muted); font-size: 0.79rem; margin-top: 3px; }
    .stat-sub.good { color: var(--pll-good); font-weight: 600; }
    .stat-sub.bad  { color: var(--pll-bad);  font-weight: 600; }

    .profile-card {
        border: 1px solid var(--pll-border);
        border-radius: 16px;
        padding: 16px 20px;
        background: var(--pll-surface-strong);
        margin-bottom: 14px;
    }

    .profile-title { font-size: 1.5rem; font-weight: 800; margin-bottom: 3px; }
    .profile-subtitle { color: var(--pll-muted); font-size: 0.95rem; }

    .mini-card {
        border: 1px solid var(--pll-border);
        border-radius: 14px;
        padding: 13px 15px;
        background: var(--pll-surface);
        margin-bottom: 10px;
        min-height: 120px;
    }

    .mini-title { font-size: 1.02rem; font-weight: 800; margin-bottom: 7px; }
    .mini-line { font-size: 0.85rem; line-height: 1.5; }
    .mini-label { color: var(--pll-muted); font-weight: 700; }

    .note-box {
        border: 1px solid var(--pll-border);
        border-left: 3px solid var(--pll-accent);
        border-radius: 12px;
        padding: 13px 16px;
        background: var(--pll-surface);
        margin-bottom: 13px;
    }

    .note-title { font-size: 1.0rem; font-weight: 800; margin-bottom: 5px; }

    /* Home-page routing card. Fixed height so the links below a row of cards
       line up instead of stepping down with each blurb's length. */
    .nav-card {
        border: 1px solid var(--pll-border);
        border-radius: 14px;
        padding: 13px 15px 11px 15px;
        background: var(--pll-surface);
        margin-bottom: 8px;
        min-height: 118px;
    }

    .nav-question { font-size: 0.98rem; font-weight: 800; margin-bottom: 6px; }

    /* Scoreboard — replaces the light-theme CSS page 07 used to inject inline */
    .scoreboard {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        border: 1px solid var(--pll-border);
        border-radius: 16px;
        padding: 16px 22px;
        background: var(--pll-surface-strong);
        margin-bottom: 14px;
    }

    .scoreboard-team { flex: 1; }
    .scoreboard-team.away { text-align: left; }
    .scoreboard-team.home { text-align: right; }
    .scoreboard-name { font-size: 1.12rem; font-weight: 800; }
    .scoreboard-record { color: var(--pll-muted); font-size: 0.82rem; }
    .scoreboard-score { font-size: 2.3rem; font-weight: 800; line-height: 1; }
    .scoreboard-meta {
        text-align: center;
        color: var(--pll-muted);
        font-size: 0.83rem;
        min-width: 130px;
    }

    div[data-testid="stDataFrame"] { border-radius: 12px; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
</style>
"""


def apply_css() -> None:
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ============================================================
# LABELS AND FORMATTING (delegated to shared.metrics)
# ============================================================

def pretty_col(col) -> str:
    """Display label for a column. Kept as the app-wide label entry point."""
    return M.label(str(col))


def fmt_value(x, digits=2, pct=False) -> str:
    """
    Format a bare number with no column context.

    Prefer `fmt_metric(key, value)` — it knows the unit. This exists for values
    that genuinely have no metric key (counts of rows, ad-hoc arithmetic).
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(value):
        return "—"
    if pct:
        return f"{value * 100:.1f}%"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def fmt_metric(key: str, value) -> str:
    """Format `value` using the registered unit for `key`."""
    return M.format_value(key, value)


def nice_num(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return ""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.2f}"


# Time formatting — one implementation. Page 07 used to carry its own copy of
# mmss_from_seconds and format_pct_safe.
def mmss_from_seconds(x) -> str:
    return M.format_seconds(x, total=False)


def format_seconds_for_table(x, total: bool = False) -> str:
    return M.format_seconds(x, total=total, dash="")


def format_pct_safe(x) -> str:
    return M.format_as(M.UNIT_PCT01, x)


def _pll_seconds_to_mmss(value) -> str:
    return M.format_seconds(value, total=False)


def _pll_pct_text(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:.{digits}f}%"


DEFAULT_HIDE_COLS = {
    "player_id", "team_id", "opponent_team_id", "game_id", "event_id", "event_numeric_id",
    "schedule_slug", "game_slug", "source_path", "profile_url", "player_slug",
    "player_name_key", "first_name", "last_name", "team_id_raw", "team_name_raw",
    "opponent_team_id_raw", "opponent_team_name_raw", "away_team_id_raw", "home_team_id_raw",
    "winner_team_id_raw", "loser_team_id_raw", "winner_team_id", "loser_team_id",
    "event_summary_path", "team_game_stats_path", "player_game_stats_path",
    "discovery_source", "source", "source_name", "raw_path",
    "profile_sort_order", "profile_context_sort", "ranking_sort_order",
    "ranking_context_sort", "games_played_source",
}


# ============================================================
# HEADINGS AND CARDS
# ============================================================

def page_heading(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="page-title">{escape(str(title))}</div>',
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{escape(str(subtitle))}</div>',
                    unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    """Section heading with an optional explanatory line."""
    st.markdown(f"#### {title}")
    if note:
        st.markdown(f'<div class="section-note">{escape(str(note))}</div>',
                    unsafe_allow_html=True)


def stat_card(label: str, value: str, sub: str | None = None,
              tone: str | None = None) -> None:
    """
    A single KPI card. `tone` of "good"/"bad" colours the sub-line — used for
    trend deltas and rank context.
    """
    sub_html = ""
    if sub:
        cls = f"stat-sub {tone}" if tone in {"good", "bad"} else "stat-sub"
        sub_html = f'<div class="{cls}">{escape(str(sub))}</div>'
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{escape(str(label))}</div>
            <div class="stat-value">{escape(str(value))}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(row, key: str, label: str | None = None,
                sub: str | None = None, tone: str | None = None) -> None:
    """Stat card for a metric key — label, value and formatting from the registry."""
    value = row.get(key, np.nan) if hasattr(row, "get") else np.nan
    stat_card(label or M.label(key), M.format_value(key, value), sub=sub, tone=tone)


def metric_grid(row, keys: Sequence[str], columns: int = 4,
                context: pd.DataFrame | None = None,
                row_index=None, skip_missing: bool = True) -> None:
    """
    Grid of metric cards driven by the registry.

    Replaces `stat_grid`'s (label, key, digits, pct) tuples, where the digits and
    pct flags were set per call site and disagreed between pages. When `context`
    is supplied each card also shows the value's rank within it.

    `skip_missing` drops keys the row has no value for, so a caller can pass a
    generous candidate list — the season and career marts do not carry identical
    columns — without printing a row of dashes for the ones that are absent.
    """
    keys = [k for k in keys if k]
    if row is None or not keys:
        return
    if skip_missing and hasattr(row, "index"):
        present = set(row.index)
        keys = [k for k in keys if k in present and pd.notna(row.get(k))]
        if not keys:
            return
    cols = st.columns(columns)
    for i, key in enumerate(keys):
        sub = None
        if context is not None and row_index is not None:
            from shared import analysis  # local import avoids a circular import
            sub = analysis.rank_text(context, key, row_index) or None
        with cols[i % columns]:
            metric_card(row, key, sub=sub)


def stat_grid(row, specs, columns: int = 4) -> None:
    """
    Legacy grid taking (label, key[, digits[, pct]]) tuples.

    Formatting now comes from the registry; the digits/pct entries are ignored so
    a percentage can't be rendered as a raw decimal by an out-of-date call site.
    """
    if row is None or len(specs) == 0:
        return
    cols = st.columns(columns)
    for i, spec in enumerate(specs):
        label, key = spec[0], spec[1]
        value = row.get(key, np.nan) if hasattr(row, "get") else np.nan
        with cols[i % columns]:
            stat_card(label, M.format_value(key, value))


def profile_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-title">{escape(str(title))}</div>
            <div class="profile-subtitle">{escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="note-box">
            <div class="note-title">{escape(str(title))}</div>
            <div class="mini-line">{escape(str(body))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pll_page_note(title: str, body: str) -> None:
    note_box(title, body)


def nav_card(question: str, body: str) -> None:
    """Routing card for the home page: the question, then what the page does."""
    st.markdown(
        f'<div class="nav-card">'
        f'<div class="nav-question">{escape(str(question))}</div>'
        f'<div class="mini-line">{escape(str(body))}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def profile_summary_cards(df: pd.DataFrame, title_col: str,
                          specs: Sequence, columns: int = 3) -> None:
    """
    One mini-card per row. `specs` entries are (label, key) — any third element
    is ignored, since the registry decides formatting.
    """
    if df is None or len(df) == 0:
        return
    n_cols = max(1, min(columns, len(df)))
    cols = st.columns(n_cols)
    for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        title = escape(str(row.get(title_col, "Unknown")))
        lines = []
        for spec in specs:
            label, key = spec[0], spec[1]
            value = M.format_value(key, row.get(key, np.nan))
            lines.append(
                f'<div class="mini-line"><span class="mini-label">'
                f'{escape(str(label))}:</span> {escape(value)}</div>'
            )
        with cols[i % n_cols]:
            st.markdown(
                f'<div class="mini-card"><div class="mini-title">{title}</div>'
                f'{"".join(lines)}</div>',
                unsafe_allow_html=True,
            )


def definition_caption(keys: Iterable[str]) -> None:
    """
    Expander listing what each metric on screen means. Definitions live with the
    metrics, so a table can explain itself instead of relying on the Data Guide.
    """
    keys = [k for k in keys if M.definition(k)]
    if not keys:
        return
    with st.expander("What do these metrics mean?", expanded=False):
        for key in keys:
            st.markdown(f"**{M.label(key)}** — {M.definition(key)}  \n"
                        f"<span class='section-note'>{M.direction_note(key)}</span>",
                        unsafe_allow_html=True)


# ============================================================
# TABLES
# ============================================================

def make_unique_columns(cols: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
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


def collapse_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fold `competition_type` + `round_label` into one readable `segment` column.

    Game-grain queries now select both so a mixed table can say which rows are
    playoff games. Two raw columns reading "post" and "Semifinal" is worse than
    one reading "Semifinal", and in the default regular-season scope every value
    is identical — so the column is dropped rather than repeated down the page.
    """
    if df is None or len(df) == 0:
        return df

    present = [c for c in ("competition_type", "round_label") if c in df.columns]
    if not present or "segment" in df.columns:
        return df

    from shared import segments as S

    blank = pd.Series([None] * len(df), index=df.index)
    comp = df["competition_type"] if "competition_type" in df.columns else blank
    rnd = df["round_label"] if "round_label" in df.columns else blank

    out = df.drop(columns=present)
    if not any(S.is_postseason_value(v) for v in comp):
        return out

    labels = [S.segment_display(c, r) for c, r in zip(comp, rnd)]
    at = min(list(df.columns).index(c) for c in present)
    at = min(at, len(out.columns))
    out.insert(at, "segment", labels)
    return out


def prepare_display_df(df: pd.DataFrame, hide_cols=None, date_cols=None,
                       max_cols: int | None = None,
                       clean_schema: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Return (display_df, formatter_map).

    Column keys are resolved to labels only at the very end, and the formatter
    map is keyed by the FINAL label so the Styler can format by metric. Time
    columns are pre-rendered to text because M:SS isn't a number.

    Pass `clean_schema=True` when the rows come straight from the `clean` schema:
    `clean_save_pct` is stored 0–1 there and 0–100 in every mart, so the same
    column needs a different formatter depending on where it was read from.
    """
    if df is None:
        return pd.DataFrame(), {}

    out = df.copy().reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = collapse_segment_columns(out)

    hide = set(DEFAULT_HIDE_COLS)
    if hide_cols:
        hide.update(hide_cols)
    out = out[[c for c in out.columns if c not in hide]]

    if max_cols is not None and len(out.columns) > max_cols:
        out = out.iloc[:, :max_cols]

    if date_cols is None:
        date_cols = [c for c in out.columns if "date" in str(c).lower()]
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")

    # Pre-render units the Styler can't express numerically.
    for c in list(out.columns):
        unit = M.unit(str(c))
        if unit in {M.UNIT_SEC, M.UNIT_SEC_TOTAL} and pd.api.types.is_numeric_dtype(out[c]):
            total = unit == M.UNIT_SEC_TOTAL
            out[c] = out[c].apply(lambda v, t=total: M.format_seconds(v, total=t, dash=""))

    for c in out.columns:
        name = str(c).lower()
        if name == "season":
            out[c] = (pd.to_numeric(out[c], errors="coerce")
                      .astype("Int64").astype(str).replace("<NA>", ""))
        elif name == "is_home":
            out[c] = out[c].map({1: "Home", 0: "Away", True: "Home", False: "Away"}).fillna(out[c])

    # Build the formatter map keyed by final label, then rename.
    original = list(out.columns)
    labels = make_unique_columns([M.label(str(c)) for c in original])
    fmt_map = {}
    for src, label in zip(original, labels):
        if pd.api.types.is_numeric_dtype(out[src]):
            fmt_map[label] = M.formatter_for(str(src), dash="",
                                             clean_schema=clean_schema)
    out.columns = labels

    out = out.reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out, fmt_map


def display_table(df: pd.DataFrame, height: int = 420, hide_cols=None,
                  date_cols=None, max_cols: int | None = None,
                  highlight: str | None = None, clean_schema: bool = False,
                  empty_message: str = "No rows match the current filters.") -> None:
    """
    Render `df` with per-metric formatting.

    `highlight` names a column to colour-scale, using the metric's direction so
    green always means good. See prepare_display_df for `clean_schema`.
    """
    out, fmt_map = prepare_display_df(df, hide_cols=hide_cols, date_cols=date_cols,
                                      max_cols=max_cols, clean_schema=clean_schema)
    if out is None or len(out) == 0:
        st.info(empty_message)
        return

    try:
        styler = (
            out.style
            .format(fmt_map, na_rep="")
            .set_properties(**{"text-align": "center", "vertical-align": "middle"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"),
                                             ("font-weight", "700"),
                                             ("vertical-align", "middle")]},
                {"selector": "td", "props": [("text-align", "center"),
                                             ("vertical-align", "middle")]},
            ])
        )
        if highlight:
            label = M.label(highlight)
            if label in out.columns and pd.api.types.is_numeric_dtype(out[label]):
                styler = styler.background_gradient(
                    subset=[label],
                    cmap="RdYlGn_r" if M.is_lower_better(highlight) else "RdYlGn",
                )
        st.dataframe(styler, width="stretch", hide_index=True, height=height)
    except Exception:
        st.dataframe(out, width="stretch", hide_index=True, height=height)


def comparison_matrix(df: pd.DataFrame, entity_col: str,
                      metrics: Sequence[str]) -> pd.DataFrame:
    """
    Metrics as rows, entities as columns, values formatted per metric, plus a
    "Best" column naming the leader — the old version left the reader to work
    out which end of each row was good.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().sum() == 0:
            continue

        row = {"Metric": M.label(metric)}
        for idx, r in df.iterrows():
            row[str(r.get(entity_col, "Unknown"))] = M.format_value(metric, r.get(metric))

        if M.direction(metric) is not None:
            best_idx = values.idxmin() if M.is_lower_better(metric) else values.idxmax()
            if pd.notna(best_idx):
                row["Best"] = str(df.at[best_idx, entity_col])
        else:
            row["Best"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def display_comparison_matrix(df: pd.DataFrame, entity_col: str,
                              metrics: Sequence[str], height: int = 420) -> None:
    matrix = comparison_matrix(df, entity_col, metrics)
    if matrix is None or len(matrix) == 0:
        st.info("No shared metrics available for the current selection.")
        return
    # Already formatted as strings — bypass the metric formatter.
    st.dataframe(matrix, width="stretch", hide_index=True, height=height)


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if df is None or len(df) == 0:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def add_window_summary_rows(df: pd.DataFrame, label_col: str = "row_type") -> pd.DataFrame:
    """Append Window Total / Window Avg rows to a game-log window."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy().reset_index(drop=True)
    out.insert(0, label_col, [f"Game {i + 1}" for i in range(len(out))])

    excluded = {"season", "game_number", "is_home"}
    numeric_cols = [c for c in out.select_dtypes(include=[np.number]).columns
                    if c not in excluded]

    total_row = {c: "" for c in out.columns}
    avg_row = {c: "" for c in out.columns}
    total_row[label_col] = "Window Total"
    avg_row[label_col] = "Window Avg"
    for c in numeric_cols:
        # Totals are meaningless for rates — average them instead of summing.
        if M.is_percent(c) or str(c).endswith(("_per_game", "_pct", "_rate", M.PER_100_SUFFIX)):
            total_row[c] = out[c].mean(skipna=True)
        else:
            total_row[c] = out[c].sum(skipna=True)
        avg_row[c] = out[c].mean(skipna=True)
    return pd.concat([out, pd.DataFrame([total_row, avg_row])], ignore_index=True)


# ============================================================
# CHARTS
# ============================================================

def clean_chart_x(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    out = df.copy()
    if x_col in out.columns and str(x_col).lower() == "season":
        out[x_col] = (pd.to_numeric(out[x_col], errors="coerce")
                      .astype("Int64").astype(str).replace("<NA>", ""))
    return out


def standardize_chart(fig, category_x: bool = False, y_key: str | None = None,
                      height: int = 420):
    """
    Apply consistent layout. The y-axis format follows the metric rather than a
    blanket ".2f", and hover is left intact (it used to be nulled outright).
    """
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        hovermode="x unified" if category_x else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, title=None),
    )
    if y_key:
        tickformat = M.plotly_tickformat(y_key)
        if tickformat:
            fig.update_yaxes(tickformat=tickformat)
    if category_x:
        fig.update_xaxes(type="category")
    return fig


def safe_line_chart(df: pd.DataFrame, x_col: str, y_cols: Sequence[str],
                    title: str, color_col: str | None = None,
                    height: int = 420) -> None:
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return
    if x_col not in df.columns:
        st.warning(f"Missing x-axis column: {x_col}")
        return

    available = [c for c in y_cols if c in df.columns
                 and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not available:
        st.info("No data available for the selected chart metrics.")
        return

    use_cols = [x_col] + ([color_col] if color_col and color_col in df.columns else []) + available
    chart_df = clean_chart_x(df[use_cols].copy(), x_col)

    fig = px.line(
        chart_df, x=x_col, y=available,
        color=color_col if color_col and color_col in chart_df.columns else None,
        markers=True, title=title,
        labels={c: M.label(c) for c in chart_df.columns},
    )
    # Single-metric charts can format the axis; mixed-unit charts must not.
    y_key = available[0] if len(available) == 1 else None
    fig = standardize_chart(fig, category_x=(str(x_col).lower() == "season"),
                            y_key=y_key, height=height)
    st.plotly_chart(fig, width="stretch")


def safe_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                   color_col: str | None = None, orientation: str = "v",
                   show_labels: bool = True, height: int = 420) -> None:
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return

    required = [x_col, y_col] + ([color_col] if color_col else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing chart columns: {missing}")
        return
    if pd.to_numeric(df[y_col], errors="coerce").notna().sum() == 0:
        st.info(f"No {M.label(y_col)} data available for this selection.")
        return

    chart_df = clean_chart_x(df.copy(), x_col)
    labels = {c: M.label(c) for c in chart_df.columns}
    text_arg = y_col if show_labels else None

    if orientation == "h":
        fig = px.bar(chart_df, x=y_col, y=x_col, color=color_col, text=text_arg,
                     title=title, orientation="h", labels=labels)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tickformat = M.plotly_tickformat(y_col)
        if tickformat:
            fig.update_xaxes(tickformat=tickformat)
        fig = standardize_chart(fig, height=height)
    else:
        fig = px.bar(chart_df, x=x_col, y=y_col, color=color_col, text=text_arg,
                     title=title, labels=labels)
        fig = standardize_chart(fig, category_x=(str(x_col).lower() == "season"),
                                y_key=y_col, height=height)

    if show_labels:
        fig.update_traces(texttemplate=M.plotly_texttemplate(y_col),
                          textposition="outside", cliponaxis=False)
    if color_col == x_col:
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")


def safe_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                 size_col: str | None = None, color_col: str | None = None,
                 title: str = "Scatter", hover_col: str | None = None,
                 quadrants: bool = False, height: int = 460) -> None:
    """
    Scatter with optional median quadrant lines — for offense-vs-defense style
    views where "which corner is good" is the whole point of the chart.
    """
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return

    required = [x_col, y_col] + [c for c in (size_col, color_col) if c]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing chart columns: {missing}")
        return

    plot_df = df.copy()
    # Plotly errors on negative sizes; drop the size encoding rather than the rows.
    if size_col:
        sizes = pd.to_numeric(plot_df[size_col], errors="coerce")
        if sizes.notna().sum() == 0 or (sizes.dropna() < 0).any():
            size_col = None

    hover_name = hover_col or ("full_name" if "full_name" in plot_df.columns
                               else ("team_name" if "team_name" in plot_df.columns else None))

    fig = px.scatter(plot_df, x=x_col, y=y_col, size=size_col, color=color_col,
                     hover_name=hover_name, title=title,
                     labels={c: M.label(c) for c in plot_df.columns})

    if quadrants:
        x_med = pd.to_numeric(plot_df[x_col], errors="coerce").median()
        y_med = pd.to_numeric(plot_df[y_col], errors="coerce").median()
        if pd.notna(x_med):
            fig.add_vline(x=x_med, line_dash="dot", line_width=1, opacity=0.45)
        if pd.notna(y_med):
            fig.add_hline(y=y_med, line_dash="dot", line_width=1, opacity=0.45)

    fig = standardize_chart(fig, y_key=y_col, height=height)
    tickformat = M.plotly_tickformat(x_col)
    if tickformat:
        fig.update_xaxes(tickformat=tickformat)
    st.plotly_chart(fig, width="stretch")


def metric_bar(df: pd.DataFrame, entity_col: str, metrics: Sequence[str],
               title: str, height: int = 420) -> None:
    """
    Grouped bar of several 0–100 component scores for one entity.
    Pages 13 and 14 each carried their own `_pll_metric_bar` doing this.
    """
    if df is None or len(df) == 0:
        st.info("No chart data available.")
        return
    available = [m for m in metrics if m in df.columns
                 and pd.to_numeric(df[m], errors="coerce").notna().any()]
    if not available:
        st.info("No component scores available for this selection.")
        return

    long_df = df.melt(id_vars=[entity_col], value_vars=available,
                      var_name="component", value_name="score")
    long_df["component"] = long_df["component"].map(M.label)
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")

    fig = px.bar(long_df, x="score", y="component",
                 color=entity_col if long_df[entity_col].nunique() > 1 else None,
                 orientation="h", title=title, text="score",
                 labels={"score": "Score", "component": "Component"},
                 barmode="group")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 105], tickformat=".0f")
    fig = standardize_chart(fig, height=height)
    st.plotly_chart(fig, width="stretch")


# ============================================================
# METRIC PICKERS
# ============================================================

def metric_selectbox(label: str, options: Sequence[str], key: str,
                     default: str | None = None, container=None,
                     help: str | None = None) -> str | None:
    """
    Selectbox over metric keys with registry labels and a direction caption, so
    the user can see whether high or low is good without consulting a legend.
    """
    options = [o for o in options if o]
    if not options:
        return None
    target = container if container is not None else st
    index = options.index(default) if default in options else 0
    chosen = target.selectbox(label, options=options, index=index,
                              format_func=M.label, key=key, help=help)
    if chosen and M.direction(chosen):
        caption = M.direction_note(chosen)
        definition = M.definition(chosen)
        target.caption(f"{caption}. {definition}" if definition else caption)
    return chosen


def family_metric_picker(df: pd.DataFrame, candidates: Sequence[str],
                         key: str, label: str = "Metrics",
                         default: Sequence[str] | None = None) -> list[str]:
    """
    Multiselect grouped by metric family, restricted to columns that hold data.
    Replaces the 100-column raw dumps: the user opts in to what they want.
    """
    available = M.with_data(df, candidates)
    if not available:
        return []
    grouped = M.by_family(available)
    ordered = [k for fam in grouped.values() for k in fam]
    default_keys = [d for d in (default or ordered[:8]) if d in available]

    def fmt(k: str) -> str:
        return f"{M.FAMILY_LABELS.get(M.family(k), 'Other')} · {M.label(k)}"

    return st.multiselect(label, options=ordered, default=default_keys,
                          format_func=fmt, key=key)


# ============================================================
# DATA HELPERS
# ============================================================

def _pll_select_existing(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return M.existing(df, cols)


def _pll_safe_sort(df: pd.DataFrame, metric: str,
                   lower_is_better: bool | None = None) -> pd.DataFrame:
    ascending = lower_is_better if lower_is_better is not None else None
    return M.sort_df(df, metric, ascending=ascending)


def _pll_apply_goalie_save_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute goalie save% as saves / (saves + goals against).

    Needed because the source save_pct can exceed 1 on partial-game rows. Also
    adds shots_faced_calc, since the provider doesn't supply shots faced.
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

    shots_faced = saves + against
    out["shots_faced_calc"] = shots_faced
    out["save_pct_display"] = (saves / shots_faced.replace(0, np.nan)).clip(lower=0, upper=1)
    out["save_pct_display_pct"] = out["save_pct_display"].apply(_pll_pct_text)
    if "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)
        out["shots_faced_per_game_calc"] = shots_faced / games
    return out


def _pll_add_possession_mmss(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "time_in_possession_per_game" in out.columns:
        out["time_in_possession_per_game_mmss"] = \
            out["time_in_possession_per_game"].apply(_pll_seconds_to_mmss)
    elif "time_in_possession" in out.columns and "games" in out.columns:
        games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)
        out["time_in_possession_per_game"] = \
            pd.to_numeric(out["time_in_possession"], errors="coerce") / games
        out["time_in_possession_per_game_mmss"] = \
            out["time_in_possession_per_game"].apply(_pll_seconds_to_mmss)
    return out


def scoreboard(away_name: str, away_score, home_name: str, home_score,
               meta: str = "", away_sub: str = "", home_sub: str = "") -> None:
    """
    Game scoreboard using the shared theme. Page 07 previously injected its own
    light-theme CSS here, which fought the app's card styling.
    """
    def fmt(v):
        return "—" if v is None or pd.isna(v) else f"{int(round(float(v)))}"

    st.markdown(
        f"""
        <div class="scoreboard">
            <div class="scoreboard-team away">
                <div class="scoreboard-name">{escape(str(away_name))}</div>
                <div class="scoreboard-record">{escape(str(away_sub))}</div>
                <div class="scoreboard-score">{fmt(away_score)}</div>
            </div>
            <div class="scoreboard-meta">{escape(str(meta))}</div>
            <div class="scoreboard-team home">
                <div class="scoreboard-name">{escape(str(home_name))}</div>
                <div class="scoreboard-record">{escape(str(home_sub))}</div>
                <div class="scoreboard-score">{fmt(home_score)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
