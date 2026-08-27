"""
shared/page.py — page setup, cross-page state, and shared filter context.

Every page used to open with the same ~20 lines: sys.path surgery,
st.set_page_config, apply_css, an os.path.exists check on the DB, and a
try/except around render_sidebar_filters. Fourteen copies that had already
drifted apart (page 18 dropped the blank line, page 16 imported DB_PATH from a
different place, page 11 imported `os` halfway down the file). `init_page()`
is now the single entry point.

It also carries the two things the app was missing:

* SELECTION STATE. There was no way to go from a player on the Rankings page to
  that player's profile — no shared session state, no query params. `select_*`
  / `selected_*` and `link_to()` give pages a common channel, and selections
  survive a page switch because they round-trip through st.query_params.

* HONEST FILTER SCOPE. The old sidebar ended with the admission "Global filters
  primarily affect Overview and Leaderboards. Explorer pages have their own
  filters." Pages now declare which filters they actually honour via
  `filters=(...)`, and only those are rendered — so a control that is on screen
  is a control that works.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Make the repo root importable before any `shared.*` import can fail. Pages
# import this module first, so this replaces the per-page sys.path preamble.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from shared import roles
from shared import segments
from shared.db import DB_PATH, filter_values
from shared.ui import apply_css, page_heading

PAGE_ICON = "🥍"
APP_NAME = "PLL Analytics"

# Filter identifiers a page can request.
F_SEASON = "season"
F_TEAM = "team"
F_POSITION = "position"
F_MIN_GAMES = "min_games"

ALL_FILTERS = (F_SEASON, F_TEAM, F_POSITION, F_MIN_GAMES)

# Session-state keys for cross-page selection.
_SEL_PLAYER = "pll_selected_player"
_SEL_TEAM = "pll_selected_team"
_SEL_SEASON = "pll_selected_season"
_QP_PLAYER = "player"
_QP_TEAM = "team"
_QP_SEASON = "season"


@dataclass
class PageContext:
    """Everything a page needs from setup, in one object."""
    seasons: list = field(default_factory=list)
    teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    players: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: list = field(default_factory=list)
    selected_seasons: list = field(default_factory=list)
    selected_teams: list = field(default_factory=list)
    selected_positions: list = field(default_factory=list)
    min_games: int = 1
    # Which games every stat on the page counts: see shared/segments.py.
    scope: str = segments.REGULAR

    @property
    def scope_label(self) -> str:
        return segments.scope_label(self.scope)

    @property
    def is_playoff_scope(self) -> bool:
        return self.scope != segments.REGULAR

    # ---------- convenience ----------

    @property
    def latest_season(self):
        return max(self.seasons) if self.seasons else None

    @property
    def team_names(self) -> list:
        if "team_name" not in self.teams.columns:
            return []
        return self.teams["team_name"].dropna().tolist()

    @property
    def player_names(self) -> list:
        if "full_name" not in self.players.columns:
            return []
        return self.players["full_name"].dropna().unique().tolist()

    def season_default_index(self, options: Sequence | None = None) -> int:
        """Index of the newest season — the sensible default for a selectbox."""
        opts = list(options if options is not None else self.seasons)
        if not opts:
            return 0
        return len(opts) - 1

    def player_id_for(self, full_name: str):
        rows = self.players[self.players["full_name"] == full_name]
        if len(rows) == 0:
            return None
        return rows["player_id"].iloc[0]

    def player_ids_for(self, names: Iterable[str]) -> list:
        names = list(names)
        rows = self.players[self.players["full_name"].isin(names)]
        return rows["player_id"].tolist()

    def position_for(self, full_name: str) -> str:
        rows = self.players[self.players["full_name"] == full_name]
        if len(rows) == 0 or "position" not in rows.columns:
            return ""
        return str(rows["position"].iloc[0] or "")

    def role_for(self, full_name: str) -> str:
        return roles.role_for_position(self.position_for(full_name))

    def team_id_for(self, team_name: str):
        rows = self.teams[self.teams["team_name"] == team_name]
        if len(rows) == 0:
            return None
        return rows["team_id"].iloc[0]

    def team_ids_for(self, names: Iterable[str]) -> list:
        names = list(names)
        rows = self.teams[self.teams["team_name"].isin(names)]
        return rows["team_id"].tolist()

    def seasons_or_all(self) -> list:
        """Selected seasons, or every season when nothing is selected."""
        return list(self.selected_seasons) if self.selected_seasons else list(self.seasons)

    def teams_or_all(self) -> list:
        return list(self.selected_teams) if self.selected_teams else self.team_names

    def positions_or_all(self) -> list:
        return list(self.selected_positions) if self.selected_positions else list(self.positions)


# ============================================================
# PAGE INITIALIZATION
# ============================================================

def init_page(title: str,
              subtitle: str = "",
              filters: Sequence[str] = (),
              icon: str = PAGE_ICON,
              layout: str = "wide",
              heading: bool = True,
              scope: bool = True) -> PageContext:
    """
    Configure the page, load filter values, render the sidebar, draw the heading.

    `filters` lists only the filters this page actually applies; nothing else is
    rendered, so no control on screen is decorative.

    `scope` renders the regular-season / playoffs selector. It is on by default
    because every page that reads a stat honours it for free — the resolver in
    shared/segments.py rewrites the table names underneath. Pass scope=False on
    pages that describe the warehouse rather than sample games (the schedule, the
    data guide, data QA), where the control would be a lie.
    """
    st.set_page_config(
        page_title=f"{title} · {APP_NAME}",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    apply_css()

    if not os.path.exists(DB_PATH):
        st.error(
            f"The PLL warehouse could not be found at `{DB_PATH}`.\n\n"
            "Run `python scripts/bootstrap_db.py` to rebuild it from the "
            "committed parquet files."
        )
        st.stop()

    try:
        seasons, teams_df, players_df, positions = filter_values()
    except Exception as exc:  # surfacing the real error beats a bare "failed"
        st.error("Failed to read the PLL warehouse.")
        st.exception(exc)
        st.stop()

    positions = roles.sort_positions(positions)
    ctx = PageContext(
        seasons=list(seasons),
        teams=teams_df,
        players=players_df,
        positions=positions,
    )

    _sync_selection_from_query_params(ctx)
    _render_sidebar(ctx, filters, scope=scope)

    if heading:
        page_heading(title, subtitle)

    # Said once, at the top, so no number below it has to be read twice. Only
    # when it isn't the default: "regular season" needs no announcing.
    if scope and ctx.scope != segments.REGULAR:
        st.caption(f"**{ctx.scope_label}** — {segments.scope_note(ctx.scope)}")

    return ctx


def _render_sidebar(ctx: PageContext, filters: Sequence[str],
                    scope: bool = True) -> None:
    requested = [f for f in ALL_FILTERS if f in set(filters)]

    st.sidebar.title(APP_NAME)
    st.sidebar.caption("PLL player and team analysis")

    # Above the filters, because it decides which games exist before any filter
    # narrows them.
    segments.suppress_scope(not scope)
    if scope:
        st.sidebar.divider()
        ctx.scope = segments.render_control()
    else:
        ctx.scope = segments.REGULAR

    if not requested:
        st.sidebar.divider()
        st.sidebar.caption(
            "This page has its own controls in the main panel."
        )
        _render_sidebar_footer(ctx)
        return

    st.sidebar.divider()
    st.sidebar.markdown("**Filters**")

    if F_SEASON in requested:
        default = [ctx.latest_season] if ctx.latest_season is not None else []
        ctx.selected_seasons = st.sidebar.multiselect(
            "Season",
            options=ctx.seasons,
            default=default,
            key="sidebar_seasons",
            help="Leave empty to include every season.",
        )

    if F_TEAM in requested:
        ctx.selected_teams = st.sidebar.multiselect(
            "Team",
            options=ctx.team_names,
            default=[],
            key="sidebar_teams",
            help="Leave empty to include every team.",
        )

    if F_POSITION in requested:
        ctx.selected_positions = st.sidebar.multiselect(
            "Position",
            options=ctx.positions,
            default=[],
            key="sidebar_positions",
            format_func=lambda p: f"{p} — {roles.position_label(p)}",
            help="Leave empty to include every position.",
        )

    if F_MIN_GAMES in requested:
        ctx.min_games = int(st.sidebar.number_input(
            "Minimum games",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key="sidebar_min_games",
            help="Excludes small-sample rows from rate-stat comparisons.",
        ))

    _render_sidebar_footer(ctx)


def _render_sidebar_footer(ctx: PageContext) -> None:
    st.sidebar.divider()
    player = selected_player()
    team = selected_team()
    if player or team:
        st.sidebar.caption("**Current selection**")
        if player:
            st.sidebar.caption(f"Player: {player}")
        if team:
            st.sidebar.caption(f"Team: {team}")
        if st.sidebar.button("Clear selection", width="stretch"):
            clear_selection()
            st.rerun()


# ============================================================
# CROSS-PAGE SELECTION
# ============================================================

def _sync_selection_from_query_params(ctx: PageContext) -> None:
    """
    Seed session state from the URL so a deep link works on first load, and
    so a browser refresh doesn't lose the current player/team.
    """
    try:
        params = st.query_params
    except Exception:
        return

    qp_player = params.get(_QP_PLAYER)
    if qp_player and _SEL_PLAYER not in st.session_state:
        if qp_player in set(ctx.player_names):
            st.session_state[_SEL_PLAYER] = qp_player

    qp_team = params.get(_QP_TEAM)
    if qp_team and _SEL_TEAM not in st.session_state:
        if qp_team in set(ctx.team_names):
            st.session_state[_SEL_TEAM] = qp_team

    qp_season = params.get(_QP_SEASON)
    if qp_season and _SEL_SEASON not in st.session_state:
        try:
            season = int(qp_season)
        except (TypeError, ValueError):
            season = None
        if season is not None and season in set(ctx.seasons):
            st.session_state[_SEL_SEASON] = season


def _write_query_param(key: str, value) -> None:
    try:
        if value is None:
            st.query_params.pop(key, None)
        else:
            st.query_params[key] = str(value)
    except Exception:
        # Query params are a convenience; never let them break a page.
        pass


def select_player(name: str | None) -> None:
    st.session_state[_SEL_PLAYER] = name
    _write_query_param(_QP_PLAYER, name)


def select_team(name: str | None) -> None:
    st.session_state[_SEL_TEAM] = name
    _write_query_param(_QP_TEAM, name)


def select_season(season) -> None:
    st.session_state[_SEL_SEASON] = season
    _write_query_param(_QP_SEASON, season)


def selected_player(default: str | None = None) -> str | None:
    return st.session_state.get(_SEL_PLAYER, default)


def selected_team(default: str | None = None) -> str | None:
    return st.session_state.get(_SEL_TEAM, default)


def selected_season(default=None):
    return st.session_state.get(_SEL_SEASON, default)


def clear_selection() -> None:
    for key in (_SEL_PLAYER, _SEL_TEAM, _SEL_SEASON):
        st.session_state.pop(key, None)
    for key in (_QP_PLAYER, _QP_TEAM, _QP_SEASON):
        _write_query_param(key, None)


def default_index(options: Sequence, value, fallback: int = 0) -> int:
    """
    Index of `value` in `options`, else `fallback`. Lets a page open on the
    player/team the user picked elsewhere without crashing if it isn't listed.
    """
    opts = list(options)
    if value is not None and value in opts:
        return opts.index(value)
    if fallback < 0:
        return max(0, len(opts) + fallback)
    return min(fallback, max(0, len(opts) - 1))


# ============================================================
# NAVIGATION
# ============================================================
#
# Page filenames are the navigation contract. Keeping them in one place means a
# rename breaks one dict rather than a dozen scattered st.page_link calls.

PAGES = {
    "home": "app.py",
    "league": "pages/05_League_Overview.py",
    "players": "pages/08_Player_Profiles.py",
    "teams": "pages/09_Team_Profiles.py",
    "specialists": "pages/10_Specialists.py",
    "compare_players": "pages/11_Compare_Players.py",
    "compare_teams": "pages/12_Compare_Teams.py",
    "rankings": "pages/13_Player_Rankings.py",
    "styles": "pages/14_Team_Styles.py",
    "leaderboards": "pages/15_Leaderboards.py",
    "matchup": "pages/07_Matchup_Preview.py",
    "schedule": "pages/16_Schedule.py",
    "guide": "pages/17_Data_Guide.py",
    "qa": "pages/18_Data_QA.py",
}


def link_to(page_key: str, label: str, player: str | None = None,
            team: str | None = None, season=None, icon: str | None = None,
            full_width: bool = False) -> None:
    """
    Render a link to another page, optionally setting the selection first.

    Because st.page_link navigates without a rerun of this script, the selection
    is written to session state now so the destination page reads it on arrival.
    """
    if player is not None:
        select_player(player)
    if team is not None:
        select_team(team)
    if season is not None:
        select_season(season)

    target = PAGES.get(page_key)
    if not target:
        return
    try:
        st.page_link(target, label=label, icon=icon,
                     width="stretch" if full_width else "content")
    except Exception:
        # Older Streamlit, or the page file was renamed — degrade to plain text
        # rather than taking down the page.
        st.caption(label)


def jump_button(page_key: str, label: str, player: str | None = None,
                team: str | None = None, season=None,
                key: str | None = None,
                full_width: bool = True) -> None:
    """
    A button that sets the selection and switches pages. Use where a link needs
    to look like an action ("View full profile →").
    """
    target = PAGES.get(page_key)
    if not target:
        return
    if st.button(label, key=key, width="stretch" if full_width else "content"):
        if player is not None:
            select_player(player)
        if team is not None:
            select_team(team)
        if season is not None:
            select_season(season)
        try:
            st.switch_page(target)
        except Exception:
            st.warning(f"Open **{label}** from the sidebar to continue.")
