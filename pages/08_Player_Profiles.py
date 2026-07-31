"""
Player Profiles — one player, through every lens the warehouse supports.

The sidebar filters are not requested: the player picker and the game-log filters
in the main panel are what this page acts on.

Two role bugs fixed. The page classified positions with a local literal set
`{"A", "M", "AT", "MF", "SSDM"}`, which put short-stick defensive midfield — 64
players, the third-largest position group — on the offensive side; `roles.py`
exists to settle that, and this page now asks it. And the role sections were
gated on hardcoded position strings, so the metric grids showed a goalie a
Shot % of 0.0 and an attackman an empty Save Quality panel.

The Save Quality panel also stacked `clean_save_pct` (clean ÷ saves, 40.1%) next
to `clean_save_rate` (clean ÷ shots faced, 22.9%) under the near-identical labels
"Clean Save%" and "Clean Save Rate", so the same goalie appeared to have two
contradictory clean-save numbers. Both now carry their registry labels — Clean
Save Share and Clean Saves/Shot — and a caption saying what each divides by.

Three more, matching the fixes made to Team Profiles:

* No figure on the page carried a rank. 2.4 points per game is a different story
  for an attackman than for a defenseman, and the page gave the reader no way to
  tell which one they were looking at. Every headline and rate card now ranks the
  player against others at the same position in the same context, via
  `metric_grid`'s `context=` hook.
* Recent Form read `marts.player_last5_stats`, the league-wide "last five games
  played anywhere" mart. Every row in it is 2026, so a player's 2022 profile
  reported his mid-2026 form under the heading "Last 5", and the game list under
  it — an unscoped `ORDER BY game_date_utc DESC LIMIT 5` — listed 2026 opponents.
  `player_season_last5_stats` is the season-scoped mart.
* Recent Form showed a window aggregate and a bar chart of individual games but
  no trailing mean, so a player averaging 3 points while sliding from 5 to 1 read
  identically to one holding steady at 3. `analysis.add_rolling` and
  `analysis.form_delta` existed for this and had no callers anywhere in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from shared import analysis
from shared import metrics as M
from shared import page as P
from shared import roles
from shared import ui
from shared.db import query_df
from shared.ui import (
    stat_card, safe_bar_chart, safe_line_chart, display_table,
    fmt_value, pretty_col, profile_header, add_window_summary_rows, download_csv
)

ctx = P.init_page(
    "Player Profiles",
    "Review one player through career, season, recent-form, game-log, and "
    "opponent-split lenses.",
)

player_names = ctx.player_names

selected_player = st.selectbox(
    "Select player",
    options=player_names,
    # Honours a player picked on another page, so following a name from the
    # Rankings or Leaderboards arrives on the right profile.
    index=P.default_index(player_names, P.selected_player()),
    key="player_explorer_select"
)

if selected_player:
    # Keep the choice for the next page, so a jump from here carries the player.
    P.select_player(selected_player)
    player_row = ctx.players[ctx.players["full_name"] == selected_player].iloc[0]
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

    # Role decides which stats are worth showing. Asking roles.py rather than a
    # local position set is what keeps SSDM on the defensive side of the app.
    position = str(summary.get("position", player_row.get("position", "")) or "")
    role = roles.role_for_position(position)

    # The role is only named when it adds something. "Goalie (goalie)" and
    # "Faceoff Specialist, judged as faceoff" are the position restating itself;
    # "Short-Stick D-Midfield, judged as defense" is the case worth spelling out,
    # since that position is the one the app used to file under offence.
    position_name = roles.position_label(position)
    role_name = roles.role_label(role).lower()
    scope = (position_name if role_name in position_name.lower()
             else f"{position_name}, judged as {role_name}")

    # ------------------------------------------------------------------
    # PEER CONTEXT
    #
    # Ranks are only meaningful against comparable players, so the peer group is
    # everyone at the same role in the same context with a real sample. Ranking a
    # defenseman's points per game against the whole league would put him near the
    # bottom by construction and say nothing.
    # ------------------------------------------------------------------

    if selected_context == "Career":
        peers = query_df("SELECT * FROM marts.player_career_stats")
        peer_context_label = "career"
    else:
        peers = query_df("SELECT * FROM marts.player_season_stats WHERE season = ?",
                         [int(selected_context)])
        peer_context_label = str(selected_context)

    peer_row_index = None
    if len(peers):
        peers = roles.add_role_column(peers)
        peers = peers[peers["role_group"] == role]
        # A one-game callup at 4.0 points per game would otherwise sit above every
        # full-season player in every rate rank on the page.
        if "games" in peers.columns:
            min_peer_games = 3 if selected_context != "Career" else 5
            enough = pd.to_numeric(peers["games"], errors="coerce").fillna(0) >= min_peer_games
            # Never filter the subject out of his own peer group.
            peers = peers[enough | (peers["player_id"].astype(str) == str(player_id))]
        match = peers.index[peers["player_id"].astype(str) == str(player_id)]
        peer_row_index = match[0] if len(match) else None

    peer_frame = peers if peer_row_index is not None else None
    peer_note = (f"Ranks are against the {len(peers)} {role_name} players with "
                 f"meaningful minutes in the {peer_context_label} table."
                 if peer_frame is not None else "")

    article = "an" if scope[:1].upper() in "AEIOU" else "a"
    ui.section("Headline Stats",
               f"The stats that matter for {article} {scope}. {peer_note}".strip())
    # metric_grid formats from the registry, so a percentage renders as a
    # percentage without the caller restating the digits and pct flag; context=
    # turns each card's sub-line into a peer rank.
    ui.metric_grid(summary, roles.headline_metrics(position), columns=3,
                   context=peer_frame, row_index=peer_row_index)

    ui.section("Totals")
    ui.metric_grid(
        summary,
        roles.relevant_metrics(position, [
            "games", "points", "goals", "assists", "shots", "shots_on_goal",
            "ground_balls", "caused_turnovers", "turnovers", "touches",
            "saves", "goals_against",
        ]),
        columns=4,
    )

    ui.section("Per-Game and Rate Stats",
               "Per-game figures, comparable across seasons of different lengths. "
               + peer_note)
    # relevant_metrics drops the ones that are meaningless for this role, so a
    # goalie no longer gets a Shot % of 0.0 presented as a finding.
    ui.metric_grid(
        summary,
        roles.relevant_metrics(position, [
            "points_per_game", "goals_per_game", "assists_per_game",
            "shots_per_game", "shot_pct_calc", "ground_balls_per_game",
            "turnovers_per_game", "caused_turnovers_per_game",
            "touches_per_game", "faceoff_pct_calc",
            "save_pct_calc", "saves_per_game", "goals_against_per_game",
        ]),
        columns=4,
        context=peer_frame,
        row_index=peer_row_index,
    )

    # ------------------------------------------------------------------
    # Role-specific panels. Gated on roles.py rather than a local position set,
    # which is what previously routed SSDM to the offensive panel.
    # ------------------------------------------------------------------
    if roles.is_offense(position):
        playmaking = M.with_values(summary, [
            "assist_conv_rate", "assist_opp_per_game", "two_pt_conversion",
            "assist_opportunities",
        ])
        if playmaking:
            ui.section(
                "Playmaking Efficiency",
                "Assist conversion = Assists ÷ Assist Opportunities. "
                "2PT conversion = 2PT Goals ÷ 2PT Shots attempted.",
            )
            ui.metric_grid(summary, playmaking, columns=4)

    if roles.is_goalie(position):
        save_quality = M.with_values(summary, [
            "clean_saves", "messy_saves", "clean_save_pct", "clean_save_rate",
        ])
        if save_quality:
            # These two divide by different things and used to sit side by side
            # under the labels "Clean Save%" and "Clean Save Rate", which read as
            # the same statistic reported twice with different answers.
            ui.section(
                "Save Quality",
                "Clean saves are controlled stops; messy saves include scramble "
                "and rebound saves. Clean Save Share divides by saves made; "
                "Clean Saves/Shot divides by every shot faced, so it is always "
                "the smaller number.",
            )
            ui.metric_grid(summary, save_quality, columns=4)

    if roles.is_faceoff(position):
        faceoff = M.with_values(summary, [
            "faceoff_pct_calc", "faceoffs_won", "faceoffs_lost", "faceoffs",
            "faceoffs_won_per_game", "ground_balls_per_game",
        ])
        if faceoff:
            ui.section("Faceoff Work",
                       "Win rate and volume at the X, plus the ground balls that "
                       "follow from it.")
            ui.metric_grid(summary, faceoff, columns=3)

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

    ui.section(
        "Season Trend",
        "Career arc. Raw rates answer 'what did he produce'; percentile answers "
        "'how did that compare to his peers that year', which is the version that "
        "separates a player improving from a league scoring more.",
    )

    trend_cols = ["points_per_game", "goals_per_game", "assists_per_game",
                  "shots_per_game", "ground_balls_per_game",
                  "caused_turnovers_per_game", "turnovers_per_game",
                  "touches_per_game", "shot_pct_calc", "save_pct_calc",
                  "saves_per_game", "goals_against_per_game", "faceoff_pct_calc",
                  "faceoffs_won_per_game"]
    trend_options = M.with_data(player_seasons,
                                roles.relevant_metrics(position, trend_cols))

    # The default follows the role's headline metrics. A fixed
    # points/goals/assists default drew three flat zero lines for every goalie,
    # since `relevant_metrics` keeps scoring columns for them (a goalie can score)
    # even though almost none ever do.
    trend_default = [k for k in roles.headline_metrics(position)
                     if k in trend_options][:3]
    if not trend_default:
        trend_default = trend_options[:2]

    trend_scale = st.radio(
        "Trend scale",
        options=["Raw rate", "Peer percentile"],
        horizontal=True,
        key=f"player_trend_scale_{player_id}",
        help="Peer percentile ranks the player against others in the same role "
             "that season, so 100 is the best in the league and 50 is average.",
    )

    trend_selection = st.multiselect(
        "Trend metrics",
        options=trend_options,
        default=trend_default,
        format_func=M.label,
        key=f"player_trend_metrics_{player_id}"
    )

    if len(player_seasons) and trend_selection:
        if trend_scale == "Raw rate":
            season_trend_df = player_seasons[["season"] + trend_options].copy()
            y_cols = trend_selection
            trend_title = f"{selected_player} — Season Trend"
        else:
            # Percentile needs the whole league by season, ranked within role.
            # `add_league_context` does exactly this and had no callers anywhere.
            all_seasons = query_df("SELECT * FROM marts.player_season_stats")
            all_seasons = roles.add_role_column(all_seasons)
            if "games" in all_seasons.columns:
                keep = pd.to_numeric(all_seasons["games"], errors="coerce").fillna(0) >= 3
                all_seasons = all_seasons[
                    keep | (all_seasons["player_id"].astype(str) == str(player_id))]
            for metric in trend_selection:
                all_seasons = analysis.add_league_context(
                    all_seasons, metric, group_cols=["season", "role_group"])
            mine = all_seasons[all_seasons["player_id"].astype(str) == str(player_id)]
            pct_cols = [f"{m}_percentile" for m in trend_selection
                        if f"{m}_percentile" in mine.columns]
            season_trend_df = mine[["season"] + pct_cols].sort_values("season").copy()
            # Registry labels for the legend. "percentile" spelled out, not
            # "%ile": M.label title-cases whatever it is handed, which turns
            # "%ile" into "%Ile".
            renames = {f"{m}_percentile": f"{M.label(m)} percentile"
                       for m in trend_selection}
            season_trend_df = season_trend_df.rename(columns=renames)
            y_cols = [c for c in renames.values() if c in season_trend_df.columns]
            trend_title = (f"{selected_player} — percentile among {role_name} "
                           f"players, by season")

        safe_line_chart(season_trend_df, x_col="season", y_cols=y_cols,
                        title=trend_title)
        if trend_scale == "Peer percentile":
            st.caption("100 is the best mark in the role that season, 50 average. "
                       "Seasons with fewer than three games played are excluded "
                       "from the peer pool but not from this line.")

    ui.section(
        "Recent Form",
        "The player's last games in the context being viewed — not his last games "
        "overall.",
    )

    split_choice = st.radio(
        "Recent form window",
        options=["Last 5", "Last 10"],
        horizontal=True,
        key=f"player_recent_split_{player_id}"
    )

    window_n = 5 if split_choice == "Last 5" else 10

    # Season-scoped marts, not the league-wide last5/last10 ones. Every row in
    # those is 2026, so a 2022 profile showed 2026 form under the heading
    # "Last 5". Only the Career context wants the unscoped view.
    if selected_context == "Career":
        split_table = ("marts.player_last5_stats" if split_choice == "Last 5"
                       else "marts.player_last10_stats")
        split_df = query_df(f"SELECT * FROM {split_table} WHERE player_id = ?",
                            [player_id])
        form_window_note = "most recent games in any season"
    else:
        split_table = ("marts.player_season_last5_stats" if split_choice == "Last 5"
                       else "marts.player_season_last10_stats")
        split_df = query_df(
            f"SELECT * FROM {split_table} WHERE player_id = ? AND season = ?",
            [player_id, int(selected_context)])
        form_window_note = f"last games of {selected_context}"

    if len(split_df) > 0:
        split_summary = split_df.iloc[0]
        profile_header(
            f"{selected_player} — {split_choice}",
            f"{form_window_note} | Games: "
            f"{fmt_value(split_summary.get('games', np.nan), 0)} | Opponents: "
            f"{split_summary.get('opponents', '—')} | Teams: "
            f"{split_summary.get('teams', '—')}"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Window Totals")
            ui.metric_grid(
                split_summary,
                roles.relevant_metrics(position, [
                    "points", "goals", "assists", "shots", "ground_balls",
                    "turnovers", "caused_turnovers", "touches",
                    "saves", "goals_against", "faceoffs_won",
                ]),
                columns=4,
            )

        with c2:
            st.markdown("#### Window Averages")
            ui.metric_grid(
                split_summary,
                roles.relevant_metrics(position, [
                    "points_per_game", "goals_per_game", "assists_per_game",
                    "shots_per_game", "ground_balls_per_game", "turnovers_per_game",
                    "caused_turnovers_per_game", "touches_per_game",
                    "save_pct_calc", "saves_per_game", "faceoff_pct_calc",
                ]),
                columns=4,
            )

        # Scoped like the aggregate above.
        season_clause = "" if selected_context == "Career" else "AND season = ?"
        recent_params = [player_id]
        if selected_context != "Career":
            recent_params.append(int(selected_context))
        recent_games = query_df(f"""
            SELECT season, game_number, game_date_utc, team_name, opponent_team_name, is_home,
                   points, goals, assists, assist_opportunities, shots, shots_on_goal,
                   ground_balls, turnovers, caused_turnovers,
                   saves, clean_saves, messy_saves, saa, faceoffs_won, faceoffs_lost,
                   touches, total_passes
            FROM clean.player_game_stats
            WHERE player_id = ? {season_clause}
            ORDER BY game_date_utc DESC, season DESC, game_number DESC
            LIMIT {window_n}
        """, recent_params)

        st.markdown(f"#### {split_choice} Individual Games")
        st.caption("The bottom two rows summarize the selected window across the individual games shown above.")
        recent_with_summary = add_window_summary_rows(recent_games)
        display_table(recent_with_summary, height=360, clean_schema=True)

        # Role-aware candidates, so a goalie's window chart offers saves and SAA
        # rather than a flat line of zero points.
        recent_metric = ui.metric_selectbox(
            f"{split_choice} game-by-game chart metric",
            options=M.with_data(recent_games, roles.relevant_metrics(position, [
                "points", "goals", "assists", "shots", "shots_on_goal",
                "ground_balls", "turnovers", "caused_turnovers", "touches",
                "saves", "saa", "faceoffs_won",
            ])),
            key=f"player_recent_game_metric_{player_id}_{window_n}",
            default="points",
        )

        if len(recent_games) > 0 and recent_metric:
            recent_chart = recent_games.sort_values(["season", "game_number"]).copy()
            recent_chart["game_label"] = (recent_chart["season"].astype(str) + " G"
                                          + recent_chart["game_number"].astype(str))
            safe_bar_chart(
                recent_chart, x_col="game_label", y_col=recent_metric,
                title=f"{selected_player} — {split_choice} {M.label(recent_metric)} by Game",
            )

            # The direction of travel inside the window, which neither the
            # aggregate nor the bar chart states: 5-4-3-2-1 and a flat 3 both
            # average 3.
            if len(recent_chart) > 3:
                delta = analysis.form_delta(recent_chart, recent_metric, window=3)
                if pd.notna(delta) and abs(delta) > 1e-9:
                    st.caption(
                        f"Last three games average "
                        f"{M.format_value(recent_metric, abs(delta))} "
                        f"{'above' if delta > 0 else 'below'} the earlier games in "
                        f"this window."
                    )

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
