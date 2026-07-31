"""
Sweep the form contexts on Compare Players, Compare Teams and Leaderboards.

All three default to Career, so `smoke_pages.py` never enters the Last 5 / Last 10
branches at all. Every context is crossed with every season here, and page 11 is
run both with two active players and with an active/retired pair — the case where
the old league-wide last5 mart silently compared 2022 form against 2026 form.

Usage:
    python scripts/sweep_form_scopes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from shared.db import query_df

SEASONS = sorted(int(s) for s in query_df(
    "SELECT DISTINCT season FROM marts.team_season_stats")["season"].dropna())


def problems(at) -> list[str]:
    out = []
    for e in at.exception:
        val = getattr(e, "value", "") or getattr(e, "body", "") or str(e)
        out.append(str(val).strip().splitlines()[0] if val else "unknown")
    out += ["st.error: " + str(getattr(e, "value", e)).strip().splitlines()[0]
            for e in at.error]
    return out


def run(page: str, state: dict):
    at = AppTest.from_file(str(ROOT / "pages" / page), default_timeout=240)
    at.run()
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def check(label: str, page: str, state: dict, counters: list[int]) -> None:
    at = run(page, state)
    counters[0] += 1
    errs = problems(at)
    if errs:
        counters[1] += 1
        print(f"[FAIL] {label}: {' | '.join(errs)}")


def main() -> int:
    counters = [0, 0]

    # ---- page 11: two active players, and an active vs a retired one ----
    active = query_df("""
        SELECT full_name FROM marts.player_season_stats
        WHERE season = 2026 AND games >= 5 ORDER BY points DESC LIMIT 2
    """)["full_name"].tolist()
    retired = query_df("""
        SELECT full_name FROM marts.player_season_stats
        WHERE season = 2022 AND games >= 5
          AND player_id NOT IN (SELECT player_id FROM marts.player_season_stats
                                WHERE season = 2026)
        ORDER BY points DESC LIMIT 1
    """)["full_name"].tolist()

    for pair, tag in ((active, "two active"),
                      (active[:1] + retired, "active + retired")):
        if len(pair) < 2:
            continue
        for context in ("Career", "Last 5", "Last 10", "Season"):
            for season in SEASONS:
                check(f"p11 {tag} / {context} / {season}", "11_Compare_Players.py", {
                    "compare_players": pair,
                    "player_compare_context": context,
                    "player_compare_form_season": season,
                    "player_compare_season": season,
                }, counters)

    # ---- page 12 ----
    teams = query_df(
        "SELECT DISTINCT team_name FROM marts.team_season_stats ORDER BY team_name"
    )["team_name"].tolist()[:3]
    for context in ("Career", "Last 5", "Last 10", "Season"):
        for season in SEASONS:
            check(f"p12 {context} / {season}", "12_Compare_Teams.py", {
                "compare_teams": teams,
                "team_compare_context": context,
                "team_compare_form_season": season,
                "team_compare_season": season,
            }, counters)

    # ---- page 15: every scope on all three tabs ----
    for section, scope_key, scopes in (
        ("Players", "leader_player_scope", ["Season", "Career", "Last 5", "Last 10"]),
        ("Teams", "leader_team_scope", ["Season", "Last 5", "Last 10"]),
        ("Defence", "defense_leader_scope", ["Season", "Career"]),
    ):
        for scope in scopes:
            check(f"p15 {section} / {scope}", "15_Leaderboards.py", {
                "leaderboard_section": section,
                scope_key: scope,
            }, counters)

    print(f"\n{counters[0] - counters[1]}/{counters[0]} form-scope renders clean")
    return 1 if counters[1] else 0


if __name__ == "__main__":
    raise SystemExit(main())
