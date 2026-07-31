"""
Deep sweep of Team Profiles (page 09).

Every team crossed with every context (career and each season) and both form
windows, plus every form-trend metric on one team. The season contexts are the
point: the page's league-context, four-factor and recent-form blocks all query
per-context, and a default-only render exercises none of them.

Usage:
    python scripts/sweep_team_profiles.py smoke   # one render
    python scripts/sweep_team_profiles.py         # full sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from shared.db import query_df

PAGE = str(ROOT / "pages" / "09_Team_Profiles.py")

teams = query_df(
    "SELECT DISTINCT team_name FROM marts.team_season_stats ORDER BY team_name"
)["team_name"].tolist()
seasons = sorted(
    int(s) for s in query_df(
        "SELECT DISTINCT season FROM marts.team_season_stats"
    )["season"].dropna()
)
CONTEXTS = ["Career"] + [str(s) for s in seasons]
WINDOWS = ["Last 5", "Last 10"]
TREND = ["scores", "scores_against", "shots", "shots_on_goal", "saves",
         "turnovers", "caused_turnovers", "touches", "offensive_sequence_proxy"]


def problems(at) -> list[str]:
    out = []
    for e in at.exception:
        val = getattr(e, "value", "") or getattr(e, "body", "") or str(e)
        out.append(str(val).strip().splitlines()[0] if val else "unknown")
    out += ["st.error: " + str(getattr(e, "value", e)).strip().splitlines()[0]
            for e in at.error]
    return out


def render(team: str, context: str | None = None, window: str | None = None,
           trend: str | None = None):
    at = AppTest.from_file(PAGE, default_timeout=240)
    at.session_state["team_explorer_select"] = team
    at.run()
    tid = next((r.key.split("team_context_")[1] for r in at.radio
                if (r.key or "").startswith("team_context_")), None)
    if tid is None:
        raise RuntimeError(f"{team}: no context radio rendered")
    if context is not None:
        at.session_state[f"team_context_{tid}"] = context
    if window is not None:
        at.session_state[f"team_recent_split_{tid}"] = window
    if trend is not None:
        at.session_state[f"team_form_trend_{tid}"] = trend
    at.run()
    return at


def texts(at) -> str:
    return "\n".join(str(getattr(m, "value", "")) for m in at.markdown) + "\n" + \
           "\n".join(str(getattr(c, "value", "")) for c in at.caption)


def main() -> int:
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    runs = bad = 0

    if smoke:
        at = render(teams[0], "2022", "Last 5", "scores")
        errs = problems(at)
        print(f"[{'FAIL' if errs else 'PASS'}] {teams[0]} / 2022 / Last 5: "
              f"{' | '.join(errs) if errs else f'{len(at.dataframe)} tables, {len(at.markdown)} md'}")
        return 1 if errs else 0

    for team in teams:
        for context in CONTEXTS:
            for window in WINDOWS:
                at = render(team, context, window, "scores")
                runs += 1
                errs = problems(at)
                if errs:
                    bad += 1
                    print(f"[FAIL] {team} / {context} / {window}: {' | '.join(errs)}")

    # Every trend metric, on one team in a mid-history season.
    for trend in TREND:
        at = render(teams[0], "2023", "Last 10", trend)
        runs += 1
        errs = problems(at)
        if errs:
            bad += 1
            print(f"[FAIL] {teams[0]} / 2023 / Last 10 / {trend}: {' | '.join(errs)}")

    print(f"\n{runs - bad}/{runs} page-09 renders clean")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
