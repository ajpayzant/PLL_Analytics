"""
Deep sweep of Player Profiles (page 08).

`smoke_pages.py` renders each page once on its defaults, which never reaches a
role-specific panel, a non-current season, or the peer-percentile trend. This
crosses a position-representative set of players — the busiest and thinnest-sample
player at each position, so both the peer-pool floor and the form-delta guards get
hit — with every context they played in, both form windows and both trend scales.

Usage:
    python scripts/sweep_player_profiles.py smoke   # one render
    python scripts/sweep_player_profiles.py         # full sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from shared.db import query_df

PAGE = str(ROOT / "pages" / "08_Player_Profiles.py")

# Per position: the busiest player, plus the thinnest-sample one, since the peer
# filter and the form-delta guards both key off games played.
subjects = query_df("""
    WITH ranked AS (
        SELECT full_name, position, games,
               ROW_NUMBER() OVER (PARTITION BY position ORDER BY games DESC) AS hi,
               ROW_NUMBER() OVER (PARTITION BY position ORDER BY games ASC) AS lo
        FROM marts.player_career_stats
        WHERE position IS NOT NULL
    )
    SELECT full_name, position, games FROM ranked WHERE hi <= 2 OR lo <= 2
    ORDER BY position, games DESC
""")

WINDOWS = ["Last 5", "Last 10"]


def problems(at) -> list[str]:
    out = []
    for e in at.exception:
        val = getattr(e, "value", "") or getattr(e, "body", "") or str(e)
        out.append(str(val).strip().splitlines()[0] if val else "unknown")
    out += ["st.error: " + str(getattr(e, "value", e)).strip().splitlines()[0]
            for e in at.error]
    return out


def render(player: str, context: str | None = None, window: str | None = None,
           scale: str | None = None):
    at = AppTest.from_file(PAGE, default_timeout=240)
    at.session_state["player_explorer_select"] = player
    at.run()
    pid = next((r.key.split("player_context_")[1] for r in at.radio
                if (r.key or "").startswith("player_context_")), None)
    if pid is None:
        raise RuntimeError(f"{player}: no context radio rendered")
    contexts = next(r.options for r in at.radio
                    if (r.key or "") == f"player_context_{pid}")
    if context is not None:
        at.session_state[f"player_context_{pid}"] = context
    if window is not None:
        at.session_state[f"player_recent_split_{pid}"] = window
    if scale is not None:
        at.session_state[f"player_trend_scale_{pid}"] = scale
    at.run()
    return at, pid, contexts


def main() -> int:
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    runs = bad = 0

    if smoke:
        name = subjects["full_name"].iloc[0]
        at, _, contexts = render(name, None, "Last 5")
        errs = problems(at)
        print(f"[{'FAIL' if errs else 'PASS'}] {name}: "
              f"{' | '.join(errs) if errs else f'{len(at.dataframe)} tables'}")
        return 1 if errs else 0

    for _, sub in subjects.iterrows():
        name = sub["full_name"]
        try:
            _, _, contexts = render(name)
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            bad += 1
            runs += 1
            continue
        for context in contexts:
            for window in WINDOWS:
                for scale in ("Raw rate", "Peer percentile"):
                    at, _, _ = render(name, context, window, scale)
                    runs += 1
                    errs = problems(at)
                    if errs:
                        bad += 1
                        print(f"[FAIL] {name} ({sub['position']}) / {context} / "
                              f"{window} / {scale}: {' | '.join(errs)}")

    print(f"\n{runs - bad}/{runs} page-08 renders clean "
          f"({len(subjects)} players)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
