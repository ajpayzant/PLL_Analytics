"""
Headless smoke test for the Streamlit pages.

Runs each page top to bottom with Streamlit's AppTest harness and reports any
exception. Catches the errors that matter most here — a BinderException from a
column that doesn't exist in a given mart, or a KeyError from a renamed display
column — without needing a browser.

Usage:
    python scripts/smoke_pages.py               # every page plus app.py
    python scripts/smoke_pages.py 05 08         # only pages whose name contains these
    python scripts/smoke_pages.py scope=all     # under one segment scope
    python scripts/smoke_pages.py scope=every   # under each scope in turn

The scope runs are what catch what the segment work could break. On a warehouse
with no playoff tables yet, `scope=playoffs` resolves every scoped table to an
empty result — which is exactly the state each page has to render without an
exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from shared import segments

# Only the analytics pages. The untracked projection pages (1-4) belong to the
# other app and are not part of PLL Analytics.
ANALYTICS_PREFIXES = tuple(f"{n:02d}" for n in range(5, 19))

# Pages that legitimately take a while (large marts, many charts).
TIMEOUT = 120

# The scope control hides itself when the warehouse has no playoff tables, and it
# resets a stale selection to regular season — correct in the app, useless in a
# test, because the non-regular runs would silently become regular ones. Forcing
# the answer makes the pages resolve tables that do not exist yet, which is the
# fallback path worth proving: an empty result rendered as an empty state.
_REAL_HAS_SEGMENTS = segments.has_segment_tables


def force_scopes(force: bool) -> None:
    segments.has_segment_tables = (lambda: True) if force else _REAL_HAS_SEGMENTS


def targets(filters: list[str]) -> list[Path]:
    pages = sorted(
        p for p in (ROOT / "pages").glob("*.py")
        if p.name.startswith(ANALYTICS_PREFIXES)
    )
    files = [ROOT / "app.py"] + pages
    if filters:
        files = [f for f in files if any(s in f.name for s in filters)]
    return files


def run(path: Path, scope: str = segments.REGULAR) -> tuple[bool, str]:
    try:
        at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
        # Seeded before the run so the sidebar radio picks it up as its value, the
        # same way it would after a click.
        at.session_state[segments.STATE_KEY] = scope
        at = at.run()
    except Exception as exc:  # harness-level failure (import error, syntax error)
        return False, f"{type(exc).__name__}: {exc}"

    if at.exception:
        messages = []
        for e in at.exception:
            # AppTest exposes the rendered exception element, not the traceback.
            value = getattr(e, "value", "") or getattr(e, "body", "") or str(e)
            messages.append(str(value).strip().splitlines()[0] if value else "unknown")
        return False, " | ".join(messages)

    # st.error() calls are not exceptions but usually indicate a failed guard.
    errors = [str(getattr(e, "value", e)).strip().splitlines()[0] for e in at.error]
    if errors:
        return False, "st.error: " + " | ".join(errors)

    warnings = len(at.warning)
    detail = f"{len(at.dataframe)} tables, {len(at.markdown)} md blocks"
    if warnings:
        detail += f", {warnings} warning(s)"
    return True, detail


def main() -> int:
    args = sys.argv[1:]
    requested = next((a.split("=", 1)[1] for a in args if a.startswith("scope=")),
                     segments.REGULAR)
    filters = [a for a in args if not a.startswith("scope=")]

    scopes = list(segments.SCOPES) if requested == "every" else [requested]
    unknown = [s for s in scopes if s not in segments.SCOPES]
    if unknown:
        print(f"Unknown scope {unknown[0]!r}. Use one of: "
              f"{', '.join(segments.SCOPES)}, every.")
        return 1

    files = targets(filters)
    if not files:
        print("No matching pages.")
        return 1

    failures = 0
    for scope in scopes:
        force_scopes(scope != segments.REGULAR)
        if len(scopes) > 1:
            print(f"\n--- {segments.SCOPE_LABEL[scope]} ---")
        for path in files:
            ok, detail = run(path, scope)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {path.name:34s} {detail}")
            if not ok:
                failures += 1

    runs = len(files) * len(scopes)
    print(f"\n{runs - failures}/{runs} page runs passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
