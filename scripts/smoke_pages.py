"""
Headless smoke test for the Streamlit pages.

Runs each page top to bottom with Streamlit's AppTest harness and reports any
exception. Catches the errors that matter most here — a BinderException from a
column that doesn't exist in a given mart, or a KeyError from a renamed display
column — without needing a browser.

Usage:
    python scripts/smoke_pages.py            # every page plus app.py
    python scripts/smoke_pages.py 05 08      # only pages whose name contains these
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

# Only the analytics pages. The untracked projection pages (1-4) belong to the
# other app and are not part of PLL Analytics.
ANALYTICS_PREFIXES = tuple(f"{n:02d}" for n in range(5, 19))

# Pages that legitimately take a while (large marts, many charts).
TIMEOUT = 120


def targets(filters: list[str]) -> list[Path]:
    pages = sorted(
        p for p in (ROOT / "pages").glob("*.py")
        if p.name.startswith(ANALYTICS_PREFIXES)
    )
    files = [ROOT / "app.py"] + pages
    if filters:
        files = [f for f in files if any(s in f.name for s in filters)]
    return files


def run(path: Path) -> tuple[bool, str]:
    try:
        at = AppTest.from_file(str(path), default_timeout=TIMEOUT).run()
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
    filters = sys.argv[1:]
    files = targets(filters)
    if not files:
        print("No matching pages.")
        return 1

    failures = 0
    for path in files:
        ok, detail = run(path)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {path.name:34s} {detail}")
        if not ok:
            failures += 1

    print(f"\n{len(files) - failures}/{len(files)} pages passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
