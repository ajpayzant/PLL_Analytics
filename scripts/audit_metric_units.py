"""
Cross-check every numeric warehouse column against the metric registry's unit.

The formatting layer trusts `shared/metrics.py` completely: if a column is
declared pct01 but is stored 0–100, the app renders "8500.0%" and nobody notices
until a reader does. This walks every table in clean/marts/qc, compares each
numeric column's observed range against its declared unit, and flags the
mismatches that would produce a wrong number on screen.

Two known-good exceptions are whitelisted below: columns whose scale genuinely
differs between the `clean` and `marts` schemas, which the registry handles via
`clean_schema=True` rather than by declaring one scale for both.

Usage:
    python scripts/audit_metric_units.py            # summary + suspects
    python scripts/audit_metric_units.py --verbose  # every column checked
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from shared import metrics as M  # noqa: E402
from shared.db import DB_PATH  # noqa: E402

SCHEMAS = ("clean", "marts", "qc")

# (schema, table, column) triples where a range mismatch is expected, with why.
EXPECTED = {
    ("clean", "player_game_stats", "clean_save_pct"):
        "clean.* stores 0-1, marts.* stores 0-100; handled by clean_schema=True",
    ("clean", "team_game_stats", "clean_save_pct"):
        "clean.* stores 0-1, marts.* stores 0-100; handled by clean_schema=True",
}

# A percentage stored 0-1 can legitimately reach 1.0 exactly (a perfect rate),
# so allow a little headroom before calling it a scale error.
PCT01_CEILING = 1.5


def suspect(key: str, unit_code: str, lo: float, hi: float,
            has_fraction: bool) -> str | None:
    """Reason this column's values contradict its declared unit, or None."""
    if unit_code == M.UNIT_PCT01 and hi > PCT01_CEILING:
        return f"declared pct01 (0-1) but max is {hi:,.2f}"
    if unit_code in (M.UNIT_PCT100, M.UNIT_SCORE) and pd.notna(hi) and hi <= PCT01_CEILING:
        return f"declared {unit_code} (0-100) but max is only {hi:,.4f}"
    if unit_code == M.UNIT_INT and has_fraction:
        return "declared int but holds fractional values"
    if unit_code in (M.UNIT_SEC, M.UNIT_SEC_TOTAL) and pd.notna(hi) and hi < 60:
        return f"declared seconds but max is {hi:,.2f} — may already be minutes"
    return None


def main() -> int:
    verbose = "--verbose" in sys.argv[1:]

    if not Path(DB_PATH).exists():
        print(f"ERROR: warehouse not found at {DB_PATH}")
        return 1

    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('clean', 'marts', 'qc')
        ORDER BY table_schema, table_name
        """
    ).fetchall()

    checked = 0
    suspects: list[tuple[str, str, str, str, str]] = []
    unregistered = 0

    for schema, table in tables:
        cols = con.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema, table],
        ).fetchall()

        numeric = [
            c for c, dtype in cols
            if any(t in dtype.upper() for t in
                   ("INT", "DOUBLE", "DECIMAL", "FLOAT", "REAL", "NUMERIC", "HUGEINT"))
        ]
        if not numeric:
            continue

        quoted = ", ".join(f'"{c}"' for c in numeric)
        df = con.execute(f'SELECT {quoted} FROM "{schema}"."{table}"').fetchdf()

        for col in numeric:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            checked += 1

            described = M.describe(col)
            if described.family == "meta":
                unregistered += 1
            unit_code = M.unit(col)
            lo, hi = float(values.min()), float(values.max())
            has_fraction = bool(((values - values.round()).abs() > 1e-9).any())

            reason = suspect(col, unit_code, lo, hi, has_fraction)
            if reason and (schema, table, col) not in EXPECTED:
                suspects.append((f"{schema}.{table}", col, unit_code,
                                 f"[{lo:,.4g} .. {hi:,.4g}]", reason))
            if verbose:
                mark = "!" if reason else " "
                print(f"{mark} {schema}.{table:38s} {col:44s} "
                      f"{unit_code:9s} [{lo:,.4g} .. {hi:,.4g}]")

    print(f"\n{checked:,} numeric columns checked across {len(tables)} tables "
          f"({unregistered:,} not explicitly registered, formatted by inference).")

    if not suspects:
        print(f"{len(EXPECTED)} known clean-vs-marts scale differences whitelisted.")
        print("No unit mismatches.")
        return 0

    print(f"\n{len(suspects)} suspect column(s):\n")
    for table, col, unit_code, rng, reason in suspects:
        print(f"  {table}.{col}")
        print(f"      declared {unit_code}, observed {rng}")
        print(f"      {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
