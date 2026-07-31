# ============================================================
# PLL DATA PLATFORM — GITHUB WAREHOUSE BUILDER
# Source of truth: final Colab notebook builder blocks, ported for GitHub.
# ============================================================

from __future__ import annotations

import os
import re
import json
import gzip
import time
import hashlib
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import duckdb

from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 250)


def display(obj=None, *args, **kwargs):
    """Notebook-compatible display shim for GitHub Actions logs."""
    if obj is None:
        print()
        return

    try:
        if isinstance(obj, pd.DataFrame):
            if len(obj) > 40:
                print(obj.head(40).to_string(index=False))
                print(f"... ({len(obj)} rows total)")
            else:
                print(obj.to_string(index=False))
        else:
            print(obj)
    except Exception:
        print(repr(obj))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out or default


REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# BLOCK 1 — CONFIG, PATHS, TOKEN, SESSION, HELPERS
# ============================================================

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = REPO_ROOT / "data"

SOURCE_DATA_DIR = PROJECT_ROOT / "source_data"
API_RESPONSES_DIR = SOURCE_DATA_DIR / "api_responses"

STANDARDIZED_DATA_DIR = PROJECT_ROOT / "standardized_data"
GAME_TABLES_DIR = STANDARDIZED_DATA_DIR / "game_tables"
REFERENCE_TABLES_DIR = STANDARDIZED_DATA_DIR / "reference_tables"

CURATED_DATA_DIR = PROJECT_ROOT / "curated_data"
CURATED_ALL_DIR = CURATED_DATA_DIR / "all_requested_seasons"

ANALYTICS_DATABASE_DIR = PROJECT_ROOT / "analytics_database"
QUALITY_CHECKS_DIR = PROJECT_ROOT / "quality_checks"
CONFIG_DIR = PROJECT_ROOT / "config"
EXPORT_DIR = PROJECT_ROOT / "exports"

RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")
RUN_CHECK_DIR = QUALITY_CHECKS_DIR / RUN_ID

for p in [
    PROJECT_ROOT,
    SOURCE_DATA_DIR,
    API_RESPONSES_DIR,
    STANDARDIZED_DATA_DIR,
    GAME_TABLES_DIR,
    REFERENCE_TABLES_DIR,
    CURATED_DATA_DIR,
    CURATED_ALL_DIR,
    ANALYTICS_DATABASE_DIR,
    QUALITY_CHECKS_DIR,
    RUN_CHECK_DIR,
    CONFIG_DIR,
    EXPORT_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Run check dir:", RUN_CHECK_DIR)

# -----------------------------
# Main config
# -----------------------------
TARGET_SEASONS = env_int_list("PLL_TARGET_SEASONS", [2022, 2023, 2024, 2025, 2026])
COMPETITION_TYPE = os.getenv("PLL_COMPETITION_TYPE", "regular").strip().lower()

EXPECTED_REGULAR_GAMES = {
    2022: 40,
    2023: 40,
    2024: 40,
    2025: 40,
    2026: None,   # ongoing / schedule-aware
}

PLL_STATS_SITE = "https://stats.premierlacrosseleague.com"
PLL_API_BASE = "https://api.stats.premierlacrosseleague.com/api/v4"
TIME_ZONE = "America/Los_Angeles"

FORCE_RECOLLECT = env_bool("PLL_FORCE_RECOLLECT", False)
FORCE_REDISCOVER = env_bool("PLL_FORCE_REDISCOVER", False)

MANUAL_SLUG_INVENTORY_FILE = CONFIG_DIR / "manual_slug_inventory.csv"

# -----------------------------
# Team mappings
# -----------------------------
TEAM_ID_CANONICAL_MAP = {
    "ATL": "ATL",
    "OUT": "OUT",
    "CAN": "CAN",
    "RED": "RED",
    "WAT": "WAT",
    "WHP": "WHP",
    "CHA": "CHA",
    "ARC": "ARC",
    "CHR": "OUT",   # Chrome historical franchise rolls into Outlaws.
}

TEAM_NAME_CANONICAL_MAP = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
    "CHR": "Outlaws",
}

TEAM_NAME_LOOKUP_RAW = {
    "ATL": "Atlas",
    "OUT": "Outlaws",
    "CAN": "Cannons",
    "RED": "Redwoods",
    "WAT": "Waterdogs",
    "WHP": "Whipsnakes",
    "CHA": "Chaos",
    "ARC": "Archers",
    "CHR": "Chrome",
}

TEAM_DISPLAY_NAME_LOOKUP = {
    "ATL": "New York Atlas",
    "OUT": "Denver Outlaws",
    "CAN": "Boston Cannons",
    "RED": "California Redwoods",
    "WAT": "Philadelphia Waterdogs",
    "WHP": "Maryland Whipsnakes",
    "CHA": "Carolina Chaos",
    "ARC": "Utah Archers",
}

def canonical_team_id(team_id):
    if pd.isna(team_id):
        return pd.NA
    return TEAM_ID_CANONICAL_MAP.get(str(team_id).strip(), str(team_id).strip())

def canonical_team_name(team_id_raw, fallback_name=None):
    if pd.isna(team_id_raw):
        return fallback_name if fallback_name is not None else pd.NA
    team_id_raw = str(team_id_raw).strip()
    return TEAM_NAME_CANONICAL_MAP.get(
        team_id_raw,
        fallback_name if fallback_name is not None else team_id_raw
    )

def resolve_team_name_raw(team_id_raw, candidate_name=None):
    if pd.isna(team_id_raw) and pd.isna(candidate_name):
        return pd.NA

    raw_id = None if pd.isna(team_id_raw) else str(team_id_raw).strip()
    raw_name = None if pd.isna(candidate_name) else str(candidate_name).strip()

    if raw_name and raw_id and raw_name != raw_id:
        return raw_name

    if raw_name and not raw_id:
        return raw_name

    if raw_id:
        return TEAM_NAME_LOOKUP_RAW.get(raw_id, raw_id)

    return pd.NA

# -----------------------------
# Token
# -----------------------------
def clean_token_value(x):
    if x is None:
        return ""
    x = str(x).strip()
    x = x.replace("^", "").strip()
    x = re.sub(r"\s+", " ", x).strip()
    return x

PLL_BEARER_TOKEN = clean_token_value(
    os.environ.get("PLL_BEARER_TOKEN", "")
    or os.environ.get("PLL_API_TOKEN", "")
    or os.environ.get("PLL_TOKEN", "")
)

if not PLL_BEARER_TOKEN:
    raise RuntimeError(
        "PLL_BEARER_TOKEN is missing. Add it as a GitHub Actions repository secret."
    )

def token_preview(tok):
    return "SET" if tok else "MISSING"

print("Token loaded:", bool(PLL_BEARER_TOKEN))
print("Token preview:", token_preview(PLL_BEARER_TOKEN))

# -----------------------------
# HTTP session
# -----------------------------
def build_session(bearer_token=""):
    s = requests.Session()

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "origin": "https://stats.premierlacrosseleague.com",
        "pragma": "no-cache",
        "referer": "https://stats.premierlacrosseleague.com/",
        "time-zone": TIME_ZONE,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
    }

    if bearer_token:
        tok = clean_token_value(bearer_token)
        headers["authorization"] = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
        headers["authsource"] = "stats"

    s.headers.update(headers)
    return s

SESSION = build_session(PLL_BEARER_TOKEN)

print("Authorization header present:", "authorization" in SESSION.headers)

# -----------------------------
# URL builders
# -----------------------------
def event_list_url(year, season_segment=COMPETITION_TYPE):
    return f"{PLL_API_BASE}/events?year={year}&seasonSegment={season_segment}"

def event_summary_url(slug):
    return f"{PLL_API_BASE}/events/{slug}"

def player_game_stats_url(slug):
    return f"{PLL_API_BASE}/events/{slug}/players/stats"

def team_game_stats_url(slug):
    return f"{PLL_API_BASE}/events/{slug}/teams/stats"

# -----------------------------
# General helpers
# -----------------------------
def now_utc_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def write_gzip_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def fetch_url(url, session=None, timeout=30):
    if session is None:
        session = SESSION
    return session.get(url, timeout=timeout)

def fetch_json_with_cache(url, cache_path, session=None, timeout=30, force=False):
    if session is None:
        session = SESSION

    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        try:
            payload = read_gzip_json(cache_path)
            return payload, 200, "cached"
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass

    r = fetch_url(url, session=session, timeout=timeout)

    try:
        payload = r.json()
    except Exception:
        payload = None

    if r.status_code == 200 and payload is not None:
        write_gzip_json(cache_path, payload)

    return payload, r.status_code, "downloaded"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def snake_case(s):
    s = str(s)
    s = re.sub(r"[%/\-]+", "_", s)
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s

def to_num_scalar(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def safe_nullable_int(series):
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()
    if non_null.empty:
        return s.astype("Int64")
    if np.isclose(non_null % 1, 0).all():
        return s.round().astype("Int64")
    return s

def normalize_person_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None

def extract_game_number_from_slug(slug):
    if pd.isna(slug):
        return pd.NA

    slug = str(slug)

    m1 = re.search(r"_game_(\d+)$", slug)
    if m1:
        return int(m1.group(1))

    m2 = re.search(r"^game-(\d+)-\d{4}-\d{2}-\d{2}$", slug)
    if m2:
        return int(m2.group(1))

    m3 = re.search(r"^(\d{4})-ev-(\d+)$", slug)
    if m3:
        return int(m3.group(2))

    return pd.NA

def extract_home_team_obj(data):
    return data.get("homeTeam", {}) or {}

def extract_away_team_obj(data):
    for key in ["visitorTeam", "awayTeam", "visitor", "away"]:
        obj = data.get(key, {}) or {}
        if obj:
            return obj
    return {}

def extract_team_id_from_obj(obj):
    if not isinstance(obj, dict):
        return pd.NA
    return obj.get("officialId") or obj.get("teamId") or obj.get("id")

def extract_team_name_from_obj(obj):
    if not isinstance(obj, dict):
        return pd.NA
    return (
        obj.get("name")
        or obj.get("fullName")
        or obj.get("teamName")
        or obj.get("nickname")
        or obj.get("officialId")
        or obj.get("teamId")
        or obj.get("id")
    )

def validate_event_payload(payload, season):
    data = safe_get(payload, "data", default={}) if payload else {}

    year_val = to_num_scalar(data.get("year"))
    event_id = data.get("eventId")
    event_numeric_id = data.get("id")
    season_segment = data.get("seasonSegment")
    slugname = data.get("slugname")
    start_time_unix = to_num_scalar(data.get("startTime"))
    event_status = data.get("eventStatus")

    valid = bool(
        not pd.isna(year_val)
        and int(year_val) == int(season)
        and event_id
        and season_segment == COMPETITION_TYPE
    )

    return {
        "valid": valid,
        "year": None if pd.isna(year_val) else int(year_val),
        "event_id": event_id,
        "event_numeric_id": event_numeric_id,
        "competition_type": season_segment,
        "slugname": slugname,
        "start_time_unix": None if pd.isna(start_time_unix) else int(start_time_unix),
        "event_status": event_status,
    }

def recursive_leaf_pairs(obj, prefix=""):
    pairs = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            pairs.extend(recursive_leaf_pairs(v, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            pairs.extend(recursive_leaf_pairs(v, p))

    else:
        pairs.append((prefix, obj))

    return pairs

def find_numeric_leaf_candidates(obj, normalized_terms):
    pairs = recursive_leaf_pairs(obj)
    out = []

    for raw_path, val in pairs:
        path_norm = snake_case(raw_path)
        if all(term in path_norm for term in normalized_terms):
            num = to_num_scalar(val)
            if not pd.isna(num):
                out.append((raw_path, num))

    return out

def coalesce_numeric_with_alt(item, direct_keys, alt_term_groups, allow_zero=True):
    for k in direct_keys:
        if k in item:
            val = to_num_scalar(item.get(k))
            if not pd.isna(val):
                if allow_zero or val != 0:
                    return val

    for term_group in alt_term_groups:
        cands = find_numeric_leaf_candidates(item, term_group)
        if cands:
            cands_sorted = sorted(cands, key=lambda x: (x[1] == 0, len(x[0])))
            best_val = cands_sorted[0][1]
            if allow_zero or best_val != 0:
                return best_val

    return np.nan

def derive_one_point_goals(total_goals, raw_one_point_goals, two_point_goals):
    tg = to_num_scalar(total_goals)
    rg = to_num_scalar(raw_one_point_goals)
    tw = to_num_scalar(two_point_goals)

    if not pd.isna(tg) and not pd.isna(tw):
        calc = tg - tw
        if pd.isna(rg) or not np.isclose(rg, calc):
            return calc

    return rg

def derive_scoring_points(one_point_goals, two_point_goals):
    one = to_num_scalar(one_point_goals)
    two = to_num_scalar(two_point_goals)

    if pd.isna(one) and pd.isna(two):
        return np.nan

    return (0 if pd.isna(one) else one) + 2 * (0 if pd.isna(two) else two)

def derive_player_points(raw_points, scoring_points, assists):
    rp = to_num_scalar(raw_points)
    sp = to_num_scalar(scoring_points)
    ast = to_num_scalar(assists)

    if not pd.isna(sp) and not pd.isna(ast):
        calc = sp + ast
        if pd.isna(rp) or not np.isclose(rp, calc):
            return calc

    return rp

def mode_or_first(s):
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA
    mode = s2.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return s2.iloc[0]

def latest_non_null_by_game(g, col):
    if col not in g.columns:
        return pd.NA
    s = g.sort_values(["season", "game_number", "game_id"])[col].dropna()
    if len(s) == 0:
        return pd.NA
    return s.iloc[-1]

print("Config/helper block complete.")

# ============================================================
# BLOCK 2 — API SANITY CHECK
# ============================================================

sanity_rows = []

test_urls = [
    ("event_list_2026", event_list_url(2026)),
    ("event_summary_2025_game_1", event_summary_url("2025_game_1")),
    ("event_summary_2026_ev_1", event_summary_url("2026-ev-1")),
]

for label, url in test_urls:
    try:
        r = SESSION.get(url, timeout=30)

        try:
            payload = r.json()
        except Exception:
            payload = None

        items = safe_get(payload, "data", "items", default=None) if payload else None
        data = safe_get(payload, "data", default=None) if payload else None

        sanity_rows.append({
            "label": label,
            "url": url,
            "status_code": r.status_code,
            "has_json": payload is not None,
            "has_data": data is not None,
            "has_items": isinstance(items, list),
            "items_count": len(items) if isinstance(items, list) else None,
            "text_preview": r.text[:300],
        })

    except Exception as e:
        sanity_rows.append({
            "label": label,
            "url": url,
            "status_code": None,
            "has_json": False,
            "has_data": False,
            "has_items": False,
            "items_count": None,
            "text_preview": str(e)[:300],
        })

api_sanity_check = pd.DataFrame(sanity_rows)
api_sanity_check.to_csv(RUN_CHECK_DIR / "api_sanity_check.csv", index=False)

display(api_sanity_check)

if not api_sanity_check["has_data"].any():
    raise RuntimeError("API sanity check failed. Check token/API access before continuing.")

print("API sanity check passed.")

# ============================================================
# BLOCK 3 — DISCOVERY: FULL SCHEDULE + COMPLETED STAT INVENTORY
# ============================================================

def ensure_manual_slug_template():
    if not MANUAL_SLUG_INVENTORY_FILE.exists():
        pd.DataFrame(columns=["season", "slug", "note"]).to_csv(MANUAL_SLUG_INVENTORY_FILE, index=False)
        print(f"Created optional manual slug template: {MANUAL_SLUG_INVENTORY_FILE}")

ensure_manual_slug_template()

def load_manual_slug_inventory():
    if not MANUAL_SLUG_INVENTORY_FILE.exists():
        return pd.DataFrame(columns=["season", "slug", "note"])

    df = pd.read_csv(MANUAL_SLUG_INVENTORY_FILE)

    for c in ["season", "slug", "note"]:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.dropna(subset=["season", "slug"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["slug"] = df["slug"].astype(str).str.strip()
    df = df[df["season"].isin(TARGET_SEASONS)].copy()

    return df[["season", "slug", "note"]]

def fetch_event_list_for_year(year, season_segment=COMPETITION_TYPE):
    candidate_urls = [
        f"{PLL_API_BASE}/events?year={year}&seasonSegment={season_segment}",
        f"{PLL_API_BASE}/events?year={year}",
    ]

    probe_rows = []
    best_payload = None

    for url in candidate_urls:
        try:
            r = SESSION.get(url, timeout=30)

            try:
                payload = r.json()
            except Exception:
                payload = None

            items = safe_get(payload, "data", "items", default=[]) if payload else []

            probe_rows.append({
                "season": year,
                "url": url,
                "status_code": r.status_code,
                "ok": r.ok,
                "has_json": payload is not None,
                "items_count": len(items) if isinstance(items, list) else 0,
                "text_preview": r.text[:500],
            })

            if r.status_code == 200 and isinstance(items, list) and len(items) > 0 and best_payload is None:
                best_payload = payload

        except Exception as e:
            probe_rows.append({
                "season": year,
                "url": url,
                "status_code": None,
                "ok": False,
                "has_json": False,
                "items_count": 0,
                "text_preview": str(e)[:500],
            })

    return best_payload, pd.DataFrame(probe_rows)

def parse_event_list_payload(payload, year, season_segment=COMPETITION_TYPE):
    items = safe_get(payload, "data", "items", default=[]) if payload else []
    rows = []

    for item in items:
        if not isinstance(item, dict):
            continue

        item_year = item.get("year")
        item_segment = item.get("seasonSegment")

        if item_year is not None:
            try:
                if int(item_year) != int(year):
                    continue
            except Exception:
                pass

        if item_segment is not None and item_segment != season_segment:
            continue

        slug = item.get("slugname") or item.get("slug") or item.get("eventSlug")
        if not slug:
            continue

        start_time_unix = to_num_scalar(item.get("startTime"))
        game_date_guess = (
            pd.to_datetime(start_time_unix, unit="s", utc=True)
            if not pd.isna(start_time_unix)
            else pd.NaT
        )

        home_obj = extract_home_team_obj(item)
        away_obj = extract_away_team_obj(item)

        home_team_id_raw = extract_team_id_from_obj(home_obj)
        away_team_id_raw = extract_team_id_from_obj(away_obj)

        home_team_name_raw = extract_team_name_from_obj(home_obj)
        away_team_name_raw = extract_team_name_from_obj(away_obj)

        event_status = item.get("eventStatus")
        event_status_num = to_num_scalar(event_status)

        rows.append({
            "season": int(year),
            "slug": str(slug),
            "event_id": item.get("eventId"),
            "event_numeric_id": item.get("id"),
            "year": item_year,
            "competition_type": item_segment,
            "slugname": slug,
            "start_time_unix": start_time_unix,
            "game_date_guess": game_date_guess,
            "away_team_id_raw": away_team_id_raw,
            "away_team_name_raw": away_team_name_raw,
            "home_team_id_raw": home_team_id_raw,
            "home_team_name_raw": home_team_name_raw,
            "away_score": item.get("visitorScore") if item.get("visitorScore") is not None else item.get("awayScore"),
            "home_score": item.get("homeScore"),
            "event_status": event_status,
            "event_status_num": event_status_num,
            "event_status_label": "final" if event_status_num == 3 else ("scheduled" if event_status_num == 0 else "unknown"),
            "source": "event_list_endpoint",
            "discovery_source": "event_list_endpoint",
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    out = out.drop_duplicates(subset=["season", "slug"]).copy()
    out = out.sort_values(["game_date_guess", "event_numeric_id", "slug"], na_position="last").reset_index(drop=True)
    out["game_number"] = np.arange(1, len(out) + 1)
    out["game_number_guess"] = out["game_number"]
    out["valid"] = True
    out["status_code"] = 200
    out["error"] = None

    return out

def probe_slug_summary(slug, season, discovery_source):
    cache_path = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}" / "event_summary.json.gz"

    payload, status_code, fetch_mode = fetch_json_with_cache(
        event_summary_url(slug),
        cache_path,
        force=FORCE_RECOLLECT
    )

    meta = validate_event_payload(payload, season)

    return {
        "season": season,
        "slug": slug,
        "status_code": status_code,
        "valid": bool(status_code == 200 and meta["valid"]),
        "year": meta["year"],
        "event_id": meta["event_id"],
        "event_numeric_id": meta["event_numeric_id"],
        "competition_type": meta["competition_type"],
        "slugname": meta["slugname"],
        "source": fetch_mode,
        "discovery_source": discovery_source,
        "start_time_unix": meta["start_time_unix"],
        "game_date_guess": pd.to_datetime(meta["start_time_unix"], unit="s", utc=True) if meta["start_time_unix"] else pd.NaT,
        "game_number_guess": extract_game_number_from_slug(slug),
        "event_status": meta["event_status"],
        "event_status_num": to_num_scalar(meta["event_status"]),
        "error": None,
    }

def validate_event_list_slugs(schedule_df, year):
    rows = []

    if len(schedule_df) == 0:
        return pd.DataFrame()

    for _, r in tqdm(schedule_df.iterrows(), total=len(schedule_df), desc=f"Validating event-list slugs {year}"):
        slug = str(r["slug"])

        try:
            row = probe_slug_summary(
                slug=slug,
                season=year,
                discovery_source="event_list_endpoint_validated_summary"
            )

            row["game_number"] = r.get("game_number")
            row["event_list_event_id"] = r.get("event_id")
            row["event_list_event_numeric_id"] = r.get("event_numeric_id")
            row["event_list_event_status"] = r.get("event_status")
            row["event_list_event_status_num"] = r.get("event_status_num")
            row["event_list_event_status_label"] = r.get("event_status_label")
            row["event_list_game_date_guess"] = r.get("game_date_guess")

            rows.append(row)

        except Exception as e:
            rows.append({
                "season": year,
                "slug": slug,
                "status_code": None,
                "valid": False,
                "year": None,
                "event_id": None,
                "event_numeric_id": None,
                "competition_type": None,
                "slugname": None,
                "source": "error",
                "discovery_source": "event_list_endpoint_validated_summary",
                "start_time_unix": None,
                "game_date_guess": pd.NaT,
                "game_number_guess": r.get("game_number"),
                "game_number": r.get("game_number"),
                "event_status": r.get("event_status"),
                "event_status_num": r.get("event_status_num"),
                "event_list_event_status": r.get("event_status"),
                "event_list_event_status_num": r.get("event_status_num"),
                "event_list_event_status_label": r.get("event_status_label"),
                "error": str(e)[:500],
            })

        time.sleep(0.03)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["game_number", "game_date_guess", "slug"], na_position="last").reset_index(drop=True)

    return out

def scan_cached_event_summaries(season):
    rows = []
    season_dir = API_RESPONSES_DIR / f"season_{season}"

    if not season_dir.exists():
        return pd.DataFrame(rows)

    for fp in sorted(season_dir.rglob("event_summary.json.gz")):
        slug = fp.parent.name.replace("game_", "", 1)

        try:
            payload = read_gzip_json(fp)
            meta = validate_event_payload(payload, season)

            rows.append({
                "season": season,
                "slug": slug,
                "status_code": 200,
                "valid": bool(meta["valid"]),
                "year": meta["year"],
                "event_id": meta["event_id"],
                "event_numeric_id": meta["event_numeric_id"],
                "competition_type": meta["competition_type"],
                "slugname": meta["slugname"],
                "source": "cached",
                "discovery_source": "cached_summary_scan",
                "start_time_unix": meta["start_time_unix"],
                "game_date_guess": pd.to_datetime(meta["start_time_unix"], unit="s", utc=True) if meta["start_time_unix"] else pd.NaT,
                "game_number_guess": extract_game_number_from_slug(slug),
                "game_number": pd.NA,
                "event_status": meta["event_status"],
                "event_status_num": to_num_scalar(meta["event_status"]),
                "error": None,
            })

        except Exception as e:
            rows.append({
                "season": season,
                "slug": slug,
                "status_code": None,
                "valid": False,
                "source": "cached",
                "discovery_source": "cached_summary_scan",
                "error": str(e)[:500],
            })

    return pd.DataFrame(rows)

def discover_numeric_season(season, max_guess=90, stop_after_consecutive_misses=12):
    rows = []
    valid_count = 0
    consecutive_misses = 0

    for game_number in range(1, max_guess + 1):
        slug = f"{season}_game_{game_number}"

        row = probe_slug_summary(slug, season, "numeric_probe")
        row["game_number"] = game_number
        row["game_number_guess"] = game_number
        rows.append(row)

        if row["valid"]:
            valid_count += 1
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        if valid_count > 0 and consecutive_misses >= stop_after_consecutive_misses:
            break

        time.sleep(0.03)

    return pd.DataFrame(rows)

def discover_dated_season(season, start_date, end_date, max_game_number=65, stop_after_consecutive_missing_numbers=10):
    rows = []
    valid_count = 0
    consecutive_missing_numbers = 0
    date_list = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()

    for game_number in range(1, max_game_number + 1):
        found_this_number = False

        for d in date_list:
            slug = f"game-{game_number}-{d}"

            row = probe_slug_summary(slug, season, f"dated_probe_{season}")
            row["game_number"] = game_number
            row["game_number_guess"] = game_number
            rows.append(row)

            if row["valid"]:
                valid_count += 1
                found_this_number = True
                break

            time.sleep(0.01)

        if found_this_number:
            consecutive_missing_numbers = 0
        else:
            consecutive_missing_numbers += 1

        if valid_count > 0 and consecutive_missing_numbers >= stop_after_consecutive_missing_numbers:
            break

    return pd.DataFrame(rows)

def build_discovery_inventories():
    event_list_probe_frames = []
    schedule_frames = []
    validated_frames = []

    # 1. Preferred discovery: official event-list endpoint.
    for season in TARGET_SEASONS:
        payload, probe_df = fetch_event_list_for_year(season, COMPETITION_TYPE)
        event_list_probe_frames.append(probe_df)

        parsed_schedule = parse_event_list_payload(payload, season, COMPETITION_TYPE)

        if len(parsed_schedule) > 0:
            schedule_frames.append(parsed_schedule)

            validated = validate_event_list_slugs(parsed_schedule, season)
            if len(validated) > 0:
                validated_frames.append(validated)

    event_list_probe_summary = (
        pd.concat(event_list_probe_frames, ignore_index=True)
        if event_list_probe_frames
        else pd.DataFrame()
    )

    event_list_schedule_inventory = (
        pd.concat(schedule_frames, ignore_index=True)
        if schedule_frames
        else pd.DataFrame()
    )

    validated_inventory = (
        pd.concat(validated_frames, ignore_index=True)
        if validated_frames
        else pd.DataFrame()
    )

    # 2. Fallback discovery where event-list endpoint fails or is incomplete.
    fallback_frames = []

    manual_df = load_manual_slug_inventory()
    if len(manual_df) > 0:
        manual_rows = []
        for _, r in manual_df.iterrows():
            season = int(r["season"])
            slug = str(r["slug"]).strip()
            row = probe_slug_summary(slug, season, "manual_slug_inventory")
            row["manual_note"] = r.get("note")
            manual_rows.append(row)
        fallback_frames.append(pd.DataFrame(manual_rows))

    for season in TARGET_SEASONS:
        expected = EXPECTED_REGULAR_GAMES.get(season)

        current_valid = validated_inventory[
            (pd.to_numeric(validated_inventory.get("season", pd.Series(dtype=float)), errors="coerce") == season)
            & (validated_inventory.get("valid", pd.Series(dtype=bool)) == True)
        ] if len(validated_inventory) > 0 else pd.DataFrame()

        need_fallback = len(current_valid) == 0 or (expected is not None and len(current_valid) < expected)

        if not need_fallback:
            continue

        cached_df = scan_cached_event_summaries(season)
        if len(cached_df) > 0:
            fallback_frames.append(cached_df)

        numeric_df = discover_numeric_season(season)
        if len(numeric_df) > 0:
            fallback_frames.append(numeric_df)

        dated_ranges = {
            2022: ("2022-06-01", "2022-09-30"),
            2023: ("2023-06-01", "2023-09-30"),
            2024: ("2024-05-01", "2024-09-30"),
            2025: ("2025-05-01", "2025-09-30"),
        }
        if season in dated_ranges:
            start, end = dated_ranges[season]
            dated_df = discover_dated_season(season, start, end)
            if len(dated_df) > 0:
                fallback_frames.append(dated_df)

    fallback_inventory = (
        pd.concat(fallback_frames, ignore_index=True)
        if fallback_frames
        else pd.DataFrame()
    )

    # 3. Combine validated event-list and fallback.
    discovery_log_parts = []
    if len(validated_inventory) > 0:
        discovery_log_parts.append(validated_inventory)
    if len(fallback_inventory) > 0:
        discovery_log_parts.append(fallback_inventory)

    game_discovery_log = (
        pd.concat(discovery_log_parts, ignore_index=True)
        if discovery_log_parts
        else pd.DataFrame()
    )

    if len(game_discovery_log) == 0:
        return (
            event_list_probe_summary,
            event_list_schedule_inventory,
            game_discovery_log,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    valid_discovered = game_discovery_log[game_discovery_log["valid"] == True].copy()

    # Prefer event-list validated rows over fallback rows.
    source_rank = {
        "event_list_endpoint_validated_summary": 1,
        "manual_slug_inventory": 2,
        "cached_summary_scan": 3,
        "numeric_probe": 4,
    }

    valid_discovered["discovery_rank"] = valid_discovered["discovery_source"].map(source_rank).fillna(9)

    valid_discovered = valid_discovered.sort_values(
        ["season", "event_id", "discovery_rank", "game_date_guess", "slug"],
        na_position="last"
    )

    valid_discovered = valid_discovered.drop_duplicates(
        subset=["season", "event_id"],
        keep="first"
    ).copy()

    valid_discovered = valid_discovered.sort_values(
        ["season", "game_date_guess", "game_number", "slug"],
        na_position="last"
    ).reset_index(drop=True)

    # Fill game_number by season if missing.
    valid_discovered["game_number"] = pd.to_numeric(valid_discovered["game_number"], errors="coerce")
    valid_discovered["game_number"] = valid_discovered.groupby("season").cumcount() + 1

    # 4. Build full schedule inventory.
    # If event-list schedule exists, use it for schedule. Otherwise use valid discovered rows.
    if len(event_list_schedule_inventory) > 0:
        schedule_inventory = event_list_schedule_inventory.copy()
    else:
        schedule_inventory = valid_discovered.copy()

    schedule_inventory = schedule_inventory.sort_values(
        ["season", "game_number", "game_date_guess", "slug"],
        na_position="last"
    ).reset_index(drop=True)

    # 5. Build stat-available inventory.
    #
    # Final rule:
    # - Historical completed seasons 2022-2025: use all validated regular-season games.
    #   Some historical PLL event-list rows can have imperfect event_status values even though stats exist.
    # - Ongoing/current/future seasons 2026+: use only event_status == 3 so scheduled games do not pollute stat tables.

    stat_parts = []

    for season in TARGET_SEASONS:
        discovered_season = valid_discovered[
            pd.to_numeric(valid_discovered["season"], errors="coerce") == season
        ].copy()

        schedule_season = schedule_inventory[
            pd.to_numeric(schedule_inventory["season"], errors="coerce") == season
        ].copy()

        if len(discovered_season) == 0:
            continue

        if season <= 2025:
            # Historical seasons: all validated regular-season games should be stat-available.
            stat_season = discovered_season.copy()

        else:
            # Ongoing/future seasons: only final games should be included in stat tables.
            if len(schedule_season) > 0 and "event_status_num" in schedule_season.columns:
                final_slugs = (
                    schedule_season[
                        pd.to_numeric(schedule_season["event_status_num"], errors="coerce") == 3
                    ]["slug"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                stat_season = discovered_season[
                    discovered_season["slug"].astype(str).isin(final_slugs)
                ].copy()

            else:
                stat_season = pd.DataFrame(columns=discovered_season.columns)

        if len(stat_season) > 0:
            stat_season = stat_season.sort_values(
                ["game_date_guess", "game_number", "slug"],
                na_position="last"
            ).copy()

            stat_season["game_number"] = np.arange(1, len(stat_season) + 1)

            stat_parts.append(stat_season)

    stat_inventory = (
        pd.concat(stat_parts, ignore_index=True)
        if stat_parts
        else pd.DataFrame()
    )

    stat_inventory = stat_inventory.sort_values(
        ["season", "game_number", "game_date_guess", "slug"],
        na_position="last"
    ).reset_index(drop=True)

    return (
        event_list_probe_summary,
        schedule_inventory,
        game_discovery_log,
        valid_discovered,
        stat_inventory,
    )

event_list_probe_summary, season_schedule_inventory, game_discovery_log, season_slug_inventory, stat_slug_inventory = build_discovery_inventories()

# Save discovery outputs.
event_list_probe_summary.to_csv(RUN_CHECK_DIR / "event_list_probe_summary.csv", index=False)
season_schedule_inventory.to_csv(RUN_CHECK_DIR / "season_schedule_inventory_all_games.csv", index=False)
game_discovery_log.to_csv(RUN_CHECK_DIR / "game_discovery_log.csv", index=False)
season_slug_inventory.to_csv(RUN_CHECK_DIR / "season_slug_inventory_validated.csv", index=False)
stat_slug_inventory.to_csv(RUN_CHECK_DIR / "stat_slug_inventory_completed_games.csv", index=False)

# Build season_to_slugs for stat collection only.
season_to_slugs = {}

for season in TARGET_SEASONS:
    df = stat_slug_inventory[pd.to_numeric(stat_slug_inventory["season"], errors="coerce") == season].copy()
    df = df.sort_values(["game_number", "slug"])
    season_to_slugs[season] = df["slug"].dropna().astype(str).tolist()

print("Full schedule games by season:")
display(
    season_schedule_inventory
    .groupby("season", dropna=False)
    .agg(full_schedule_games=("slug", "nunique"))
    .reset_index()
)

print("Stat-available/completed games by season:")
display(
    stat_slug_inventory
    .groupby("season", dropna=False)
    .agg(stat_available_games=("slug", "nunique"))
    .reset_index()
)

print("Resolved stat-available slugs by season:")
for season in TARGET_SEASONS:
    expected = EXPECTED_REGULAR_GAMES.get(season)
    found = len(season_to_slugs.get(season, []))
    print(f" - {season}: {found} stat games found | expected={expected}")

display(
    season_schedule_inventory[
        [c for c in [
            "season", "game_number", "slug", "event_id", "game_date_guess",
            "away_team_id_raw", "home_team_id_raw", "away_score", "home_score",
            "event_status", "event_status_label"
        ] if c in season_schedule_inventory.columns]
    ].tail(60)
)

print("Discovery complete.")

# ============================================================
# BLOCK 4 — API COLLECTION, COMPLETED GAMES ONLY, NO PLAY-BY-PLAY
# ============================================================

def scrape_game_surfaces_no_pbp(season_to_slugs_map):
    rows = []

    for season, slug_list in season_to_slugs_map.items():
        for slug in tqdm(slug_list, desc=f"Collecting season {season}"):
            game_dir = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}"
            game_dir.mkdir(parents=True, exist_ok=True)

            surfaces = [
                ("event_summary", event_summary_url, "event_summary.json.gz"),
                ("player_game_stats", player_game_stats_url, "player_game_stats.json.gz"),
                ("team_game_stats", team_game_stats_url, "team_game_stats.json.gz"),
            ]

            for source_name, url_builder, filename in surfaces:
                cache_path = game_dir / filename

                try:
                    payload, status_code, fetch_mode = fetch_json_with_cache(
                        url_builder(slug),
                        cache_path,
                        force=FORCE_RECOLLECT
                    )

                    items = safe_get(payload, "data", "items", default=None) if payload else None

                    rows.append({
                        "season": season,
                        "game_slug": slug,
                        "source_name": source_name,
                        "http_status": status_code,
                        "fetch_mode": fetch_mode,
                        "raw_path": str(cache_path) if status_code == 200 else None,
                        "has_payload": payload is not None,
                        "has_items": isinstance(items, list),
                        "items_count": len(items) if isinstance(items, list) else None,
                        "error": None,
                    })

                    if fetch_mode == "downloaded":
                        time.sleep(0.04)

                except Exception as e:
                    rows.append({
                        "season": season,
                        "game_slug": slug,
                        "source_name": source_name,
                        "http_status": None,
                        "fetch_mode": "error",
                        "raw_path": None,
                        "has_payload": False,
                        "has_items": False,
                        "items_count": None,
                        "error": str(e)[:500],
                    })

    return pd.DataFrame(rows)

api_collection_log = scrape_game_surfaces_no_pbp(season_to_slugs)
api_collection_log.to_csv(RUN_CHECK_DIR / "api_collection_log.csv", index=False)

print("Collection status counts:")
display(
    api_collection_log
    .groupby(["season", "source_name", "fetch_mode", "http_status"], dropna=False)
    .size()
    .reset_index(name="n")
    .sort_values(["season", "source_name", "fetch_mode", "http_status"])
)

print("Collection item counts:")
display(
    api_collection_log
    .groupby(["season", "source_name"], dropna=False)
    .agg(
        games=("game_slug", "nunique"),
        min_items=("items_count", "min"),
        max_items=("items_count", "max"),
        total_items=("items_count", "sum"),
    )
    .reset_index()
)

display(api_collection_log.head(25))

print("API collection complete.")

# ============================================================
# BLOCK 5 — STANDARDIZED GAME TABLES
# ============================================================

game_manifest_rows = []
team_game_rows = []
player_game_rows = []
skipped_game_rows = []
orphan_stat_rows = []

for season in TARGET_SEASONS:
    for slug in season_to_slugs.get(season, []):
        game_dir = API_RESPONSES_DIR / f"season_{season}" / f"game_{slug}"

        summary_path = game_dir / "event_summary.json.gz"
        team_path = game_dir / "team_game_stats.json.gz"
        player_path = game_dir / "player_game_stats.json.gz"

        if not all([summary_path.exists(), team_path.exists(), player_path.exists()]):
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "missing_required_api_surface",
                "summary_exists": summary_path.exists(),
                "team_exists": team_path.exists(),
                "player_exists": player_path.exists(),
            })
            continue

        try:
            summary_payload = read_gzip_json(summary_path)
            team_payload = read_gzip_json(team_path)
            player_payload = read_gzip_json(player_path)
        except Exception as e:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "could_not_read_cached_json",
                "error": str(e)[:500],
            })
            continue

        summary_data = safe_get(summary_payload, "data", default={}) or {}
        team_items = safe_get(team_payload, "data", "items", default=[]) or []
        player_items = safe_get(player_payload, "data", "items", default=[]) or []

        season_segment = summary_data.get("seasonSegment")

        if season_segment != COMPETITION_TYPE:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "non_regular_season_segment",
                "season_segment": season_segment,
            })
            continue

        # PLL can leave an orphan team row on an event. While 2026-ev-47
        # (WAT vs ATL, 2026-08-15) was being scored live the stats were briefly
        # entered against WAT vs WHP. The attribution was corrected quickly, but
        # a fully zeroed WHP row stayed on the event, so team_items had 3 entries
        # and the guard below dropped the whole game -- including 37 valid player
        # rows -- leaving the box score silently missing from the warehouse.
        #
        # The summary's own homeTeam/awayTeam stayed correct throughout, so treat
        # it as the authority on who actually played and discard rows belonging to
        # anyone else. Drops are recorded in qc.orphan_stat_rows.
        participants = []

        for team_obj in (
            extract_home_team_obj(summary_data),
            extract_away_team_obj(summary_data),
        ):
            participant_id = extract_team_id_from_obj(team_obj)

            if participant_id is None or participant_id is pd.NA:
                continue

            participant_id = str(participant_id).strip()

            if participant_id and participant_id.lower() not in ("nan", "<na>", "none"):
                participants.append(participant_id)

        participants = list(dict.fromkeys(participants))

        if len(participants) == 2 and len(team_items) > 2:
            team_keep = [x for x in team_items if str(x.get("officialId")) in participants]
            team_drop = [x for x in team_items if str(x.get("officialId")) not in participants]

            # Only trust the reconciliation when it leaves a clean two-team game.
            if len(team_keep) == 2:
                for x in team_drop:
                    orphan_stat_rows.append({
                        "season": season,
                        "slug": slug,
                        "surface": "team_game_stats",
                        "dropped_team_id": x.get("officialId"),
                        "participants": ",".join(participants),
                        "nonzero_numeric_fields": sum(
                            1 for v in x.values()
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and v
                        ),
                        "items_before": len(team_items),
                    })

                print(
                    f"  {slug}: dropped {len(team_drop)} orphan team row(s) "
                    f"[{', '.join(str(x.get('officialId')) for x in team_drop)}], "
                    f"kept {','.join(participants)}"
                )

                team_items = team_keep

        # Same correction on the player surface. A live mis-attribution can leave
        # player rows on the wrong matchup too. If *every* row looks foreign the
        # id vocabularies disagree rather than the data being wrong, so keep all.
        if len(participants) == 2:
            player_drop = [x for x in player_items if str(x.get("teamId")) not in participants]

            if player_drop and len(player_drop) < len(player_items):
                for x in player_drop:
                    orphan_stat_rows.append({
                        "season": season,
                        "slug": slug,
                        "surface": "player_game_stats",
                        "dropped_team_id": x.get("teamId"),
                        "participants": ",".join(participants),
                        "nonzero_numeric_fields": None,
                        "items_before": len(player_items),
                    })

                print(
                    f"  {slug}: dropped {len(player_drop)} player row(s) for "
                    f"non-participant team(s)"
                )

                player_items = [x for x in player_items if str(x.get("teamId")) in participants]

        if len(team_items) != 2:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "team_items_not_equal_2",
                "team_items": len(team_items),
                "player_items": len(player_items),
            })
            continue

        if len(player_items) == 0:
            skipped_game_rows.append({
                "season": season,
                "slug": slug,
                "reason": "no_player_items",
                "team_items": len(team_items),
                "player_items": len(player_items),
            })
            continue

        participant_ids_raw = [x.get("officialId") for x in team_items if x.get("officialId")]
        participant_ids_raw = list(dict.fromkeys(participant_ids_raw))

        home_obj = extract_home_team_obj(summary_data)
        away_obj = extract_away_team_obj(summary_data)

        home_team_id_raw = extract_team_id_from_obj(home_obj)
        away_team_id_raw = extract_team_id_from_obj(away_obj)

        home_team_name_raw_candidate = extract_team_name_from_obj(home_obj)
        away_team_name_raw_candidate = extract_team_name_from_obj(away_obj)

        if home_team_id_raw and not away_team_id_raw and len(participant_ids_raw) == 2:
            other_ids = [tid for tid in participant_ids_raw if tid != home_team_id_raw]
            if len(other_ids) == 1:
                away_team_id_raw = other_ids[0]

        if not home_team_id_raw and len(participant_ids_raw) == 2:
            home_team_id_raw = participant_ids_raw[0]
            away_team_id_raw = participant_ids_raw[1]

        home_team_name_raw = resolve_team_name_raw(home_team_id_raw, home_team_name_raw_candidate)
        away_team_name_raw = resolve_team_name_raw(away_team_id_raw, away_team_name_raw_candidate)

        home_team_id = canonical_team_id(home_team_id_raw)
        away_team_id = canonical_team_id(away_team_id_raw)

        home_team_name = canonical_team_name(home_team_id_raw, home_team_name_raw)
        away_team_name = canonical_team_name(away_team_id_raw, away_team_name_raw)

        game_slug = summary_data.get("slugname", slug)
        game_id = summary_data.get("eventId")
        event_numeric_id = summary_data.get("id")
        week = summary_data.get("week")
        league = summary_data.get("league")

        start_time_unix = to_num_scalar(summary_data.get("startTime"))
        start_time_utc = (
            pd.to_datetime(pd.Series([start_time_unix]), unit="s", utc=True).iloc[0]
            if not pd.isna(start_time_unix)
            else pd.NaT
        )
        game_date_utc = start_time_utc.date() if pd.notna(start_time_utc) else pd.NaT

        inv_row = stat_slug_inventory[
            (pd.to_numeric(stat_slug_inventory["season"], errors="coerce") == season)
            & (stat_slug_inventory["slug"].astype(str) == str(slug))
        ]

        if len(inv_row) > 0:
            game_number = int(pd.to_numeric(inv_row["game_number"], errors="coerce").iloc[0])
            game_number_from_slug = pd.to_numeric(inv_row["game_number_guess"], errors="coerce").iloc[0]
            schedule_slug = inv_row["slug"].iloc[0]
            schedule_event_status = inv_row["event_status"].iloc[0] if "event_status" in inv_row.columns else pd.NA
        else:
            game_number = extract_game_number_from_slug(game_slug)
            game_number_from_slug = extract_game_number_from_slug(game_slug)
            schedule_slug = slug
            schedule_event_status = summary_data.get("eventStatus")

        away_score = to_num_scalar(summary_data.get("visitorScore"))
        home_score = to_num_scalar(summary_data.get("homeScore"))

        winner_team_id_raw = pd.NA
        loser_team_id_raw = pd.NA
        winner_team_id = pd.NA
        loser_team_id = pd.NA

        if pd.notna(home_score) and pd.notna(away_score):
            if home_score > away_score:
                winner_team_id_raw = home_team_id_raw
                loser_team_id_raw = away_team_id_raw
                winner_team_id = home_team_id
                loser_team_id = away_team_id
            elif away_score > home_score:
                winner_team_id_raw = away_team_id_raw
                loser_team_id_raw = home_team_id_raw
                winner_team_id = away_team_id
                loser_team_id = home_team_id

        game_manifest_rows.append({
            "season": season,
            "competition_type": season_segment,
            "schedule_slug": schedule_slug,
            "game_slug": game_slug,
            "game_number_from_slug": game_number_from_slug,
            "game_number": game_number,
            "game_id": game_id,
            "event_numeric_id": event_numeric_id,
            "event_status": summary_data.get("eventStatus"),
            "schedule_event_status": schedule_event_status,
            "week": week,
            "league": league,
            "start_time_unix": start_time_unix,
            "start_time_utc": start_time_utc,
            "game_date_utc": game_date_utc,
            "venue": summary_data.get("venue"),
            "venue_location": summary_data.get("venueLocation"),
            "location": summary_data.get("location"),
            "period": summary_data.get("period"),
            "clock_minutes": summary_data.get("clockMinutes"),
            "clock_seconds": summary_data.get("clockSeconds"),
            "away_team_id_raw": away_team_id_raw,
            "away_team_name_raw": away_team_name_raw,
            "home_team_id_raw": home_team_id_raw,
            "home_team_name_raw": home_team_name_raw,
            "away_team_id": away_team_id,
            "away_team_name": away_team_name,
            "home_team_id": home_team_id,
            "home_team_name": home_team_name,
            "away_score": away_score,
            "home_score": home_score,
            "winner_team_id_raw": winner_team_id_raw,
            "loser_team_id_raw": loser_team_id_raw,
            "winner_team_id": winner_team_id,
            "loser_team_id": loser_team_id,
            "event_summary_path": str(summary_path),
            "team_game_stats_path": str(team_path),
            "player_game_stats_path": str(player_path),
        })

        side_map = {
            home_team_id_raw: {
                "team_id_raw": home_team_id_raw,
                "team_name_raw": home_team_name_raw,
                "team_id": home_team_id,
                "team_name": home_team_name,
                "opponent_team_id_raw": away_team_id_raw,
                "opponent_team_name_raw": away_team_name_raw,
                "opponent_team_id": away_team_id,
                "opponent_team_name": away_team_name,
                "is_home": 1,
            },
            away_team_id_raw: {
                "team_id_raw": away_team_id_raw,
                "team_name_raw": away_team_name_raw,
                "team_id": away_team_id,
                "team_name": away_team_name,
                "opponent_team_id_raw": home_team_id_raw,
                "opponent_team_name_raw": home_team_name_raw,
                "opponent_team_id": home_team_id,
                "opponent_team_name": home_team_name,
                "is_home": 0,
            },
        }

        # -----------------------------
        # Team rows
        # -----------------------------
        for item in team_items:
            team_id_raw = item.get("officialId")
            side = side_map.get(team_id_raw, {})

            resolved_team_name_raw = side.get(
                "team_name_raw",
                resolve_team_name_raw(team_id_raw)
            )

            goals = to_num_scalar(item.get("goals"))
            two_point_goals = to_num_scalar(item.get("twoPointGoals"))
            one_point_goals = derive_one_point_goals(goals, item.get("onePointGoals"), two_point_goals)

            touches = coalesce_numeric_with_alt(
                item,
                direct_keys=["touches"],
                alt_term_groups=[["touches"]],
                allow_zero=True,
            )

            total_passes = coalesce_numeric_with_alt(
                item,
                direct_keys=["totalPasses"],
                alt_term_groups=[["totalpasses"], ["passes"]],
                allow_zero=True,
            )

            time_in_possession = coalesce_numeric_with_alt(
                item,
                direct_keys=["timeInPossesion", "timeInPossession"],
                alt_term_groups=[["timeinpossesion"], ["timeinpossession"]],
                allow_zero=True,
            )

            time_in_possession_pct = coalesce_numeric_with_alt(
                item,
                direct_keys=["timeInPossesionPct", "timeInPossessionPct"],
                alt_term_groups=[["timeinpossesionpct"], ["timeinpossessionpct"]],
                allow_zero=True,
            )

            total_possessions = coalesce_numeric_with_alt(
                item,
                direct_keys=["totalPossessions"],
                alt_term_groups=[["totalpossessions"], ["possessions"]],
                allow_zero=True,
            )

            team_game_rows.append({
                "season": season,
                "competition_type": season_segment,
                "game_id": game_id,
                "schedule_slug": schedule_slug,
                "game_slug": game_slug,
                "game_number": game_number,
                "week": week,
                "game_date_utc": game_date_utc,
                "team_id_raw": team_id_raw,
                "team_name_raw": resolved_team_name_raw,
                "opponent_team_id_raw": side.get("opponent_team_id_raw"),
                "opponent_team_name_raw": side.get("opponent_team_name_raw"),
                "team_id": side.get("team_id", canonical_team_id(team_id_raw)),
                "team_name": side.get("team_name", canonical_team_name(team_id_raw, resolved_team_name_raw)),
                "opponent_team_id": side.get("opponent_team_id"),
                "opponent_team_name": side.get("opponent_team_name"),
                "is_home": side.get("is_home"),
                "scores": to_num_scalar(item.get("scores")),
                "goals": goals,
                "one_point_goals": one_point_goals,
                "two_point_goals": two_point_goals,
                "assists": to_num_scalar(item.get("assists")),
                "shots": to_num_scalar(item.get("shots")),
                "shot_pct": to_num_scalar(item.get("shotPct")),
                "shots_on_goal": to_num_scalar(item.get("shotsOnGoal")),
                "shots_on_goal_pct": to_num_scalar(item.get("shotsOnGoalPct")),
                "two_point_shots": to_num_scalar(item.get("twoPointShots")),
                "two_point_shot_pct": to_num_scalar(item.get("twoPointShotPct")),
                "two_point_shots_on_goal": to_num_scalar(item.get("twoPointShotsOnGoal")),
                "ground_balls": to_num_scalar(item.get("groundBalls")),
                "turnovers": to_num_scalar(item.get("turnovers")),
                "caused_turnovers": to_num_scalar(item.get("causedTurnovers")),
                "faceoff_pct": to_num_scalar(item.get("faceoffPct")),
                "faceoffs": to_num_scalar(item.get("faceoffs")),
                "faceoffs_won": to_num_scalar(item.get("faceoffsWon")),
                "faceoffs_lost": to_num_scalar(item.get("faceoffsLost")),
                "saves": to_num_scalar(item.get("saves")),
                "clean_saves": to_num_scalar(item.get("cleanSaves")),
                "messy_saves": to_num_scalar(item.get("messySaves")),
                "save_pct": to_num_scalar(item.get("savePct")),
                "clean_save_pct": to_num_scalar(item.get("cleanSavePct")),
                "scores_against": to_num_scalar(item.get("scoresAgainst")),
                "goals_against": to_num_scalar(item.get("goalsAgainst")),
                "num_penalties": to_num_scalar(item.get("numPenalties")),
                "pim": to_num_scalar(item.get("pim")),
                "power_play_pct": to_num_scalar(item.get("powerPlayPct")),
                "power_play_goals": to_num_scalar(item.get("powerPlayGoals")),
                "power_play_shots": to_num_scalar(item.get("powerPlayShots")),
                "power_play_goals_against": to_num_scalar(item.get("powerPlayGoalsAgainst")),
                "power_play_goals_against_pct": to_num_scalar(item.get("powerPlayGoalsAgainstPct")),
                "times_man_up": to_num_scalar(item.get("timesManUp")),
                "times_short_handed": to_num_scalar(item.get("timesShortHanded")),
                "man_down_pct": to_num_scalar(item.get("manDownPct")),
                "ride_attempts": to_num_scalar(item.get("rideAttempts")),
                "clear_attempts": to_num_scalar(item.get("clearAttempts")),
                "clears": to_num_scalar(item.get("clears")),
                "clear_pct": to_num_scalar(item.get("clearPct")),
                "shot_clock_expirations": to_num_scalar(item.get("shotClockExpirations")),
                "two_point_goals_against": to_num_scalar(item.get("twoPointGoalsAgainst")),
                "touches": touches,
                "total_passes": total_passes,
                "time_in_possession": time_in_possession,
                "time_in_possession_pct": time_in_possession_pct,
                "total_possessions": total_possessions,
                "source_path": str(team_path),
            })

        # -----------------------------
        # Player rows
        # -----------------------------
        for item in player_items:
            team_id_raw = item.get("teamId")
            side = side_map.get(team_id_raw, {})

            resolved_team_name_raw = side.get(
                "team_name_raw",
                resolve_team_name_raw(team_id_raw)
            )

            first_name = item.get("firstName")
            last_name = item.get("lastName")
            full_name = f"{first_name or ''} {last_name or ''}".strip()

            goals = to_num_scalar(item.get("goals"))
            two_point_goals = to_num_scalar(item.get("twoPointGoals"))
            one_point_goals = derive_one_point_goals(goals, item.get("onePointGoals"), two_point_goals)
            scoring_points = derive_scoring_points(one_point_goals, two_point_goals)
            points_total = derive_player_points(item.get("points"), scoring_points, item.get("assists"))

            shots = to_num_scalar(item.get("shots"))
            shots_on_goal = to_num_scalar(item.get("shotsOnGoal"))
            shots_on_goal_rate = np.nan if pd.isna(shots) or shots == 0 else shots_on_goal / shots

            player_game_rows.append({
                "season": season,
                "competition_type": season_segment,
                "game_id": game_id,
                "schedule_slug": schedule_slug,
                "game_slug": game_slug,
                "game_number": game_number,
                "week": week,
                "game_date_utc": game_date_utc,
                "team_id_raw": team_id_raw,
                "team_name_raw": resolved_team_name_raw,
                "opponent_team_id_raw": side.get("opponent_team_id_raw"),
                "opponent_team_name_raw": side.get("opponent_team_name_raw"),
                "team_id": side.get("team_id", canonical_team_id(team_id_raw)),
                "team_name": side.get("team_name", canonical_team_name(team_id_raw, resolved_team_name_raw)),
                "opponent_team_id": side.get("opponent_team_id"),
                "opponent_team_name": side.get("opponent_team_name"),
                "is_home": side.get("is_home"),
                "player_id": item.get("officialId"),
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "player_name_key": normalize_person_name(full_name),
                "player_slug": item.get("slug"),
                "profile_url": item.get("profileUrl"),
                "position": item.get("position"),
                "position_name": item.get("positionName"),
                "jersey_number": to_num_scalar(item.get("jerseyNum")),
                "games_played_source": to_num_scalar(item.get("gamesPlayed")),
                "points": points_total,
                "scoring_points": scoring_points,
                "one_point_goals": one_point_goals,
                "two_point_goals": two_point_goals,
                "goals": goals,
                "assists": to_num_scalar(item.get("assists")),
                "shots": shots,
                "shot_pct": to_num_scalar(item.get("shotPct")),
                "shots_on_goal": shots_on_goal,
                "shots_on_goal_rate": shots_on_goal_rate,
                "two_point_shots": to_num_scalar(item.get("twoPointShots")),
                "saves": to_num_scalar(item.get("saves")),
                "clean_saves": to_num_scalar(item.get("cleanSaves")),
                "messy_saves": to_num_scalar(item.get("messySaves")),
                "save_pct": to_num_scalar(item.get("savePct")),
                "clean_save_pct": to_num_scalar(item.get("cleanSavePct")),
                "scores_against_average": to_num_scalar(item.get("GAA")),
                "two_point_gaa": to_num_scalar(item.get("twoPtGaa")),
                "scores_against": to_num_scalar(item.get("scoresAgainst")),
                "saa": to_num_scalar(item.get("saa")),
                "ground_balls": to_num_scalar(item.get("groundBalls")),
                "turnovers": to_num_scalar(item.get("turnovers")),
                "caused_turnovers": to_num_scalar(item.get("causedTurnovers")),
                "faceoffs_won": to_num_scalar(item.get("faceoffsWon")),
                "faceoffs_lost": to_num_scalar(item.get("faceoffsLost")),
                "faceoffs": to_num_scalar(item.get("faceoffs")),
                "faceoff_pct": to_num_scalar(item.get("faceoffPct")),
                "goals_against": to_num_scalar(item.get("goalsAgainst")),
                "two_point_goals_against": to_num_scalar(item.get("twoPointGoalsAgainst")),
                "num_penalties": to_num_scalar(item.get("numPenalties")),
                "pim": to_num_scalar(item.get("pim")),
                "fo_record": item.get("foRecord"),
                "assist_opportunities": to_num_scalar(item.get("assistOpportunities")),
                "touches": to_num_scalar(item.get("touches")),
                "total_passes": to_num_scalar(item.get("totalPasses")),
                "source_path": str(player_path),
            })

game_manifest = pd.DataFrame(game_manifest_rows)
team_game_stats = pd.DataFrame(team_game_rows)
player_game_stats = pd.DataFrame(player_game_rows)
skipped_games = pd.DataFrame(skipped_game_rows)
orphan_stat_rows_df = pd.DataFrame(orphan_stat_rows)

if len(game_manifest) > 0:
    game_manifest = game_manifest.sort_values(["season", "game_number", "game_slug"]).reset_index(drop=True)

if len(team_game_stats) > 0:
    team_game_stats = team_game_stats.sort_values(["season", "game_number", "team_id", "game_id"]).reset_index(drop=True)

if len(player_game_stats) > 0:
    player_game_stats = player_game_stats.sort_values(["season", "game_number", "team_id", "full_name", "game_id"]).reset_index(drop=True)

team_non_numeric = {
    "competition_type", "game_id", "schedule_slug", "game_slug", "game_date_utc",
    "team_id_raw", "team_name_raw", "opponent_team_id_raw", "opponent_team_name_raw",
    "team_id", "team_name", "opponent_team_id", "opponent_team_name", "source_path"
}

player_non_numeric = {
    "competition_type", "game_id", "schedule_slug", "game_slug", "game_date_utc",
    "team_id_raw", "team_name_raw", "opponent_team_id_raw", "opponent_team_name_raw",
    "team_id", "team_name", "opponent_team_id", "opponent_team_name",
    "player_id", "first_name", "last_name", "full_name", "player_name_key",
    "player_slug", "profile_url", "position", "position_name", "fo_record", "source_path"
}

if len(team_game_stats) > 0:
    team_game_stats = coerce_numeric(team_game_stats, [c for c in team_game_stats.columns if c not in team_non_numeric])

if len(player_game_stats) > 0:
    player_game_stats = coerce_numeric(player_game_stats, [c for c in player_game_stats.columns if c not in player_non_numeric])

for c in ["season", "game_number", "week", "is_home"]:
    if c in team_game_stats.columns:
        team_game_stats[c] = safe_nullable_int(team_game_stats[c])

for c in ["season", "game_number", "week", "is_home", "jersey_number", "games_played_source", "faceoffs", "faceoffs_won", "faceoffs_lost", "shots_on_goal"]:
    if c in player_game_stats.columns:
        player_game_stats[c] = safe_nullable_int(player_game_stats[c])

print("Standardized table shapes:")
print("game_manifest:", game_manifest.shape)
print("team_game_stats:", team_game_stats.shape)
print("player_game_stats:", player_game_stats.shape)
print("skipped_games:", skipped_games.shape)
print("orphan_stat_rows:", orphan_stat_rows_df.shape)

display(game_manifest.head())
display(team_game_stats.head())
display(player_game_stats.head())

# Save standardized tables.
game_manifest.to_parquet(GAME_TABLES_DIR / "game_manifest.parquet", index=False)
team_game_stats.to_parquet(GAME_TABLES_DIR / "team_game_stats.parquet", index=False)
player_game_stats.to_parquet(GAME_TABLES_DIR / "player_game_stats.parquet", index=False)

game_manifest.to_csv(GAME_TABLES_DIR / "game_manifest.csv", index=False)
team_game_stats.to_csv(GAME_TABLES_DIR / "team_game_stats.csv", index=False)
player_game_stats.to_csv(GAME_TABLES_DIR / "player_game_stats.csv", index=False)
skipped_games.to_csv(RUN_CHECK_DIR / "skipped_games.csv", index=False)
orphan_stat_rows_df.to_csv(RUN_CHECK_DIR / "orphan_stat_rows.csv", index=False)

print("Standardized tables saved.")

# ============================================================
# BLOCK 6 — POSSESSION CLEANUP + STAT COLUMN DEFINITIONS
# ============================================================
# Run after:
# - game_manifest
# - team_game_stats
# - player_game_stats
#
# Run before:
# - curated season/career/split tables
# - defensive/opponent marts
# - DuckDB warehouse save

def seconds_to_mmss_value(x):
    if x is None or pd.isna(x):
        return None

    x = int(round(float(x)))
    sign = "-" if x < 0 else ""
    x = abs(x)

    return f"{sign}{x // 60}:{x % 60:02d}"




def seconds_to_mmss_safe(x):
    """Compatibility alias used by the team style profile builder.

    The Colab notebook used seconds_to_mmss_value() in the possession cleanup
    block and later referenced seconds_to_mmss_safe() in the team style profile
    block. Keeping this alias preserves the same formatting behavior while
    making the GitHub script executable as one consolidated file.
    """
    return seconds_to_mmss_value(x)

def seconds_to_hhmmss_value(x):
    if x is None or pd.isna(x):
        return None

    x = int(round(float(x)))
    sign = "-" if x < 0 else ""
    x = abs(x)

    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60

    if h > 0:
        return f"{sign}{h}:{m:02d}:{s:02d}"

    return f"{sign}{m}:{s:02d}"


def pct_display_value(x):
    if x is None or pd.isna(x):
        return None

    return f"{float(x) * 100:.2f}%"


def patch_team_possession_fields(team_df):
    out = team_df.copy()

    if len(out) == 0:
        return out, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------
    # Ensure expected columns exist
    # ------------------------------------------------------------
    required_cols = [
        "season",
        "game_id",
        "game_slug",
        "schedule_slug",
        "game_number",
        "game_date_utc",
        "team_id",
        "team_name",
        "opponent_team_id",
        "opponent_team_name",
        "scores",
        "scores_against",
        "shots",
        "turnovers",
        "shot_clock_expirations",
        "touches",
        "total_passes",
        "time_in_possession",
        "time_in_possession_pct",
        "total_possessions",
    ]

    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan

    numeric_cols = [
        "season",
        "game_number",
        "scores",
        "scores_against",
        "shots",
        "turnovers",
        "shot_clock_expirations",
        "touches",
        "total_passes",
        "time_in_possession",
        "time_in_possession_pct",
        "total_possessions",
    ]

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # ------------------------------------------------------------
    # Preserve raw possession values
    # ------------------------------------------------------------
    out["total_possessions_raw"] = pd.to_numeric(out["total_possessions"], errors="coerce")
    out["time_in_possession_raw"] = pd.to_numeric(out["time_in_possession"], errors="coerce")
    out["time_in_possession_pct_raw"] = pd.to_numeric(out["time_in_possession_pct"], errors="coerce")

    # ------------------------------------------------------------
    # Official possession handling
    # Only trust totalPossessions in seasons where it is populated.
    # ------------------------------------------------------------
    possession_field_quality = (
        out
        .groupby("season", dropna=False)
        .agg(
            games=("game_id", "nunique"),
            team_rows=("game_id", "count"),
            total_possessions_nonzero=(
                "total_possessions_raw",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            ),
            total_possessions_sum=(
                "total_possessions_raw",
                lambda s: pd.to_numeric(s, errors="coerce").sum()
            )
        )
        .reset_index()
    )

    usable_possession_seasons = possession_field_quality[
        possession_field_quality["total_possessions_nonzero"] > 0
    ]["season"].tolist()

    out["official_total_possessions"] = np.where(
        out["season"].isin(usable_possession_seasons),
        out["total_possessions_raw"],
        np.nan
    )

    out["official_total_possessions"] = np.where(
        pd.to_numeric(out["official_total_possessions"], errors="coerce") > 0,
        out["official_total_possessions"],
        np.nan
    )

    # ------------------------------------------------------------
    # Offensive sequence proxy
    # ------------------------------------------------------------
    out["offensive_sequence_proxy"] = (
        out["shots"].fillna(0)
        + out["turnovers"].fillna(0)
        + out["shot_clock_expirations"].fillna(0)
    )

    # ------------------------------------------------------------
    # Implied possession clock
    # ------------------------------------------------------------
    out["implied_game_clock_seconds"] = np.where(
        (out["time_in_possession_raw"].notna())
        & (out["time_in_possession_raw"] > 0)
        & (out["time_in_possession_pct_raw"].notna())
        & (out["time_in_possession_pct_raw"] > 0),
        out["time_in_possession_raw"] / out["time_in_possession_pct_raw"],
        np.nan
    )

    # ------------------------------------------------------------
    # Game-level possession quality
    # ------------------------------------------------------------
    game_possession_quality = (
        out
        .groupby(
            ["season", "game_id", "game_slug", "game_number", "game_date_utc"],
            dropna=False
        )
        .agg(
            team_rows=("team_id", "count"),
            combined_time_in_possession_raw=("time_in_possession_raw", "sum"),
            combined_time_in_possession_pct_raw=("time_in_possession_pct_raw", "sum"),
            combined_touches=("touches", "sum"),
            combined_passes=("total_passes", "sum"),
            combined_offensive_sequence_proxy=("offensive_sequence_proxy", "sum"),
            min_implied_game_clock=("implied_game_clock_seconds", "min"),
            max_implied_game_clock=("implied_game_clock_seconds", "max"),
            median_implied_game_clock=("implied_game_clock_seconds", "median"),
        )
        .reset_index()
    )

    game_possession_quality["implied_clock_range"] = (
        game_possession_quality["max_implied_game_clock"]
        - game_possession_quality["min_implied_game_clock"]
    )

    game_possession_quality["possession_data_status"] = np.select(
        [
            game_possession_quality["team_rows"] != 2,

            (
                game_possession_quality["combined_time_in_possession_raw"].fillna(0).eq(0)
                & game_possession_quality["combined_touches"].fillna(0).gt(0)
            ),

            game_possession_quality["implied_clock_range"] > 90,

            game_possession_quality["median_implied_game_clock"] > 3100,

            game_possession_quality["median_implied_game_clock"] < 2500,
        ],
        [
            "bad_team_row_count",
            "missing_possession_time",
            "team_denominator_mismatch",
            "extended_or_ot_clock",
            "short_or_provider_clock",
        ],
        default="normal"
    )

    game_possession_quality["possession_time_available"] = (
        game_possession_quality["possession_data_status"] != "missing_possession_time"
    )

    game_possession_quality["combined_time_in_possession_display"] = np.where(
        game_possession_quality["possession_time_available"],
        game_possession_quality["combined_time_in_possession_raw"].apply(seconds_to_mmss_value),
        None
    )

    game_possession_quality["median_implied_game_clock_display"] = (
        game_possession_quality["median_implied_game_clock"].apply(seconds_to_mmss_value)
    )

    game_possession_quality["possession_data_note"] = np.select(
        [
            game_possession_quality["possession_data_status"].eq("normal"),
            game_possession_quality["possession_data_status"].eq("extended_or_ot_clock"),
            game_possession_quality["possession_data_status"].eq("short_or_provider_clock"),
            game_possession_quality["possession_data_status"].eq("missing_possession_time"),
            game_possession_quality["possession_data_status"].eq("team_denominator_mismatch"),
            game_possession_quality["possession_data_status"].eq("bad_team_row_count"),
        ],
        [
            "Normal possession clock.",
            "Provider clock appears longer than regulation; likely overtime or extended provider denominator.",
            "Provider clock appears shorter than regulation; review before using possession time heavily.",
            "Possession time is unavailable even though touches/passes exist.",
            "Teams imply different possession-clock denominators; review manually.",
            "Game does not have exactly two team rows.",
        ],
        default="Review possession data."
    )

    # ------------------------------------------------------------
    # Merge possession quality back to team rows
    # ------------------------------------------------------------
    merge_cols = [
        "season",
        "game_id",
        "possession_data_status",
        "possession_time_available",
        "median_implied_game_clock",
        "median_implied_game_clock_display",
        "implied_clock_range",
        "possession_data_note",
    ]

    out = out.merge(
        game_possession_quality[merge_cols],
        on=["season", "game_id"],
        how="left"
    )

    # ------------------------------------------------------------
    # Clean possession time
    # True missing TOP games should be NaN, not 0.
    # ------------------------------------------------------------
    missing_top_mask = out["possession_data_status"].eq("missing_possession_time")

    out["time_in_possession"] = np.where(
        missing_top_mask,
        np.nan,
        out["time_in_possession_raw"]
    )

    out["time_in_possession_pct"] = np.where(
        missing_top_mask,
        np.nan,
        out["time_in_possession_pct_raw"]
    )

    out["time_in_possession_available_game"] = np.where(
        out["time_in_possession"].notna()
        & out["time_in_possession_pct"].notna()
        & out["possession_time_available"].fillna(False),
        1,
        0
    )

    out["time_in_possession_display"] = out["time_in_possession"].apply(seconds_to_mmss_value)
    out.loc[out["time_in_possession_available_game"].eq(0), "time_in_possession_display"] = None

    out["time_in_possession_pct_display"] = out["time_in_possession_pct"].apply(pct_display_value)
    out.loc[out["time_in_possession_available_game"].eq(0), "time_in_possession_pct_display"] = None

    out["implied_game_clock_display"] = out["implied_game_clock_seconds"].apply(seconds_to_mmss_value)

    # ------------------------------------------------------------
    # Possession style fields
    # ------------------------------------------------------------
    out["passes_per_touch"] = np.where(
        out["touches"] > 0,
        out["total_passes"] / out["touches"],
        np.nan
    )

    out["seconds_possession_per_touch"] = np.where(
        (out["touches"] > 0) & out["time_in_possession"].notna(),
        out["time_in_possession"] / out["touches"],
        np.nan
    )

    out["touches_per_offensive_sequence_proxy"] = np.where(
        out["offensive_sequence_proxy"] > 0,
        out["touches"] / out["offensive_sequence_proxy"],
        np.nan
    )

    out["passes_per_offensive_sequence_proxy"] = np.where(
        out["offensive_sequence_proxy"] > 0,
        out["total_passes"] / out["offensive_sequence_proxy"],
        np.nan
    )

    # ------------------------------------------------------------
    # Team-game possession quality mart
    # ------------------------------------------------------------
    possession_cols = [
        "season",
        "game_id",
        "game_slug",
        "schedule_slug",
        "game_number",
        "game_date_utc",
        "team_id",
        "team_name",
        "opponent_team_id",
        "opponent_team_name",
        "scores",
        "scores_against",
        "touches",
        "total_passes",
        "time_in_possession_raw",
        "time_in_possession",
        "time_in_possession_display",
        "time_in_possession_pct_raw",
        "time_in_possession_pct",
        "time_in_possession_pct_display",
        "implied_game_clock_seconds",
        "implied_game_clock_display",
        "median_implied_game_clock",
        "median_implied_game_clock_display",
        "total_possessions_raw",
        "official_total_possessions",
        "offensive_sequence_proxy",
        "passes_per_touch",
        "seconds_possession_per_touch",
        "touches_per_offensive_sequence_proxy",
        "passes_per_offensive_sequence_proxy",
        "time_in_possession_available_game",
        "possession_time_available",
        "possession_data_status",
        "possession_data_note",
    ]

    possession_cols = [c for c in possession_cols if c in out.columns]
    team_game_possession_quality = out[possession_cols].copy()

    # ------------------------------------------------------------
    # Season-level possession field quality summary
    # ------------------------------------------------------------
    possession_field_quality = (
        out
        .groupby("season", dropna=False)
        .agg(
            games=("game_id", "nunique"),
            team_rows=("game_id", "count"),
            time_in_possession_available_team_rows=("time_in_possession_available_game", "sum"),
            time_in_possession_raw_nonzero=(
                "time_in_possession_raw",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            ),
            missing_possession_team_rows=(
                "possession_data_status",
                lambda s: int((s == "missing_possession_time").sum())
            ),
            total_possessions_nonzero=(
                "total_possessions_raw",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            ),
            total_possessions_sum=("total_possessions_raw", "sum"),
        )
        .reset_index()
    )

    return out, possession_field_quality, game_possession_quality, team_game_possession_quality


team_game_stats, possession_field_quality, game_possession_quality, team_game_possession_quality = patch_team_possession_fields(team_game_stats)

try:
    possession_field_quality.to_csv(RUN_CHECK_DIR / "possession_field_quality_by_season.csv", index=False)
    game_possession_quality.to_csv(RUN_CHECK_DIR / "game_possession_quality.csv", index=False)
    team_game_possession_quality.to_csv(RUN_CHECK_DIR / "team_game_possession_quality.csv", index=False)
except Exception as e:
    print("Could not save possession QC files:", e)

print("Possession field quality:")
display(possession_field_quality)

print("\nGame possession quality status counts:")
display(
    game_possession_quality["possession_data_status"]
    .value_counts()
    .rename_axis("possession_data_status")
    .reset_index(name="games")
)

print("\nNon-normal possession games:")
display(
    game_possession_quality.loc[
        game_possession_quality["possession_data_status"] != "normal",
        [
            "possession_data_status",
            "season",
            "game_number",
            "game_date_utc",
            "game_slug",
            "combined_time_in_possession_raw",
            "combined_time_in_possession_display",
            "combined_time_in_possession_pct_raw",
            "median_implied_game_clock_display",
            "combined_touches",
            "combined_passes",
            "combined_offensive_sequence_proxy",
            "possession_data_note",
        ]
    ].sort_values(["season", "game_number"])
)


# ============================================================
# PLAYER / TEAM SUM COLUMN DEFINITIONS
# ============================================================

PLAYER_SUM_COLS = [
    "points", "scoring_points", "one_point_goals", "two_point_goals", "goals", "assists",
    "shots", "shots_on_goal", "two_point_shots",
    "saves", "clean_saves", "messy_saves", "scores_against", "saa",
    "ground_balls", "turnovers", "caused_turnovers",
    "faceoffs_won", "faceoffs_lost", "faceoffs",
    "goals_against", "two_point_goals_against",
    "num_penalties", "pim", "assist_opportunities", "touches", "total_passes"
]

TEAM_SUM_COLS = [
    "scores", "goals", "one_point_goals", "two_point_goals", "assists",
    "shots", "shots_on_goal", "two_point_shots", "two_point_shots_on_goal",
    "ground_balls", "turnovers", "caused_turnovers",
    "faceoffs", "faceoffs_won", "faceoffs_lost",
    "saves", "clean_saves", "messy_saves", "scores_against", "goals_against",
    "num_penalties", "pim", "power_play_goals", "power_play_shots",
    "power_play_goals_against", "times_man_up", "times_short_handed",
    "ride_attempts", "clear_attempts", "clears", "shot_clock_expirations",
    "two_point_goals_against", "touches", "total_passes",
    "time_in_possession", "time_in_possession_available_game",
    "official_total_possessions", "offensive_sequence_proxy"
]

PLAYER_SUM_COLS = [c for c in PLAYER_SUM_COLS if c in player_game_stats.columns]
TEAM_SUM_COLS = [c for c in TEAM_SUM_COLS if c in team_game_stats.columns]

print("PLAYER_SUM_COLS:", PLAYER_SUM_COLS)
print("TEAM_SUM_COLS:", TEAM_SUM_COLS)

# ============================================================
# BLOCK 7 — CURATED TABLES, SEASON TOTALS, CAREER TOTALS, SPLITS
# ============================================================

def add_player_rate_columns(df):
    out = df.copy()

    if "shots" in out.columns and "goals" in out.columns:
        out["shot_pct_calc"] = np.where(out["shots"] > 0, out["goals"] / out["shots"], np.nan)

    if "shots" in out.columns and "shots_on_goal" in out.columns:
        out["shots_on_goal_rate_calc"] = np.where(out["shots"] > 0, out["shots_on_goal"] / out["shots"], np.nan)

    if "faceoffs" in out.columns and "faceoffs_won" in out.columns:
        out["faceoff_pct_calc"] = np.where(out["faceoffs"] > 0, out["faceoffs_won"] / out["faceoffs"], np.nan)

    if "saves" in out.columns and "goals_against" in out.columns:
        out["save_pct_calc"] = np.where(
            (out["saves"] + out["goals_against"]) > 0,
            out["saves"] / (out["saves"] + out["goals_against"]),
            np.nan
        )

    if "clean_saves" in out.columns and "saves" in out.columns:
        out["clean_save_pct"] = np.where(
            pd.to_numeric(out["saves"], errors="coerce") > 0,
            pd.to_numeric(out["clean_saves"], errors="coerce") / pd.to_numeric(out["saves"], errors="coerce") * 100,
            np.nan
        )

    if "clean_saves" in out.columns and "saves" in out.columns and "goals_against" in out.columns:
        _shots_faced = (
            pd.to_numeric(out["saves"], errors="coerce").fillna(0)
            + pd.to_numeric(out["goals_against"], errors="coerce").fillna(0)
        )
        out["clean_save_rate"] = np.where(
            _shots_faced > 0,
            pd.to_numeric(out["clean_saves"], errors="coerce") / _shots_faced,
            np.nan
        )

    if "assist_opportunities" in out.columns and "assists" in out.columns:
        out["assist_conv_rate"] = np.where(
            pd.to_numeric(out["assist_opportunities"], errors="coerce") > 0,
            pd.to_numeric(out["assists"], errors="coerce") / pd.to_numeric(out["assist_opportunities"], errors="coerce"),
            np.nan
        )

    if "two_point_goals" in out.columns and "two_point_shots" in out.columns:
        out["two_pt_conversion"] = np.where(
            pd.to_numeric(out["two_point_shots"], errors="coerce") > 0,
            pd.to_numeric(out["two_point_goals"], errors="coerce") / pd.to_numeric(out["two_point_shots"], errors="coerce"),
            np.nan
        )

    if "games" in out.columns:
        for c in PLAYER_SUM_COLS:
            if c in out.columns:
                out[f"{c}_per_game"] = np.where(out["games"] > 0, out[c] / out["games"], np.nan)

    # Convenience alias for assist opportunities per game
    if "assist_opportunities_per_game" in out.columns:
        out["assist_opp_per_game"] = out["assist_opportunities_per_game"]
    elif "assist_opportunities" in out.columns and "games" in out.columns:
        out["assist_opp_per_game"] = np.where(
            pd.to_numeric(out["games"], errors="coerce") > 0,
            pd.to_numeric(out["assist_opportunities"], errors="coerce") / pd.to_numeric(out["games"], errors="coerce"),
            np.nan
        )

    return out


def add_team_rate_columns(df):
    out = df.copy()

    if "shots" in out.columns and "goals" in out.columns:
        out["shot_pct_calc"] = np.where(out["shots"] > 0, out["goals"] / out["shots"], np.nan)

    if "shots" in out.columns and "shots_on_goal" in out.columns:
        out["shots_on_goal_rate_calc"] = np.where(out["shots"] > 0, out["shots_on_goal"] / out["shots"], np.nan)

    if "faceoffs" in out.columns and "faceoffs_won" in out.columns:
        out["faceoff_pct_calc"] = np.where(out["faceoffs"] > 0, out["faceoffs_won"] / out["faceoffs"], np.nan)

    if "clear_attempts" in out.columns and "clears" in out.columns:
        out["clear_pct_calc"] = np.where(out["clear_attempts"] > 0, out["clears"] / out["clear_attempts"], np.nan)

    if "games" in out.columns:
        for c in TEAM_SUM_COLS:
            if c not in out.columns:
                continue

            if c == "time_in_possession" and "time_in_possession_available_game" in out.columns:
                denom = pd.to_numeric(out["time_in_possession_available_game"], errors="coerce")
                out[f"{c}_per_game"] = np.where(denom > 0, out[c] / denom, np.nan)

                # If no valid TOP games exist, do not let all-missing sums appear as 0.
                out[c] = np.where(denom > 0, out[c], np.nan)

            else:
                out[f"{c}_per_game"] = np.where(out["games"] > 0, out[c] / out["games"], np.nan)

    # Possession style ratios from aggregated totals
    if "touches" in out.columns and "total_passes" in out.columns:
        out["passes_per_touch"] = np.where(out["touches"] > 0, out["total_passes"] / out["touches"], np.nan)

    if "touches" in out.columns and "time_in_possession" in out.columns:
        out["seconds_possession_per_touch"] = np.where(
            out["touches"] > 0,
            out["time_in_possession"] / out["touches"],
            np.nan
        )

    if "offensive_sequence_proxy" in out.columns and "touches" in out.columns:
        out["touches_per_offensive_sequence_proxy"] = np.where(
            out["offensive_sequence_proxy"] > 0,
            out["touches"] / out["offensive_sequence_proxy"],
            np.nan
        )

    if "offensive_sequence_proxy" in out.columns and "total_passes" in out.columns:
        out["passes_per_offensive_sequence_proxy"] = np.where(
            out["offensive_sequence_proxy"] > 0,
            out["total_passes"] / out["offensive_sequence_proxy"],
            np.nan
        )

    # Display-ready possession fields for downstream tables/app
    if "time_in_possession" in out.columns:
        out["time_in_possession_total_display"] = out["time_in_possession"].apply(seconds_to_hhmmss_value)

    if "time_in_possession_per_game" in out.columns:
        out["time_in_possession_per_game_display"] = out["time_in_possession_per_game"].apply(seconds_to_mmss_value)

    return out

# -----------------------------
# Team alias mapping
# -----------------------------
if len(team_game_stats) > 0:
    observed_team_id_raw = sorted(set(team_game_stats["team_id_raw"].dropna().astype(str).tolist()))
else:
    observed_team_id_raw = []

team_alias_mapping = pd.DataFrame({"team_id_raw": observed_team_id_raw})

if len(team_alias_mapping) > 0:
    team_alias_mapping["team_name_raw"] = team_alias_mapping["team_id_raw"].map(lambda x: TEAM_NAME_LOOKUP_RAW.get(x, x))
    team_alias_mapping["team_id"] = team_alias_mapping["team_id_raw"].map(canonical_team_id)
    team_alias_mapping["team_name"] = team_alias_mapping.apply(
        lambda r: canonical_team_name(r["team_id_raw"], r["team_name_raw"]),
        axis=1
    )

team_alias_mapping = team_alias_mapping.drop_duplicates().sort_values(["team_id", "team_id_raw"]).reset_index(drop=True)

# -----------------------------
# Team directory
# -----------------------------
if len(team_game_stats) > 0:
    observed_canonical_team_ids = sorted(set(team_game_stats["team_id"].dropna().astype(str).tolist()))
else:
    observed_canonical_team_ids = []

team_directory = pd.DataFrame({"team_id": observed_canonical_team_ids})

if len(team_directory) > 0:
    team_directory["team_name"] = team_directory["team_id"].map(lambda x: canonical_team_name(x, x))
    team_directory["team_display_name"] = team_directory["team_id"].map(TEAM_DISPLAY_NAME_LOOKUP).fillna(team_directory["team_name"])

team_directory = team_directory.drop_duplicates().sort_values("team_id").reset_index(drop=True)

# -----------------------------
# Player directory
# -----------------------------
if len(player_game_stats) > 0:
    player_directory = (
        player_game_stats
        .sort_values(["season", "game_number", "game_id"])
        .groupby("player_id", dropna=False)
        .agg({
            "full_name": "last",
            "first_name": "last",
            "last_name": "last",
            "player_name_key": "last",
            "player_slug": "last",
            "profile_url": "last",
            "position": mode_or_first,
            "position_name": mode_or_first,
            "jersey_number": mode_or_first,
            "team_id": lambda s: "|".join(sorted(set([str(x) for x in s.dropna()]))),
            "team_name": lambda s: "|".join(sorted(set([str(x) for x in s.dropna()]))),
            "game_id": "nunique",
            "season": "nunique",
        })
        .reset_index()
        .rename(columns={
            "game_id": "career_games_in_database",
            "season": "seasons_in_database",
        })
        .sort_values(["full_name", "player_id"])
        .reset_index(drop=True)
    )
else:
    player_directory = pd.DataFrame()

# -----------------------------
# Player season by team
# -----------------------------
if len(player_game_stats) > 0:
    player_season_stats_by_team = (
        player_game_stats
        .groupby(["season", "player_id", "full_name", "team_id", "team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            position=("position", mode_or_first),
            position_name=("position_name", mode_or_first),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in PLAYER_SUM_COLS}
        )
        .reset_index()
    )
    player_season_stats_by_team = add_player_rate_columns(player_season_stats_by_team)
else:
    player_season_stats_by_team = pd.DataFrame()

# -----------------------------
# Player season, one row per player-season
# -----------------------------
player_season_rows = []

if len(player_game_stats) > 0:
    for keys, g in player_game_stats.groupby(["season", "player_id"], dropna=False):
        season, player_id = keys

        row = {
            "season": season,
            "player_id": player_id,
            "full_name": latest_non_null_by_game(g, "full_name"),
            "first_name": latest_non_null_by_game(g, "first_name"),
            "last_name": latest_non_null_by_game(g, "last_name"),
            "position": mode_or_first(g["position"]) if "position" in g.columns else pd.NA,
            "position_name": mode_or_first(g["position_name"]) if "position_name" in g.columns else pd.NA,
            "games": g["game_id"].nunique(),
            "teams": "|".join(sorted(set([str(x) for x in g["team_id"].dropna()]))),
            "team_names": "|".join(sorted(set([str(x) for x in g["team_name"].dropna()]))),
            "first_game_date": g["game_date_utc"].min(),
            "last_game_date": g["game_date_utc"].max(),
        }

        for c in PLAYER_SUM_COLS:
            row[c] = pd.to_numeric(g[c], errors="coerce").sum()

        player_season_rows.append(row)

player_season_stats = pd.DataFrame(player_season_rows)
player_season_stats = add_player_rate_columns(player_season_stats) if len(player_season_stats) > 0 else player_season_stats

# -----------------------------
# Player career, one row per player
# -----------------------------
player_career_rows = []

if len(player_game_stats) > 0:
    for player_id, g in player_game_stats.groupby("player_id", dropna=False):
        row = {
            "player_id": player_id,
            "full_name": latest_non_null_by_game(g, "full_name"),
            "first_name": latest_non_null_by_game(g, "first_name"),
            "last_name": latest_non_null_by_game(g, "last_name"),
            "position": mode_or_first(g["position"]) if "position" in g.columns else pd.NA,
            "position_name": mode_or_first(g["position_name"]) if "position_name" in g.columns else pd.NA,
            "games": g["game_id"].nunique(),
            "seasons": g["season"].nunique(),
            "teams": "|".join(sorted(set([str(x) for x in g["team_id"].dropna()]))),
            "team_names": "|".join(sorted(set([str(x) for x in g["team_name"].dropna()]))),
            "first_game_date": g["game_date_utc"].min(),
            "last_game_date": g["game_date_utc"].max(),
        }

        for c in PLAYER_SUM_COLS:
            row[c] = pd.to_numeric(g[c], errors="coerce").sum()

        player_career_rows.append(row)

player_career_stats = pd.DataFrame(player_career_rows)
player_career_stats = add_player_rate_columns(player_career_stats) if len(player_career_stats) > 0 else player_career_stats

# -----------------------------
# Score-based team game results
# -----------------------------
def add_score_based_team_result_flags(team_games):
    """
    Adds authoritative team-game win/loss/tie flags from the team-game score itself.

    This avoids using game_manifest winner_team_id / loser_team_id, which can be stale,
    missing, or inconsistent for newly completed games.
    """
    if team_games is None or len(team_games) == 0:
        return pd.DataFrame() if team_games is None else team_games.copy()

    out = team_games.copy()

    if "scores" not in out.columns or "scores_against" not in out.columns:
        raise KeyError("team_game_stats must contain 'scores' and 'scores_against' before record flags can be created.")

    out["scores"] = pd.to_numeric(out["scores"], errors="coerce")
    out["scores_against"] = pd.to_numeric(out["scores_against"], errors="coerce")

    valid_score = out["scores"].notna() & out["scores_against"].notna()

    out["win_flag"] = np.where(
        valid_score & (out["scores"] > out["scores_against"]),
        1,
        np.where(valid_score, 0, np.nan)
    )

    out["loss_flag"] = np.where(
        valid_score & (out["scores"] < out["scores_against"]),
        1,
        np.where(valid_score, 0, np.nan)
    )

    out["tie_flag"] = np.where(
        valid_score & (out["scores"] == out["scores_against"]),
        1,
        np.where(valid_score, 0, np.nan)
    )

    out["score_margin"] = np.where(
        valid_score,
        out["scores"] - out["scores_against"],
        np.nan
    )

    out["result"] = np.select(
        [
            out["win_flag"].eq(1),
            out["loss_flag"].eq(1),
            out["tie_flag"].eq(1),
        ],
        ["W", "L", "T"],
        default=pd.NA
    )

    return out


team_game_stats = add_score_based_team_result_flags(team_game_stats)

print("Score-based team result flags added to team_game_stats.")

if len(team_game_stats) > 0:
    display(
        team_game_stats[
            [
                "season",
                "game_number",
                "team_id",
                "team_name",
                "opponent_team_id",
                "opponent_team_name",
                "scores",
                "scores_against",
                "win_flag",
                "loss_flag",
                "tie_flag",
                "score_margin",
                "result",
            ]
        ]
        .sort_values(["season", "game_number", "team_name"])
        .tail(30)
    )


# -----------------------------
# Team season
# -----------------------------
if len(team_game_stats) > 0:
    team_season_stats = (
        team_game_stats
        .groupby(["season", "team_id", "team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            ties=("tie_flag", "sum"),
            score_margin=("score_margin", "sum"),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in TEAM_SUM_COLS}
        )
        .reset_index()
    )

    team_season_stats["win_pct"] = np.where(
        team_season_stats["games"] > 0,
        team_season_stats["wins"] / team_season_stats["games"],
        np.nan
    )

    team_season_stats["score_margin_per_game"] = np.where(
        team_season_stats["games"] > 0,
        team_season_stats["score_margin"] / team_season_stats["games"],
        np.nan
    )

    team_season_stats = add_team_rate_columns(team_season_stats)

    # Keep record fields in a clean numeric format.
    for c in ["wins", "losses", "ties"]:
        if c in team_season_stats.columns:
            team_season_stats[c] = pd.to_numeric(team_season_stats[c], errors="coerce")

else:
    team_season_stats = pd.DataFrame()


# -----------------------------
# Team career/franchise
# -----------------------------
if len(team_game_stats) > 0:
    team_career_stats = (
        team_game_stats
        .groupby(["team_id", "team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            seasons=("season", "nunique"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            ties=("tie_flag", "sum"),
            score_margin=("score_margin", "sum"),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in TEAM_SUM_COLS}
        )
        .reset_index()
    )

    team_career_stats["win_pct"] = np.where(
        team_career_stats["games"] > 0,
        team_career_stats["wins"] / team_career_stats["games"],
        np.nan
    )

    team_career_stats["score_margin_per_game"] = np.where(
        team_career_stats["games"] > 0,
        team_career_stats["score_margin"] / team_career_stats["games"],
        np.nan
    )

    team_career_stats = add_team_rate_columns(team_career_stats)

    for c in ["wins", "losses", "ties"]:
        if c in team_career_stats.columns:
            team_career_stats[c] = pd.to_numeric(team_career_stats[c], errors="coerce")

else:
    team_career_stats = pd.DataFrame()


# -----------------------------
# Opponent splits
# -----------------------------
if len(player_game_stats) > 0:
    player_vs_opponent_stats = (
        player_game_stats
        .groupby(["player_id", "full_name", "opponent_team_id", "opponent_team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            position=("position", mode_or_first),
            position_name=("position_name", mode_or_first),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in PLAYER_SUM_COLS}
        )
        .reset_index()
    )

    player_vs_opponent_stats = add_player_rate_columns(player_vs_opponent_stats)

else:
    player_vs_opponent_stats = pd.DataFrame()


if len(team_game_stats) > 0:
    team_vs_opponent_stats = (
        team_game_stats
        .groupby(["team_id", "team_name", "opponent_team_id", "opponent_team_name"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("win_flag", "sum"),
            losses=("loss_flag", "sum"),
            ties=("tie_flag", "sum"),
            score_margin=("score_margin", "sum"),
            first_game_date=("game_date_utc", "min"),
            last_game_date=("game_date_utc", "max"),
            **{c: (c, "sum") for c in TEAM_SUM_COLS}
        )
        .reset_index()
    )

    team_vs_opponent_stats["win_pct"] = np.where(
        team_vs_opponent_stats["games"] > 0,
        team_vs_opponent_stats["wins"] / team_vs_opponent_stats["games"],
        np.nan
    )

    team_vs_opponent_stats["score_margin_per_game"] = np.where(
        team_vs_opponent_stats["games"] > 0,
        team_vs_opponent_stats["score_margin"] / team_vs_opponent_stats["games"],
        np.nan
    )

    team_vs_opponent_stats = add_team_rate_columns(team_vs_opponent_stats)

else:
    team_vs_opponent_stats = pd.DataFrame()


print("Curated base tables created.")
print("player_directory:", player_directory.shape)
print("player_season_stats:", player_season_stats.shape)
print("player_career_stats:", player_career_stats.shape)
print("team_season_stats:", team_season_stats.shape)
print("team_career_stats:", team_career_stats.shape)
print("player_vs_opponent_stats:", player_vs_opponent_stats.shape)
print("team_vs_opponent_stats:", team_vs_opponent_stats.shape)

if len(team_season_stats) > 0:
    print("\nScore-based team records check:")
    display(
        team_season_stats[
            ["season", "team_id", "team_name", "games", "wins", "losses", "ties", "win_pct", "scores", "scores_against"]
        ]
        .sort_values(["season", "team_name"])
        .tail(40)
    )

# ============================================================
# BLOCK 8 — LAST 5 / LAST 10 SPLIT TABLES
# ============================================================

def build_player_last_n_stats(player_games, n=5, by_season=False):
    if len(player_games) == 0:
        return pd.DataFrame()

    base = player_games.sort_values(["season", "game_date_utc", "game_number", "game_id"], na_position="first").copy()

    group_cols = ["player_id"]
    if by_season:
        group_cols = ["season", "player_id"]

    rows = []

    for keys, g in base.groupby(group_cols, dropna=False):
        g_last = g.tail(n).copy()

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["full_name"] = latest_non_null_by_game(g_last, "full_name")
        row["position"] = mode_or_first(g_last["position"]) if "position" in g_last.columns else pd.NA
        row["position_name"] = mode_or_first(g_last["position_name"]) if "position_name" in g_last.columns else pd.NA
        row["split_type"] = f"last_{n}"
        row["games"] = g_last["game_id"].nunique()
        row["first_game_date"] = g_last["game_date_utc"].min()
        row["last_game_date"] = g_last["game_date_utc"].max()
        row["opponents"] = "|".join(sorted(set([str(x) for x in g_last["opponent_team_id"].dropna()])))
        row["teams"] = "|".join(sorted(set([str(x) for x in g_last["team_id"].dropna()])))

        for c in PLAYER_SUM_COLS:
            row[c] = pd.to_numeric(g_last[c], errors="coerce").sum()

        rows.append(row)

    out = pd.DataFrame(rows)
    out = add_player_rate_columns(out)

    return out

def build_team_last_n_stats(team_games, n=5, by_season=False):
    if len(team_games) == 0:
        return pd.DataFrame()

    base = team_games.sort_values(["season", "game_date_utc", "game_number", "game_id"], na_position="first").copy()

    group_cols = ["team_id"]
    if by_season:
        group_cols = ["season", "team_id"]

    rows = []

    for keys, g in base.groupby(group_cols, dropna=False):
        g_last = g.tail(n).copy()

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["team_name"] = latest_non_null_by_game(g_last, "team_name")
        row["split_type"] = f"last_{n}"
        row["games"] = g_last["game_id"].nunique()
        row["first_game_date"] = g_last["game_date_utc"].min()
        row["last_game_date"] = g_last["game_date_utc"].max()
        row["opponents"] = "|".join(sorted(set([str(x) for x in g_last["opponent_team_id"].dropna()])))

        for c in TEAM_SUM_COLS:
            row[c] = pd.to_numeric(g_last[c], errors="coerce").sum()

        rows.append(row)

    out = pd.DataFrame(rows)
    out = add_team_rate_columns(out)

    return out

player_last5_stats = build_player_last_n_stats(player_game_stats, n=5, by_season=False)
player_last10_stats = build_player_last_n_stats(player_game_stats, n=10, by_season=False)

player_season_last5_stats = build_player_last_n_stats(player_game_stats, n=5, by_season=True)
player_season_last10_stats = build_player_last_n_stats(player_game_stats, n=10, by_season=True)

team_last5_stats = build_team_last_n_stats(team_game_stats, n=5, by_season=False)
team_last10_stats = build_team_last_n_stats(team_game_stats, n=10, by_season=False)

team_season_last5_stats = build_team_last_n_stats(team_game_stats, n=5, by_season=True)
team_season_last10_stats = build_team_last_n_stats(team_game_stats, n=10, by_season=True)

print("Rolling split tables created.")
print("player_last5_stats:", player_last5_stats.shape)
print("player_last10_stats:", player_last10_stats.shape)
print("player_season_last5_stats:", player_season_last5_stats.shape)
print("player_season_last10_stats:", player_season_last10_stats.shape)
print("team_last5_stats:", team_last5_stats.shape)
print("team_last10_stats:", team_last10_stats.shape)
print("team_season_last5_stats:", team_season_last5_stats.shape)
print("team_season_last10_stats:", team_season_last10_stats.shape)

display(player_last5_stats.head())
display(team_last5_stats.head())

# ============================================================
# BLOCK 9 — CLEAN SCHEDULE TABLES
# ============================================================

def build_clean_schedule_table(schedule_inventory):
    if len(schedule_inventory) == 0:
        return pd.DataFrame()

    out = schedule_inventory.copy()

    for c in ["event_status_num", "event_status", "away_score", "home_score"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "event_status_label" not in out.columns:
        out["event_status_label"] = np.select(
            [
                out.get("event_status_num", pd.Series(index=out.index, dtype=float)) == 3,
                out.get("event_status_num", pd.Series(index=out.index, dtype=float)) == 0,
            ],
            ["final", "scheduled"],
            default="unknown"
        )

    out["away_team_id"] = out["away_team_id_raw"].map(canonical_team_id) if "away_team_id_raw" in out.columns else pd.NA
    out["home_team_id"] = out["home_team_id_raw"].map(canonical_team_id) if "home_team_id_raw" in out.columns else pd.NA

    out["away_team_name"] = out.apply(
        lambda r: canonical_team_name(r.get("away_team_id_raw"), r.get("away_team_name_raw")),
        axis=1
    )

    out["home_team_name"] = out.apply(
        lambda r: canonical_team_name(r.get("home_team_id_raw"), r.get("home_team_name_raw")),
        axis=1
    )

    keep_cols = [
        "season",
        "game_number",
        "slug",
        "event_id",
        "event_numeric_id",
        "game_date_guess",
        "away_team_id_raw",
        "away_team_name_raw",
        "home_team_id_raw",
        "home_team_name_raw",
        "away_team_id",
        "away_team_name",
        "home_team_id",
        "home_team_name",
        "away_score",
        "home_score",
        "event_status",
        "event_status_num",
        "event_status_label",
        "discovery_source",
        "source",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols].copy()
    out = out.sort_values(["season", "game_number", "game_date_guess", "slug"], na_position="last").reset_index(drop=True)

    return out

game_schedule_all = build_clean_schedule_table(season_schedule_inventory)

game_schedule_2026 = game_schedule_all[
    pd.to_numeric(game_schedule_all["season"], errors="coerce") == 2026
].copy()

print("game_schedule_all:", game_schedule_all.shape)
print("game_schedule_2026:", game_schedule_2026.shape)

display(game_schedule_2026.head(25))

# ============================================================
# BLOCK 10 — QUALITY CHECKS
# ============================================================

quality_rows = []

def add_qc_check(check_name, status, actual=None, expected=None, notes=None):
    quality_rows.append({
        "check_name": check_name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "notes": notes,
    })

add_qc_check("game_manifest_rows", "info", len(game_manifest), None, "Number of completed/stat-available regular-season games parsed.")
add_qc_check("team_game_stats_rows", "info", len(team_game_stats), None, "Should usually be 2x game_manifest rows.")
add_qc_check("player_game_stats_rows", "info", len(player_game_stats), None, "One row per player-game.")
add_qc_check("skipped_games_rows", "info", len(skipped_games), None, "Games skipped due to missing surfaces or invalid stats.")
add_qc_check("orphan_stat_rows", "info", len(orphan_stat_rows_df), None, "Stat rows dropped for teams that did not play in the event.")
add_qc_check("full_schedule_inventory_rows", "info", len(season_schedule_inventory), None, "Full schedule, including future games.")
add_qc_check("stat_slug_inventory_rows", "info", len(stat_slug_inventory), None, "Completed/stat-available games only.")

# Expected completed games by season.
if len(game_manifest) > 0:
    games_by_season = game_manifest.groupby("season")["game_id"].nunique().reset_index(name="games")

    for _, r in games_by_season.iterrows():
        season = int(r["season"])
        actual_games = int(r["games"])
        expected_games = EXPECTED_REGULAR_GAMES.get(season)

        if expected_games is None:
            status = "info" if actual_games > 0 else "warning"
            notes = "Ongoing season; only completed games are expected."
        else:
            status = "pass" if actual_games == expected_games else "warning"
            notes = "Completed regular season expected count."

        add_qc_check(
            f"expected_stat_game_count_{season}",
            status,
            actual_games,
            expected_games,
            notes
        )

# Full schedule by season.
if len(season_schedule_inventory) > 0:
    sched_by_season = season_schedule_inventory.groupby("season")["slug"].nunique().reset_index(name="schedule_games")

    for _, r in sched_by_season.iterrows():
        season = int(r["season"])
        add_qc_check(
            f"full_schedule_game_count_{season}",
            "info",
            int(r["schedule_games"]),
            None,
            "Full schedule count, including scheduled/future games."
        )

# Team rows per game.
if len(team_game_stats) > 0:
    team_rows_per_game = team_game_stats.groupby("game_id").size().reset_index(name="team_rows")
    bad_team_row_games = team_rows_per_game[team_rows_per_game["team_rows"] != 2].copy()

    add_qc_check(
        "exactly_two_team_rows_per_game",
        "pass" if len(bad_team_row_games) == 0 else "warning",
        len(bad_team_row_games),
        0,
        "Each completed game should have exactly two team-game rows."
    )

    bad_team_row_games.to_csv(RUN_CHECK_DIR / "bad_team_row_games.csv", index=False)

# Duplicate keys.
if len(team_game_stats) > 0:
    team_dupes = team_game_stats.duplicated(subset=["game_id", "team_id"], keep=False).sum()
    add_qc_check(
        "duplicate_team_game_keys",
        "pass" if team_dupes == 0 else "fail",
        int(team_dupes),
        0,
        "Duplicate game_id/team_id rows."
    )

if len(player_game_stats) > 0:
    player_dupes = player_game_stats.duplicated(subset=["game_id", "player_id", "team_id"], keep=False).sum()
    add_qc_check(
        "duplicate_player_game_keys",
        "pass" if player_dupes == 0 else "fail",
        int(player_dupes),
        0,
        "Duplicate game_id/player_id/team_id rows."
    )

# Team scoring formula.
if len(team_game_stats) > 0:
    scoring_check = team_game_stats.copy()
    scoring_check["scores_calc"] = scoring_check["one_point_goals"] + 2 * scoring_check["two_point_goals"]
    scoring_check["score_diff"] = scoring_check["scores"] - scoring_check["scores_calc"]

    bad_score_formula = scoring_check[
        scoring_check["score_diff"].notna() &
        (scoring_check["score_diff"].abs() > 0.001)
    ].copy()

    add_qc_check(
        "team_scores_formula",
        "pass" if len(bad_score_formula) == 0 else "warning",
        len(bad_score_formula),
        0,
        "scores should equal one_point_goals + 2 * two_point_goals."
    )

    bad_score_formula.to_csv(RUN_CHECK_DIR / "bad_team_score_formula_rows.csv", index=False)

# Player scoring formulas.
if len(player_game_stats) > 0:
    player_scoring_check = player_game_stats.copy()
    player_scoring_check["scoring_points_calc"] = player_scoring_check["one_point_goals"] + 2 * player_scoring_check["two_point_goals"]
    player_scoring_check["scoring_points_diff"] = player_scoring_check["scoring_points"] - player_scoring_check["scoring_points_calc"]

    bad_player_scoring = player_scoring_check[
        player_scoring_check["scoring_points_diff"].notna() &
        (player_scoring_check["scoring_points_diff"].abs() > 0.001)
    ].copy()

    player_scoring_check["points_calc"] = player_scoring_check["scoring_points"] + player_scoring_check["assists"]
    player_scoring_check["points_diff"] = player_scoring_check["points"] - player_scoring_check["points_calc"]

    bad_player_points = player_scoring_check[
        player_scoring_check["points_diff"].notna() &
        (player_scoring_check["points_diff"].abs() > 0.001)
    ].copy()

    add_qc_check(
        "player_scoring_points_formula",
        "pass" if len(bad_player_scoring) == 0 else "warning",
        len(bad_player_scoring),
        0,
        "scoring_points should equal one_point_goals + 2 * two_point_goals."
    )

    add_qc_check(
        "player_total_points_formula",
        "pass" if len(bad_player_points) == 0 else "warning",
        len(bad_player_points),
        0,
        "points should equal scoring_points + assists."
    )

    bad_player_scoring.to_csv(RUN_CHECK_DIR / "bad_player_scoring_points_formula_rows.csv", index=False)
    bad_player_points.to_csv(RUN_CHECK_DIR / "bad_player_total_points_formula_rows.csv", index=False)

# Team score vs manifest.
if len(game_manifest) > 0 and len(team_game_stats) > 0:
    team_scores_compare = team_game_stats.merge(
        game_manifest[["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]],
        on="game_id",
        how="left"
    )

    team_scores_compare["manifest_score"] = np.where(
        team_scores_compare["team_id"] == team_scores_compare["home_team_id"],
        team_scores_compare["home_score"],
        np.where(
            team_scores_compare["team_id"] == team_scores_compare["away_team_id"],
            team_scores_compare["away_score"],
            np.nan
        )
    )

    team_scores_compare["manifest_score_diff"] = team_scores_compare["scores"] - team_scores_compare["manifest_score"]

    bad_manifest_score = team_scores_compare[
        team_scores_compare["manifest_score_diff"].notna() &
        (team_scores_compare["manifest_score_diff"].abs() > 0.001)
    ].copy()

    add_qc_check(
        "team_scores_match_game_manifest",
        "pass" if len(bad_manifest_score) == 0 else "warning",
        len(bad_manifest_score),
        0,
        "Team stats score should match home/away score in game manifest."
    )

    bad_manifest_score.to_csv(RUN_CHECK_DIR / "bad_manifest_score_match_rows.csv", index=False)

# Possession field quality.
if len(possession_field_quality) > 0:
    for _, r in possession_field_quality.iterrows():
        season = int(r["season"])
        total_nonzero = int(r["total_possessions_nonzero"])

        status = "info" if total_nonzero > 0 else "warning"

        add_qc_check(
            f"official_total_possessions_populated_{season}",
            status,
            total_nonzero,
            None,
            "Raw totalPossessions is only reliable in seasons where non-zero values exist."
        )

quality_summary = pd.DataFrame(quality_rows)
quality_summary.to_csv(RUN_CHECK_DIR / "quality_summary.csv", index=False)

display(quality_summary)

print("QC files saved to:", RUN_CHECK_DIR)

# ============================================================
# BLOCK 10.5 — DEFENSIVE / OPPONENT METRICS MARTS
# ============================================================

def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    return np.where(
        denominator.notna() & (denominator != 0),
        numerator / denominator,
        np.nan
    )


def build_team_game_opponent_context(team_df):
    """
    Builds one defensive/opponent-context row per team-game by self-joining
    each team row to the opposing team row from the same game.

    This table is the foundation for defensive rankings, matchup previews,
    opponent allowances, and team style/profile work.
    """
    if team_df is None or len(team_df) == 0:
        return pd.DataFrame()

    left = team_df.copy()
    right = team_df.copy()

    merged = left.merge(
        right,
        on="game_id",
        how="left",
        suffixes=("", "_opp_row")
    )

    merged = merged[
        merged["team_id"].astype(str) != merged["team_id_opp_row"].astype(str)
    ].copy()

    # Safety: one opponent row per team-game.
    merged = merged.drop_duplicates(subset=["game_id", "team_id"], keep="first").copy()

    ctx = pd.DataFrame()

    base_cols = [
        "season", "competition_type", "game_id", "schedule_slug", "game_slug",
        "game_number", "week", "game_date_utc", "team_id", "team_name",
        "opponent_team_id", "opponent_team_name", "is_home"
    ]

    for c in base_cols:
        ctx[c] = merged[c] if c in merged.columns else pd.NA

    # -----------------------------
    # Team/offensive values
    # -----------------------------
    team_value_map = {
        "team_scores": "scores",
        "team_goals": "goals",
        "team_one_point_goals": "one_point_goals",
        "team_two_point_goals": "two_point_goals",
        "team_assists": "assists",
        "team_shots": "shots",
        "team_shots_on_goal": "shots_on_goal",
        "team_ground_balls": "ground_balls",
        "team_turnovers": "turnovers",
        "caused_turnovers_for": "caused_turnovers",
        "team_faceoffs": "faceoffs",
        "team_faceoffs_won": "faceoffs_won",
        "team_faceoffs_lost": "faceoffs_lost",
        "saves_for": "saves",
        "team_clears": "clears",
        "team_clear_attempts": "clear_attempts",
        "team_touches": "touches",
        "team_total_passes": "total_passes",
        "team_time_in_possession": "time_in_possession",
        "team_offensive_sequence_proxy": "offensive_sequence_proxy",
    }

    for new_col, old_col in team_value_map.items():
        ctx[new_col] = pd.to_numeric(merged[old_col], errors="coerce") if old_col in merged.columns else np.nan

    # -----------------------------
    # Opponent/offense allowed values
    # -----------------------------
    opponent_value_map = {
        "scores_allowed": "scores_opp_row",
        "goals_allowed": "goals_opp_row",
        "one_point_goals_allowed": "one_point_goals_opp_row",
        "two_point_goals_allowed": "two_point_goals_opp_row",
        "assists_allowed": "assists_opp_row",
        "opponent_shots": "shots_opp_row",
        "opponent_shots_on_goal": "shots_on_goal_opp_row",
        "opponent_two_point_shots": "two_point_shots_opp_row",
        "opponent_two_point_shots_on_goal": "two_point_shots_on_goal_opp_row",
        "opponent_ground_balls": "ground_balls_opp_row",
        "opponent_turnovers": "turnovers_opp_row",
        "opponent_caused_turnovers": "caused_turnovers_opp_row",
        "opponent_faceoffs": "faceoffs_opp_row",
        "opponent_faceoffs_won": "faceoffs_won_opp_row",
        "opponent_faceoffs_lost": "faceoffs_lost_opp_row",
        "opponent_saves": "saves_opp_row",
        "opponent_clears": "clears_opp_row",
        "opponent_clear_attempts": "clear_attempts_opp_row",
        "opponent_touches": "touches_opp_row",
        "opponent_total_passes": "total_passes_opp_row",
        "opponent_time_in_possession": "time_in_possession_opp_row",
        "opponent_offensive_sequence_proxy": "offensive_sequence_proxy_opp_row",
    }

    for new_col, old_col in opponent_value_map.items():
        ctx[new_col] = pd.to_numeric(merged[old_col], errors="coerce") if old_col in merged.columns else np.nan

    # Explicit opponent checks used by QC.
    ctx["opponent_scores_check"] = ctx["scores_allowed"]
    ctx["opponent_goals_check"] = ctx["goals_allowed"]

    # -----------------------------
    # Game-level defensive metrics
    # -----------------------------
    ctx["score_margin"] = ctx["team_scores"] - ctx["scores_allowed"]
    ctx["win_flag"] = np.where(ctx["score_margin"] > 0, 1, np.where(ctx["score_margin"] < 0, 0, np.nan))
    ctx["loss_flag"] = np.where(ctx["score_margin"] < 0, 1, np.where(ctx["score_margin"] > 0, 0, np.nan))

    ctx["opponent_goal_pct"] = safe_divide(ctx["goals_allowed"], ctx["opponent_shots"])
    ctx["opponent_sog_rate"] = safe_divide(ctx["opponent_shots_on_goal"], ctx["opponent_shots"])
    ctx["opponent_sog_goal_pct"] = safe_divide(ctx["goals_allowed"], ctx["opponent_shots_on_goal"])

    # Save percentage proxy uses goals allowed, not PLL "scores" allowed, because 2PT goals count as one goal but two score units.
    ctx["save_pct_proxy"] = safe_divide(ctx["saves_for"], ctx["saves_for"] + ctx["goals_allowed"])

    ctx["ct_per_opponent_turnover"] = safe_divide(ctx["caused_turnovers_for"], ctx["opponent_turnovers"])
    ctx["opponent_scores_per_offensive_sequence_proxy"] = safe_divide(
        ctx["scores_allowed"],
        ctx["opponent_offensive_sequence_proxy"]
    )
    ctx["opponent_goals_per_shot"] = safe_divide(ctx["goals_allowed"], ctx["opponent_shots"])

    return ctx.sort_values(["season", "game_number", "team_name"]).reset_index(drop=True)


TEAM_DEFENSE_SUM_COLS = [
    "team_scores", "scores_allowed", "team_goals", "goals_allowed",
    "team_one_point_goals", "one_point_goals_allowed",
    "team_two_point_goals", "two_point_goals_allowed",
    "team_assists", "assists_allowed",
    "team_shots", "opponent_shots",
    "team_shots_on_goal", "opponent_shots_on_goal",
    "opponent_two_point_shots", "opponent_two_point_shots_on_goal",
    "team_ground_balls", "opponent_ground_balls",
    "team_turnovers", "opponent_turnovers",
    "caused_turnovers_for", "opponent_caused_turnovers",
    "team_faceoffs", "opponent_faceoffs",
    "team_faceoffs_won", "opponent_faceoffs_won",
    "team_faceoffs_lost", "opponent_faceoffs_lost",
    "saves_for", "opponent_saves",
    "team_clears", "opponent_clears",
    "team_clear_attempts", "opponent_clear_attempts",
    "team_touches", "opponent_touches",
    "team_total_passes", "opponent_total_passes",
    "team_time_in_possession", "opponent_time_in_possession",
    "team_offensive_sequence_proxy", "opponent_offensive_sequence_proxy",
    "score_margin",
]


def add_team_defense_rate_columns(df):
    out = df.copy()

    if "games" in out.columns:
        for c in TEAM_DEFENSE_SUM_COLS:
            if c in out.columns:
                out[f"{c}_per_game"] = safe_divide(out[c], out["games"])

    out["win_pct"] = safe_divide(out["wins"], out["games"]) if {"wins", "games"}.issubset(out.columns) else np.nan

    out["opponent_goal_pct"] = safe_divide(out["goals_allowed"], out["opponent_shots"])
    out["opponent_sog_rate"] = safe_divide(out["opponent_shots_on_goal"], out["opponent_shots"])
    out["opponent_sog_goal_pct"] = safe_divide(out["goals_allowed"], out["opponent_shots_on_goal"])
    out["save_pct_proxy"] = safe_divide(out["saves_for"], out["saves_for"] + out["goals_allowed"])
    out["ct_per_opponent_turnover"] = safe_divide(out["caused_turnovers_for"], out["opponent_turnovers"])
    out["opponent_scores_per_offensive_sequence_proxy"] = safe_divide(
        out["scores_allowed"],
        out["opponent_offensive_sequence_proxy"]
    )
    out["opponent_goals_per_shot"] = safe_divide(out["goals_allowed"], out["opponent_shots"])

    return out


def build_team_defense_agg(context_df, group_cols):
    if context_df is None or len(context_df) == 0:
        return pd.DataFrame()

    sum_cols = [c for c in TEAM_DEFENSE_SUM_COLS if c in context_df.columns]

    agg_dict = {
        "games": ("game_id", "nunique"),
        "first_game_date": ("game_date_utc", "min"),
        "last_game_date": ("game_date_utc", "max"),
        "wins": ("win_flag", "sum"),
        "losses": ("loss_flag", "sum"),
    }

    for c in sum_cols:
        agg_dict[c] = (c, "sum")

    out = (
        context_df
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    out = add_team_defense_rate_columns(out)

    sort_cols = [c for c in ["season", "scores_allowed_per_game", "team_name"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[True, True, True][:len(sort_cols)]).reset_index(drop=True)

    return out


team_game_opponent_context = build_team_game_opponent_context(team_game_stats)

team_defense_season_stats = build_team_defense_agg(
    team_game_opponent_context,
    ["season", "team_id", "team_name"]
)

team_defense_career_stats = build_team_defense_agg(
    team_game_opponent_context,
    ["team_id", "team_name"]
)

# -----------------------------
# Defensive/opponent QC
# -----------------------------
def add_def_qc(check_name, status, actual=None, expected=None, notes=None):
    return {
        "check_name": check_name,
        "status": status,
        "actual": actual,
        "expected": expected,
        "notes": notes,
    }


defensive_qc_rows = []

expected_context_rows = len(team_game_stats)
actual_context_rows = len(team_game_opponent_context)

defensive_qc_rows.append(add_def_qc(
    "context_rows_match_team_game_rows",
    "pass" if actual_context_rows == expected_context_rows else "fail",
    actual_context_rows,
    expected_context_rows,
    "Opponent context should have one row per team-game."
))

defensive_qc_rows.append(add_def_qc(
    "season_defense_rows",
    "info",
    len(team_defense_season_stats),
    None,
    "One row per season/team with completed stat games."
))

defensive_qc_rows.append(add_def_qc(
    "career_defense_rows",
    "info",
    len(team_defense_career_stats),
    None,
    "One row per team across all completed stat games."
))

missing_opponent_join_rows = max(expected_context_rows - actual_context_rows, 0)

defensive_qc_rows.append(add_def_qc(
    "missing_opponent_join_rows",
    "pass" if missing_opponent_join_rows == 0 else "fail",
    missing_opponent_join_rows,
    0,
    "Every team-game row should find opponent row from same game."
))

duplicate_context_keys = (
    team_game_opponent_context.duplicated(subset=["game_id", "team_id"], keep=False).sum()
    if len(team_game_opponent_context) > 0 and {"game_id", "team_id"}.issubset(team_game_opponent_context.columns)
    else 0
)

defensive_qc_rows.append(add_def_qc(
    "duplicate_team_game_context_keys",
    "pass" if duplicate_context_keys == 0 else "fail",
    int(duplicate_context_keys),
    0,
    "No duplicate game_id/team_id rows in opponent context."
))

context_rows_per_game = (
    team_game_opponent_context.groupby("game_id").size().reset_index(name="rows")
    if len(team_game_opponent_context) > 0
    else pd.DataFrame(columns=["game_id", "rows"])
)

bad_context_game_count = int((context_rows_per_game["rows"] != 2).sum()) if len(context_rows_per_game) > 0 else 0

defensive_qc_rows.append(add_def_qc(
    "exactly_two_context_rows_per_game",
    "pass" if bad_context_game_count == 0 else "fail",
    bad_context_game_count,
    0,
    "Each completed/stat-available game should have two context rows."
))

if len(team_game_opponent_context) > 0:
    bad_scores_allowed = team_game_opponent_context[
        team_game_opponent_context["scores_allowed"].notna()
        & team_game_opponent_context["opponent_scores_check"].notna()
        & ((team_game_opponent_context["scores_allowed"] - team_game_opponent_context["opponent_scores_check"]).abs() > 0.001)
    ].copy()

    bad_goals_allowed = team_game_opponent_context[
        team_game_opponent_context["goals_allowed"].notna()
        & team_game_opponent_context["opponent_goals_check"].notna()
        & ((team_game_opponent_context["goals_allowed"] - team_game_opponent_context["opponent_goals_check"]).abs() > 0.001)
    ].copy()

else:
    bad_scores_allowed = pd.DataFrame()
    bad_goals_allowed = pd.DataFrame()

defensive_qc_rows.append(add_def_qc(
    "scores_allowed_matches_opponent_scores",
    "pass" if len(bad_scores_allowed) == 0 else "fail",
    len(bad_scores_allowed),
    0,
    "scores_allowed should equal opponent team scores."
))

defensive_qc_rows.append(add_def_qc(
    "goals_allowed_matches_opponent_goals",
    "pass" if len(bad_goals_allowed) == 0 else "fail",
    len(bad_goals_allowed),
    0,
    "goals_allowed should equal opponent team goals."
))

season_coverage_created = (
    team_defense_season_stats["season"].nunique()
    if len(team_defense_season_stats) > 0 and "season" in team_defense_season_stats.columns
    else 0
)

defensive_qc_rows.append(add_def_qc(
    "season_coverage_created",
    "info",
    int(season_coverage_created),
    None,
    "Number of seasons with defensive/opponent data."
))

defensive_opponent_build_quality = pd.DataFrame(defensive_qc_rows)

# Save early QC copies to run directory.
team_game_opponent_context.to_csv(RUN_CHECK_DIR / "team_game_opponent_context_preview.csv", index=False)
team_defense_season_stats.to_csv(RUN_CHECK_DIR / "team_defense_season_stats_preview.csv", index=False)
team_defense_career_stats.to_csv(RUN_CHECK_DIR / "team_defense_career_stats_preview.csv", index=False)
defensive_opponent_build_quality.to_csv(RUN_CHECK_DIR / "defensive_opponent_build_quality.csv", index=False)

# Append defensive QC into existing quality_summary so Data Quality shows it with the rest.
if "quality_summary" in globals() and isinstance(quality_summary, pd.DataFrame):
    existing_checks = set(quality_summary["check_name"].astype(str)) if "check_name" in quality_summary.columns else set()
    add_rows = defensive_opponent_build_quality[
        ~defensive_opponent_build_quality["check_name"].astype(str).isin(existing_checks)
    ].copy()

    quality_summary = pd.concat([quality_summary, add_rows], ignore_index=True)
    quality_summary.to_csv(RUN_CHECK_DIR / "quality_summary.csv", index=False)

print("Defensive/opponent marts created.")
print("team_game_opponent_context:", team_game_opponent_context.shape)
print("team_defense_season_stats:", team_defense_season_stats.shape)
print("team_defense_career_stats:", team_defense_career_stats.shape)

print("\nDefensive/opponent QC:")
display(defensive_opponent_build_quality)

print("\nBest defensive seasons by Scores Allowed/G:")
display(
    team_defense_season_stats[
        [
            c for c in [
                "season", "team_name", "games", "scores_allowed_per_game",
                "goals_allowed_per_game", "opponent_shots_per_game",
                "opponent_goal_pct", "save_pct_proxy", "caused_turnovers_for_per_game"
            ] if c in team_defense_season_stats.columns
        ]
    ]
    .sort_values(["scores_allowed_per_game", "opponent_goal_pct"], ascending=[True, True])
    .head(25)
)

print("\nRecent team-game opponent context:")
display(
    team_game_opponent_context[
        [
            c for c in [
                "season", "game_number", "game_date_utc", "team_name", "opponent_team_name",
                "team_scores", "scores_allowed", "opponent_shots",
                "opponent_shots_on_goal", "opponent_turnovers",
                "caused_turnovers_for", "save_pct_proxy"
            ] if c in team_game_opponent_context.columns
        ]
    ]
    .sort_values(["season", "game_number"], ascending=[False, False])
    .head(30)
)


# ============================================================
# GITHUB PORT ADD-ON — PLAYER RANKING + TEAM STYLE MARTS
# Mirrors the tested Streamlit pages' expected marts/columns.
# ============================================================

def _rank_pct(series, higher_is_better=True):
    """
    Percentile rank where the best value scores highest.

    `ascending=higher_is_better` is the correct pairing and is not a typo waiting
    to be tidied: pandas' `ascending` orders the *ranks*, so for a higher-is-better
    metric the largest value must sort last to receive the top percentile. This
    line previously read `ascending=not higher_is_better`, which inverted every
    score in the ranking mart — the league's best scorer took a points score of
    0.8 while a 0.25 points-per-game midfielder took 92.2, and overall score
    correlated -0.93 with points per game. Every component score, peer standing
    and percentile column in `player_ranking_profiles` flows through here, so the
    sign of this one comparison decided the whole board.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    return s.rank(pct=True, ascending=higher_is_better, method="average", na_option="keep") * 100


def _sigmoid_stretch(percentile_series):
    """
    Map a 0-100 percentile to an interpretable 0-100 score with spread.

    Uses a logistic curve centered at 50 so the output distribution is
    intuitive:
        99th pct  -> ~99    (historically elite)
        90th pct  -> ~92    (excellent / all-star)
        75th pct  -> ~77    (solid starter)
        50th pct  -> 50     (league average)
        25th pct  -> ~23    (below average)
        10th pct  -> ~8     (low impact)

    k=2.5 gives the best balance: top players reach the high 80s/low 90s
    in overall score while average players sit near 50 and below-average
    players score in the 20s-30s (still readable, not crushingly low).
    """
    p = pd.to_numeric(percentile_series, errors="coerce")
    x = (p / 100.0 - 0.5)
    k = 2.5
    stretched = 1.0 / (1.0 + np.exp(-k * x))
    lo = 1.0 / (1.0 + np.exp(-k * (-0.5)))
    hi = 1.0 / (1.0 + np.exp(-k * 0.5))
    result = (stretched - lo) / (hi - lo) * 100
    return result.clip(0, 100).where(p.notna(), np.nan)


def _normalize_within_role(series, role_group, calibrate_mask=None,
                           center=50.0, spread=32.5, clip_z=4.0):
    """
    Put each role's composite on a common median/IQR scale so roles are comparable.

    A composite's ceiling is set by how redundant its inputs are, not by how good
    its best player is. Averaging three faceoff stats that correlate at r=0.65 is
    close to averaging one stat, so its leader reaches ~99; averaging eight
    offensive stats correlating at r=0.37 is closer to averaging two, so a player
    who is merely average at one of them cannot get there — the best attackman
    tops out around 92. Measured on the 2026 board: Faceoff had 1.30 effective
    independent components against Offense's 2.25, and RPS maxima of 98.9 vs 91.9
    with medians of 52.0 vs 59.9.

    That is an artefact of component count and correlation, and it was putting
    specialists above attackmen on the cross-role board before a game was played.
    It also contradicted the scale the app documents (`shared/scoring.py`: "each on
    a 0-100 scale where 50 is league average for the ranking context"), which
    per-role medians of 52-60 and maxima of 78-99 plainly were not.

    Rescaling on the median and IQR fixes the comparison without touching the
    formula: it is monotonic within a role, so every player's standing among their
    own peers is unchanged (measured Spearman 0.997-1.000), and only the
    cross-role scale moves. The IQR is used rather than the standard deviation for
    the same reason the sigmoid path does — one outlier season should not stretch
    the scale for everyone else.

    `calibrate_mask` selects the players whose median and IQR set the scale
    (pass the games-threshold mask): a role containing four one-game players would
    otherwise let them define what "average" means. Their scores are still
    computed, they just do not get a vote. `clip_z` bounds how far a genuine
    outlier can travel, so a 1-of-1 faceoff man cannot land at 400.

    `spread` is not a free tuning knob, and the role mix of the resulting top 25 is
    the wrong thing to tune it against — picking the number that produces a
    pleasing answer is how the artefact got here. It is fixed by the scale the
    other two components already use. `_sigmoid_stretch` maps the 90th percentile
    to 91.7, and the 90th percentile of a normal distribution is 1.282 standard
    deviations above the median, so that curve is worth (91.7 - 50) / 1.282 = 32.5
    points per standard deviation. Matching it keeps a weighted average honest: in
    a blend, a component's real influence is its weight times its spread, so
    scoring RPS at 12.5 per SD against a PSS at ~35 would give the nominal 0.60 RPS
    weight less pull than the nominal 0.25 PSS weight.
    """
    s = pd.to_numeric(series, errors="coerce")
    roles_ = pd.Series(role_group, index=s.index).astype("object")
    if calibrate_mask is None:
        calibrate_mask = pd.Series(True, index=s.index)
    calibrate_mask = pd.Series(calibrate_mask, index=s.index).fillna(False).astype(bool)

    out = pd.Series(np.nan, index=s.index, dtype="float64")
    for _, idx in roles_.groupby(roles_, dropna=False).groups.items():
        idx = list(idx)
        vals = s.loc[idx]
        basis = vals[calibrate_mask.loc[idx]].dropna()
        # Too few qualified players to characterise a distribution: fall back to
        # the whole role rather than calibrating on two rows.
        if len(basis) < 5:
            basis = vals.dropna()
        if len(basis) < 2:
            out.loc[idx] = vals
            continue

        med = float(basis.median())
        q1, q3 = float(basis.quantile(0.25)), float(basis.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or np.isclose(iqr, 0.0):
            out.loc[idx] = vals
            continue

        # 1.349 IQRs span one standard deviation for a normal distribution, so
        # this reads as a z-score without inheriting the SD's outlier sensitivity.
        z = ((vals - med) / (iqr / 1.349)).clip(-clip_z, clip_z)

        # Logistic rather than `(center + spread * z).clip(0, 100)`. A linear map
        # at this spread sends the top of a wide-tailed role past 100, and the clip
        # then ties everyone above the ceiling — which silently destroys the order
        # this function exists to preserve (it dropped within-role Spearman to
        # 0.9991). The logistic is bounded and strictly increasing, so no player
        # ever ties another and the top of a role compresses smoothly instead of
        # hitting a wall. k = 4 * spread / 100 matches the linear slope at the
        # median, so the scale still reads as `spread` points per SD in the middle,
        # where nearly everyone sits.
        k = 4.0 * spread / 100.0
        out.loc[idx] = 100.0 / (1.0 + np.exp(-k * z))

    return out.where(s.notna(), np.nan)


# Games at which a composite score is trusted at (roughly) full weight.
#
# Measured, not chosen. Scoring each player on their first k games of a season and
# regressing that against a score built on their *remaining* games gives the
# empirically optimal shrink factor directly — it is the slope of that regression,
# because regression to the mean is exactly what the slope measures:
#
#     k games   out-of-sample r   optimal shrink
#        2           0.537            0.556
#        3           0.592            0.616
#        4           0.598            0.605
#        5           0.627            0.630
#        6           0.622            0.616
#
# Two lessons. First, the curve plateaus near 0.62 rather than climbing to 1.0 —
# a full-season composite is still only ~62% signal, so shrinking a 10-game score
# hard would be over-correcting for noise that a 10-game sample does not have.
# Second, the *tails* are where small samples actually lie. Splitting first-5-game
# scores into bands and looking at where each band lands over the rest of the year:
#
#     first-5 band   n     mean first 5   mean rest
#        85+         31        90.3          78.6
#        75-85       57        80.3          69.7
#        60-75       69        66.8          57.8
#        <60        274        35.3          41.1
#
# An 85+ on five games is a ~79 the rest of the way; a sub-60 is a 41. Both tails
# collapse inward by 10-12 points while the sample is thin. So the correction
# belongs on the deviation from average, which is what shrinkage does, and it
# should be substantial at 1-3 games and nearly absent by 8-10.
#
# GAMES_FOR_FULL_TRUST = 8 with a floor of 0.35 reproduces that: 1 game keeps 35%
# of the deviation, 3 games 52%, 5 games 71%, 8+ games 100%. Deliberately NOT
# shrinking a full-season score toward 50 — the plateau above says the 10-game
# number is as good as this data gets, and the context median shift already
# handles the absolute level.
GAMES_FOR_FULL_TRUST = 8
MIN_SAMPLE_TRUST = 0.35


def _sample_trust(games, full_trust_games=GAMES_FOR_FULL_TRUST,
                  floor=MIN_SAMPLE_TRUST):
    """
    Fraction of a player's deviation from average to keep, given their game count.

    Linear from `floor` at one game to 1.0 at `full_trust_games`, then flat. See
    GAMES_FOR_FULL_TRUST for the measurements behind both constants.
    """
    g = pd.to_numeric(games, errors="coerce").fillna(0).clip(lower=0)
    ramp = floor + (1.0 - floor) * (g / float(full_trust_games)).clip(0, 1)
    return ramp.clip(floor, 1.0)


def _shrink_to_average(series, games, center=50.0,
                       full_trust_games=GAMES_FOR_FULL_TRUST,
                       floor=MIN_SAMPLE_TRUST):
    """
    Pull a 0-100 score toward `center` in proportion to how few games back it.

    A six-game career and a sixty-game career produced the same score before this:
    corr(games, overall_score) was 0.181 and seven of the top 25 had fewer than
    seven games, including the all-time #1 on six career games. This does not
    penalise a short career — it declines to claim a six-game player is the best
    in the league, which is a different statement.
    """
    s = pd.to_numeric(series, errors="coerce")
    trust = _sample_trust(games, full_trust_games, floor).reindex(s.index)
    return (center + trust * (s - center)).clip(0, 100).where(s.notna(), np.nan)


def _reliable_rate(numerator, denominator, min_denominator, prior=None,
                   index=None):
    """
    Rate stat with a denominator-scaled prior — an empirical-Bayes shrunk rate.

    Raw ratios on tiny denominators are noise wearing a precise-looking number.
    Measured split-half reliability of the rates this system scores, alongside the
    median denominator each one actually gets:

        rate                split-half r    median denom
        faceoff_pct             0.757            large
        clean_save_rate         0.625            large
        points_per_touch        0.470             37
        turnovers_per_touch     0.192             37
        sog_rate                0.054              4
        two_pt_conversion       0.037              1
        assist_conv_rate        0.017              2
        shot_pct               -0.038              4

    The bottom four are indistinguishable from random: 57% of player-seasons have
    fewer than 10 shots, 74% fewer than 10 assist opportunities, 95% fewer than 10
    two-point shots. A 1-for-1 two-point shooter scored a perfect two-point
    conversion. Adding `min_denominator` phantom attempts at the league rate makes
    a player earn their way off the average — one make in one attempt lands just
    above league average instead of at the ceiling, while a 40-shot season is
    barely moved.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    if index is None:
        index = num.index
    num = num.reindex(index)
    den = den.reindex(index)

    valid = num.notna() & den.notna() & den.gt(0)
    if prior is None:
        # League rate from the pooled totals, which is the correct prior for a
        # ratio: it weights players by how much they actually attempted.
        total_den = float(den[valid].sum())
        prior = float(num[valid].sum()) / total_den if total_den > 0 else np.nan

    if not np.isfinite(prior):
        return pd.Series(np.nan, index=index, dtype="float64")

    m = float(min_denominator)
    shrunk = (num + m * prior) / (den + m)
    return shrunk.where(valid, np.nan)


# Two-way credit: the most a player can add to their RPS for genuine production
# on the side of the ball their role score ignores, and the gate they must clear.
#
# Sizing comes from the overlap in the data. Per-game means by listed position
# (2022+, 5+ games) show the two axes are close to disjoint:
#
#     pos    n    points/g   CT/g   GB/g
#     A    131      2.77     0.16   1.39
#     M    201      1.57     0.12   0.97
#     SSDM 132      0.30     0.41   1.25
#     LSM   75      0.27     0.75   2.14
#     D    135      0.11     0.86   1.77
#
# Only 14% of midfield seasons clear the SSDM median on defensive production, and
# *zero* SSDM seasons clear the midfield median on points. So this cannot be a
# symmetric cross-role comparison — judged against midfielders, no defender would
# ever earn offensive credit and that half of the feature would be dead on arrival.
#
# Each player is therefore scored within their own role (an SSDM's offence measured
# against other SSDMs, where the spread is real: median 0.20 points/g, p90 0.60,
# max 1.33) and separately *gated* on absolute production against the pool that
# owns the axis. The within-role scoring is what makes the credit reachable; the
# absolute gate is what stops the best-of-a-weak-pool from qualifying on noise.
#
# THE GATES ARE ASYMMETRIC because the two distributions are not mirror images. A
# single shared percentile cannot serve both directions — at the 40th percentile,
# 14% of offensive players clear the defensive gate while only 0.6% of defenders
# clear the offensive one. Calibrated instead to equal selectivity, ~4% each way,
# measured over all 2022+ player-seasons with 5+ games:
#
#     defender earning offensive credit:  >= 0.89 points/g  (Offense pool p10)
#                                         -> 13 of 342 defender-seasons qualify
#     offensive earning defensive credit: >= 0.50 CT/g      (Defense pool p40)
#                                         -> 13 of 332 offensive-seasons qualify
#
# THE GATES ARE ABSOLUTE CONSTANTS, not percentiles recomputed per context. This
# matters: computing them per context made the bar move with the pool, and the
# thinner contexts have softer pools. The Offense p10 of points/game came out at
# 0.50 in Career, 0.29 in the 2026 season and 0.20 in Last 5, so the share of
# defenders qualifying drifted from 10% to 19% to 38% — the same player earned
# two-way credit or not depending on which view you were looking at, and in Last 5
# more than a third of all defenders were "two-way". Per-game rates are directly
# comparable across contexts, so a fixed threshold is both simpler and correct.
#
# The defensive axis is caused turnovers ALONE, not caused turnovers plus ground
# balls. Ground balls are already credited to every role by Cross-Role Impact, so
# including them here would pay an offensive player twice for one recovery, and CT
# is also the stat that actually separates defenders (standardized beta +0.219 on
# score margin against +0.121 for ground balls once faceoff men are excluded).
#
# The players this finds: Zach Currier (M, 0.75 CT/g in 2026, six times the
# midfield median) whose Offense RPS counted none of his defending; Jeff Trainor
# and Ian MacKay (SSDM, 1.0-1.3 points/g, five times the SSDM median) whose Defense
# RPS counted none of their offence. And who it now correctly excludes: Ryan
# Terefenko (0.57 points/g, 0 caused turnovers) and Aidan Maguire (0.75 points/g),
# both of whom cleared a drifting per-context gate on production that is ordinary
# for the position.
#
# Capped at 6 points because it is a supplement to a role score, not a second role
# score. 6 points is about a third of the gap between the median and the top of a
# role — enough to move a genuine two-way player up meaningfully, not enough to let
# defensive production outrank being an elite attackman.
TWO_WAY_MAX_CREDIT = 6.0
TWO_WAY_GATE_POINTS_PER_GAME = 0.89   # defender must reach this to earn offensive credit
TWO_WAY_GATE_CT_PER_GAME = 0.50       # attacker must reach this to earn defensive credit

# Two-way play is a claim about season-long deployment, so it needs a real sample —
# more than the ranking board's own games threshold, which is deliberately permissive
# (4 games in 2026) so that fringe players still appear somewhere on the list.
# A share of the context's own length rather than a flat number, because the contexts
# are different lengths: 70% is 7 of 10 games in a season view and 4 of 5 in Last 5.
# Capped at 10 so the Career context, where the longest career sets the maximum, does
# not demand 30-odd games and exclude everyone still playing.
TWO_WAY_MIN_GAMES_SHARE = 0.70
TWO_WAY_MIN_GAMES_CAP = 10


def _two_way_min_games(max_games_in_context):
    """Games a player needs before two-way credit is available to them."""
    m = float(max_games_in_context or 0)
    if m <= 0:
        return 0
    return int(min(np.ceil(TWO_WAY_MIN_GAMES_SHARE * m), TWO_WAY_MIN_GAMES_CAP))


def _two_way_credit(secondary_score, secondary_raw, games, gate, eligible=None,
                    max_credit=TWO_WAY_MAX_CREDIT):
    """
    Bounded, additive credit for real production on a player's secondary axis.

    `secondary_score` is the player's 0-100 standing on that axis *within their own
    role*; `secondary_raw` is their absolute per-game production on it; `gate` is
    the fixed absolute threshold they must reach. Returns 0 for anyone who fails the
    gate, and is shrunk by game count so a one-game outlier cannot buy the bonus.

    Additive rather than blended into the role weights on purpose: a two-way
    midfielder should not have their offensive score diluted to make room for
    defensive credit. Playing both ways is extra value, so it is scored as extra.
    """
    sec = pd.to_numeric(secondary_score, errors="coerce")
    raw = pd.to_numeric(secondary_raw, errors="coerce")

    if sec.notna().sum() == 0 or not np.isfinite(gate):
        return pd.Series(0.0, index=sec.index, dtype="float64")

    # Size the credit on within-role standing, with the absolute gate as a pass/fail
    # requirement rather than a second sliding scale.
    #
    # An earlier version also ramped the credit by how far past the gate a player
    # was, as a multiple of the gate. That silently capped the defensive half of the
    # feature: a defender needs 0.89 points/game to qualify, so full credit would
    # have required 1.78 — and the highest any defender has ever managed is 1.33. No
    # defender could ever have earned more than half credit, which is exactly the
    # asymmetry the two separate gates exist to correct. It also inverted the ordering
    # it was meant to refine, paying a 4-game midfielder more than the league's
    # actual two-way midfielder because a short sample produces a higher rate.
    above = ((sec - 50.0) / 50.0).clip(0, 1).fillna(0.0)

    credit = max_credit * above
    credit = credit.where(raw.fillna(-np.inf) >= float(gate), 0.0)

    # Two-way play is a claim about how a player is deployed, and three games cannot
    # support it. Without this, the top two-way "midfielders" were Ty English on 4
    # games and Adam Charalambides on 1 — a single game with one caused turnover
    # reads as 1.00 CT/game, double the gate — while Zach Currier, the actual
    # two-way midfielder in the league, ranked below them. See _two_way_min_games for
    # the bar; _sample_trust then scales what remains.
    if eligible is not None:
        gate_mask = pd.Series(eligible, index=credit.index).fillna(False).astype(bool)
        credit = credit.where(gate_mask, 0.0)

    trust = _sample_trust(games).reindex(credit.index).fillna(0.0)
    return (credit * trust).fillna(0.0)


def _minmax_score(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(lo, hi):
        return pd.Series(50.0, index=series.index).where(s.notna(), np.nan)
    score = (s - lo) / (hi - lo) * 100
    if not higher_is_better:
        score = 100 - score
    return score.clip(0, 100)


def _robust_z(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=series.index).where(s.notna(), np.nan)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if pd.isna(mad) or np.isclose(mad, 0):
        std = s.std(skipna=True)
        if pd.isna(std) or np.isclose(std, 0):
            return pd.Series(0.0, index=series.index).where(s.notna(), np.nan)
        return ((s - s.mean(skipna=True)) / std).clip(-4, 4)
    return (0.6745 * (s - med) / mad).clip(-4, 4)


def _z_to_score(z):
    return (50 + 12.5 * pd.to_numeric(z, errors="coerce")).clip(0, 100)


def _value_tier_from_z(z):
    """
    Legacy fallback tiering by peer-separation z-score only.

    The official role tier below uses a fuller role-context profile. This
    fallback remains for safety if a row-level tier cannot be calculated.
    """
    try:
        z = float(z)
    except Exception:
        return "Unrated"
    if pd.isna(z):
        return "Unrated"
    if z >= 1.75:
        return "Outlier Elite"
    if z >= 0.95:
        return "Elite"
    if z >= 0.45:
        return "High-End"
    if z >= -0.35:
        return "Average / Starter"
    if z >= -1.00:
        return "Below Average"
    return "Low Impact"


def _official_role_value_tier(row):
    """
    Assign the final role tier using the same language used by the Streamlit
    page. This avoids the previous issue where tiers were driven only by
    adjusted z-score, which made it too hard for anyone to be labeled Elite
    in small contexts.

    The tier now considers:
      - role_context_value_score: blended role value;
      - role_primary_percentile: rank within role peers;
      - role_adjusted_z: actual separation from role peers.
    """
    try:
        score = float(row.get("role_context_value_score", np.nan))
    except Exception:
        score = np.nan

    try:
        pct = float(row.get("role_primary_percentile", np.nan))
    except Exception:
        pct = np.nan

    try:
        z = float(row.get("role_adjusted_z", np.nan))
    except Exception:
        z = np.nan

    if pd.isna(score):
        return _value_tier_from_z(z)

    # Truly elite separator: great blended role score plus either top percentile
    # or clear peer separation.
    if score >= 90 or (score >= 84 and pct >= 95 and z >= 0.85):
        return "Outlier Elite"

    # Elite role value: top role performers without requiring extreme z-scores
    # during small-sample season contexts.
    if score >= 78 and (pd.isna(pct) or pct >= 80):
        return "Elite"

    if score >= 65 or (not pd.isna(pct) and pct >= 70) or (not pd.isna(z) and z >= 0.45):
        return "High-End"

    if score >= 48:
        return "Average / Starter"

    if score >= 34:
        return "Below Average"

    return "Low Impact"


def _label_from_score(score, labels):
    try:
        s = float(score)
    except Exception:
        return "Unrated"
    if pd.isna(s):
        return "Unrated"
    if s >= 80:
        return labels[0]
    if s >= 65:
        return labels[1]
    if s >= 45:
        return labels[2]
    if s >= 30:
        return labels[3]
    return labels[4]


def _ensure_cols(df, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ranking_context_min_games(context_type, max_games):
    if context_type in {"Last 5", "Last 10"}:
        return 1
    if context_type == "Career":
        return 5 if max_games >= 5 else 2
    if max_games <= 3:
        return 2
    if max_games <= 6:
        return 3
    if max_games <= 10:
        return 4
    return 5




def _score_metric(series, higher_is_better=True, percentile_weight=0.35):
    """
    Context-local score for a raw metric.

    Uses pure percentile rank (0-100) passed through a sigmoid stretch so the
    output is interpretable: ~50 = league average, ~85 = 90th percentile,
    ~95 = 99th percentile. The percentile_weight parameter is retained for
    API compatibility but no longer used (pure percentile is always applied).
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    pct = _rank_pct(s, higher_is_better=higher_is_better)
    return _sigmoid_stretch(pct).where(s.notna(), np.nan)


def _weighted_available_score(df, weights, fallback=np.nan, neutral=None):
    """
    Weighted row score tolerant of missing component columns/values.

    `neutral` decides what a missing component means. With `neutral=None` (the
    default) the remaining weights are renormalized, which is right when the
    component is missing because it is *inapplicable* — a goalie has no faceoff
    percentage, and pretending it is average would be worse than ignoring it.

    Pass `neutral=50.0` when a missing component instead means the player did not
    do the thing, because renormalizing then hands out a silent bonus. Measured on
    the current mart: `two_pt_conv_score` is absent for 41% of offensive rows and
    `assist_conv_score` for 7%. A player with no two-point attempts had their other
    seven weights scaled up by 1/0.98, so *not attempting* was scored identically to
    attempting at exactly their own average — and strictly better than attempting
    and missing. Substituting 50 makes a non-attempt read as "no evidence, assume
    average", which is what it is.

    If every component is missing, returns `fallback` (default NaN).
    """
    score = pd.Series(0.0, index=df.index, dtype="float64")
    weight_sum = pd.Series(0.0, index=df.index, dtype="float64")
    any_valid = pd.Series(False, index=df.index)

    for col, weight in weights.items():
        if isinstance(col, pd.Series):
            vals = pd.to_numeric(col, errors="coerce")
        elif col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
        else:
            continue

        valid = vals.notna()
        any_valid |= valid

        if neutral is not None:
            # Every listed component keeps its full weight; absent values take the
            # neutral level rather than redistributing their weight to the others.
            filled = vals.fillna(float(neutral))
            score += filled * float(weight)
            weight_sum += float(weight)
        elif valid.any():
            score.loc[valid] += vals.loc[valid] * float(weight)
            weight_sum.loc[valid] += float(weight)

    out = score / weight_sum.replace(0, np.nan)
    if neutral is not None:
        # A row with no component at all present is unscored, not average.
        out = out.where(any_valid, np.nan)

    if fallback is not None and not (isinstance(fallback, float) and pd.isna(fallback)):
        out = out.fillna(float(fallback))

    return out.clip(0, 100)


def _role_metric_score(out, col, higher_is_better=True, role_col="role_group", percentile_weight=0.35):
    """Score a metric inside role groups for secondary peer context."""
    if col not in out.columns:
        return pd.Series(np.nan, index=out.index, dtype="float64")

    s = pd.to_numeric(out[col], errors="coerce")
    result = pd.Series(np.nan, index=out.index, dtype="float64")

    for _, idx in out.groupby(role_col, dropna=False).groups.items():
        idx = list(idx)
        result.loc[idx] = _score_metric(
            s.loc[idx],
            higher_is_better=higher_is_better,
            percentile_weight=percentile_weight,
        ).values

    return result


def _add_test_style_role_separation(out):
    """
    Add the role percentile and peer-separation fields using the same structure
    as the final Colab testing app:

    robust_scale = IQR / 1.349, with standard deviation fallback
    role_reliability = min(role_group_size, 8) / 8
    role_separation_score = 50 + role_reliability * ((50 + 12.5*z) - 50)
    """
    out = out.copy()

    out["role_primary_percentile"] = np.nan
    out["role_robust_z"] = np.nan
    out["role_adjusted_z"] = np.nan
    out["role_separation_score"] = np.nan
    out["role_group_size"] = np.nan
    out["role_reliability"] = np.nan

    for _, idx in out.groupby("role_group", dropna=False).groups.items():
        idx = list(idx)
        role_scores = pd.to_numeric(out.loc[idx, "role_primary_score"], errors="coerce")
        valid = role_scores.dropna()
        n = int(valid.shape[0])

        out.loc[idx, "role_group_size"] = n

        if n == 0:
            out.loc[idx, "role_primary_percentile"] = np.nan
            out.loc[idx, "role_robust_z"] = 0.0
            out.loc[idx, "role_adjusted_z"] = 0.0
            out.loc[idx, "role_separation_score"] = 50.0
            out.loc[idx, "role_reliability"] = 0.0
            continue

        percentile = _rank_pct(role_scores, True)

        median = valid.median()
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        robust_scale = iqr / 1.349 if pd.notna(iqr) and iqr > 0 else np.nan
        std_scale = valid.std(ddof=0)

        if pd.notna(robust_scale) and robust_scale > 1e-9:
            scale = robust_scale
        elif pd.notna(std_scale) and std_scale > 1e-9:
            scale = std_scale
        else:
            scale = np.nan

        if pd.isna(scale):
            z = pd.Series(0.0, index=role_scores.index, dtype="float64")
        else:
            z = ((role_scores - median) / scale).clip(-4, 4).fillna(0.0)

        reliability = min(n, 8) / 8.0
        raw_sep = (50.0 + 12.5 * z).clip(0, 100)
        sep = (50.0 + reliability * (raw_sep - 50.0)).clip(0, 100)

        out.loc[idx, "role_primary_percentile"] = percentile.values
        out.loc[idx, "role_robust_z"] = z.values
        out.loc[idx, "role_adjusted_z"] = (z * reliability).values
        out.loc[idx, "role_separation_score"] = sep.values
        out.loc[idx, "role_reliability"] = reliability * 100.0

    out["role_value_tier"] = out.apply(_official_role_value_tier, axis=1)
    return out


def _add_player_ranking_scores(df):
    """
    Build the official player ranking mart using the final architecture but
    closer to the Colab-tested math:

    - warehouse calculates rankings once;
    - app only displays rankings;
    - no experimental version labels;
    - component scores are context-local and mostly min-max based like the
      tested scoring scale;
    - usage blends global usage with role-peer usage so offensive players are
      not crushed and specialists are not artificially boosted;
    - role context uses role score + role percentile + IQR-based peer separation;
    - scoring_value_score captures direct scoring value;
    - playmaking_value_score captures assist/creation value;
    - overall_score is the final official displayed rating.
    """
    out = df.copy()

    needed = [
        "points_per_game", "scoring_points_per_game", "one_point_goals_per_game",
        "two_point_goals_per_game", "goals_per_game", "assists_per_game",
        "shots_per_game", "shots_on_goal_per_game", "shot_pct_calc",
        "ground_balls_per_game", "turnovers_per_game", "caused_turnovers_per_game",
        "touches_per_game", "total_passes_per_game", "faceoff_pct_calc",
        "faceoffs_per_game", "faceoffs_won_per_game", "faceoffs_lost_per_game",
        "saves_per_game", "goals_against_per_game", "scores_against_per_game",
        "save_pct_calc", "two_point_shots", "two_point_goals", "games",
        # Phase 2 new columns
        "assist_opportunities", "assists", "clean_saves", "messy_saves",
        "two_point_shots", "saves", "goals_against", "clean_save_pct",
    ]
    out = _ensure_cols(out, needed)

    pos = out.get("position", pd.Series("", index=out.index)).astype(str).str.upper().str.strip()
    out["role_group"] = np.select(
        [pos.eq("G"), pos.isin(["FO", "FOS"]), pos.isin(["D", "LSM", "SSDM"])],
        ["Goalie", "Faceoff", "Defense"],
        default="Offense",
    )

    # ------------------------------------------------------------------
    # Derived ratios retained from the tested mart contract.
    # ------------------------------------------------------------------
    touches_pg = pd.to_numeric(out["touches_per_game"], errors="coerce")
    shots_pg = pd.to_numeric(out["shots_per_game"], errors="coerce")

    # Season/career totals, which are the denominators these rates deserve. The
    # per-game forms above divide two averages and so carry no information about
    # how many attempts stood behind them — a 1-for-1 two-point shooter and a
    # 20-for-40 shooter both arrived as a bare ratio.
    _tot = {}
    for name in ("touches", "shots", "shots_on_goal", "goals", "points", "assists",
                 "turnovers", "faceoffs", "faceoffs_won", "assist_opportunities",
                 "two_point_goals", "two_point_shots", "saves", "clean_saves",
                 "goals_against"):
        _tot[name] = pd.to_numeric(
            out.get(name, pd.Series(np.nan, index=out.index)), errors="coerce")

    # Prior strength per rate, in phantom attempts at the league rate. Each is set
    # against that rate's measured reliability — solving n/(n+m) = reliability at
    # the rate's own median denominator gives the m that matches observed noise:
    #
    #   rate              split-half r   median denom   implied m   used
    #   faceoff_pct           0.757           large        ~2         20
    #   clean_save_rate       0.625            89          27         30
    #   points_per_touch      0.470            71          40         40
    #   turnovers_per_touch   0.192            71         150         60
    #   sog_rate              0.054            10          87         20
    #   two_pt_conversion     0.037             2          26         10
    #   assist_conv_rate      0.017             5         145         15
    #   shot_pct             -0.038            10           inf       25
    #
    # Held below the implied value wherever the implied one is enormous. An
    # implied m of infinity says shot percentage carries no repeatable signal at
    # all, and the honest response to that is to stop scoring it rather than to
    # shrink it to a constant — but its weight is small (0.05) and its face
    # validity high, so it is shrunk firmly and left in. faceoff_pct is shrunk at
    # m=20 despite implying ~2: real specialists take 200+ faceoffs a season so it
    # barely moves them, while it stops a midfielder who won his only draw from
    # scoring a perfect faceoff rate.
    RATE_PRIOR = {
        "points_per_touch": 40.0, "assists_per_touch": 40.0,
        "turnovers_per_touch": 60.0, "goals_per_shot": 25.0,
        "sog_rate": 20.0, "faceoff_pct": 20.0, "save_pct": 40.0,
        "clean_save_rate": 30.0, "assist_conv": 15.0, "two_pt_conv": 10.0,
    }

    # The raw ratios stay exactly as they were: the Rankings, Leaderboards and
    # Profiles pages display these columns as the player's actual shot percentage
    # and faceoff percentage, and a shrunk value under a "FO Win %" header would be
    # a lie about what the player did. Shrinkage applies to the `_shrunk` twins
    # below, which only the scorers read. Display shows what happened; scoring uses
    # what is repeatable.
    out["points_per_touch"] = np.where(
        touches_pg > 0,
        pd.to_numeric(out["points_per_game"], errors="coerce") / touches_pg,
        np.nan,
    )
    out["assists_per_touch"] = np.where(
        touches_pg > 0,
        pd.to_numeric(out["assists_per_game"], errors="coerce") / touches_pg,
        np.nan,
    )
    out["turnovers_per_touch"] = np.where(
        touches_pg > 0,
        pd.to_numeric(out["turnovers_per_game"], errors="coerce") / touches_pg,
        np.nan,
    )
    out["goals_per_shot"] = np.where(
        shots_pg > 0,
        pd.to_numeric(out["goals_per_game"], errors="coerce") / shots_pg,
        np.nan,
    )
    out["sog_rate_for_ranking"] = np.where(
        shots_pg > 0,
        pd.to_numeric(out["shots_on_goal_per_game"], errors="coerce") / shots_pg,
        np.nan,
    )
    out["faceoff_pct_for_ranking"] = pd.to_numeric(out["faceoff_pct_calc"], errors="coerce")
    out["save_pct_for_ranking"] = pd.to_numeric(out["save_pct_calc"], errors="coerce")

    out["two_point_goal_pct_calc"] = (
        pd.to_numeric(out["two_point_goals"], errors="coerce")
        / pd.to_numeric(out["two_point_shots"], errors="coerce").replace(0, np.nan)
    )

    # Shrunk twins — scoring inputs only.
    out["points_per_touch_shrunk"] = _reliable_rate(
        _tot["points"], _tot["touches"], RATE_PRIOR["points_per_touch"],
        index=out.index)
    out["assists_per_touch_shrunk"] = _reliable_rate(
        _tot["assists"], _tot["touches"], RATE_PRIOR["assists_per_touch"],
        index=out.index)
    out["turnovers_per_touch_shrunk"] = _reliable_rate(
        _tot["turnovers"], _tot["touches"], RATE_PRIOR["turnovers_per_touch"],
        index=out.index)
    out["shot_pct_shrunk"] = _reliable_rate(
        _tot["goals"], _tot["shots"], RATE_PRIOR["goals_per_shot"], index=out.index)
    out["sog_rate_shrunk"] = _reliable_rate(
        _tot["shots_on_goal"], _tot["shots"], RATE_PRIOR["sog_rate"], index=out.index)
    out["faceoff_pct_shrunk"] = _reliable_rate(
        _tot["faceoffs_won"], _tot["faceoffs"], RATE_PRIOR["faceoff_pct"],
        index=out.index)

    # Shots faced, the denominator for both goalie rates. `saa` is not present in
    # every context, so it is rebuilt from the parts.
    _goalie_faced = _tot["saves"].fillna(0) + _tot["goals_against"].fillna(0)
    _goalie_faced = _goalie_faced.where(_goalie_faced > 0, np.nan)
    out["save_pct_shrunk"] = _reliable_rate(
        _tot["saves"], _goalie_faced, RATE_PRIOR["save_pct"], index=out.index)
    out["two_pt_conversion_shrunk"] = _reliable_rate(
        _tot["two_point_goals"], _tot["two_point_shots"],
        RATE_PRIOR["two_pt_conv"], index=out.index)

    # ------------------------------------------------------------------
    # Phase 2 derived fields.
    # ------------------------------------------------------------------
    # Assist conversion rate: assists / assist_opportunities
    out["assist_conv_rate"] = np.where(
        pd.to_numeric(out.get("assist_opportunities", pd.Series(np.nan, index=out.index)), errors="coerce") > 0,
        pd.to_numeric(out.get("assists", pd.Series(np.nan, index=out.index)), errors="coerce")
        / pd.to_numeric(out.get("assist_opportunities", pd.Series(np.nan, index=out.index)), errors="coerce"),
        np.nan,
    )

    # 2PT shot conversion: two_point_goals / two_point_shots
    out["two_pt_conversion"] = np.where(
        pd.to_numeric(out.get("two_point_shots", pd.Series(np.nan, index=out.index)), errors="coerce") > 0,
        pd.to_numeric(out.get("two_point_goals", pd.Series(np.nan, index=out.index)), errors="coerce")
        / pd.to_numeric(out.get("two_point_shots", pd.Series(np.nan, index=out.index)), errors="coerce"),
        np.nan,
    )

    # Clean save rate: clean_saves / (saves + goals_against)
    _goalie_shots_faced = (
        pd.to_numeric(out.get("saves", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("goals_against", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    )
    out["clean_save_rate"] = np.where(
        _goalie_shots_faced > 0,
        pd.to_numeric(out.get("clean_saves", pd.Series(np.nan, index=out.index)), errors="coerce") / _goalie_shots_faced,
        np.nan,
    )

    # Shrunk twins for the two remaining scored rates.
    out["assist_conv_rate_shrunk"] = _reliable_rate(
        _tot["assists"], _tot["assist_opportunities"], RATE_PRIOR["assist_conv"],
        index=out.index)
    out["clean_save_rate_shrunk"] = _reliable_rate(
        _tot["clean_saves"], _goalie_faced, RATE_PRIOR["clean_save_rate"],
        index=out.index)

    # Assist opportunities per game
    out["assist_opp_per_game"] = np.where(
        pd.to_numeric(out.get("games", pd.Series(0, index=out.index)), errors="coerce") > 0,
        pd.to_numeric(out.get("assist_opportunities", pd.Series(np.nan, index=out.index)), errors="coerce")
        / pd.to_numeric(out.get("games", pd.Series(0, index=out.index)), errors="coerce"),
        np.nan,
    )

    # ------------------------------------------------------------------
    # Component scores.
    #
    # KEY DESIGN PRINCIPLE: role-primary metrics are scored within each
    # role group, not globally. Scoring a defender's CT rate against all
    # 200 players (most of whom have 0 CTs) inflates defender scores
    # artificially. An SSDM at the 97th pct globally on CTs is not more
    # valuable than an attacker at the 90th pct globally on points —
    # but global scoring made it look that way.
    #
    # Global scoring is used only for usage/touches (all players compete
    # for possession time and ground balls across role lines).
    # ------------------------------------------------------------------

    # Global usage/possession metrics — these are legitimately cross-role
    out["touches_score_global"] = _score_metric(out["touches_per_game"], True)
    out["passes_score_global"] = _score_metric(out["total_passes_per_game"], True)
    out["ground_ball_score_global"] = _score_metric(out["ground_balls_per_game"], True)
    out["turnover_security_score"] = _score_metric(out["turnovers_per_touch_shrunk"], False)

    # Role-peer scores — each metric scored within role group only
    out["points_score"] = _role_metric_score(out, "points_per_game", True)
    out["scoring_points_score"] = _role_metric_score(out, "scoring_points_per_game", True)
    out["one_point_goal_score"] = _role_metric_score(out, "one_point_goals_per_game", True)
    out["two_point_goal_score"] = _role_metric_score(out, "two_point_goals_per_game", True)
    out["goals_score"] = _role_metric_score(out, "goals_per_game", True)
    out["assists_score"] = _role_metric_score(out, "assists_per_game", True)
    out["shots_score"] = _role_metric_score(out, "shots_per_game", True)
    out["sog_score"] = _role_metric_score(out, "shots_on_goal_per_game", True)
    out["shot_pct_score"] = _role_metric_score(out, "shot_pct_shrunk", True)
    out["ct_score"] = _role_metric_score(out, "caused_turnovers_per_game", True)
    out["two_point_goal_efficiency_score"] = _role_metric_score(out, "two_pt_conversion_shrunk", True)
    out["points_per_touch_score"] = _role_metric_score(out, "points_per_touch_shrunk", True)
    out["assists_per_touch_score"] = _role_metric_score(out, "assists_per_touch_shrunk", True)

    # Specialist metrics — only meaningful within their own role group
    out["faceoff_pct_score"] = _role_metric_score(out, "faceoff_pct_shrunk", True)
    out["faceoff_volume_score"] = _role_metric_score(out, "faceoffs_per_game", True)
    out["faceoff_wins_score"] = _role_metric_score(out, "faceoffs_won_per_game", True)
    out["save_pct_score"] = _role_metric_score(out, "save_pct_shrunk", True)
    out["saves_score"] = _role_metric_score(out, "saves_per_game", True)
    out["goals_against_score"] = _role_metric_score(out, "goals_against_per_game", False)
    out["scores_against_score"] = _role_metric_score(out, "scores_against_per_game", False)

    # Phase 2 new metric scores (role-peer)
    out["assist_conv_score"] = _role_metric_score(out, "assist_conv_rate_shrunk", True)
    out["two_pt_conv_score"] = _role_metric_score(out, "two_pt_conversion_shrunk", True)
    out["clean_save_rate_score"] = _role_metric_score(out, "clean_save_rate_shrunk", True)
    out["assist_opp_score"] = _role_metric_score(out, "assist_opp_per_game", True)

    # Discipline — penalties conceded per game, scored within role because a
    # close defender and an attackman are not policed for the same things.
    # `num_penalties` was collected at player level and never scored; Piper Bond
    # took 6 penalties in 10 games with no effect on his rating.
    _pen_pg = pd.to_numeric(
        out.get("num_penalties_per_game", pd.Series(np.nan, index=out.index)),
        errors="coerce")
    if _pen_pg.isna().all():
        _games_for_pen = pd.to_numeric(out.get("games", pd.Series(np.nan, index=out.index)),
                                       errors="coerce")
        _pen_pg = (pd.to_numeric(out.get("num_penalties", pd.Series(np.nan, index=out.index)),
                                 errors="coerce")
                   / _games_for_pen.replace(0, np.nan))
    out["penalties_per_game_for_ranking"] = _pen_pg
    out["discipline_score"] = _role_metric_score(
        out, "penalties_per_game_for_ranking", higher_is_better=False)

    # Role-peer usage for blending
    out["touches_score_role"] = _role_metric_score(out, "touches_per_game", True)
    out["shots_score_role"] = _role_metric_score(out, "shots_per_game", True)
    out["passes_score_role"] = _role_metric_score(out, "total_passes_per_game", True)
    out["ground_ball_score_role"] = _role_metric_score(out, "ground_balls_per_game", True)
    out["sog_score_role"] = _role_metric_score(out, "shots_on_goal_per_game", True)

    # Public helper scores
    out["touches_score"] = _weighted_available_score(out, {"touches_score_global": 0.70, "touches_score_role": 0.30})
    out["passes_score"] = _weighted_available_score(out, {"passes_score_global": 0.70, "passes_score_role": 0.30})
    out["ground_ball_score"] = _weighted_available_score(out, {"ground_ball_score_global": 0.70, "ground_ball_score_role": 0.30})

    # ------------------------------------------------------------------
    # Legacy composite scores — retained for backward compatibility
    # ------------------------------------------------------------------
    out["scoring_value_score"] = _weighted_available_score(out, {
        "scoring_points_score": 0.34,
        "points_score": 0.22,
        "one_point_goal_score": 0.14,
        "two_point_goal_score": 0.22,
        "two_point_goal_efficiency_score": 0.08,
    })

    out["goal_value_score"] = out["scoring_value_score"]

    out["playmaking_value_score"] = _weighted_available_score(out, {
        "assists_score": 0.45,
        "assists_per_touch_score": 0.20,
        "points_per_touch_score": 0.20,
        "passes_score_global": 0.10,
        "turnover_security_score": 0.05,
    })

    out["offensive_creation_score"] = _weighted_available_score(out, {
        "scoring_value_score": 0.60,
        "playmaking_value_score": 0.40,
    })

    # Usage: blend global (how much possession vs all players) with
    # role-peer (how much vs role peers) to keep offensive players
    # from being penalized for lower touches than defenders/FO
    usage_global_score = _weighted_available_score(out, {
        "touches_score_global": 0.45,
        "shots_score_role": 0.20,
        "passes_score_global": 0.15,
        "ground_ball_score_global": 0.10,
        "sog_score_role": 0.10,
    })

    usage_role_score = _weighted_available_score(out, {
        "touches_score_role": 0.45,
        "shots_score_role": 0.20,
        "passes_score_role": 0.15,
        "ground_ball_score_role": 0.10,
        "sog_score_role": 0.10,
    })

    out["usage_global_score"] = usage_global_score
    out["usage_role_score"] = usage_role_score
    out["usage_possession_score"] = _weighted_available_score(out, {
        "usage_global_score": 0.70,
        "usage_role_score": 0.30,
    })
    out["usage_score"] = out["usage_possession_score"]

    # ------------------------------------------------------------------
    # Phase 2 — Role Performance Score (RPS) per role.
    #
    # Clean 3-component architecture replacing the old flat composite:
    #   RPS  = pure role-specific production score
    #   PSS  = peer standing (where player ranks among role peers)
    #   CIS  = cross-role impact (global contributions)
    # ------------------------------------------------------------------

    # Offense RPS: points production + creation + shot quality.
    #
    # `neutral=50.0` is the fix for the silent-reweighting bug. two_pt_conv_score is
    # absent for 41% of offensive rows and assist_conv_score for 7%; renormalizing
    # meant a player who never attempted a two-pointer had their other seven weights
    # scaled up, scoring the same as attempting at their own average and strictly
    # better than attempting and missing. Absent now reads as average.
    out["offense_rps"] = _weighted_available_score(out, {
        "points_score": 0.28,
        "scoring_points_score": 0.18,
        "goals_score": 0.15,
        "assists_score": 0.12,
        "assist_conv_score": 0.12,
        "shots_score": 0.08,
        "shot_pct_score": 0.05,
        "two_pt_conv_score": 0.02,
    }, neutral=50.0)

    # Defense RPS: disruption + possession recovery + discipline.
    #
    # Reweighted against measured impact rather than intuition. Aggregating player
    # production to team-seasons (n=40) and correlating with score margin per game:
    #
    #     stat                  all players   excl. faceoff men   defenders only
    #     ground balls/g          +0.357           +0.467             +0.121
    #     caused turnovers/g      +0.152           +0.155             +0.219
    #     turnovers/g             -0.152           -0.187             -0.161
    #
    # Ground balls look like the strongest team-level signal, but that signal is
    # largely *possession*, not defence: faceoff specialists alone account for 22%
    # of a team's ground balls, and once you restrict to actual defenders the ground
    # ball correlation collapses to +0.121 while caused turnovers rises to +0.219.
    # Against team scores-against per game — the outcome a defender is supposed to
    # influence — defenders' ground balls come in at -0.120 and caused turnovers at
    # +0.097, both weak, neither favouring ground balls.
    #
    # So ground balls fall from 0.30 to 0.20: they are real value, but they are
    # possession value that CIS already credits for every role, and inside the
    # Defense score they were the second-largest term on the weakest evidence.
    # Caused turnovers rise to 0.58 as the one stat that discriminates defenders.
    #
    # Discipline enters at 0.07. Penalties conceded is a genuine repeatable player
    # trait — year-over-year r across four season pairs is 0.30, 0.38, 0.39, 0.27,
    # in the same range as caused turnovers — and it was collected at player level
    # and never used. Kept small: it is a real trait but a minor share of defending,
    # and team penalty volume does not correlate negatively with margin (+0.134), so
    # weighting it heavily would be punishing aggression the data does not condemn.
    out["defense_rps"] = _weighted_available_score(out, {
        "ct_score": 0.58,
        "ground_ball_score": 0.20,
        "turnover_security_score": 0.15,
        "discipline_score": 0.07,
    }, neutral=50.0)

    # Faceoff RPS: winning possessions
    out["faceoff_rps"] = _weighted_available_score(out, {
        "faceoff_pct_score": 0.55,
        "faceoff_wins_score": 0.30,
        "faceoff_volume_score": 0.15,
    })

    # Goalie RPS: stopping the ball — clean_save_rate as primary
    out["goalie_rps"] = _weighted_available_score(out, {
        "clean_save_rate_score": 0.40,
        "save_pct_score": 0.35,
        "saves_score": 0.15,
        "goals_against_score": 0.10,
    })

    # Aliases for backward-compatibility with existing page columns
    out["offensive_score"] = out["offense_rps"]
    out["offensive_score_raw"] = out["offense_rps"]
    out["defensive_score"] = out["defense_rps"]
    out["defensive_score_raw"] = out["defense_rps"]
    out["faceoff_score"] = out["faceoff_rps"]
    out["faceoff_score_raw"] = out["faceoff_rps"]
    out["goalie_score"] = out["goalie_rps"]
    out["goalie_score_raw"] = out["goalie_rps"]

    # Cross-Role Impact Score (CIS) — global contributions all players make.
    #
    # Rebalanced to stop double counting. CIS is supposed to be the role-neutral
    # pillar, but two of its three components were already inside Defense RPS, so a
    # defender was paid twice for the same play while an attackman was paid once:
    #
    #     ground balls    0.65 x 0.30 + 0.10 x 0.45 = 0.240 of a defender's score
    #     ball security   0.65 x 0.15 + 0.10 x 0.20 = 0.118 of a defender's score
    #
    # The consequence was visible in the output: CIS medians spanned 21 points
    # across roles, which is a role bonus, not a cross-role measure.
    #
    # Touches becomes the primary term because it is the one genuinely universal
    # contribution and by far the most reliable stat in the dataset (split-half
    # r = 0.949, against 0.836 for ground balls and 0.577 for caused turnovers).
    # Ground balls stay in at a reduced 0.25 — they are legitimately cross-role, and
    # the Defense weight above came down at the same time, so the combined share for
    # a defender drops from 0.240 to 0.155. Passing volume joins at 0.15 as a second
    # possession contribution that no role score counts, which dilutes the remaining
    # overlap rather than concentrating it.
    out["cross_role_impact_raw"] = _weighted_available_score(out, {
        "touches_score_global": 0.45,
        "ground_ball_score_global": 0.25,
        "passes_score_global": 0.15,
        "turnover_security_score": 0.15,
    }, neutral=50.0)

    # Then centred within role, which is the part that makes it role-neutral in
    # fact rather than only in intent.
    #
    # Reweighting the components alone did not fix the role tilt, and measuring the
    # inputs shows why it cannot: every globally-scored possession stat is itself
    # a near-perfect proxy for position. Median global score by role, 2026:
    #
    #     component                  Def   FO    G     Off    spread
    #     touches_score_global      22.8  25.5  60.9  79.8     57.0
    #     ground_ball_score_global  58.0  97.7  64.6  30.5     67.2
    #     passes_score_global       26.5  17.1  68.7  77.2     60.1
    #     turnover_security_score   50.8   3.4  88.3  50.0     84.9
    #
    # A close defender handles the ball a third as often as an attackman because of
    # where he stands, not because he contributes less. So any weighted average of
    # these lands a defender ~27 points below an attackman before either plays, and
    # no choice of weights avoids it — my first attempt at this actually widened the
    # role spread from 16.7 to 28.6 while making each individual component more
    # defensible. The component weights were never the problem.
    #
    # Centring within role keeps the question CIS is good at ("does this player
    # contribute more possession value than others in his position?") and drops the
    # one it was answering by accident ("does this player's position touch the ball
    # a lot?"). It matters because Overall Score is RPS_normalized + CIS, and
    # RPS_normalized is already role-centred by design — leaving CIS role-tilted
    # meant it silently re-imposed a role bonus that no weight in this file chose
    # and that the Data Guide does not document.
    out["cross_role_impact"] = _normalize_within_role(
        out["cross_role_impact_raw"], out["role_group"],
        calibrate_mask=pd.to_numeric(
            out.get("eligible_for_default_ranking", pd.Series(1, index=out.index)),
            errors="coerce").fillna(1).eq(1),
        spread=20.0)

    # ------------------------------------------------------------------
    # Role context = RPS as role_primary_score + peer separation.
    # ------------------------------------------------------------------
    role_rps_map = {
        "Offense": "offense_rps",
        "Defense": "defense_rps",
        "Faceoff": "faceoff_rps",
        "Goalie": "goalie_rps",
    }

    out["role_primary_score"] = np.nan
    for role_name, col in role_rps_map.items():
        mask = out["role_group"].eq(role_name)
        out.loc[mask, "role_primary_score"] = pd.to_numeric(out.loc[mask, col], errors="coerce")

    # ------------------------------------------------------------------
    # Two-way credit.
    #
    # A role score by construction ignores half the field: Defense RPS counts no
    # points, so Jeff Trainor's 1.0 points/game as an SSDM — five times the SSDM
    # median — earned him nothing, and Offense RPS counts no ground balls or caused
    # turnovers, so Zach Currier's 7.1 recoveries/game as a midfielder, triple the
    # SSDM median, earned him nothing either. Both are two-way players and the
    # system was blind to the half of their game that made them one.
    #
    # Each player is scored on their secondary axis *within their own role* and must
    # additionally clear the 40th percentile of the role that owns that axis in
    # absolute per-game terms. See _two_way_credit for why both conditions are
    # needed: cross-role comparison alone makes the credit unreachable for
    # defenders (zero SSDM seasons reach the midfield points median), and within-role
    # comparison alone would hand a bonus to whoever tops a pure specialist pool's
    # noise. Goalies and faceoff specialists are excluded — their secondary
    # production is a function of their position, not two-way play.
    # ------------------------------------------------------------------
    # Games-threshold players define each role's centre, so a pool full of
    # one-game call-ups cannot drag the median the shrinkage aims at.
    _rps_eligible_for_center = pd.to_numeric(
        out.get("eligible_for_default_ranking", pd.Series(1, index=out.index)),
        errors="coerce").fillna(1).eq(1)

    _off_axis = pd.to_numeric(out.get("points_per_game", pd.Series(np.nan, index=out.index)),
                              errors="coerce")
    # Caused turnovers only — ground balls are already paid to every role by CIS.
    _def_axis = pd.to_numeric(
        out.get("caused_turnovers_per_game", pd.Series(np.nan, index=out.index)),
        errors="coerce")
    out["two_way_defensive_activity_per_game"] = _def_axis

    _is_off = out["role_group"].eq("Offense")
    _is_def = out["role_group"].eq("Defense")

    # Within-role standing on the secondary axis.
    _off_players_def_score = _role_metric_score(
        out, "two_way_defensive_activity_per_game", higher_is_better=True)
    out["two_way_secondary_score"] = np.nan
    _def_players_off_score = _role_metric_score(out, "points_per_game", higher_is_better=True)

    _games_for_two_way = pd.to_numeric(out.get("games", pd.Series(0, index=out.index)),
                                       errors="coerce").fillna(0)

    # The board's own games threshold is too low to support a two-way claim, so the
    # credit sets its own, higher bar. Without it the top two-way "midfielders" in
    # 2026 were 4-game players, and in Last 5 a 2-game player, ahead of Zach Currier.
    _two_way_games_floor = _two_way_min_games(
        float(_games_for_two_way.max()) if len(_games_for_two_way) else 0)
    out["two_way_min_games"] = _two_way_games_floor
    _two_way_eligible = _rps_eligible_for_center & _games_for_two_way.ge(_two_way_games_floor)

    out["two_way_credit"] = 0.0

    # Offensive players earning defensive credit, gated on the Defense pool's CT.
    if _is_off.any() and _is_def.any():
        out.loc[_is_off, "two_way_credit"] = _two_way_credit(
            _off_players_def_score.loc[_is_off],
            _def_axis.loc[_is_off],
            _games_for_two_way.loc[_is_off],
            gate=TWO_WAY_GATE_CT_PER_GAME,
            eligible=_two_way_eligible.loc[_is_off],
        ).values
        out.loc[_is_off, "two_way_secondary_score"] = _off_players_def_score.loc[_is_off].values

        # Defensive players earning offensive credit. A different absolute gate:
        # see TWO_WAY_GATE_POINTS_PER_GAME for why the two directions cannot
        # share one threshold.
        out.loc[_is_def, "two_way_credit"] = _two_way_credit(
            _def_players_off_score.loc[_is_def],
            _off_axis.loc[_is_def],
            _games_for_two_way.loc[_is_def],
            gate=TWO_WAY_GATE_POINTS_PER_GAME,
            eligible=_two_way_eligible.loc[_is_def],
        ).values
        out.loc[_is_def, "two_way_secondary_score"] = _def_players_off_score.loc[_is_def].values

    out["two_way_credit"] = pd.to_numeric(out["two_way_credit"], errors="coerce").fillna(0.0)
    out["is_two_way_player"] = out["two_way_credit"].gt(1.0).astype(int)

    # The credit is added before normalization so it competes on the raw role scale
    # the rest of the role's distribution lives on.
    out["role_primary_score_before_two_way"] = out["role_primary_score"].copy()
    out["role_primary_score"] = (
        pd.to_numeric(out["role_primary_score"], errors="coerce") + out["two_way_credit"]
    ).clip(0, 100)

    # ------------------------------------------------------------------
    # Sample-size shrinkage.
    #
    # Applied to the role score, before normalization and before it feeds Peer
    # Standing, so one correction covers every downstream use instead of being
    # bolted onto the final number. corr(games, overall_score) was 0.181 with seven
    # of the top 25 under seven games; an 85+ score built on five games was worth
    # ~79 over the rest of the season, and a sub-60 was worth 41 — both tails
    # collapse inward while the sample is thin, which is what this removes.
    #
    # Shrinking toward the role's own median rather than a flat 50: the role scales
    # differ (that is why _normalize_within_role exists), so 50 is not the average
    # of every role's raw RPS, and pulling a defender toward a number that is not
    # his pool's centre would be a role adjustment wearing a sample-size label.
    # ------------------------------------------------------------------
    out["role_primary_score_before_shrink"] = out["role_primary_score"].copy()
    _shrunk_rps = pd.to_numeric(out["role_primary_score"], errors="coerce")
    for _, _idx in out.groupby("role_group", dropna=False).groups.items():
        _idx = list(_idx)
        _vals = _shrunk_rps.loc[_idx]
        _basis = _vals[_rps_eligible_for_center.loc[_idx]].dropna()
        if len(_basis) < 5:
            _basis = _vals.dropna()
        if len(_basis) < 2:
            continue
        _shrunk_rps.loc[_idx] = _shrink_to_average(
            _vals, _games_for_two_way.loc[_idx], center=float(_basis.median())).values
    out["role_primary_score"] = _shrunk_rps
    out["sample_trust"] = _sample_trust(_games_for_two_way)

    # RPS on a common cross-role scale, for the all-player board only.
    #
    # Each role's RPS is a weighted average of a different number of differently
    # correlated stats, so the raw numbers are not comparable between roles — see
    # _normalize_within_role for the measurements. The role-specific views
    # (offense_rps, defense_rps, faceoff_rps, goalie_rps, and the Goalie/Faceoff
    # pages built on them) keep using the raw values: within a role the raw scale
    # is the meaningful one, and normalising is monotonic there anyway. Only the
    # cross-role Overall blend reads this column.
    _rps_calibration_mask = pd.to_numeric(
        out.get("eligible_for_default_ranking", pd.Series(1, index=out.index)),
        errors="coerce").fillna(1).eq(1)
    out["role_primary_score_normalized"] = _normalize_within_role(
        out["role_primary_score"], out["role_group"],
        calibrate_mask=_rps_calibration_mask)

    out = _add_test_style_role_separation(out)

    # Peer Standing Score (PSS): role_primary_score through sigmoid, ranked
    # *within role group* — "where they rank among players in the same role".
    #
    # PUBLISHED FOR DISPLAY ONLY. It is no longer a term in the Overall Score, and
    # the reason is that it never functioned as one. PSS is a monotone transform of
    # a within-role ranking of RPS, and RPS is the other component, so the two carry
    # the same ordering by construction. Measured on the shipped mart, in all eight
    # role-by-context pairs:
    #
    #     Spearman(RPS, PSS) = 1.0000        (every role, every context)
    #     Pearson(RPS_normalized, PSS) = 0.982 - 0.998
    #
    # A "three-component blend" whose second component cannot reorder the first is a
    # two-component blend that describes itself inaccurately: the real split was
    # ~85% role performance and 10-15% cross-role impact. So the 0.25 is folded back
    # into RPS, which is where it was already going. This changes no player's rank
    # relative to another within a role; it makes the published weights honest and
    # stops the tiny nonlinear disagreement between the two scales (Pearson < 1)
    # from acting as unexplained jitter in the final number.
    #
    # This block used to rank role_primary_score across the whole league in one
    # pool. That is not a peer standing, and it was not even a coherent ranking:
    # each role's RPS is a weighted average of a different number of differently
    # correlated components, so the scales are not comparable. Pooling them put
    # every specialist above every attackman before a single game was played.
    out["peer_standing_score"] = np.nan
    for _, idx in out.groupby("role_group", dropna=False).groups.items():
        idx = list(idx)
        rps = out.loc[idx, "role_primary_score"]
        out.loc[idx, "peer_standing_score"] = _sigmoid_stretch(
            _rank_pct(rps.where(rps.notna()), True)).values

    out["role_context_value_score"] = _weighted_available_score(out, {
        "role_primary_score": 0.50,
        "role_primary_percentile": 0.25,
        "role_separation_score": 0.25,
    })
    out["role_context_percentile"] = out["role_primary_percentile"]

    # ------------------------------------------------------------------
    # Final overall score — role-specific 3-component blend.
    # ------------------------------------------------------------------
    out["base_impact_score"] = out["role_primary_score"].copy()
    out["base_impact_score"] = pd.to_numeric(out["base_impact_score"], errors="coerce").clip(0, 100)

    # Keep overall_impact_score as the base-impact alias for compatibility.
    out["overall_impact_score"] = out["base_impact_score"]

    # ------------------------------------------------------------------
    # Official overall score.
    #
    # This keeps the tested role-specific architecture. Goalies use a
    # transfer-adjusted version of their goalie-specific inputs for the all-player
    # Overall view rather than a flat subtraction. This preserves objective goalie
    # evaluation in the Goalie view while preventing a small goalie peer pool from
    # transferring too aggressively into cross-position rankings.
    # ------------------------------------------------------------------

    def _transfer_toward_average(series, factor):
        s = pd.to_numeric(series, errors="coerce")
        return (50.0 + float(factor) * (s - 50.0)).clip(0, 100)

    # Full goalie skill metrics remain available in goalie_score / role context.
    # These transfer-adjusted fields are used only for the all-player Overall
    # score calculation.
    # The goalie transfer is applied on top of the cross-role normalized RPS, since
    # that is the scale the Overall blend reads. Compressing the raw goalie scale
    # toward 50 and then comparing it against a differently-scaled offensive RPS
    # would leave the artefact this normalization removes.
    _rps_norm = pd.to_numeric(out["role_primary_score_normalized"],
                              errors="coerce").clip(0, 100)
    out["goalie_base_for_overall"] = _rps_norm.copy()
    out["goalie_role_context_for_overall"] = out["role_context_value_score"].copy()
    out["goalie_save_pct_for_overall"] = out["save_pct_score"].copy()
    out["goalie_saves_for_overall"] = out["saves_score"].copy()

    # Scale transfer compression by sample size: specialists ranked within a
    # small peer pool (8-10 goalies, 5-8 FO players) should not dominate the
    # cross-role Overall ranking in early season. At 6+ games the factors
    # relax toward the calibrated full-season values.
    max_games_in_ctx = float(out["games"].max()) if len(out) else 0
    sample_factor = float(np.clip(max_games_in_ctx / 10.0, 0.15, 1.0))

    # Goalie compression eased on measured impact.
    #
    # The compression was calibrated to stop a small goalie pool from flooding the
    # top of the board, which is a real risk, but it was set so hard that no goalie
    # could reach the top at all: the best goalie ranked 43rd of 161 in 2026 and
    # 60th of 314 all-time. That is not a defensible ceiling for the position. Save
    # percentage carries a standardized beta of +0.574 on team score margin against
    # +0.586 for scoring — goalkeeping is as close to winning as offence is — and
    # goalie mean score has the second-strongest correlation with team wins of any
    # role (+0.463, behind only offence).
    #
    # 0.70 -> 0.88 at full sample keeps a deliberate discount, because a goalie's
    # save percentage is partly his defence's shot quality and this data cannot
    # separate the two, but it lets an elite goalie into the top 15 instead of
    # capping him outside the top 40. The early-season end stays aggressive (0.55 ->
    # 0.62): the pool is 8-15 goalies and one hot week should not lead the league.
    goalie_rps_factor = 0.62 + 0.26 * sample_factor   # 0.62 early → 0.88 full
    goalie_ctx_factor = 0.30 + 0.20 * sample_factor   # 0.30 early → 0.50 full
    goalie_savepct_factor = 0.45 + 0.25 * sample_factor  # 0.45 early → 0.70 full
    goalie_saves_factor = 0.45 + 0.25 * sample_factor    # 0.45 early → 0.70 full

    # FO specialists: compress role context more in small samples
    fo_ctx_factor = 0.45 + 0.15 * sample_factor  # 0.45 early → 0.60 full

    goalie_mask = out["role_group"].eq("Goalie")
    fo_mask = out["role_group"].eq("Faceoff")

    out.loc[goalie_mask, "goalie_base_for_overall"] = _transfer_toward_average(_rps_norm.loc[goalie_mask], goalie_rps_factor)
    out.loc[goalie_mask, "goalie_role_context_for_overall"] = _transfer_toward_average(out.loc[goalie_mask, "role_context_value_score"], goalie_ctx_factor)
    out.loc[goalie_mask, "goalie_save_pct_for_overall"] = _transfer_toward_average(out.loc[goalie_mask, "save_pct_score"], goalie_savepct_factor)
    out.loc[goalie_mask, "goalie_saves_for_overall"] = _transfer_toward_average(out.loc[goalie_mask, "saves_score"], goalie_saves_factor)

    # FO role context: apply sample-scaled compression for overall ranking
    out["fo_role_context_for_overall"] = out["role_context_value_score"].copy()
    out.loc[fo_mask, "fo_role_context_for_overall"] = _transfer_toward_average(out.loc[fo_mask, "role_context_value_score"], fo_ctx_factor)

    # 2-component overall formula per role, after folding the redundant Peer
    # Standing term into Role Performance (see the PSS block above — Spearman with
    # RPS was exactly 1.0000 in all eight role-by-context pairs, so it could not
    # reorder anything):
    #   Offense:  0.85 RPS + 0.15 CIS
    #   Defense:  0.90 RPS + 0.10 CIS
    #   Faceoff:  0.90 RPS + 0.10 CIS
    #   Goalie:   0.95 RPS + 0.05 CIS  (with specialist compression)
    _cis = pd.to_numeric(out.get("cross_role_impact", pd.Series(50.0, index=out.index)), errors="coerce").fillna(50.0)
    # All three take the cross-role normalized RPS rather than the raw role column.
    # role_primary_score_normalized already holds each player's own role's RPS, so
    # one column serves every branch of the np.select below.
    _rps_norm_blend = _rps_norm.fillna(50.0)
    _rps_off = _rps_norm_blend
    _rps_def = _rps_norm_blend
    _rps_fo = _rps_norm_blend
    _rps_g_adj = pd.to_numeric(out["goalie_base_for_overall"], errors="coerce").fillna(50.0)

    out["overall_score_raw"] = np.select(
        [
            out["role_group"].eq("Offense"),
            out["role_group"].eq("Defense"),
            out["role_group"].eq("Faceoff"),
            out["role_group"].eq("Goalie"),
        ],
        [
            0.85 * _rps_off + 0.15 * _cis,
            0.90 * _rps_def + 0.10 * _cis,
            0.90 * _rps_fo + 0.10 * _cis,
            0.95 * _rps_g_adj + 0.05 * _cis,
        ],
        default=out["base_impact_score"],
    )
    out["overall_score_raw"] = pd.to_numeric(out["overall_score_raw"], errors="coerce")

    out["role_overall_adjustment"] = 0.0
    out["overall_score_raw"] = pd.to_numeric(out["overall_score_raw"], errors="coerce").clip(0, 100)

    # Context calibration: shift the eligible context median toward 50 so the
    # scale stays interpretable across different seasons and sample sizes.
    # The sigmoid stretch already expands the distribution — this final shift
    # ensures 50 always means "league average" in the selected context.
    # Cap widened to ±20 so small early-season samples calibrate properly.
    out["overall_score"] = out["overall_score_raw"].copy()
    eligible_mask = pd.to_numeric(out.get("eligible_for_default_ranking", pd.Series(1, index=out.index)), errors="coerce").fillna(1).eq(1)
    raw_eligible = pd.to_numeric(out.loc[eligible_mask, "overall_score_raw"], errors="coerce")
    if raw_eligible.notna().sum() >= 5:
        context_median = raw_eligible.median()
        if pd.notna(context_median):
            shift = float(np.clip(50.0 - context_median, -20.0, 20.0))
            out["overall_score"] = (pd.to_numeric(out["overall_score_raw"], errors="coerce") + shift).clip(0, 100)
            out["overall_score_context_shift"] = shift
        else:
            out["overall_score_context_shift"] = 0.0
    else:
        out["overall_score_context_shift"] = 0.0

    # Flag the players the 0-100 clip flattened. Six players sat at exactly 0.0 in
    # the shipped mart, which reads as a measured score and is really "this player
    # is somewhere at or below the bottom of the scale" — they are not tied, the
    # scale simply ran out. Published so the pages can mark them rather than
    # presenting a clip as a measurement.
    _pre_clip = pd.to_numeric(out.get("overall_score", pd.Series(np.nan, index=out.index)),
                              errors="coerce")
    out["overall_score_at_scale_bound"] = (
        _pre_clip.le(0.0) | _pre_clip.ge(100.0)).fillna(False).astype(int)

    out["overall_score"] = pd.to_numeric(out["overall_score"], errors="coerce").clip(0, 100)

    out["official_overall_score"] = out["overall_score"]
    out["ranking_formula_version"] = "official_player_ranking_test_aligned"

    if "save_pct_calc" in out.columns:
        out["save_pct"] = out["save_pct_calc"]
    if "faceoff_pct_calc" in out.columns:
        out["faceoff_pct"] = out["faceoff_pct_calc"]
    if "shot_pct_calc" in out.columns:
        out["shot_pct"] = out["shot_pct_calc"]

    return out


def _build_player_ranking_context(df, context_type, context_label, sort_order):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    if "games" not in out.columns:
        out["games"] = 0

    out["games"] = pd.to_numeric(out["games"], errors="coerce").fillna(0)

    max_games = int(out["games"].max()) if len(out) else 0
    min_games = _ranking_context_min_games(context_type, max_games)

    out["ranking_context_type"] = context_type
    out["ranking_context"] = context_label
    out["ranking_sort_order"] = sort_order

    # What "Career" actually spans.
    #
    # The PLL began in 2019 but this warehouse holds 2022-2026, so the Career
    # context is a five-year window presented as a career total: Brett Dobson shows
    # 45 games, Marcus Holman 49, and Lyle Thompson — a founding player with a
    # decade in the league — shows 9. Anyone reading "Career" as career is being
    # misled about players whose best years predate the data.
    #
    # Published as a column rather than folded into the label, because
    # `ranking_context == "Career"` is a join key in six pages and renaming the value
    # would break every one of them. The pages show this span next to the label; the
    # honest fix upstream is to scrape 2019-2021, at which point this string
    # updates itself from the data.
    _ctx_seasons = pd.to_numeric(out.get("season", pd.Series(dtype="float64")),
                                 errors="coerce").dropna()
    if context_type == "Career" and "seasons_played_list" in out.columns:
        _ctx_seasons = pd.Series(dtype="float64")
    if len(_ctx_seasons):
        _lo, _hi = int(_ctx_seasons.min()), int(_ctx_seasons.max())
    else:
        _lo, _hi = int(min(TARGET_SEASONS)), int(max(TARGET_SEASONS))
    out["ranking_context_season_span"] = (
        f"{_lo}" if _lo == _hi else f"{_lo}–{_hi}")
    out["ranking_context_covers_full_history"] = int(_lo <= 2019)
    out["ranking_context_sort"] = sort_order
    out["max_games_in_context"] = max_games
    out["ranking_context_max_games"] = max_games
    out["default_min_games_used"] = min_games
    out["min_games_default"] = min_games
    out["is_ranking_eligible"] = (out["games"].fillna(0) >= min_games).astype(int)
    out["eligible_for_default_ranking"] = out["is_ranking_eligible"]
    out["sample_size_note"] = np.where(
        max_games <= 2,
        "Early season: rankings include players with 1+ game.",
        np.where(
            max_games <= 5,
            f"Small sample: default ranking requires {min_games}+ games.",
            ""
        )
    )

    out = _add_player_ranking_scores(out)

    # Rank every player with a valid score in the selected context, but rank the
    # players who meet the games threshold *first* so their numbers are contiguous.
    #
    # This block used to rank the whole pool together while the app displayed only
    # the eligible rows, so the visible 2026 board read 1, 6, 12, 13, 16 — ranks
    # 2-5 belonged to four one-game players who were then filtered out. `eligible`
    # was computed here and never used. Small-sample players still receive a rank
    # (the reason the pool is not simply filtered), they just queue behind the
    # qualified ones instead of interleaving with them.
    eligible = pd.to_numeric(out["eligible_for_default_ranking"], errors="coerce").fillna(0).eq(1)
    rankable = pd.to_numeric(out.get("overall_score", pd.Series(np.nan, index=out.index)), errors="coerce").notna()

    def _official_rank(scores, eligible_flags):
        """
        Rank descending with eligible players taking 1..N before ineligible ones.

        Ranking on the tuple (ineligible, -score) via a lexicographic sort key is
        what keeps the two groups from interleaving.
        """
        s = pd.to_numeric(scores, errors="coerce")
        ineligible = (~eligible_flags.reindex(s.index).fillna(False)).astype(int)
        order = pd.DataFrame({"blocked": ineligible, "score": s})
        return (order.sort_values(["blocked", "score"], ascending=[True, False])
                     .assign(rk=lambda d: np.arange(1, len(d) + 1))
                     .loc[s.index, "rk"]
                     .where(s.notna()))

    out["overall_rank"] = np.nan
    out["overall_percentile"] = np.nan
    out["position_rank"] = np.nan
    out["position_percentile"] = np.nan
    out["role_context_rank"] = np.nan
    out["role_context_percentile"] = np.nan
    out["offensive_rank"] = np.nan
    out["defensive_rank"] = np.nan
    out["faceoff_rank"] = np.nan
    out["goalie_rank"] = np.nan

    if rankable.any():
        out.loc[rankable, "overall_rank"] = _official_rank(
            out.loc[rankable, "overall_score"], eligible).values
        out.loc[rankable, "overall_percentile"] = _rank_pct(out.loc[rankable, "overall_score"], True).values

        for _, idx in out.loc[rankable].groupby("position", dropna=False).groups.items():
            idx = list(idx)
            out.loc[idx, "position_rank"] = _official_rank(
                out.loc[idx, "overall_score"], eligible).values
            out.loc[idx, "position_percentile"] = _rank_pct(out.loc[idx, "overall_score"], True).values

        if "role_context_value_score" in out.columns:
            role_rankable = pd.to_numeric(out["role_context_value_score"], errors="coerce").notna()
            for _, idx in out.loc[role_rankable].groupby("role_group", dropna=False).groups.items():
                idx = list(idx)
                out.loc[idx, "role_context_rank"] = _official_rank(
                    out.loc[idx, "role_context_value_score"], eligible).values
                out.loc[idx, "role_context_percentile"] = _rank_pct(out.loc[idx, "role_context_value_score"], True).values

        off_rankable = pd.to_numeric(out.get("offensive_score", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
        if off_rankable.any():
            out.loc[off_rankable, "offensive_rank"] = out.loc[off_rankable, "offensive_score"].rank(method="min", ascending=False)

        mask = out["role_group"].eq("Defense") & pd.to_numeric(out.get("defensive_score", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
        if mask.any():
            out.loc[mask, "defensive_rank"] = out.loc[mask, "defensive_score"].rank(method="min", ascending=False)

        mask = out["role_group"].eq("Faceoff") & pd.to_numeric(out.get("faceoff_score", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
        if mask.any():
            out.loc[mask, "faceoff_rank"] = out.loc[mask, "faceoff_score"].rank(method="min", ascending=False)

        mask = out["role_group"].eq("Goalie") & pd.to_numeric(out.get("goalie_score", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
        if mask.any():
            out.loc[mask, "goalie_rank"] = out.loc[mask, "goalie_score"].rank(method="min", ascending=False)

    score_cols = [c for c in out.columns if c.endswith("_score") or c.endswith("_percentile") or c.endswith("_rank") or c in ["overall_impact_score", "base_impact_score", "usage_possession_score", "usage_score"]]
    for c in score_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    return out.sort_values(["ranking_sort_order", "overall_rank", "full_name"], na_position="last").reset_index(drop=True)


ranking_contexts = []

if "player_career_stats" in globals() and len(player_career_stats) > 0:
    ranking_contexts.append(_build_player_ranking_context(player_career_stats, "Career", "Career", 0))

if "player_last10_stats" in globals() and len(player_last10_stats) > 0:
    ranking_contexts.append(_build_player_ranking_context(player_last10_stats, "Last 10", "Last 10", 1))

if "player_last5_stats" in globals() and len(player_last5_stats) > 0:
    ranking_contexts.append(_build_player_ranking_context(player_last5_stats, "Last 5", "Last 5", 2))

if "player_season_stats" in globals() and len(player_season_stats) > 0 and "season" in player_season_stats.columns:
    for i, season in enumerate(sorted(pd.to_numeric(player_season_stats["season"], errors="coerce").dropna().astype(int).unique())):
        sdf = player_season_stats[pd.to_numeric(player_season_stats["season"], errors="coerce").eq(season)].copy()
        ranking_contexts.append(_build_player_ranking_context(sdf, "Season", f"{season} Season", 100 + i))

player_ranking_profiles = pd.concat(ranking_contexts, ignore_index=True, sort=False) if ranking_contexts else pd.DataFrame()


def _build_team_style_context(team_stats, defense_stats, context_type, context_label, sort_order):
    if team_stats is None or len(team_stats) == 0:
        return pd.DataFrame()

    teams = team_stats.copy()
    defense = defense_stats.copy() if defense_stats is not None else pd.DataFrame()

    if len(defense) > 0:
        merge_keys = ["team_id"]
        if "season" in teams.columns and "season" in defense.columns:
            merge_keys = ["season", "team_id"]

        keep_cols = [
            c for c in [
                *merge_keys,
                "scores_allowed_per_game",
                "goals_allowed_per_game",
                "opponent_shots_per_game",
                "def_opponent_shots_per_game",
                "opponent_goal_pct",
                "def_opponent_goal_pct",
                "opponent_sog_rate",
                "save_pct_proxy",
                "def_save_pct_proxy",
            ]
            if c in defense.columns
        ]

        if all(k in defense.columns for k in merge_keys):
            teams = teams.merge(defense[keep_cols].drop_duplicates(merge_keys), on=merge_keys, how="left", suffixes=("", "_def"))

    alias_pairs = {
        "scores_allowed_per_game": ["scores_allowed_per_game_def", "scores_against_per_game"],
        "goals_allowed_per_game": ["goals_allowed_per_game_def", "goals_against_per_game"],
        "opponent_shots_per_game": ["opponent_shots_per_game_def", "def_opponent_shots_per_game", "shots_against_per_game"],
        "opponent_goal_pct": ["opponent_goal_pct_def", "def_opponent_goal_pct"],
        "save_pct_proxy": ["save_pct_proxy_def", "def_save_pct_proxy"],
    }

    for primary, fallbacks in alias_pairs.items():
        if primary not in teams.columns:
            teams[primary] = np.nan
        for fallback in fallbacks:
            if fallback in teams.columns:
                teams[primary] = teams[primary].fillna(teams[fallback])

    numeric_cols = [
        "scores_per_game", "shots_per_game", "touches_per_game",
        "time_in_possession_per_game", "offensive_sequence_proxy_per_game",
        "turnovers_per_game", "assists_per_game", "total_passes_per_game",
        "scores_allowed_per_game", "opponent_shots_per_game", "opponent_goal_pct",
        "save_pct_proxy", "faceoff_pct_calc", "score_margin_per_game", "shot_pct_calc"
    ]
    teams = _ensure_cols(teams, numeric_cols)

    if teams["opponent_goal_pct"].isna().all() and "goals_against" in teams.columns and "shots_against" in teams.columns:
        teams["opponent_goal_pct"] = pd.to_numeric(teams["goals_against"], errors="coerce") / pd.to_numeric(teams["shots_against"], errors="coerce").replace(0, np.nan)

    if teams["save_pct_proxy"].isna().all() and "saves" in teams.columns and "goals_against" in teams.columns:
        saves = pd.to_numeric(teams["saves"], errors="coerce")
        ga = pd.to_numeric(teams["goals_against"], errors="coerce")
        teams["save_pct_proxy"] = (saves / (saves + ga).replace(0, np.nan)).clip(0, 1)

    teams["profile_context_type"] = context_type
    teams["profile_context"] = context_label
    teams["profile_sort_order"] = sort_order
    teams["profile_context_sort"] = sort_order

    teams["offensive_volume_score"] = (
        0.35 * _minmax_score(teams["scores_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(teams["shots_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["touches_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["offensive_sequence_proxy_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["offensive_efficiency_score"] = (
        0.45 * _minmax_score(teams["scores_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(teams["shot_pct_calc"], True).fillna(50)
        + 0.20 * _minmax_score(teams["turnovers_per_game"], False).fillna(50)
        + 0.10 * _minmax_score(teams["score_margin_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["ball_movement_score"] = (
        0.55 * _minmax_score(teams["assists_per_game"], True).fillna(50)
        + 0.25 * _minmax_score(teams["total_passes_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["touches_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["possession_control_score"] = (
        0.45 * _minmax_score(teams["touches_per_game"], True).fillna(50)
        + 0.35 * _minmax_score(teams["time_in_possession_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["faceoff_pct_calc"], True).fillna(50)
    ).clip(0, 100)

    teams["defensive_suppression_score"] = (
        0.40 * _minmax_score(teams["scores_allowed_per_game"], False).fillna(50)
        + 0.25 * _minmax_score(teams["opponent_shots_per_game"], False).fillna(50)
        + 0.20 * _minmax_score(teams["opponent_goal_pct"], False).fillna(50)
        + 0.15 * _minmax_score(teams["save_pct_proxy"], True).fillna(50)
    ).clip(0, 100)

    teams["pace_tempo_score"] = (
        0.35 * _minmax_score(teams["shots_per_game"], True).fillna(50)
        + 0.30 * _minmax_score(teams["touches_per_game"], True).fillna(50)
        + 0.20 * _minmax_score(teams["offensive_sequence_proxy_per_game"], True).fillna(50)
        + 0.15 * _minmax_score(teams["time_in_possession_per_game"], True).fillna(50)
    ).clip(0, 100)

    teams["team_style_overall_score"] = (
        0.22 * teams["offensive_volume_score"]
        + 0.20 * teams["offensive_efficiency_score"]
        + 0.16 * teams["ball_movement_score"]
        + 0.18 * teams["possession_control_score"]
        + 0.18 * teams["defensive_suppression_score"]
        + 0.06 * teams["pace_tempo_score"]
    ).clip(0, 100)

    teams["overall_score"] = teams["team_style_overall_score"]
    teams["overall_style"] = teams["team_style_overall_score"]
    teams["profile_rank"] = teams["team_style_overall_score"].rank(method="min", ascending=False)
    teams["def_scores_allowed_per_game"] = teams["scores_allowed_per_game"]
    teams["def_opponent_shots_per_game"] = teams["opponent_shots_per_game"]
    teams["def_opponent_goal_pct"] = teams["opponent_goal_pct"]
    teams["def_save_pct_proxy"] = teams["save_pct_proxy"]
    teams["net_scores_per_game"] = teams["scores_per_game"] - teams["scores_allowed_per_game"]
    teams["time_in_possession_per_game_mmss"] = teams["time_in_possession_per_game"].apply(seconds_to_mmss_safe)
    teams["possession_pg"] = teams["time_in_possession_per_game_mmss"]

    teams["pace_label"] = teams["pace_tempo_score"].apply(lambda x: _label_from_score(x, ("High Tempo", "Above-Average Tempo", "Balanced Tempo", "Slower Tempo", "Very Slow Tempo")))
    teams["offensive_profile_label"] = teams["offensive_efficiency_score"].apply(lambda x: _label_from_score(x, ("Elite Offense", "Above-Average Offense", "Middle Tier", "Low-Output Offense", "Poor Offense")))
    teams["defensive_profile_label"] = teams["defensive_suppression_score"].apply(lambda x: _label_from_score(x, ("Elite Defense", "Above-Average Defense", "Middle Tier", "Below-Average Defense", "Vulnerable Defense")))
    teams["possession_profile_label"] = teams["possession_control_score"].apply(lambda x: _label_from_score(x, ("Elite Possession", "Above-Average Possession", "Middle Tier", "Below-Average Possession", "Poor Possession")))
    teams["style_summary"] = teams["pace_label"].astype(str) + " | " + teams["offensive_profile_label"].astype(str) + " | " + teams["defensive_profile_label"].astype(str) + " | " + teams["possession_profile_label"].astype(str)
    teams["sample_size_note"] = np.where(pd.to_numeric(teams.get("games", pd.Series(0, index=teams.index)), errors="coerce").fillna(0) <= 2, "Early-season sample.", "")

    for c in [c for c in teams.columns if c.endswith("_score") or c in ["profile_rank"]]:
        teams[c] = pd.to_numeric(teams[c], errors="coerce").round(2)

    return teams.sort_values(["profile_sort_order", "profile_rank"], na_position="last").reset_index(drop=True)


style_contexts = []
if "team_career_stats" in globals() and len(team_career_stats) > 0:
    style_contexts.append(_build_team_style_context(team_career_stats, team_defense_career_stats if "team_defense_career_stats" in globals() else pd.DataFrame(), "Career", "Career", 0))
if "team_season_stats" in globals() and len(team_season_stats) > 0 and "season" in team_season_stats.columns:
    for i, season in enumerate(sorted(pd.to_numeric(team_season_stats["season"], errors="coerce").dropna().astype(int).unique())):
        sdf = team_season_stats[pd.to_numeric(team_season_stats["season"], errors="coerce").eq(season)].copy()
        if "team_defense_season_stats" in globals() and len(team_defense_season_stats) > 0 and "season" in team_defense_season_stats.columns:
            ddf = team_defense_season_stats[pd.to_numeric(team_defense_season_stats["season"], errors="coerce").eq(season)].copy()
        else:
            ddf = pd.DataFrame()
        style_contexts.append(_build_team_style_context(sdf, ddf, "Season", f"{season} Season", 100 + i))

team_style_profiles = pd.concat(style_contexts, ignore_index=True, sort=False) if style_contexts else pd.DataFrame()

print("Player ranking and team style profile marts created.")
print("player_ranking_profiles:", player_ranking_profiles.shape)
print("team_style_profiles:", team_style_profiles.shape)



# ============================================================
# GITHUB PORT ADD-ON — PARQUET-SAFE STORAGE
# ============================================================

def _make_unique_storage_columns(columns):
    seen = {}
    clean_cols = []
    for col in columns:
        base = str(col)
        if base not in seen:
            seen[base] = 0
            clean_cols.append(base)
        else:
            seen[base] += 1
            clean_cols.append(f"{base}_{seen[base]}")
    return clean_cols


def sanitize_dataframe_for_storage(df):
    if df is None:
        return pd.DataFrame({"_empty_placeholder": pd.Series(dtype="string")})

    out = df.copy()

    if len(out.columns) == 0:
        return pd.DataFrame({"_empty_placeholder": pd.Series(dtype="string")})

    out.columns = _make_unique_storage_columns(list(out.columns))

    def storage_scalar(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, default=str, sort_keys=True)
            except Exception:
                return str(value)
        if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
            return value.isoformat()
        return value

    for col in out.columns:
        s = out[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("boolean")
            continue
        if pd.api.types.is_integer_dtype(s):
            out[col] = s.astype("Int64")
            continue
        if pd.api.types.is_float_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce")
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            mapped = s.map(storage_scalar)
            non_null = mapped.dropna()

            if len(non_null) == 0:
                out[col] = mapped.astype("string")
                continue

            numeric_types = (int, float, np.integer, np.floating)
            if non_null.map(lambda x: isinstance(x, numeric_types) and not isinstance(x, bool)).all():
                numeric = pd.to_numeric(mapped, errors="coerce")
                if len(numeric.dropna()) > 0 and numeric.dropna().map(lambda x: float(x).is_integer()).all():
                    out[col] = numeric.astype("Int64")
                else:
                    out[col] = numeric
                continue

            if non_null.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                out[col] = mapped.astype("boolean")
                continue

            out[col] = mapped.astype("string")
            continue

        try:
            out[col] = s.map(storage_scalar).astype("string")
        except Exception:
            out[col] = s.astype("string")

    return out


# ============================================================
# BLOCK 11 — SAVE CURATED TABLES + DUCKDB WAREHOUSE
# DEFENSIVE / OPPONENT + POSSESSION QC INCLUDED
# ============================================================

def ensure_non_empty_schema(df, table_name):
    """
    DuckDB cannot read Parquet files with zero columns.
    This guarantees every exported table has at least one column.
    """
    if df is None:
        return pd.DataFrame({
            "_empty_table_name": [table_name],
            "_note": ["table_was_none_or_not_created"]
        })

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame({
            "_empty_table_name": [table_name],
            "_note": [f"not_a_dataframe__type={type(df)}"]
        })

    if len(df.columns) == 0:
        return pd.DataFrame(columns=[
            "_empty_table_name",
            "_note",
            "season",
            "game_id",
            "game_slug",
            "reason",
            "error",
        ])

    return df.copy()


def get_table_var(var_name, required=False):
    """
    Safely pulls a dataframe variable from notebook globals.
    If required=True, raises a clear error if missing.
    If required=False, creates an empty note dataframe if missing.
    """
    if var_name in globals():
        return globals()[var_name]

    msg = f"Variable `{var_name}` was not found when saving curated tables."

    if required:
        raise NameError(msg)

    print("WARNING:", msg)

    return pd.DataFrame({
        "_empty_table_name": [var_name],
        "_note": ["variable_missing_when_block_11_ran"]
    })


# ============================================================
# CURATED TABLE REGISTRY
# ============================================================

curated_tables = {
    # ------------------------------------------------------------
    # Clean/base game-level data
    # ------------------------------------------------------------
    "game_manifest": get_table_var("game_manifest", required=True),
    "team_game_stats": get_table_var("team_game_stats", required=True),
    "player_game_stats": get_table_var("player_game_stats", required=True),

    # ------------------------------------------------------------
    # Reference/directories
    # ------------------------------------------------------------
    "team_alias_mapping": get_table_var("team_alias_mapping", required=True),
    "team_directory": get_table_var("team_directory", required=True),
    "player_directory": get_table_var("player_directory", required=True),

    # ------------------------------------------------------------
    # Player marts
    # ------------------------------------------------------------
    "player_season_stats_by_team": get_table_var("player_season_stats_by_team", required=True),
    "player_season_stats": get_table_var("player_season_stats", required=True),
    "player_career_stats": get_table_var("player_career_stats", required=True),
    "player_vs_opponent_stats": get_table_var("player_vs_opponent_stats", required=True),
    "player_last5_stats": get_table_var("player_last5_stats", required=True),
    "player_last10_stats": get_table_var("player_last10_stats", required=True),
    "player_season_last5_stats": get_table_var("player_season_last5_stats", required=True),
    "player_season_last10_stats": get_table_var("player_season_last10_stats", required=True),
    "player_ranking_profiles": get_table_var("player_ranking_profiles", required=False),

    # ------------------------------------------------------------
    # Team offensive / existing marts
    # ------------------------------------------------------------
    "team_season_stats": get_table_var("team_season_stats", required=True),
    "team_career_stats": get_table_var("team_career_stats", required=True),
    "team_vs_opponent_stats": get_table_var("team_vs_opponent_stats", required=True),
    "team_last5_stats": get_table_var("team_last5_stats", required=True),
    "team_last10_stats": get_table_var("team_last10_stats", required=True),
    "team_season_last5_stats": get_table_var("team_season_last5_stats", required=True),
    "team_season_last10_stats": get_table_var("team_season_last10_stats", required=True),
    "team_style_profiles": get_table_var("team_style_profiles", required=False),

    # ------------------------------------------------------------
    # Team defensive / opponent marts
    # Created in defensive/opponent build block
    # ------------------------------------------------------------
    "team_game_opponent_context": get_table_var("team_game_opponent_context", required=True),
    "team_defense_season_stats": get_table_var("team_defense_season_stats", required=True),
    "team_defense_career_stats": get_table_var("team_defense_career_stats", required=True),

    # ------------------------------------------------------------
    # Possession marts / QC
    # Created in updated Block 6
    # ------------------------------------------------------------
    "team_game_possession_quality": get_table_var("team_game_possession_quality", required=False),
    "game_possession_quality": get_table_var("game_possession_quality", required=False),
    "possession_field_quality": get_table_var("possession_field_quality", required=False),

    # ------------------------------------------------------------
    # Schedule/discovery
    # ------------------------------------------------------------
    "season_schedule_inventory": get_table_var("season_schedule_inventory", required=True),
    "stat_slug_inventory": get_table_var("stat_slug_inventory", required=True),
    "game_schedule_all": get_table_var("game_schedule_all", required=True),
    "game_schedule_2026": get_table_var("game_schedule_2026", required=True),

    # ------------------------------------------------------------
    # QC/logs
    # ------------------------------------------------------------
    "event_list_probe_summary": get_table_var("event_list_probe_summary", required=False),
    "game_discovery_log": get_table_var("game_discovery_log", required=False),
    "season_slug_inventory": get_table_var("season_slug_inventory", required=False),
    "api_collection_log": get_table_var("api_collection_log", required=False),
    "quality_summary": get_table_var("quality_summary", required=False),
    "defensive_opponent_build_quality": get_table_var("defensive_opponent_build_quality", required=False),
    "skipped_games": get_table_var("skipped_games", required=False),
    "orphan_stat_rows": get_table_var("orphan_stat_rows_df", required=False),
}


# ============================================================
# SAVE CURATED TABLES TO PARQUET + CSV
# ============================================================

artifact_rows = []

for name, df in curated_tables.items():
    df_safe = sanitize_dataframe_for_storage(ensure_non_empty_schema(df, name))

    parquet_path = CURATED_ALL_DIR / f"{name}.parquet"
    csv_path = CURATED_ALL_DIR / f"{name}.csv"

    df_safe.to_parquet(parquet_path, index=False)
    df_safe.to_csv(csv_path, index=False)

    artifact_rows.append({
        "table_name": name,
        "rows": len(df_safe),
        "columns": len(df_safe.columns),
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
    })

artifact_index = (
    pd.DataFrame(artifact_rows)
    .sort_values("table_name")
    .reset_index(drop=True)
)

artifact_index.to_csv(CURATED_ALL_DIR / "artifact_index.csv", index=False)

print("Saved curated table artifacts:")
display(artifact_index)


# ============================================================
# DUCKDB WAREHOUSE BUILD
# ============================================================

DB_PATH = ANALYTICS_DATABASE_DIR / "pll_warehouse.duckdb"

# Close any existing notebook connection named con if it exists.
try:
    con.close()
except Exception:
    pass

con = duckdb.connect(str(DB_PATH))

con.execute("CREATE SCHEMA IF NOT EXISTS clean;")
con.execute("CREATE SCHEMA IF NOT EXISTS marts;")
con.execute("CREATE SCHEMA IF NOT EXISTS qc;")


# ------------------------------------------------------------
# Clean schema tables
# ------------------------------------------------------------

clean_table_names = [
    "game_manifest",
    "team_game_stats",
    "player_game_stats",
    "team_alias_mapping",
    "team_directory",
    "player_directory",
    "game_schedule_all",
    "game_schedule_2026",
]


# ------------------------------------------------------------
# Marts schema tables
# ------------------------------------------------------------

mart_table_names = [
    # Player marts
    "player_season_stats_by_team",
    "player_season_stats",
    "player_career_stats",
    "player_vs_opponent_stats",
    "player_last5_stats",
    "player_last10_stats",
    "player_season_last5_stats",
    "player_season_last10_stats",
    "player_ranking_profiles",

    # Team offensive / existing marts
    "team_season_stats",
    "team_career_stats",
    "team_vs_opponent_stats",
    "team_last5_stats",
    "team_last10_stats",
    "team_season_last5_stats",
    "team_season_last10_stats",
    "team_style_profiles",

    # Possession marts
    "team_game_possession_quality",

    # Defensive / opponent marts
    "team_game_opponent_context",
    "team_defense_season_stats",
    "team_defense_career_stats",
]


# ------------------------------------------------------------
# QC schema tables
# ------------------------------------------------------------

qc_table_names = [
    "season_schedule_inventory",
    "stat_slug_inventory",
    "event_list_probe_summary",
    "game_discovery_log",
    "season_slug_inventory",
    "api_collection_log",
    "possession_field_quality",
    "game_possession_quality",
    "quality_summary",
    "defensive_opponent_build_quality",
    "skipped_games",
    "orphan_stat_rows",
]


def duckdb_load_parquet(con, schema_name, table_name):
    fp = CURATED_ALL_DIR / f"{table_name}.parquet"

    if not fp.exists():
        print(f"Skipping missing file: {fp}")
        return False

    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {schema_name}.{table_name} AS
            SELECT *
            FROM read_parquet(?);
            """,
            [str(fp)]
        )
        return True

    except Exception as e:
        print(f"FAILED loading {schema_name}.{table_name}: {e}")
        return False


load_rows = []

for table_name in clean_table_names:
    loaded = duckdb_load_parquet(con, "clean", table_name)
    load_rows.append({
        "schema": "clean",
        "table_name": table_name,
        "loaded": loaded,
    })

for table_name in mart_table_names:
    loaded = duckdb_load_parquet(con, "marts", table_name)
    load_rows.append({
        "schema": "marts",
        "table_name": table_name,
        "loaded": loaded,
    })

for table_name in qc_table_names:
    loaded = duckdb_load_parquet(con, "qc", table_name)
    load_rows.append({
        "schema": "qc",
        "table_name": table_name,
        "loaded": loaded,
    })

duckdb_load_summary = pd.DataFrame(load_rows)


# ============================================================
# WAREHOUSE INDEX + BASIC VALIDATION
# ============================================================

warehouse_tables = con.execute("""
SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema IN ('clean', 'marts', 'qc')
ORDER BY table_schema, table_name
""").df()

warehouse_tables.to_csv(CURATED_ALL_DIR / "duckdb_table_index.csv", index=False)
duckdb_load_summary.to_csv(CURATED_ALL_DIR / "duckdb_load_summary.csv", index=False)


print("DuckDB load summary:")
display(duckdb_load_summary)

print("DuckDB warehouse tables:")
display(warehouse_tables)


# ------------------------------------------------------------
# Quick row-count validation
# ------------------------------------------------------------

validation_queries = {
    "clean.game_manifest": "SELECT COUNT(*) AS rows FROM clean.game_manifest",
    "clean.team_game_stats": "SELECT COUNT(*) AS rows FROM clean.team_game_stats",
    "clean.player_game_stats": "SELECT COUNT(*) AS rows FROM clean.player_game_stats",
    "marts.team_season_stats": "SELECT COUNT(*) AS rows FROM marts.team_season_stats",
    "marts.team_defense_season_stats": "SELECT COUNT(*) AS rows FROM marts.team_defense_season_stats",
    "marts.team_game_opponent_context": "SELECT COUNT(*) AS rows FROM marts.team_game_opponent_context",
    "marts.team_game_possession_quality": "SELECT COUNT(*) AS rows FROM marts.team_game_possession_quality",
    "marts.player_ranking_profiles": "SELECT COUNT(*) AS rows FROM marts.player_ranking_profiles",
    "marts.team_style_profiles": "SELECT COUNT(*) AS rows FROM marts.team_style_profiles",
    "qc.game_possession_quality": "SELECT COUNT(*) AS rows FROM qc.game_possession_quality",
    "qc.defensive_opponent_build_quality": "SELECT COUNT(*) AS rows FROM qc.defensive_opponent_build_quality",
}

validation_rows = []

for label, sql in validation_queries.items():
    try:
        n = con.execute(sql).df()["rows"].iloc[0]
        validation_rows.append({
            "table": label,
            "rows": int(n),
            "status": "ok",
        })
    except Exception as e:
        validation_rows.append({
            "table": label,
            "rows": None,
            "status": f"error: {e}",
        })

warehouse_validation = pd.DataFrame(validation_rows)
warehouse_validation.to_csv(CURATED_ALL_DIR / "warehouse_validation_summary.csv", index=False)

print("Warehouse validation summary:")
display(warehouse_validation)


# ------------------------------------------------------------
# Useful final checks
# ------------------------------------------------------------

try:
    completed_games_by_season = con.execute("""
        SELECT
            season,
            COUNT(DISTINCT game_id) AS completed_stat_games
        FROM clean.game_manifest
        GROUP BY season
        ORDER BY season
    """).df()

    print("Completed stat games by season:")
    display(completed_games_by_season)

except Exception as e:
    print("Could not display completed games by season:", e)


try:
    defensive_check = con.execute("""
        SELECT
            season,
            COUNT(*) AS team_defense_rows,
            COUNT(DISTINCT team_id) AS teams
        FROM marts.team_defense_season_stats
        GROUP BY season
        ORDER BY season
    """).df()

    print("Defensive season rows by season:")
    display(defensive_check)

except Exception as e:
    print("Could not display defensive season rows:", e)


try:
    possession_check = con.execute("""
        SELECT
            possession_data_status,
            COUNT(*) AS games
        FROM qc.game_possession_quality
        GROUP BY possession_data_status
        ORDER BY games DESC
    """).df()

    print("Game possession QC status counts:")
    display(possession_check)

except Exception as e:
    print("Could not display possession QC status counts:", e)


con.close()

print("Curated tables saved to:", CURATED_ALL_DIR)
print("DuckDB warehouse saved to:", DB_PATH)
