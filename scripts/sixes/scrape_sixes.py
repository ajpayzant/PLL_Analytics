"""Scrape PLL 6v6 data (men): Championship Series + Global Lacrosse Games.

Runs in GitHub Actions with the repo's PLL_BEARER_TOKEN secret (never printed).
Writes raw JSON under sixes_out/raw/ plus sixes_out/summary.json, then the
workflow tars sixes_out/ and publishes it to the `sixes-data` release.

Collected:
  events?year=Y                      all events per year (every segment)
  events/{slug}                      event detail, for each non-10v10 men's event
  events/{slug}/players/stats        player box scores
  events/{slug}/teams/stats          team box scores
  players?year=Y                     season player lists (ids / positions for linking)
  a few candidate GLG roster endpoints (status recorded, kept if 200)
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

import requests

BASE = "https://api.stats.premierlacrosseleague.com/api/v4"
OUT = Path("sixes_out")
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
TENV10 = {"regular", "post", "allstar"}

token = os.environ.get("PLL_BEARER_TOKEN", "").strip().strip('"').strip("'")
if not token:
    raise SystemExit("PLL_BEARER_TOKEN missing")
if not token.lower().startswith("bearer "):
    token = "Bearer " + token

S = requests.Session()
S.headers.update({
    "authorization": token,
    "authsource": "stats",
    "origin": "https://stats.premierlacrosseleague.com",
    "referer": "https://stats.premierlacrosseleague.com/",
    "accept": "application/json",
    "user-agent": "Mozilla/5.0",
})
status = {}


def get(path, name):
    url = f"{BASE}/{path}"
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30)
        except requests.RequestException as e:
            status[path] = f"error {type(e).__name__}"
            time.sleep(2)
            continue
        time.sleep(0.4)
        status[path] = r.status_code
        if r.status_code == 200:
            data = r.json()
            (RAW / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
            return data
        if r.status_code in (429, 500, 502, 503):
            time.sleep(5 * (attempt + 1))
            continue
        return None
    return None


def items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "items", "events"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                for k2 in ("items", "events"):
                    if isinstance(v.get(k2), list):
                        return v[k2]
    return []


def is_mens(e):
    slug = str(e.get("slugname", "")).lower()
    return "women" not in slug and str(e.get("league", "PLL")).upper() != "WLL"


events = []
for y in YEARS:
    for e in items(get(f"events?year={y}&includeCS=true", f"events_{y}")):
        e["_year"] = y
        events.append(e)
    get(f"players?year={y}", f"players_{y}")

# The plain events list omits Championship Series / GLG. Try list variants
# (record counts) and then probe event slugs by the site's naming pattern.
list_variants = {}
for y in (2023, 2024, 2025, 2026):
    for q in (f"events?year={y}&seasonSegment=champseries", f"events?year={y}&segment=champseries",
              f"events?year={y}&seasonSegment=glg", f"events?year={y}&includeCS=true",
              f"events?year={y}&league=GLG", f"events?year={y}&seasonSegment=champseries&league=PLL"):
        list_variants[q] = len(items(get(q, "list_" + q.replace("?", "_").replace("&", "_").replace("=", "-"))))
known = {e.get("slugname") for e in events}
patterns = []
for y in (2023, 2024, 2025, 2026):
    patterns += [f"{y}-championship-series-ev-{i}" for i in range(1, 13)]
    patterns += [f"championship-series-{y}-ev-{i}" for i in range(1, 13)]
    patterns += [f"{y}-cs-ev-{i}" for i in range(1, 13)]
patterns = []  # Championship Series now come from includeCS=true
probed = {}
# GLG: not in the events lists. Try list flags and the games/{slug} resource the site uses.
for y in (2026, 2027):
    for q in (f"events?year={y}&includeGLG=true", f"events?year={y}&includeAll=true",
              f"events?year={y}&includeCS=true&league=GLG", f"games?year={y}", f"games?year={y}&includeCS=true"):
        list_variants[q] = len(items(get(q, "list_" + q.replace("?", "_").replace("&", "_").replace("=", "-"))))
glg_found = []
for i in range(1, 17):
    slug = f"2026-mens-glg-ev-{i}"
    for res in ("games", "events"):
        d = get(f"{res}/{slug}", f"{res}_{slug}")
        if d is not None:
            glg_found.append(f"{res}/{slug}")
            for sub in ("players/stats", "teams/stats"):
                get(f"{res}/{slug}/{sub}", f"{res}_{slug}_{sub.replace('/', '_')}")
            break
# games/{slug} for one known CS game, to compare payload shape with events/{slug}
for sub in ("", "/players/stats", "/teams/stats"):
    get(f"games/2026-championship-series-ev-1{sub}", f"games_2026-cs-ev-1{sub.replace('/', '_')}")
for slug in patterns:
    if slug in known:
        continue
    d = get(f"events/{slug}", f"ev_{slug}")
    probed[slug] = d is not None
    if d is not None:
        ev = d.get("data", d) if isinstance(d, dict) else {}
        ev = ev if isinstance(ev, dict) else {}
        ev.setdefault("slugname", slug)
        ev["_year"] = int(slug[:4]) if slug[:4].isdigit() else int(slug.split("-")[2])
        ev.setdefault("seasonSegment", ev.get("seasonSegment") or "champseries")
        events.append(ev)

segs = Counter(f"{e['_year']}|{e.get('seasonSegment')}|{e.get('league')}" for e in events)
sixes = [e for e in events if e.get("seasonSegment") not in TENV10 and is_mens(e)]
women_skipped = sum(1 for e in events if e.get("seasonSegment") not in TENV10 and not is_mens(e))

games = []
for e in sixes:
    slug = e["slugname"]
    det = get(f"events/{slug}", f"ev_{slug}") if not (RAW / f"ev_{slug}.json").exists() else json.loads((RAW / f"ev_{slug}.json").read_text())
    ps = get(f"events/{slug}/players/stats", f"ps_{slug}")
    ts = get(f"events/{slug}/teams/stats", f"ts_{slug}")
    games.append({
        "slug": slug, "year": e["_year"], "segment": e.get("seasonSegment"), "league": e.get("league"),
        "start": e.get("startTime"), "desc": e.get("description"),
        "home": (e.get("homeTeam") or {}).get("officialId") or (e.get("homeTeam") or {}).get("fullName"),
        "away": (e.get("awayTeam") or {}).get("officialId") or (e.get("awayTeam") or {}).get("fullName"),
        "home_score": e.get("homeScore"), "away_score": e.get("visitorScore", e.get("awayScore")),
        "detail_ok": det is not None, "players_ok": ps is not None, "teams_ok": ts is not None,
        "player_rows": len(items(ps)) if ps is not None else 0,
    })

# Candidate GLG roster / team endpoints: record what exists.
for y in (2026, 2027):
    for seg in ("glg", "global", "champseries"):
        get(f"players?year={y}&seasonSegment={seg}", f"players_{y}_{seg}")
        get(f"standings?year={y}&seasonSegment={seg}", f"standings_{y}_{seg}")

# 10v10 box scores for 2019-2021 (the analytics warehouse starts in 2022): needed as priors for
# veterans whose best PLL seasons predate 2022 and for 2023 Championship Series players.
early = []
for e in events:
    if e.get("_year") in (2019, 2020, 2021) and e.get("seasonSegment") in ("regular", "post", "champseries"):
        slug = e["slugname"]
        ps = get(f"events/{slug}/players/stats", f"ps10_{slug}")
        ts = get(f"events/{slug}/teams/stats", f"ts10_{slug}")
        early.append({"slug": slug, "year": e["_year"], "segment": e.get("seasonSegment"),
                      "players_ok": ps is not None, "teams_ok": ts is not None,
                      "player_rows": len(items(ps)) if ps is not None else 0})

summary = {
    "early_10v10": early,
    "segments": dict(sorted(segs.items())),
    "mens_sixes_events": len(sixes),
    "womens_sixes_skipped": women_skipped,
    "games": games,
    "list_variants": list_variants,
    "glg_found": glg_found,
    "status": status,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "status"}, indent=1)[:20000])
bad = {k: v for k, v in status.items() if v != 200}
print("non-200 endpoints:", json.dumps(bad, indent=1))
