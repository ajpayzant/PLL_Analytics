"""Live scrape of the men's Global Lacrosse Games (Dec 8-13 2026) from the PLL stats API.

Runs in GitHub Actions with the repo's PLL_BEARER_TOKEN secret (never printed). Discovers GLG events
several ways (the API did not expose them before the event), keeps the games that have a score, and
writes glg_live/:
  results.csv       game_number, slug, date, home, away, home_goals, away_goals, status, period
  player_box.csv    game_number, slug, team, official_id, first_name, last_name, position, goals, assists,
                    shots, shots_on_goal, ground_balls, turnovers, caused_turnovers, saves, goals_against, touches
  team_box.csv      per team-game API team stats (possessions, touches, turnovers, ...)
  status.json       discovery log (which endpoints answered), scrape time
Women's events are skipped.
"""
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.stats.premierlacrosseleague.com/api/v4"
OUT = Path("glg_live")
OUT.mkdir(exist_ok=True)
token = os.environ.get("PLL_BEARER_TOKEN", "").strip().strip('"').strip("'")
if not token:
    raise SystemExit("PLL_BEARER_TOKEN missing")
if not token.lower().startswith("bearer "):
    token = "Bearer " + token
S = requests.Session()
S.headers.update({"authorization": token, "authsource": "stats", "origin": "https://stats.premierlacrosseleague.com",
                  "referer": "https://stats.premierlacrosseleague.com/", "accept": "application/json", "user-agent": "Mozilla/5.0"})
log = {}


def get(path):
    for attempt in range(3):
        try:
            r = S.get(f"{BASE}/{path}", timeout=30)
        except requests.RequestException as e:
            log[path] = f"error {type(e).__name__}"
            time.sleep(2)
            continue
        log[path] = r.status_code
        time.sleep(0.3)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(4 * (attempt + 1))
            continue
        return None
    return None


def items(d):
    d = d.get("data", d) if isinstance(d, dict) else d
    if isinstance(d, dict):
        d = d.get("items", d.get("events", d))
    return d if isinstance(d, list) else ([d] if isinstance(d, dict) and d else [])


def is_glg_mens(e):
    slug = str(e.get("slugname", "")).lower()
    seg = str(e.get("seasonSegment", "")).lower()
    lg = str(e.get("league", "")).upper()
    if "women" in slug or lg == "WLL":
        return False
    return "glg" in slug or "global" in slug or seg in ("glg", "global") or lg == "GLG"


# ---- discovery: list endpoints, then known slug pattern (from the PLL site: 2026-mens-glg-ev-N)
found = {}
for q in ("events?year=2026&includeCS=true", "events?year=2026&includeGLG=true", "events?year=2026&includeAll=true",
          "events?year=2026&seasonSegment=glg", "events?year=2026&league=GLG", "events?year=2027&includeCS=true"):
    for e in items(get(q)) if get(q) is not None else []:
        if isinstance(e, dict) and is_glg_mens(e):
            found[e["slugname"]] = e
for i in range(1, 8):
    slug = f"2026-mens-glg-ev-{i}"
    if slug in found:
        continue
    for res in ("events", "games"):
        d = get(f"{res}/{slug}")
        if d is not None:
            ev = d.get("data", d) if isinstance(d, dict) else {}
            if isinstance(ev, dict):
                ev.setdefault("slugname", slug)
                ev["_res"] = res
                found[slug] = ev
            break


def gnum(slug, e):
    m = re.search(r"ev-(\d+)$", slug)
    return int(m.group(1)) if m else int(e.get("gameNumber") or 0)


results, players, teams = [], [], []
for slug, e in sorted(found.items(), key=lambda kv: gnum(*kv)):
    res = e.get("_res", "events")
    det = get(f"{res}/{slug}") or {}
    det = det.get("data", det) if isinstance(det, dict) else {}
    ev = {**e, **(det if isinstance(det, dict) else {})}
    home = ev.get("homeTeam") or {}
    away = ev.get("awayTeam") or {}
    hs, as_ = ev.get("homeScore"), ev.get("visitorScore", ev.get("awayScore"))
    status = ev.get("eventStatus", ev.get("gameStatus"))
    start = ev.get("startTime")
    date = datetime.fromtimestamp(int(start), tz=timezone.utc).strftime("%Y-%m-%d") if start else ""
    n = gnum(slug, ev)
    results.append({"game_number": n, "slug": slug, "date": date,
                    "home": home.get("fullName") or home.get("officialId") or "", "home_id": home.get("officialId", ""),
                    "away": away.get("fullName") or away.get("officialId") or "", "away_id": away.get("officialId", ""),
                    "home_goals": hs, "away_goals": as_, "status": status, "period": ev.get("period"),
                    "clock": f"{ev.get('clockMinutes')}:{ev.get('clockSeconds')}"})
    for p in items(get(f"{res}/{slug}/players/stats")):
        players.append({"game_number": n, "slug": slug, "team": p.get("teamId"), "official_id": p.get("officialId"),
                        "first_name": p.get("firstName"), "last_name": p.get("lastName"), "position": p.get("position"),
                        "goals": p.get("goals"), "assists": p.get("assists"), "shots": p.get("shots"),
                        "shots_on_goal": p.get("shotsOnGoal"), "ground_balls": p.get("groundBalls"),
                        "turnovers": p.get("turnovers"), "caused_turnovers": p.get("causedTurnovers"),
                        "saves": p.get("saves"), "goals_against": p.get("goalsAgainst"), "touches": p.get("touches"),
                        "two_point_goals": p.get("twoPointGoals")})
    for t in items(get(f"{res}/{slug}/teams/stats")):
        teams.append({"game_number": n, "slug": slug, **{k: v for k, v in t.items() if not isinstance(v, (dict, list))}})


def write(name, rows):
    if not rows:
        (OUT / name).write_text("", encoding="utf-8")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


write("results.csv", results)
write("player_box.csv", players)
write("team_box.csv", teams)
status = {"scraped_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games_found": len(found),
          "games_with_scores": sum(1 for r in results if r["home_goals"] is not None), "player_rows": len(players),
          "endpoints": log}
(OUT / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in status.items() if k != "endpoints"}, indent=1))
