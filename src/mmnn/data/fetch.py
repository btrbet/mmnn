# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Fetch NCAA tournament data from Sports Reference."""
import csv
import io
import re
import sys
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from sportsipy.ncaab.teams import Team, Teams

from mmnn import paths

BASE_URL = "https://www.sports-reference.com"


def _gender_segment(women: bool) -> str:
    return "women" if women else "men"


def _tournament_url(year: int, women: bool) -> str:
    g = _gender_segment(women)
    return f"{BASE_URL}/cbb/postseason/{g}/{year}-ncaa.html"


def _school_stats_url(year: int, women: bool) -> str:
    g = _gender_segment(women)
    return f"{BASE_URL}/cbb/seasons/{g}/{year}-school-stats.html"

# Sports Reference school_id -> sportsipy abbreviation (for teams that don't match by name)
SCHOOL_ID_TO_ABBR: dict[str, str] = {
    "fairleigh-dickinson": "FDU",
    "texas-am": "TEXAM",
    "miami-fl": "MIA-FL",
    "pittsburgh": "PITT",
    "southern-california": "USC",
    "louisiana-lafayette": "ULL",
    "saint-marys-ca": "SMC",
    "north-carolina": "UNC",
    "connecticut": "UCONN",
    "brigham-young": "BYU",
    "mississippi": "MISS",
    "virginia-commonwealth": "VCU",
    "oral-roberts": "ORU",
    "texas-southern": "TXSO",
    "northern-kentucky": "NKU",
    "kennesaw-state": "KSU",
    "north-carolina-wilmington": "UNCW",
    "north-carolina-asheville": "UNCA",
    "california-san-diego": "UCSD",
    "saint-francis-pa": "SFPA",
    "st-johns-ny": "STJ",
    "nebraska-omaha": "OMAHA",
}

# Bracket display names that map to sportsipy abbreviations or name fragments
BRACKET_NAME_ALIASES: dict[str, str] = {
    "FDU": "Fairleigh Dickinson",
    "Pitt": "Pittsburgh",
    "UConn": "Connecticut",
    "UNC": "North Carolina",
    "BYU": "Brigham Young",
    "USC": "Southern California",
    "Ole Miss": "Mississippi",
    "Saint Mary's": "Saint Mary's (CA)",
    "Saint Mary's (CA)": "Saint Mary's (CA)",
    "Louisiana": "Louisiana-Lafayette",
    "Miami (FL)": "Miami (FL)",
    "Texas A&M": "Texas A&M",
    "UNC Asheville": "North Carolina-Asheville",
    "UC-San Diego": "California-San Diego",
}

TEAMS_CSV_COLUMNS = [
    "ID", "Year", "Team", "Rank", "Wins", "Losses", "WL%",
    "SOS", "SRS", "FG/G", "OREB/G", "Total Points", "FGA", "FTA",
    "AST", "TOV", "TS%", "TOV%", "AST%", "URL",
]
GAMES_CSV_COLUMNS = ["Team 1", "Team 1 Score", "Team 2", "Team 2 Score", "Winner"]


@dataclass
class BracketTeam:
    """A team appearance in the bracket with seed and optional score."""
    name: str
    seed: int
    school_id: str
    score: int | None = None


@dataclass
class BracketGame:
    """A single game in the bracket."""
    team1: BracketTeam
    team2: BracketTeam


def _fetch_tournament_page(year: int, women: bool) -> str:
    """Fetch the NCAA tournament bracket page HTML."""
    url = _tournament_url(year, women)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_bracket(
    html: str, year: int, women: bool
) -> tuple[list[BracketGame], dict[str, int], dict[str, str]]:
    """
    Parse bracket HTML to extract games, team seeds, and school IDs.

    Returns:
        - List of BracketGame (team1, team2 with names and scores)
        - Dict of team_name -> best_seed
        - Dict of team_name -> school_id (for URL construction)
    """
    soup = BeautifulSoup(html, "html.parser")
    games: list[BracketGame] = []
    team_seeds: dict[str, int] = {}
    team_school_ids: dict[str, str] = {}

    # Pattern: seed, school link (team name), boxscore link (score)
    # Find all links to schools and boxscores in the bracket area
    g = _gender_segment(women)
    school_pattern = re.compile(rf"/cbb/schools/([^/]+)/{g}/(\d+)\.html")
    boxscore_pattern = re.compile(r"/cbb/boxscores/")

    # Get main content - bracket is typically in #content or similar
    content = soup.find(id="content") or soup
    all_links = content.find_all("a", href=True)

    entries: list[BracketTeam] = []
    i = 0
    while i < len(all_links):
        a = all_links[i]
        href = a.get("href", "")
        school_match = school_pattern.search(href)
        if school_match:
            school_id = school_match.group(1)
            link_year = int(school_match.group(2))
            if link_year != year:
                i += 1
                continue
            team_name = a.get_text(strip=True)
            if not team_name:
                i += 1
                continue
            # Look backwards for seed (digit before this link, within same parent)
            seed = 16
            parent = a.parent
            if parent:
                prev_text = ""
                for prev in a.previous_siblings:
                    if hasattr(prev, "get_text"):
                        prev_text = prev.get_text() + prev_text
                    elif isinstance(prev, str):
                        prev_text = prev + prev_text
                seed_match = re.search(r"(\d{1,2})\s*$", prev_text)
                if seed_match:
                    seed = int(seed_match.group(1))
            # Look for following boxscore link for score
            score: int | None = None
            next_elem = a.next_sibling
            scan_count = 0
            while next_elem is not None and scan_count < 5:
                if hasattr(next_elem, "name") and next_elem.name == "a":
                    href = getattr(next_elem, "get", lambda k: "")( "href") or ""
                    if boxscore_pattern.search(href) and next_elem.get_text(strip=True).isdigit():
                        try:
                            score = int(next_elem.get_text(strip=True))
                            break
                        except (ValueError, TypeError):
                            pass
                if hasattr(next_elem, "find_all"):
                    box_links = next_elem.find_all("a", href=boxscore_pattern)
                    for bl in box_links:
                        try:
                            score = int(bl.get_text(strip=True))
                            break
                        except (ValueError, TypeError):
                            pass
                elif isinstance(next_elem, str) and next_elem.strip():
                    num_match = re.search(r"^(\d+)\s*$", next_elem.strip())
                    if num_match:
                        score = int(num_match.group(1))
                        break
                next_elem = getattr(next_elem, "next_sibling", None)
                scan_count += 1
            # Also check next sibling element (same parent)
            if score is None and parent:
                for sibling in parent.find_next_siblings():
                    for bl in sibling.find_all("a", href=boxscore_pattern):
                        try:
                            score = int(bl.get_text(strip=True))
                            break
                        except (ValueError, TypeError):
                            pass
                    if score is not None:
                        break
            # If still no score, check if next link is boxscore (same parent text)
            if score is None:
                next_link = a.find_next("a", href=boxscore_pattern)
                if next_link and next_link.get_text(strip=True).isdigit():
                    score = int(next_link.get_text(strip=True))
            entries.append(BracketTeam(name=team_name, seed=seed, school_id=school_id, score=score))
            team_seeds[team_name] = min(team_seeds.get(team_name, 16), seed)
            if team_name not in team_school_ids:
                team_school_ids[team_name] = school_id
        i += 1

    # Pair entries into games: consecutive pairs with scores form games
    # First Four: "A score, B score" - single "game" with two teams
    # Standard: two consecutive lines, each with team+score
    j = 0
    while j < len(entries):
        e1 = entries[j]
        if j + 1 < len(entries):
            e2 = entries[j + 1]
            # Both have scores -> standard game or same-game (First Four on one line)
            if e1.score is not None and e2.score is not None:
                games.append(BracketGame(team1=e1, team2=e2))
                j += 2
                continue
            # e1 has score, e2 doesn't - e2 might be next round matchup, skip e2 for now
            # Or e2 is "team advances" placeholder - look for paired score
            if e1.score is not None:
                # Find next entry with score
                k = j + 1
                while k < len(entries) and entries[k].score is None:
                    k += 1
                if k < len(entries):
                    games.append(BracketGame(team1=e1, team2=entries[k]))
                    j = k + 1
                    continue
        j += 1

    return games, team_seeds, team_school_ids


def _build_team_lookup(year: int, women: bool) -> dict[str, "object"]:
    """Build mapping from team display names to sportsipy Team objects (men's only)."""
    lookup: dict[str, object] = {}
    if women:
        print(
            "Women's mode: using Sports Reference stats only (sportsipy is men's D-I).",
            file=sys.stderr,
        )
        return lookup
    try:
        with redirect_stderr(io.StringIO()):
            teams = Teams(year)
    except Exception as e:
        print(f"Warning: Could not load sportsipy Teams({year}): {e}", file=sys.stderr)
        return lookup

    for team in teams:
        name = team.name
        abbr = team.abbreviation
        # Primary name: first part before common suffixes
        for suffix in [" Boilermakers", " Owls", " Wildcats", " Tigers", " Bulldogs",
                       " Seminoles", " Tar Heels", " Jayhawks", " Cougars", " Huskies",
                       " Shockers", " Fighting Irish", " Bruins", " Trojans", " Aggies",
                       " Bears", " Volunteers", " Crimson Tide", " Bulldogs", " Spartans",
                       " Hoosiers", " Hawkeyes", " Badgers", " Buckeyes", " Wolverines",
                       " Golden Eagles", " Pirates", " Midshipmen", " Cavaliers",
                       " Knights", " Bearcats", " Musketeers", " Hoyas", " Friars",
                       " Red Storm", " Blue Demons", " Panthers", " Orange", " Cardinals",
                       " Mountaineers", " Demon Deacons", " Terrapins", " Yellow Jackets",
                       " Hurricanes", " Hokies", " Wolfpack", " Commodores", " Rebels",
                       " Razorbacks", " Gators", " Gamecocks", " Volunteers", " Volunteers",
                       " Runnin' Rebels", " Lobos", " Aztecs", " Gaels", " Broncos",
                       " Zags", " Ducks", " Beavers", " Buffaloes", " Utes", " Sun Devils",
                       " Wildcats", " Trojans", " Cardinal", " Bruins"]:
            if name.endswith(suffix):
                short = name[: -len(suffix)].strip()
                if short and short not in lookup:
                    lookup[short] = team
                break
        if name and name not in lookup:
            lookup[name] = team
        if abbr and abbr not in lookup:
            lookup[abbr] = team
        # Aliases
        for bracket_name, canonical in BRACKET_NAME_ALIASES.items():
            if canonical in (name, abbr) or name.startswith(canonical) or canonical in name:
                if bracket_name not in lookup:
                    lookup[bracket_name] = team

    return lookup


def _scrape_team_stats(year: int, women: bool) -> dict[str, dict]:
    """
    Scrape team stats from Sports Reference school-stats page.
    TS%, TOV%, and AST% are calculated from basic stats (not on the page).
    Returns dict mapping school_id -> stats dict (compatible with _team_to_row).
    """
    g = _gender_segment(women)
    school_pattern = re.compile(rf"/cbb/schools/([^/]+)/{g}/(\d+)\.html")
    result: dict[str, dict] = {}

    def _parse_float(s: str) -> float:
        try:
            return float(s.strip())
        except (ValueError, TypeError):
            return 0.0

    def _parse_int(s: str) -> int:
        try:
            return int(s.strip())
        except (ValueError, TypeError):
            return 0

    def _parse_row(tr, stats_map: list[tuple[str, str, str]]) -> None:
        school_cell = tr.find("td", {"data-stat": "school_name"}) or tr.find(
            "th", {"data-stat": "school_name"}
        )
        if not school_cell:
            return
        a = school_cell.find("a", href=True)
        if not a:
            return
        m = school_pattern.search(a.get("href", ""))
        if not m or int(m.group(2)) != year:
            return
        sid = m.group(1).lower()
        if sid not in result:
            result[sid] = {}
        for data_stat, canonical_key, typ in stats_map:
            cell = tr.find("td", {"data-stat": data_stat})
            if not cell:
                continue
            text = cell.get_text(strip=True).replace("%", "")
            if typ == "float":
                result[sid][canonical_key] = _parse_float(text)
            else:
                result[sid][canonical_key] = _parse_int(text)

    stats_map = [
        ("g", "games_played", "int"), ("wins", "wins", "int"), ("losses", "losses", "int"),
        ("srs", "simple_rating_system", "float"), ("sos", "strength_of_schedule", "float"),
        ("fg", "field_goals", "int"), ("fga", "field_goal_attempts", "int"),
        ("fta", "free_throw_attempts", "int"), ("orb", "offensive_rebounds", "int"),
        ("ast", "assists", "int"), ("tov", "turnovers", "int"),
        ("pts", "points", "int"), ("tm", "points", "int"),  # pts/tm = team total points
    ]
    try:
        resp = requests.get(_school_stats_url(year, women), timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Warning: Could not scrape team stats: {e}", file=sys.stderr)
        return result
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find(
        "table",
        id=re.compile(r"school_stats|basic_school_stats"),
    )
    if not table:
        return result
    tbody = table.find("tbody")
    if not tbody:
        return result
    for tr in tbody.find_all("tr"):
        _parse_row(tr, stats_map)

    # Calculate TS%, TOV%, AST% from basic stats
    for sid, row in result.items():
        pts = row.get("points", 0) or 0
        fga = row.get("field_goal_attempts", 0) or 0
        fta = row.get("free_throw_attempts", 0) or 0
        tov = row.get("turnovers", 0) or 0
        ast = row.get("assists", 0) or 0
        fg = row.get("field_goals", 0) or 0
        tsa = fga + 0.44 * fta
        poss = fga + 0.44 * fta + tov
        row["true_shooting_percentage"] = pts / (2 * tsa) if tsa > 0 else 0.0
        row["turnover_percentage"] = (100 * tov / poss) if poss > 0 else 0.0
        row["assist_percentage"] = (100 * ast / fg) if fg > 0 else 0.0

    return result


def _resolve_team(
    team: BracketTeam, lookup: dict[str, object], year: int, women: bool
) -> object | None:
    """Resolve bracket team to sportsipy Team object or None (use scraped stats after)."""
    name = team.name.strip()
    # Try aliases first
    canonical = BRACKET_NAME_ALIASES.get(name, name)
    if canonical in lookup:
        return lookup[canonical]
    if name in lookup:
        return lookup[name]
    # Try partial match
    for key, t in lookup.items():
        if key in name or name in key:
            return t
    if women:
        return None
    from sportsipy.ncaab.teams import Team

    # Fallback: try Team(abbreviation, year) using school_id (men's sportsipy only)
    school_id = (team.school_id or "").strip().lower()
    if school_id:
        abbr = SCHOOL_ID_TO_ABBR.get(school_id)
        if not abbr:
            abbr = school_id.upper().replace("-", "")[:10]
        try:
            return Team(abbr, year=str(year))
        except (ValueError, KeyError, TypeError):
            pass
    return None


def _fallback_school_id(team_obj: object) -> str:
    """Fallback: derive school ID from sportsipy abbreviation when bracket lacks it."""
    abbr = getattr(team_obj, "abbreviation", None) or ""
    return abbr.lower().replace(" ", "-") if abbr else ""


def _get_stat(obj: object, key: str, default: float | int = 0) -> float | int:
    """Get stat from sportsipy Team or scraped dict."""
    if isinstance(obj, dict):
        return obj.get(key, default) or default
    return getattr(obj, key, default) or default


def _team_to_row(
    team_obj: object,
    team_name: str,
    seed: int,
    year: int,
    row_id: int,
    school_id: str,
    women: bool,
) -> dict:
    """Convert sportsipy Team or scraped stats dict to CSV row dict."""
    gp = _get_stat(team_obj, "games_played", 1) or 1
    wins = _get_stat(team_obj, "wins", 0)
    losses = _get_stat(team_obj, "losses", 0)
    wl_pct = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0
    fg_per_g = round((_get_stat(team_obj, "field_goals", 0) or 0) / gp, 1)
    oreb_per_g = round((_get_stat(team_obj, "offensive_rebounds", 0) or 0) / gp, 1)
    ts_pct = _get_stat(team_obj, "true_shooting_percentage", 0)
    tov_pct = _get_stat(team_obj, "turnover_percentage", 0)
    ast_pct = _get_stat(team_obj, "assist_percentage", 0)
    g = _gender_segment(women)
    url = f"{BASE_URL}/cbb/schools/{school_id}/{g}/{year}.html"
    return {
        "ID": row_id,
        "Year": year,
        "Team": team_name,
        "Rank": seed,
        "Wins": wins,
        "Losses": losses,
        "WL%": wl_pct,
        "SOS": _get_stat(team_obj, "strength_of_schedule", 0),
        "SRS": _get_stat(team_obj, "simple_rating_system", 0),
        "FG/G": fg_per_g,
        "OREB/G": oreb_per_g,
        "Total Points": _get_stat(team_obj, "points", 0),
        "FGA": _get_stat(team_obj, "field_goal_attempts", 0),
        "FTA": _get_stat(team_obj, "free_throw_attempts", 0),
        "AST": _get_stat(team_obj, "assists", 0),
        "TOV": _get_stat(team_obj, "turnovers", 0),
        "TS%": ts_pct,
        "TOV%": tov_pct,
        "AST%": ast_pct,
        "URL": url,
    }


def fetch_year(year: int, data_dir: Path | None = None, *, women: bool = False) -> None:
    """
    Fetch NCAA tournament data for the specified year.

    Fetches bracket from Sports Reference, uses sportsipy for men's team stats,
    and writes YEAR-teams.csv and YEAR-games.csv under data/men or data/women.
    """
    if data_dir is None:
        data_dir = paths.data_dir(women=women)
    data_dir.mkdir(parents=True, exist_ok=True)

    if year < 2002 or year > 2030:
        raise ValueError(f"Year {year} out of supported range (2002-2030)")

    print(f"Fetching tournament bracket for {year}...", file=sys.stderr)
    html = _fetch_tournament_page(year, women)
    games, team_seeds, team_school_ids = _parse_bracket(html, year, women)

    print(f"Building team lookup via sportsipy...", file=sys.stderr)
    lookup = _build_team_lookup(year, women)

    print(f"Scraping team stats from Sports Reference...", file=sys.stderr)
    scraped_stats = _scrape_team_stats(year, women)

    # Dedupe teams by display name, keep best (lowest) seed
    teams_seen: set[str] = set()
    teams_data: list[tuple[str, int, object, str]] = []
    for team_name, seed in team_seeds.items():
        canonical = BRACKET_NAME_ALIASES.get(team_name, team_name)
        if canonical in teams_seen:
            continue
        school_id = team_school_ids.get(team_name, "").strip().lower()
        bt = BracketTeam(name=team_name, seed=seed, school_id=school_id, score=None)
        team_obj = _resolve_team(bt, lookup, year, women)
        if team_obj is None and school_id and school_id in scraped_stats:
            team_obj = scraped_stats[school_id]
        if team_obj is None:
            print(f"Warning: Could not resolve team '{team_name}'", file=sys.stderr)
            continue
        teams_seen.add(canonical)
        school_id = team_school_ids.get(team_name) or _fallback_school_id(team_obj) or school_id
        if isinstance(team_obj, dict) and not school_id:
            school_id = team_school_ids.get(team_name, "")
        teams_data.append((team_name, seed, team_obj, school_id))

    # Write teams CSV
    teams_path = data_dir / f"{year}-teams.csv"
    teams_rows: list[dict] = []
    for idx, (team_name, seed, team_obj, school_id) in enumerate(teams_data, 1):
        row = _team_to_row(team_obj, team_name, seed, year, idx, school_id, women)
        teams_rows.append(row)
    with teams_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEAMS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(teams_rows)
    print(f"Wrote {teams_path}", file=sys.stderr)

    # Write games CSV
    games_path = data_dir / f"{year}-games.csv"
    games_rows: list[dict] = []
    for g in games:
        t1, t2 = g.team1, g.team2
        s1, s2 = t1.score or 0, t2.score or 0
        winner = t1.name if s1 > s2 else t2.name
        games_rows.append({
            "Team 1": t1.name,
            "Team 1 Score": s1,
            "Team 2": t2.name,
            "Team 2 Score": s2,
            "Winner": winner,
        })
    with games_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GAMES_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(games_rows)
    print(f"Wrote {games_path}", file=sys.stderr)
