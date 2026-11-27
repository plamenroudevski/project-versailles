"""
season.py: the season breakdown. Cuts every team's best qualifying lap into
12 areas shared by the whole field, saves one small table per event, and
adds them up over a season: where does each team lose or gain time, and
what kind of car is it (the efficiency map)?

Pipeline position:  data.py -> analysis.py -> season.py -> charts.py -> app.py

The analysis is pure functions: tables in, tables out, no downloads and no
Streamlit (tested with made-up laps in tests/test_season.py). The download
section at the bottom builds the season offline, because a season is about
1,000 OpenF1 requests (30-40 minutes). The app only reads the saved files:

    python season.py 2026            # add any new 2026 events to seasons/2026/
    python season.py 2026 --force    # rebuild every event

Steps for one event:
    1. align_field   every team's best lap on one shared distance grid (analysis.py)
    2. field_trace   the field's median speed, throttle and brake at each metre
    3. phase_map     label every metre: braking, cornering, traction, lift & coast,
                     full throttle or energy-limited, with a corner speed band
    4. event_summary each team's time in each area, plus corner/straight speeds

Sign convention everywhere: gap = the team's time minus the comparison's.
Positive = the team is slower (losing time).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import analysis
import data
from analysis import (BRAKE_ON_LEVEL, DEPLOY_DROP_KMH, DEPLOY_FLAT_GAIN_KMH, DEPLOY_MIN_RUN_M,
                      HIGH_SPEED_MIN_KMH, LOW_SPEED_MAX_KMH)

# =============================================================================
# CONFIG: every threshold the season breakdown adds, in one place.
# Shared thresholds (full throttle, brake level, speed bands, energy drop)
# come from analysis.py, so both pages agree on what they mean.
# =============================================================================

MIN_FULL_GAP_M = 50
# Two corner complexes separated by less than this much full throttle are one
# complex (linked corners like a chicane): the car barely straightens up.

MIN_COMPLEX_M = 20
# A dip off full throttle shorter than this is sensor wobble, not a corner:
# at 250 km/h, 20 m is under 0.3 s, about one sample.

TRACTION_START_PCT = 10
# After the slowest point of a corner, traction starts where the field's
# median throttle reaches this. Below it the driver is still balancing the
# car mid-corner; above it they are asking the rear tyres for drive.

BAND_LEEWAY_KMH = 30
MINIMAL_BRAKING_M = 30
HEAVY_DROP_KMH = 150
# Leeway on the 120 / 200 km/h corner bands, from Flow Racers' braking-based
# definition (flowracers.com/blog/f1-corner-types). A corner within 30 km/h
# under 200 (from 170) that needs at most a brake tap (30 m is about one
# sample at 4 Hz) is fast, like the Suzuka esses (179 km/h, 26 m of braking).
# A corner within 30 km/h over 120 (under 150) that the field slows down for
# by 150 km/h or more is slow, like a hairpin at the end of a long straight.

# =============================================================================

AREAS = [
    "Braking · slow", "Braking · medium", "Braking · fast",
    "Cornering · slow", "Cornering · medium", "Cornering · fast",
    "Traction · slow", "Traction · medium", "Traction · fast",
    "Lift & coast", "Full throttle", "Energy-limited",
]
BANDED = ("Braking", "Cornering", "Traction")


def field_trace(laps):
    """The field's typical lap: the median of every aligned lap at each metre.

    `laps` are aligned laps (analysis.align_field). Speed, throttle and
    position take the median, so one odd lap can't drag the map around.
    Brake is a vote: braking where at least BRAKE_ON_LEVEL of the laps brake.
    """
    def stack(column):
        return np.vstack([lap[column].to_numpy(dtype=float) for lap in laps])

    trace = pd.DataFrame({"Distance": laps[0]["Distance"].to_numpy()})
    for column in ("Speed", "Throttle", "X", "Y"):
        trace[column] = np.median(stack(column), axis=0)
    trace["Brake"] = stack("Brake").mean(axis=0) >= BRAKE_ON_LEVEL
    return trace


def runs(mask):
    """Start and end (exclusive) positions of each unbroken stretch of True values."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def band(min_speed, braking_m=0.0, drop_kmh=0.0):
    """Corner speed band: the slowest speed first, braking decides near the edges.

    The general bands are analysis.py's: slow under LOW_SPEED_MAX_KMH (120),
    fast from HIGH_SPEED_MIN_KMH (200). Flow Racers' F1 corner types define
    corners by braking instead ("a fast corner ... requires minimal to no
    braking"; slow corners make the driver slow down heavily), so within
    BAND_LEEWAY_KMH of an edge:
      - a corner with at most MINIMAL_BRAKING_M of braking counts as fast
      - a corner the field slows down for by HEAVY_DROP_KMH or more counts as slow
    `braking_m` = metres the field brakes in the corner; `drop_kmh` = entry
    speed minus slowest speed.
    """
    if min_speed < LOW_SPEED_MAX_KMH:
        return "slow"
    if min_speed >= HIGH_SPEED_MIN_KMH:
        return "fast"
    if braking_m <= MINIMAL_BRAKING_M and min_speed >= HIGH_SPEED_MIN_KMH - BAND_LEEWAY_KMH:
        return "fast"
    if drop_kmh >= HEAVY_DROP_KMH and min_speed < LOW_SPEED_MAX_KMH + BAND_LEEWAY_KMH:
        return "slow"
    return "medium"


def corner_complexes(trace):
    """Where the field is off full throttle: [(start, end)] grid positions, after
    joining linked corners (MIN_FULL_GAP_M) and dropping wobbles (MIN_COMPLEX_M)."""
    d = trace["Distance"].to_numpy()
    joined = []
    for start, end in runs(~analysis.flat_out(trace).to_numpy()):
        if joined and d[start] - d[joined[-1][1] - 1] < MIN_FULL_GAP_M:
            joined[-1] = (joined[-1][0], end)
        else:
            joined.append((start, end))
    return [(s, e) for s, e in joined if d[e - 1] - d[s] >= MIN_COMPLEX_M]


def phase_map(trace, corners):
    """Label every metre of the lap with what the field is doing there.

    `trace` is field_trace's output; `corners` are MultiViewer markers
    (data.get_corners) or None. Returns one row per grid metre:
      Phase    Braking, Cornering, Traction, Lift & coast, Full throttle or Energy-limited
      Band     slow / medium / fast for corner phases, None on straights
      Complex  corner complex number, -1 on straights
      Corner   corner names inside the complex, e.g. "T4/T5"
      Area     one of AREAS, e.g. "Traction · slow"

    Inside a corner complex (from leaving full throttle to being back on it):
    lift & coast before the brake, braking, cornering from brake release (or
    from the lift, if there's no braking) until the throttle reaches
    TRACTION_START_PCT after the slowest point, then traction. On straights,
    the part after the speed peaks (or levels off) is energy-limited, judged
    like analysis.straight_metrics judges a straight.
    """
    d = trace["Distance"].to_numpy()
    speed, throttle, brake = (trace[c].to_numpy() for c in ("Speed", "Throttle", "Brake"))
    n = len(trace)
    phase = np.full(n, "Full throttle", dtype=object)
    bands = np.full(n, None, dtype=object)
    complex_id = np.full(n, -1)
    tags = np.full(n, "", dtype=object)
    located = None if corners is None else analysis.locate_corners(corners, trace)

    for number, (s, e) in enumerate(corner_complexes(trace)):
        complex_id[s:e] = number
        apex = s + int(np.argmin(speed[s:e]))
        braking = s + np.flatnonzero(brake[s:e])
        bands[s:e] = band(speed[apex], float(brake[s:e].sum() * analysis.GRID_STEP_M), speed[s] - speed[apex])
        corner_from = s
        if len(braking):
            phase[s:braking[0]] = "Lift & coast"
            phase[braking[0]:braking[-1] + 1] = "Braking"
            corner_from = braking[-1] + 1
        pushing = apex + np.flatnonzero(throttle[apex:e] >= TRACTION_START_PCT)
        traction_from = max(pushing[0] if len(pushing) else e, corner_from)
        phase[corner_from:traction_from] = "Cornering"
        phase[traction_from:e] = "Traction"
        if located is not None:
            inside = located[(located["Distance"] >= d[s]) & (located["Distance"] <= d[e - 1])]
            tags[s:e] = "/".join(inside["Label"])

    # Straights. The lap array starts and ends at the line, so a straight
    # across the line is two runs here, never one.
    for s, e in runs(complex_id == -1):
        if d[e - 1] - d[s] < DEPLOY_MIN_RUN_M:
            continue
        verdict = analysis.straight_metrics(trace, d[s], d[e - 1])["deployment"]
        if verdict == "drops":
            phase[s + int(np.argmax(speed[s:e])):e] = "Energy-limited"
        elif verdict == "flattens":
            level = s + np.flatnonzero(speed[s:e] >= speed[e - 1] - DEPLOY_FLAT_GAIN_KMH)[0]
            phase[level:e] = "Energy-limited"

    area = np.array([f"{p} · {b}" if p in BANDED else p for p, b in zip(phase, bands)], dtype=object)
    return pd.DataFrame({"Distance": d, "Phase": phase, "Band": bands, "Complex": complex_id,
                         "Corner": tags, "Area": area})


def straight_runs(phases):
    """Grid positions [(start, end)] of the field's straights long enough to judge (DEPLOY_MIN_RUN_M)."""
    d = phases["Distance"].to_numpy()
    return [(s, e) for s, e in runs(phases["Complex"].to_numpy() == -1) if d[e - 1] - d[s] >= DEPLOY_MIN_RUN_M]


def event_summary(teams, laps, phases):
    """One row per team for one event: the table saved as the event's CSV.

    `teams` has one row per team (team, team_colour, driver, lap_s, teammate,
    teammate_s), in the same order as `laps` (aligned with align_field);
    `phases` is phase_map's output for this event. Adds:
      one column per name in AREAS  seconds spent in that area (0.0 if the
                                    circuit has none); they add up to the lap
      fast_corner_kmh  mean speed through "Cornering · fast" (NaN if none)
      straight_kmh     mean of the lap's top speed on each long straight. The
                       top is always before any energy drop, so running out of
                       energy can't pass for drag (NaN if no long straight)
      energy_m         metres from the lap's own top speed to the end of each
                       long straight where the speed then falls at least
                       DEPLOY_DROP_KMH: more metres = the car runs out earlier
    """
    area = phases["Area"].to_numpy()
    fast = area == "Cornering · fast"
    straights = straight_runs(phases)
    d = phases["Distance"].to_numpy()
    rows = []
    for lap in laps:
        # Time spent between each metre and the next, counted in the first metre's area.
        step = np.diff(lap["Time"].to_numpy())
        speed = lap["Speed"].to_numpy()
        row = {name: float(step[area[:-1] == name].sum()) for name in AREAS}
        row["fast_corner_kmh"] = float(speed[fast].mean()) if fast.any() else np.nan
        tops, energy = [], 0.0
        for s, e in straights:
            peak = s + int(np.argmax(speed[s:e]))
            tops.append(speed[peak])
            if speed[peak] - speed[e - 1] >= DEPLOY_DROP_KMH:
                energy += d[e - 1] - d[peak]
        row["straight_kmh"] = float(np.mean(tops)) if tops else np.nan
        row["energy_m"] = energy
        rows.append(row)
    return pd.concat([teams.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---- A season: the saved event tables added up --------------------------------

SEASONS_DIR = Path(__file__).parent / "seasons"
EVENT_FILE = re.compile(r"^(?P<round>\d{2})_(?P<event>.+)\.csv$")  # e.g. 03_suzuka.csv
EXCLUDED_COLUMNS = ["round", "event", "team", "reason"]

LEVEL_S = analysis.PATTERN_MIN_TIME_S
# Season gaps smaller than this count as level: not a loss, not a gain.


def list_seasons(root=SEASONS_DIR):
    """Years with at least one saved event, newest first; [] if there are none."""
    if not root.is_dir():
        return []
    years = [int(p.name) for p in root.iterdir()
             if p.is_dir() and p.name.isdigit() and any(EVENT_FILE.match(f.name) for f in p.iterdir())]
    return sorted(years, reverse=True)


def load_season(year, root=SEASONS_DIR):
    """(every event's team rows with `round` and `event` added, the excluded list)."""
    folder = root / str(year)
    tables = []
    for path in sorted(folder.glob("*.csv")):
        match = EVENT_FILE.match(path.name)
        if match:
            tables.append(pd.read_csv(path).assign(round=int(match["round"]), event=match["event"]))
    events = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    excluded_path = folder / "excluded.csv"
    excluded = pd.read_csv(excluded_path) if excluded_path.exists() else pd.DataFrame(columns=EXCLUDED_COLUMNS)
    return events, excluded


def area_gaps(events, team, rival=None):
    """How `team` compares in each area over the season, one row per name in AREAS.

    The comparison at each event is the fastest team in that area (default)
    or `rival`. Only events where the area exists (time > 0) for both count.
      gap_s     median gap per lap (s); positive = the team is slower
      lost_n    events where the team was slower by more than LEVEL_S
      best_n    events where the team was the fastest in that area (or, against
                a rival, faster by more than LEVEL_S)
      events_n  events counted
    """
    rows = []
    for area in AREAS:
        gaps = []
        for _, event in events.groupby("round"):
            times = event.set_index("team")[area]
            times = times[times > 0]
            if team not in times:
                continue
            if rival is None:
                gaps.append(times[team] - times.min())
            elif rival in times:
                gaps.append(times[team] - times[rival])
        gaps = np.array(gaps)
        best = (gaps <= 0) if rival is None else (gaps < -LEVEL_S)
        rows.append({"area": area, "gap_s": float(np.median(gaps)) if len(gaps) else np.nan,
                     "lost_n": int((gaps > LEVEL_S).sum()), "best_n": int(best.sum()), "events_n": len(gaps)})
    return pd.DataFrame(rows)


def efficiency(events):
    """The efficiency map: each team's corner and straight speed as % of each event's best.

    x_pct, y_pct          this event: fast-corner and straight-line speed, % of the best team
    season_x, season_y    the team's median up to and including this round: the
                          trail, which moves as upgrades arrive. NaN rounds (no
                          fast corner / long straight) are skipped.
    """
    eff = events[["team", "team_colour", "round"]].copy()
    for column, pct in (("fast_corner_kmh", "x_pct"), ("straight_kmh", "y_pct")):
        eff[pct] = 100 * events[column] / events.groupby("round")[column].transform("max")
    eff = eff.sort_values(["team", "round"]).reset_index(drop=True)
    for pct, trail in (("x_pct", "season_x"), ("y_pct", "season_y")):
        eff[trail] = eff.groupby("team")[pct].transform(lambda s: s.expanding().median())
    return eff


# =============================================================================
# Downloads: building the season offline (python season.py 2026).
# =============================================================================

RETRY_REASONS = {"no qualifying session"}
# Whole-event exclusions worth trying again on the next run: on a race
# weekend the qualifying session may simply not have happened yet.


def season_events(year):
    """The year's race weekends, oldest first, numbered 1..n (pre-season testing left out)."""
    meetings = data.list_events(year)
    meetings = meetings[~meetings["meeting_name"].str.contains("Testing", case=False)]
    meetings = meetings.sort_values("date_start").reset_index(drop=True)
    return meetings.assign(round=range(1, len(meetings) + 1))


def team_laps(session, drivers):
    """Each team's lap for the breakdown: (team rows, [(telemetry, lap time)], excluded team reasons).

    The faster driver's best valid lap; if OpenF1 has no telemetry for it,
    the teammate's best lap instead.
    """
    rows, laps, missing = [], [], {}
    for team, crew in drivers.groupby("team_name", sort=False):
        crew = crew.sort_values("best_lap")
        timed = crew[crew["best_lap"].notna()]
        if timed.empty:
            missing[team] = "no valid lap"
            continue
        for _, driver in timed.iterrows():
            lap = data.fastest_lap(data.get_laps(session, driver))
            try:
                tel = data.get_lap_telemetry(int(session["session_key"]), int(driver["driver_number"]), lap)
            except LookupError:
                continue
            mate = crew[crew["name_acronym"] != driver["name_acronym"]]
            rows.append({"team": team, "team_colour": driver["team_colour"], "driver": driver["name_acronym"],
                         "lap_s": float(lap["lap_duration"]),
                         "teammate": mate["name_acronym"].iloc[0] if len(mate) else "",
                         "teammate_s": float(mate["best_lap"].iloc[0]) if len(mate) else np.nan})
            laps.append((tel, float(lap["lap_duration"])))
            break
        else:
            missing[team] = "no telemetry"
    return pd.DataFrame(rows), laps, missing


def build_event(year, meeting):
    """One event's summary table (None if the whole event is left out) and its exclusions."""
    event = data.slug(meeting["location"])

    def excluded(team, reason):
        return {"round": int(meeting["round"]), "event": event, "team": team, "reason": reason}

    sessions = data.list_sessions(int(meeting["meeting_key"]))
    quali = sessions[sessions["session_name"] == "Qualifying"] if not sessions.empty else sessions
    if quali.empty:
        return None, [excluded("all", "no qualifying session")]
    session = quali.iloc[0]
    if data.session_had_rain(int(session["session_key"])):
        return None, [excluded("all", "wet")]

    teams, laps, missing = team_laps(session, data.list_drivers(session))
    gone = [excluded(team, reason) for team, reason in missing.items()]
    if teams.empty:
        return None, gone + [excluded("all", "no telemetry")]
    try:
        corners = data.get_corners(year, int(session["circuit_key"]))
    except LookupError:
        corners = None
    aligned = analysis.align_field(laps)
    return event_summary(teams, aligned, phase_map(field_trace(aligned), corners)), gone


def build_season(year, root=SEASONS_DIR, force=False):
    """Add any new events of `year` to root/<year>/ (every event with force=True).

    Already-built events, and events left out for good (wet, no telemetry),
    are skipped. A download error skips the event this time; the next run
    tries it again. Ends with the Ferrari check from the design.
    """
    folder = root / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    excluded_path = folder / "excluded.csv"
    excluded = pd.read_csv(excluded_path) if excluded_path.exists() else pd.DataFrame(columns=EXCLUDED_COLUMNS)
    final = excluded[(excluded["team"] == "all") & ~excluded["reason"].isin(RETRY_REASONS)]["round"]
    for _, meeting in season_events(year).iterrows():
        rnd = int(meeting["round"])
        path = folder / f"{rnd:02d}_{data.slug(meeting['location'])}.csv"
        if not force and (path.exists() or rnd in set(final)):
            continue
        try:
            summary, gone = build_event(year, meeting)
        except requests.RequestException as error:
            print(f"Round {rnd:2d} {meeting['location']}: download failed ({error}); try again later.")
            continue
        excluded = pd.concat([excluded[excluded["round"] != rnd], pd.DataFrame(gone, columns=EXCLUDED_COLUMNS)],
                             ignore_index=True)
        excluded.sort_values(["round", "team"]).to_csv(excluded_path, index=False)
        if summary is None:
            path.unlink(missing_ok=True)
            print(f"Round {rnd:2d} {meeting['location']}: left out ({gone[-1]['reason']}).")
            continue
        summary.to_csv(path, index=False, float_format="%.4f")
        note = f", left out: {', '.join(g['team'] + ' (' + g['reason'] + ')' for g in gone)}" if gone else ""
        print(f"Round {rnd:2d} {meeting['location']}: {len(summary)} teams{note}.")
    ferrari_check(year, root)


def ferrari_check(year, root=SEASONS_DIR, team="Ferrari"):
    """Print the design's sanity check: Ferrari's power-unit deficit is public, so it
    should lose time at full throttle. Printed for review, never a pass/fail gate."""
    events, _ = load_season(year, root)
    if events.empty or team not in set(events["team"]):
        print(f"{team} check: no {team} laps saved yet.")
        return
    gaps = area_gaps(events, team).set_index("area")
    print(f"\n{team} check (gap to the best team per lap, median over events; + = slower):")
    for area in ("Full throttle", "Energy-limited"):
        g = gaps.loc[area]
        print(f"  {area:15s} {g['gap_s']:+.3f} s   lost in {int(g['lost_n'])} of {int(g['events_n'])} events")
    both = events.assign(both=events["Full throttle"] + events["Energy-limited"])
    combined = both.groupby("round").apply(
        lambda e: e.loc[e["team"] == team, "both"].sum() - e["both"].min() if team in set(e["team"]) else np.nan,
        include_groups=False).dropna()
    print(f"  {'Both together':15s} {combined.median():+.3f} s   lost in {(combined > LEVEL_S).sum()} of {len(combined)} events")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        sys.exit("Usage: python season.py 2026 [--force]")
    build_season(int(sys.argv[1]), force="--force" in sys.argv)
