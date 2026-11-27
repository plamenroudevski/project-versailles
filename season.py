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
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
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


def band(min_speed):
    """Corner speed band from a corner's slowest speed (analysis.py's bands)."""
    if min_speed < LOW_SPEED_MAX_KMH:
        return "slow"
    return "fast" if min_speed >= HIGH_SPEED_MIN_KMH else "medium"


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
        bands[s:e] = band(speed[s:e].min())
        apex = s + int(np.argmin(speed[s:e]))
        braking = s + np.flatnonzero(brake[s:e])
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
