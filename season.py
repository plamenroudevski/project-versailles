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

import numpy as np
import pandas as pd

import analysis
from analysis import (BRAKE_ON_LEVEL, DEPLOY_DROP_KMH, DEPLOY_FLAT_GAIN_KMH, DEPLOY_MIN_RUN_M,
                      FULL_THROTTLE_PCT, HIGH_SPEED_MIN_KMH, LOW_SPEED_MAX_KMH)

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
