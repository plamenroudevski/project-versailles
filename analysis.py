"""
analysis.py: the analysis engine. Turns two laps of telemetry into a zone
table, a ranked list of findings and a driver-vs-car read.

Pipeline position:  data.py -> analysis.py -> charts.py -> app.py

The analysis itself is pure functions: tables in, tables out, no downloads
and no Streamlit. That's what makes it testable with made-up laps
(tests/test_analysis.py). The section at the bottom adds the download step
(load_and_compare, via data.py) and terminal output, so it also runs alone:

    python analysis.py                                   # 2026 Brazil Qualifying, NOR vs RUS
    python analysis.py 2025 Monza Qualifying LEC HAM

compare() runs the steps in order:
    1. align_laps           put both laps on one distance grid; compute the time gap
    2. locate_corners       turn corner map positions into distances along the lap
    3. split_zones          cut the lap into corner zones and straights
    4. measure_zones        per-zone metrics for each driver
    5. rank_findings        biggest time differences, each with a likely cause
    6. driver_vs_car        heuristic: do the deficits look like car or driver?
    7. mini_sector_winners  faster driver per mini-sector, for the track map

Sign convention everywhere: the gap is driver B's time minus driver A's.
Positive = B is slower (losing time).
"""

import sys

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG: every threshold the analysis uses, in one place.
# Units: metres (m), km/h, seconds (s), percent (%).
# =============================================================================

GRID_STEP_M = 1.0
# Spacing of the shared distance grid. The data has one sample every ~5-20 m
# (4 Hz), so 1 m adds no fake precision; it just keeps interpolation smooth.

FULL_THROTTLE_PCT = 95
# Throttle at or above this counts as "full". Drivers sit at 98-100 % when
# flat out; 95 tolerates small sensor wobble.

BRAKE_ON_LEVEL = 0.5
# Brake is only on/off in the data. Interpolated between samples it ramps
# 0 -> 1, and 0.5 is halfway between the last "off" and first "on" sample:
# our best guess at where braking began (error about +/- half a sample gap).

FLAT_CHECK_M = 50
# A corner marker where both drivers stay at full throttle and off the brake
# for this distance either side is a flat-out kink, treated as straight.

CORNER_MAX_BEFORE_M = 400
CORNER_MAX_AFTER_M = 400
# A corner zone runs from where the drivers come off full throttle to where
# they're both back on it, but never further than these limits from the
# corner marker. Wet braking from ~300 km/h takes over 200 m, so the limits
# are generous.

MIN_STRAIGHT_M = 150
# A gap between two corner zones shorter than this is shared between the two
# corners instead of being called a straight.

MIN_FINDING_TIME_S = 0.02
# Zone gaps smaller than this aren't reported: on one lap they're noise.

TOP_FINDINGS = 5
# How many findings to list.

MIN_POINT_DIFF_M = 15
# Braking and full-throttle points can only be placed to about +/- half a
# sample gap (+/-10 m at 300 km/h, +/-5 m at 150 km/h). Comparing two drivers
# adds both errors, so differences under 15 m aren't reported.

MIN_SPEED_DIFF_KMH = 3
# Speed differences smaller than this aren't reported: speed is recorded in
# whole km/h, and the true minimum can fall between two samples.

DEPLOY_MIN_RUN_M = 300
# A straight needs this much unbroken full throttle before we judge whether
# its speed flattens.

DEPLOY_DROP_KMH = 3
# Speed falling this far below its peak while still flat out counts as
# "drops": the car is losing drive. With 2026 power units, a likely cause is
# the battery running out of energy to deploy.

DEPLOY_FLAT_WINDOW_M = 150
DEPLOY_FLAT_GAIN_KMH = 1
# If speed rises by less than DEPLOY_FLAT_GAIN_KMH over the last
# DEPLOY_FLAT_WINDOW_M of the flat-out run, it "flattens". A car at its
# drag-limited top speed also flattens, which is why this is only "possible".

LOW_SPEED_MAX_KMH = 120
HIGH_SPEED_MIN_KMH = 200
# Corner speed bands, by the corner's minimum speed (average of both drivers):
# low-speed below 120 km/h, high-speed from 200 km/h, medium in between. The
# usual rough bands for a breakdown like AWS F1 Insights' car performance
# scores, which don't publish exact cut-offs. Also the "low-speed corners" of
# the car/driver read.

PATTERN_MIN_ZONES = 2
# At least this many straights (or low-speed corners) are needed to call a pattern.

PATTERN_MIN_TIME_S = 0.01
# Zone gaps smaller than this count as "level" in the car/driver read.

PATTERN_MIXED_SHARE = 0.25
# The corner gap only "swings both ways" if the less successful driver is
# quicker in at least this share of the corners with a clear gap. 11 corners
# to 1 is a pattern, not a swing.

MINI_SECTORS = 25
# The track map is cut into this many equal-length pieces, each coloured by
# the faster driver. ~170 m each on a 4.3 km lap: short enough to show where
# time changes hands, long enough not to flicker with sample noise.

# =============================================================================


def resample(tel, grid, stretch):
    """Put one lap's telemetry onto the shared distance grid.

    `stretch` scales this lap's distances so both laps have the same length
    (see align_laps).
    - Time, speed, throttle, RPM and position are linearly interpolated between the
      two samples either side of each grid point.
    - Brake is on/off. Interpolated, it ramps from 0 to 1 between the last
      "off" and the first "on" sample; BRAKE_ON_LEVEL marks the halfway point.
      That halves the error compared with taking the first "on" sample.
    - Gear is a whole number, so each grid point takes the gear of the last
      sample before it (no "4.5th gear").
    """
    d = tel["Distance"].to_numpy() * stretch
    out = pd.DataFrame({"Distance": grid})
    for column in ("Time", "Speed", "Throttle", "RPM", "X", "Y"):
        out[column] = np.interp(grid, d, tel[column].to_numpy(dtype=float))
    out["Brake"] = np.interp(grid, d, tel["Brake"].to_numpy(dtype=float)) >= BRAKE_ON_LEVEL
    last_sample = np.clip(np.searchsorted(d, grid, side="right") - 1, 0, len(d) - 1)
    out["Gear"] = tel["Gear"].to_numpy()[last_sample]
    return out


def align_laps(tel_a, lap_time_a, tel_b, lap_time_b):
    """Put both laps on one shared distance grid and compute the time gap (delta).

    Why: the two cars' samples are taken at different places on track, so
    they can't be subtracted row by row. Instead we ask "how long had each
    car been driving when it reached each metre of the lap?"

    Method:
    1. Each lap's distance comes from integrating speed, which drifts a few
       metres per driver (4250 m vs 4241 m for NOR and RUS in 2026 Brazil).
       Both laps are stretched to their average length, so the same corner
       sits at the same metre for both. Without this, the drift would show up
       as fake braking-point differences late in the lap.
    2. A grid of points every GRID_STEP_M metres runs from the start line
       (0 m) to the finish line.
    3. Each driver's elapsed time is interpolated onto the grid, e.g. "at
       1000 m, A had been driving for 14.21 s".
    4. Delta = B's time minus A's time at every grid point. Because time 0 is
       exactly 0 m and the lap time is exactly the finish line, the delta
       starts at 0 and ends at exactly the lap-time difference.

    Returns (a, b, delta): both laps on the grid (columns Distance, Time,
    Speed, Throttle, RPM, Brake, Gear, X, Y) and the delta in seconds per grid point.
    """
    own_lengths = [np.interp(lap_time, tel["Time"], tel["Distance"])
                   for tel, lap_time in ((tel_a, lap_time_a), (tel_b, lap_time_b))]
    length = float(np.mean(own_lengths))
    grid = np.linspace(0.0, length, int(round(length / GRID_STEP_M)) + 1)
    a = resample(tel_a, grid, length / own_lengths[0])
    b = resample(tel_b, grid, length / own_lengths[1])
    return a, b, (b["Time"] - a["Time"]).to_numpy()


def corner_labels(corners):
    """Corner names from the markers: "T" + number + letter, e.g. T2A. Shared by the zone table and the map."""
    return "T" + corners["Number"].astype(str) + corners["Letter"].fillna("").astype(str)


def locate_corners(corners, lap):
    """Turn corner markers (map X, Y) into distances along the lap.

    For each marker, find the grid point where the car passed closest to it.
    One driver's lap is enough: after align_laps both laps share the same
    distances to within a few metres, and using one avoids averaging a marker
    near the start line into the middle of the lap (5 m and 4245 m -> 2125 m).
    Returns a table of Label ("T1", "T2A", ...) and Distance (m), in lap order.
    """
    def nearest(x, y):
        return lap["Distance"].iloc[int(np.argmin((lap["X"] - x) ** 2 + (lap["Y"] - y) ** 2))]

    table = pd.DataFrame({
        "Label": corner_labels(corners),
        "Distance": [nearest(x, y) for x, y in zip(corners["X"], corners["Y"])],
    })
    return table.sort_values("Distance").reset_index(drop=True)


def flat_out(lap):
    """True wherever the driver is at full throttle and off the brake."""
    return (lap["Throttle"] >= FULL_THROTTLE_PCT) & ~lap["Brake"]


def corner_bounds(laps, lo, hi):
    """Where one corner starts and ends, searching only between `lo` and `hi`.

    For each driver: find their slowest point in the range (the apex), then
    the last full-throttle point before it (where they lift or brake) and the
    first full-throttle point after it (where they're flat out again). The
    zone runs from the earlier lift to the later return to full throttle, so
    it holds both drivers' whole braking zone and exit. A driver who never
    reaches full throttle in the range (linked corners) uses the range edge.
    """
    starts, ends = [], []
    for lap in laps:
        part = lap[(lap["Distance"] >= lo) & (lap["Distance"] <= hi)]
        flat = flat_out(part)
        apex = part["Speed"].idxmin()
        before = part.loc[:apex].loc[flat.loc[:apex], "Distance"]
        after = part.loc[apex:].loc[flat.loc[apex:], "Distance"]
        starts.append(before.iloc[-1] if len(before) else lo)
        ends.append(after.iloc[0] if len(after) else hi)
    return min(starts), max(ends)


def fastest_point(laps, lo, hi):
    """Where two linked corners split: the fastest point between their markers.

    That's where one corner's exit ends and the next corner's braking begins,
    so each corner keeps its own exit. Splitting halfway between the markers
    instead would hand part of one corner's exit to the next corner's entry.
    Uses both drivers' speeds added together, skipping points where either is
    braking, so braking for the next corner always starts after the split.
    """
    a, b = laps
    between = (a["Distance"] >= lo) & (a["Distance"] <= hi) & ~a["Brake"] & ~b["Brake"]
    if not between.any():
        return (lo + hi) / 2
    return a.at[(a["Speed"] + b["Speed"])[between].idxmax(), "Distance"]


def split_zones(corner_table, a, b):
    """Cut the lap into corner zones and straights that cover every metre once.

    1. Flat-out kinks (both drivers at full throttle and off the brake within
       FLAT_CHECK_M of the marker) aren't real corners. They become part of
       the straight around them, so a long straight stays one straight.
    2. Each remaining corner runs from where the drivers come off full
       throttle to where they're both back on it (see corner_bounds). The
       search reaches at most CORNER_MAX_BEFORE_M / CORNER_MAX_AFTER_M from
       the marker, and never past the fastest point between this corner and
       its neighbours (see fastest_point), so linked corners (no full throttle
       between them) split where one exit ends and the next braking begins.
    3. A gap between corner zones shorter than MIN_STRAIGHT_M is shared by the
       corners either side; a longer gap is a straight.

    Because the zones tile the lap with no gaps or overlaps, their time gaps
    add up exactly to the lap-time gap, which is a useful check.
    Returns a list of zones: {"name", "kind" ("corner"/"straight"), "start",
    "end"}, plus "marker" (m) for corners.
    """
    length = float(a["Distance"].iloc[-1])

    kept = []
    for label, marker in zip(corner_table["Label"], corner_table["Distance"]):
        near = (a["Distance"] - marker).abs() <= FLAT_CHECK_M
        if not all(flat_out(lap.loc[near]).all() for lap in (a, b)):
            kept.append((label, marker))

    splits = [fastest_point((a, b), kept[i][1], kept[i + 1][1]) for i in range(len(kept) - 1)]

    zones, position, previous = [], 0.0, "Line"
    for i, (label, marker) in enumerate(kept):
        lo = max(marker - CORNER_MAX_BEFORE_M, splits[i - 1] if i > 0 else 0.0)
        hi = min(marker + CORNER_MAX_AFTER_M, splits[i] if i < len(kept) - 1 else length)
        start, end = corner_bounds((a, b), max(lo, 0.0), min(hi, length))

        if start - position >= MIN_STRAIGHT_M:
            zones.append({"name": f"{previous}-{label}", "kind": "straight", "start": position, "end": start})
        elif zones:
            # Short gap: the previous corner and this one meet in the middle of it.
            start = zones[-1]["end"] = (position + start) / 2
        else:
            start = 0.0  # short gap before the first corner: that corner starts at the line
        zones.append({"name": label, "kind": "corner", "start": start, "end": end, "marker": marker})
        position, previous = end, label

    if length - position >= MIN_STRAIGHT_M or not zones:
        zones.append({"name": f"{previous}-Line", "kind": "straight", "start": position, "end": length})
    else:
        zones[-1]["end"] = length
    return zones


def corner_metrics(lap, start, end):
    """One driver's numbers for one corner zone.

    brake_point    where braking starts (m). NaN ("not a number", i.e. no
                   value) if the driver doesn't brake, or is already braking
                   when the zone starts.
    min_speed      slowest point of the corner (km/h); apex = where it is (m)
    gear_at_min    gear at the slowest point
    full_throttle  where the driver is back to full throttle, off the brake,
                   after the apex (m); NaN if not within the zone
    exit_speed     speed at the end of the zone (km/h)
    """
    z = lap[(lap["Distance"] >= start) & (lap["Distance"] <= end)]
    i_min = z["Speed"].idxmin()

    entry = z.loc[:i_min]
    brake_point = np.nan
    if not entry["Brake"].iloc[0] and entry["Brake"].any():
        brake_point = entry.loc[entry["Brake"], "Distance"].iloc[0]

    exit_part = z.loc[i_min:]
    on_full = exit_part.loc[flat_out(exit_part), "Distance"]
    return {
        "brake_point": brake_point,
        "min_speed": z.at[i_min, "Speed"],
        "apex": z.at[i_min, "Distance"],
        "gear_at_min": int(z.at[i_min, "Gear"]),
        "full_throttle": on_full.iloc[0] if len(on_full) else np.nan,
        "exit_speed": z["Speed"].iloc[-1],
    }


def straight_metrics(lap, start, end):
    """One driver's numbers for one straight.

    `end` should reach into the next corner zone, up to its marker: the
    straight ends where the *first* driver lifts, but the other driver may
    stay flat out a little longer, and the last metres before braking are
    exactly where an energy-limited car shows it.

    entry_speed speed where the straight starts (km/h): a slow start is
                carried over from the previous corner's exit
    top_speed   highest speed (km/h)
    deployment  "drops" if speed falls DEPLOY_DROP_KMH below its peak while
                still flat out, "flattens" if it gains less than
                DEPLOY_FLAT_GAIN_KMH over the last DEPLOY_FLAT_WINDOW_M of
                the flat-out run, otherwise None. Only judged on runs of at
                least DEPLOY_MIN_RUN_M.
    speed_loss  how far speed fell from its peak during the flat-out run (km/h)
    """
    z = lap[(lap["Distance"] >= start) & (lap["Distance"] <= end)]
    result = {"entry_speed": z["Speed"].iloc[0], "top_speed": z["Speed"].max(), "deployment": None, "speed_loss": 0.0}

    # The flat-out run: the last unbroken stretch at full throttle, off the brake.
    # np.flatnonzero gives the positions of the True values.
    flat = flat_out(z).to_numpy()
    if not flat.any():
        return result
    run_end = np.flatnonzero(flat)[-1]      # last flat-out sample
    lifts = np.flatnonzero(~flat[:run_end])  # samples before it that weren't flat out
    run = z.iloc[(lifts[-1] + 1 if len(lifts) else 0): run_end + 1]
    if run["Distance"].iloc[-1] - run["Distance"].iloc[0] < DEPLOY_MIN_RUN_M:
        return result

    result["speed_loss"] = run["Speed"].max() - run["Speed"].iloc[-1]
    late_gain = run["Speed"].iloc[-1] - np.interp(run["Distance"].iloc[-1] - DEPLOY_FLAT_WINDOW_M,
                                                   run["Distance"], run["Speed"])
    if result["speed_loss"] >= DEPLOY_DROP_KMH:
        result["deployment"] = "drops"
    elif late_gain < DEPLOY_FLAT_GAIN_KMH:
        result["deployment"] = "flattens"
    return result


def measure_zones(a, b, delta, zones):
    """Add the time gap and both drivers' metrics to every zone.

    time        gap change across the zone: B minus A (positive = B lost time)
    entry_time  for corners, the part of that before the apex (braking, turn-in)
    exit_time   for corners, the part after the apex
    slower_by   for corners, the biggest speed deficit (km/h) of each driver
                before and after the apex: {"entry": (A's, B's), "exit": (A's, B's)}
    a, b        each driver's corner_metrics or straight_metrics
    """
    length = float(a["Distance"].iloc[-1])

    def gap_at(distance):
        return float(np.interp(distance, a["Distance"], delta))

    results = []
    for i, zone in enumerate(zones):
        # dict(zone, time=...) makes a copy of the zone with a "time" entry added.
        z = dict(zone, time=gap_at(zone["end"]) - gap_at(zone["start"]))
        if zone["kind"] == "corner":
            z["a"] = corner_metrics(a, zone["start"], zone["end"])
            z["b"] = corner_metrics(b, zone["start"], zone["end"])
            apex = (z["a"]["apex"] + z["b"]["apex"]) / 2
            z["entry_time"] = gap_at(apex) - gap_at(zone["start"])
            z["exit_time"] = gap_at(zone["end"]) - gap_at(apex)
            b_slower = (a["Speed"] - b["Speed"]).to_numpy()  # positive where B is slower
            entry = a["Distance"].between(zone["start"], apex).to_numpy()
            exit_part = a["Distance"].between(apex, zone["end"]).to_numpy()
            z["slower_by"] = {phase: (-b_slower[part].min(), b_slower[part].max())
                              for phase, part in (("entry", entry), ("exit", exit_part))}
        else:
            look_to = zones[i + 1]["marker"] if i + 1 < len(zones) else length  # up to the next corner marker
            z["a"] = straight_metrics(a, zone["start"], look_to)
            z["b"] = straight_metrics(b, zone["start"], look_to)
        results.append(z)
    return results


# Display names for the zone table: (metric key, column label).
METRIC_COLUMNS = [
    ("brake_point", "Brake m"), ("min_speed", "Min km/h"), ("gear_at_min", "Gear"),
    ("full_throttle", "Full thr m"), ("exit_speed", "Exit km/h"),
    ("entry_speed", "Entry km/h"), ("top_speed", "Top km/h"), ("deployment", "Deploy"),
]


def zone_table(results, names):
    """Turn measured zones into one table: a row per zone, a column per metric per driver."""
    rows = []
    for z in results:
        row = {"Zone": z["name"], "Type": z["kind"], "From m": z["start"], "To m": z["end"], "Gap s": z["time"]}
        for key, label in METRIC_COLUMNS:
            for name, metrics in zip(names, (z["a"], z["b"])):
                row[f"{label} {name}"] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_corner_cause(entry_loss, exit_loss, min_speed_deficit, exit_speed_deficit):
    """Pick the corner phase that best explains a time loss.

    All four values are from the losing driver's side (positive = worse).
    - Most of the time lost before the apex -> "Braking" (braking point,
      braking and turn-in).
    - Otherwise the loss is after the apex. If the driver was already slower
      at the apex and the speed gap didn't grow on exit, the time was lost in
      the middle of the corner and carried down the exit -> "Mid-corner".
    - If the gap grew after the apex -> "Traction/exit".
    """
    if entry_loss >= exit_loss:
        return "Braking"
    if min_speed_deficit >= MIN_SPEED_DIFF_KMH and exit_speed_deficit <= min_speed_deficit + MIN_SPEED_DIFF_KMH:
        return "Mid-corner"
    return "Traction/exit"


def corner_evidence(lost, won):
    """List how the losing driver's corner differs, in lap order (braking -> apex -> exit).

    Only differences above the noise thresholds are listed. Missing values
    (NaN) fail every comparison, so they are skipped automatically.
    """
    notes = []
    earlier = won["brake_point"] - lost["brake_point"]
    if earlier >= MIN_POINT_DIFF_M:
        notes.append(f"brakes ~{earlier:.0f} m earlier")
    elif -earlier >= MIN_POINT_DIFF_M:
        notes.append(f"brakes ~{-earlier:.0f} m later")
    slower = won["min_speed"] - lost["min_speed"]
    if slower >= MIN_SPEED_DIFF_KMH:
        notes.append(f"min speed {slower:.0f} km/h lower")
    if lost["gear_at_min"] != won["gear_at_min"]:
        notes.append(f"gear {lost['gear_at_min']} vs {won['gear_at_min']} at min speed")
    later = lost["full_throttle"] - won["full_throttle"]
    if later >= MIN_POINT_DIFF_M:
        notes.append(f"full throttle ~{later:.0f} m later")
    exit_slower = won["exit_speed"] - lost["exit_speed"]
    if exit_slower >= MIN_SPEED_DIFF_KMH:
        notes.append(f"exit speed {exit_slower:.0f} km/h lower")
    return notes


def straight_evidence(lost, won, came_from):
    """List how the losing driver's straight differs: entry speed, top speed, deployment pattern.

    `came_from` is the corner before the straight (or "Line").
    """
    notes = []
    slower_in = won["entry_speed"] - lost["entry_speed"]
    if slower_in >= MIN_SPEED_DIFF_KMH and came_from != "Line":
        notes.append(f"starts the straight {slower_in:.0f} km/h slower out of {came_from}")
    slower = won["top_speed"] - lost["top_speed"]
    if slower >= MIN_SPEED_DIFF_KMH:
        notes.append(f"top speed {slower:.0f} km/h lower")
    pattern = {"drops": f"speed drops {lost['speed_loss']:.0f} km/h", "flattens": "speed stops rising"}
    if lost["deployment"]:
        both = " (so does the other car)" if won["deployment"] else ""
        notes.append(f"{pattern[lost['deployment']]} late in the straight at full throttle{both}: "
                     f"possible energy-deployment limit")
    return notes


def rank_findings(results, names):
    """The biggest time differences, largest first, each with a cause tag and a one-line explanation.

    Each finding: zone, time (s, always positive), loser, cause, text.
    """
    findings = []
    for z in results:
        if abs(z["time"]) < MIN_FINDING_TIME_S:
            continue
        b_lost = z["time"] > 0
        loser, winner = (names[1], names[0]) if b_lost else (names[0], names[1])
        lost, won = (z["b"], z["a"]) if b_lost else (z["a"], z["b"])
        if z["kind"] == "corner":
            side = 1 if b_lost else -1  # flip A-vs-B gaps to the loser's point of view
            cause = classify_corner_cause(side * z["entry_time"], side * z["exit_time"],
                                          won["min_speed"] - lost["min_speed"],
                                          won["exit_speed"] - lost["exit_speed"])
            evidence = corner_evidence(lost, won)
            # No single metric stands out (common in the wet, where nobody reaches
            # full throttle): say where the speed went instead.
            phase = "entry" if cause == "Braking" else "exit"
            deficit = z["slower_by"][phase][1 if b_lost else 0]
            if not evidence and deficit >= MIN_SPEED_DIFF_KMH:
                evidence = [f"up to {deficit:.0f} km/h slower {'before' if phase == 'entry' else 'after'} the apex"]
        else:
            # Time on a straight is mostly decided by speed at its start. If the
            # loser started slower but wasn't slower at the top, the time was
            # really lost on the previous corner's exit.
            came_from = z["name"].split("-")[0]
            carried = (came_from != "Line"
                       and won["entry_speed"] - lost["entry_speed"] >= MIN_SPEED_DIFF_KMH
                       and won["top_speed"] - lost["top_speed"] < MIN_SPEED_DIFF_KMH)
            cause = "Traction/exit" if carried else "Straight-line/deployment"
            evidence = straight_evidence(lost, won, came_from)
        detail = ", ".join(evidence) if evidence else "no single difference above the noise thresholds"
        findings.append({
            "zone": z["name"], "time": abs(z["time"]), "loser": loser, "cause": cause,
            "text": f"{loser} loses ~{abs(z['time']):.2f} s at {z['name']}: {detail}.",
        })
    findings.sort(key=lambda f: f["time"], reverse=True)
    return findings[:TOP_FINDINGS]


ZONE_TYPES = ("Low-speed corners", "Medium-speed corners", "High-speed corners", "Straights")


def zone_type(zone):
    """Which of ZONE_TYPES a measured zone is. Corners go by minimum speed, averaged over both drivers."""
    if zone["kind"] == "straight":
        return "Straights"
    speed = (zone["a"]["min_speed"] + zone["b"]["min_speed"]) / 2
    if speed < LOW_SPEED_MAX_KMH:
        return "Low-speed corners"
    return "High-speed corners" if speed >= HIGH_SPEED_MIN_KMH else "Medium-speed corners"


def type_breakdown(results, names):
    """Time gained and lost by zone type, like a broadcast "car performance" breakdown.

    One row per ZONE_TYPES entry: Type, Zones (their names), Gap s (B minus A,
    positive = B slower) and Faster (who gained there overall). Because the
    zones cover the lap exactly once, the four gaps add up to the lap-time gap.
    """
    rows = []
    for kind in ZONE_TYPES:
        zones = [z for z in results if zone_type(z) == kind]
        gap = sum(z["time"] for z in zones)
        faster = names[0] if gap > 0 else names[1] if gap < 0 else "-"
        rows.append({"Type": kind, "Zones": ", ".join(z["name"] for z in zones), "Gap s": gap, "Faster": faster})
    return pd.DataFrame(rows)


def consistent_loser(deficits, names, threshold):
    """The driver who is worse in every one of these zones, or None if it's mixed.

    `deficits` are B's deficits (positive = B worse, negative = A worse);
    each must be bigger than `threshold` to count.
    """
    if len(deficits) < PATTERN_MIN_ZONES:
        return None
    if all(d > threshold for d in deficits):
        return names[1]
    if all(d < -threshold for d in deficits):
        return names[0]
    return None


def driver_vs_car(results, names):
    """Heuristic read of whether the deficits look like the car or the driver.

    - A deficit that repeats on every straight, or in every low-speed corner,
      points to the car: a driver's mistakes rarely repeat that neatly.
      Straights are judged on top speed, not time: time on a straight is
      mostly decided by the exit of the corner before it, which is driving.
    - A deficit that swings from corner to corner points to the driver
      (lines, braking, commitment): the car is the same car in every corner.
    One lap can't prove either, so every line is hedged.
    Returns a list of sentences.
    """
    straights = [z for z in results if z["kind"] == "straight"]
    corners = [z for z in results if z["kind"] == "corner"]
    top_speed_deficits = [z["a"]["top_speed"] - z["b"]["top_speed"] for z in straights]
    slow_corner_times = [z["time"] for z in corners if zone_type(z) == "Low-speed corners"]

    lines = []
    who = consistent_loser(top_speed_deficits, names, MIN_SPEED_DIFF_KMH)
    if who:
        lines.append(f"{who} has a lower top speed on all {len(straights)} straights: telemetry suggests a "
                     f"car-side straight-line limit (power unit, deployment or drag) rather than driving.")
    who = consistent_loser(slow_corner_times, names, PATTERN_MIN_TIME_S)
    if who:
        lines.append(f"{who} loses time in all {len(slow_corner_times)} low-speed corners: telemetry suggests a "
                     f"car-side limit in low-speed grip or traction rather than driving.")

    a_quicker = sum(z["time"] > PATTERN_MIN_TIME_S for z in corners)
    b_quicker = sum(z["time"] < -PATTERN_MIN_TIME_S for z in corners)
    clear = a_quicker + b_quicker
    if clear >= PATTERN_MIN_ZONES and min(a_quicker, b_quicker) >= PATTERN_MIXED_SHARE * clear:
        lines.append(f"Corner by corner the gap swings both ways ({names[0]} quicker in {a_quicker}, "
                     f"{names[1]} in {b_quicker} of {len(corners)} corners): telemetry suggests the corner "
                     f"differences come mostly from driving (lines, braking, commitment) rather than the car.")
    elif clear >= PATTERN_MIN_ZONES:
        who = names[1] if a_quicker > b_quicker else names[0]
        lines.append(f"{who} is slower in {max(a_quicker, b_quicker)} of the {clear} corners with a clear gap: "
                     f"that could be the car's overall grip or a consistent driving difference, and one lap "
                     f"can't separate them.")
    return lines or ["No clear pattern: the gaps are too small or too few to read."]


def mini_sector_winners(a, delta, count=MINI_SECTORS):
    """Cut the lap into `count` equal-length pieces and say who was faster in each.

    Returns a list of (start m, end m, winner), winner 0 for A or 1 for B.
    If the gap (B minus A) grows across a piece, B lost time there, so A was faster.
    """
    edges = np.linspace(0.0, float(a["Distance"].iloc[-1]), count + 1)
    gaps = np.interp(edges, a["Distance"], delta)
    return [(edges[i], edges[i + 1], 0 if gaps[i + 1] > gaps[i] else 1) for i in range(count)]


def compare(tel_a, lap_time_a, tel_b, lap_time_b, corners, names):
    """Run the whole analysis on two laps. `names` is (A, B), e.g. ("NOR", "RUS").

    Returns a dict:
      a, b      both laps on the shared distance grid (for charts)
      delta     time gap B minus A at every grid point (s)
      corners   corner labels and distances along the lap
      zones     measured zones (see measure_zones)
      table     zone table (see zone_table)
      findings  ranked findings (see rank_findings)
      read      driver-vs-car sentences (see driver_vs_car)
      breakdown time gained by zone type: low/medium/high-speed corners, straights (see type_breakdown)
      mini_sectors  faster driver per mini-sector (see mini_sector_winners)
    """
    a, b, delta = align_laps(tel_a, lap_time_a, tel_b, lap_time_b)
    corner_table = locate_corners(corners, a)
    zones = measure_zones(a, b, delta, split_zones(corner_table, a, b))
    return {
        "a": a, "b": b, "delta": delta, "corners": corner_table, "zones": zones,
        "table": zone_table(zones, names),
        "findings": rank_findings(zones, names),
        "read": driver_vs_car(zones, names),
        "breakdown": type_breakdown(zones, names),
        "mini_sectors": mini_sector_winners(a, delta),
    }


# ----------------------------------------------------------------------------
# Loading and text output. load_and_compare() downloads (through data.py);
# the app uses it, title_text() and lap_time_text() too. main() is what
# `python analysis.py` runs.
# ----------------------------------------------------------------------------

def lap_time_text(seconds):
    """87.47 -> '1:27.470'"""
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def print_zone_table(table, names):
    """Print corners and straights as two compact tables, pairing each metric as 'A / B'."""
    for kind, labels in (("corner", ["Brake m", "Min km/h", "Gear", "Full thr m", "Exit km/h"]),
                         ("straight", ["Entry km/h", "Top km/h", "Deploy"])):
        rows = table[table["Type"] == kind]
        shown = pd.DataFrame({
            "Zone": rows["Zone"],
            "From-To m": rows["From m"].map("{:.0f}".format) + "-" + rows["To m"].map("{:.0f}".format),
            "Gap s": rows["Gap s"].map("{:+.3f}".format),
        })
        for label in labels:
            fmt = "{}" if label == "Deploy" else "{:.0f}"
            shown[label] = [" / ".join("-" if pd.isna(row[f"{label} {n}"]) else fmt.format(row[f"{label} {n}"])
                                       for n in names) for _, row in rows.iterrows()]
        print(f"\n{kind.upper()}S  (metrics shown as {names[0]} / {names[1]}; Gap + = {names[1]} slower)")
        print(shown.to_string(index=False))


def load_and_compare(session, name_a, name_b, lap_a=None, lap_b=None):
    """Download two drivers' laps from a session and compare them.

    `session` is a row from data.find_session or data.get_session. Each
    driver's fastest valid lap is used unless a lap number is given. Used by
    the terminal scripts and the app. Returns a dict with the session, both
    laps (from data.load_lap), the raw corner markers, the lap times, whether
    it rained, and the compare() result.
    """
    import data  # imported here so the rest of this file never needs the internet

    laps = [data.load_lap(session, name, lap) for name, lap in ((name_a, lap_a), (name_b, lap_b))]
    corners = data.get_corners(int(session["year"]), int(session["circuit_key"]))
    times = [float(lap["lap"]["lap_duration"]) for lap in laps]
    result = compare(laps[0]["telemetry"], times[0], laps[1]["telemetry"], times[1], corners, (name_a, name_b))
    return {"session": session, "laps": laps, "corners": corners, "times": times,
            "rain": data.session_had_rain(int(session["session_key"])), "result": result}


def title_text(run, names):
    """One-line heading, e.g.
    '2026 São Paulo Qualifying: NOR lap 23 (Q3) 1:27.470 vs RUS lap 18 (Q2) 1:26.745 (gap -0.725 s)'.
    The qualifying part shows when laps come from different parts of the session.
    """
    session, laps, times = run["session"], run["laps"], run["times"]
    labels = [f"lap {lap['lap']['lap_number']}" + (f" ({lap['lap'].get('part')})" if lap["lap"].get("part") else "")
              for lap in laps]
    return (f"{session['year']} {session['location']} {session['session_name']}: "
            f"{names[0]} {labels[0]} {lap_time_text(times[0])} vs "
            f"{names[1]} {labels[1]} {lap_time_text(times[1])} "
            f"(gap {times[1] - times[0]:+.3f} s)")


def main(year, place, session_name, name_a, name_b):
    import fastf1

    import data

    fastf1.set_log_level("ERROR")
    run = load_and_compare(data.find_session(year, place, session_name), name_a, name_b)
    laps, times, result = run["laps"], run["times"], run["result"]

    print(title_text(run, (name_a, name_b)))
    if run["rain"]:
        print("NOTE: rain fell during this session. Wet laps vary more, so read everything with extra caution.")
    if all(lap["telemetry"]["DRS"].isna().all() for lap in laps):
        print("NOTE: no DRS/active-aero channel in this data, so straight-line findings can't separate "
              "aero mode from energy deployment.")

    print("\nRANKED FINDINGS (biggest first)")
    for i, finding in enumerate(result["findings"], 1):
        print(f"  {i}. [{finding['cause']}] {finding['text']}")

    print(f"\nBY ZONE TYPE (corners by minimum speed: low < {LOW_SPEED_MAX_KMH}, high >= {HIGH_SPEED_MIN_KMH} km/h)")
    for _, row in result["breakdown"].iterrows():
        gain = f"{row['Faster']} faster by {abs(row['Gap s']):.3f} s" if row["Faster"] != "-" else "level"
        print(f"  {row['Type']:<21} {gain:<26} {row['Zones'] or 'none'}")

    print("\nDRIVER VS CAR (heuristic: one lap can't prove cause)")
    for line in result["read"]:
        print(f"  - {line}")

    print_zone_table(result["table"], (name_a, name_b))
    print(f"\nCheck: zone gaps add up to {result['table']['Gap s'].sum():+.3f} s; "
          f"lap-time gap is {times[1] - times[0]:+.3f} s.")


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026", "Brazil", "Qualifying", "NOR", "RUS"]
    try:
        main(int(args[0]), args[1], args[2], args[3], args[4])
    except LookupError as error:
        sys.exit(str(error))
