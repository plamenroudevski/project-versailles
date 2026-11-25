"""
replay.py: the data behind the Lap replay page, and the page's player.

Pipeline position:  data.py -> analysis.py -> replay.py -> app.py

The replay shows two drivers' laps at the same time, like a live-timing
screen, so you can see where on track any moment of the telemetry happened.
Python does all the maths once; the browser only animates.

  1. analysis.load_and_compare() (the same as the Qualifying page) already
     downloads both laps, puts them on one 1 m distance grid, works out the
     running gap and picks the 25 mini-sector winners. Nothing is re-derived.
  2. build_payload() adds what only the replay needs: the official sector
     times, where each sector ends on track, the map turned to its official
     orientation, and each corner's label position. It packs everything into
     one dict of plain numbers and text (JSON).
  3. player_html() pours that JSON into replay_player.html, a self-contained
     HTML/JavaScript player (no internet needed, no CDN). The app embeds it
     with st.components.v1.html. Play, pause and scrub happen entirely in the
     browser, so they never re-run the Streamlit script.

Run it alone to check the payload and get a standalone player:

    python replay.py                                       # 2026 Las Vegas Qualifying, ANT vs VER
    python replay.py 2026 "Las Vegas" Race ANT VER 30 30   # optional lap numbers for A and B

It prints the payload's lap and sector times next to OpenF1's official ones
and writes output/..._replay.html (open it in a browser).
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import charts

HERE = Path(__file__).parent
PLAYER_TEMPLATE = HERE / "replay_player.html"

# =============================================================================
# ASSETS SLOT: the typeface for the driver numbers on the car markers.
# Archivo (SIL Open Font License, free to bundle) squeezed to 75 % width and
# extra bold is the closest free look to the condensed numbers teams use.
# To use a licensed team font instead, drop its .woff2 file into assets/fonts/
# and change the three values below. For a font that isn't "variable", set
# "settings" to "normal". Never commit a font you don't have a licence for.
# =============================================================================
NUMBER_FONT = {
    "file": HERE / "assets" / "fonts" / "Archivo-latin-var.woff2",
    "family": "Archivo",
    "settings": '"wdth" 75, "wght" 800',
}

PLAYER_HEIGHT = 1500
# Height in pixels the app gives the player (an embedded frame can't grow to
# fit its content by itself). The player fills it exactly: the track map takes
# whatever height the other blocks leave, ~700 px at this value with the app
# 1440 px wide. A smaller value shrinks the map, never cuts off the controls.


# ---- Small helpers -------------------------------------------------------------

def numbers(values, digits):
    """Round an array and make it JSON-ready: a plain list, with missing values as None (JSON null).

    Rounding matters for size: a 6 km lap has ~6,000 grid points per channel,
    and 3 decimals on a time (1 ms) or 1 on a speed is all the data carries.
    """
    rounded = np.round(np.asarray(values, dtype=float), digits)
    return [None if np.isnan(v) else (int(v) if digits == 0 else float(v)) for v in rounded]


def rotate(x, y, degrees):
    """Turn map points about the origin, anticlockwise by `degrees` (the standard rotation)."""
    angle = np.radians(degrees)
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


# ---- Sectors -------------------------------------------------------------------

def official_sectors(lap):
    """S1, S2, S3 in seconds from OpenF1's lap row, or None if any of them is missing."""
    times = [lap.get(f"duration_sector_{n}") for n in (1, 2, 3)]
    if any(t is None or pd.isna(t) for t in times):
        return None
    return [float(t) for t in times]


def sector_ends(time_on_grid, distance, sectors):
    """Where sectors 1 and 2 end on the lap (m), for a lap whose sector times are known.

    OpenF1 gives sector *times* but not where the timing lines are. A car's
    elapsed time reaches S1 exactly as it crosses the first timing line, so
    that line is at the distance where its time-at-distance equals S1 (and
    the second at S1 + S2). np.interp reads the grid "backwards" (time ->
    distance), which works because elapsed time only ever increases.
    """
    return [float(np.interp(t, time_on_grid, distance)) for t in np.cumsum(sectors[:2])]


def sector_table(laps_on_grid, lap_rows, lap_times):
    """Both drivers' sector times and where the sectors end on track.

    Returns (ends, sectors, derived):
      ends     [S1 end m, S2 end m], or None if neither lap has sector times
      sectors  per driver [S1, S2, S3] in seconds (None if unknown)
      derived  per driver True if the times were worked out here, not official

    The timing lines are placed using the faster lap that has official
    sectors. A driver with no official sector times gets them derived from
    those lines: their own elapsed time at each line.
    """
    official = [official_sectors(lap) for lap in lap_rows]
    order = np.argsort(lap_times)  # faster driver first
    source = next((i for i in order if official[i] is not None), None)
    if source is None:
        return None, [None, None], [False, False]
    lap = laps_on_grid[source]
    ends = sector_ends(lap["Time"].to_numpy(), lap["Distance"].to_numpy(), official[source])

    sectors, derived = [], []
    for i, lap in enumerate(laps_on_grid):
        if official[i] is not None:
            sectors.append(official[i])
            derived.append(False)
        else:
            at_lines = np.interp(ends, lap["Distance"], lap["Time"])
            sectors.append(np.diff([0.0, *at_lines, lap_times[i]]).tolist())
            derived.append(True)
    return ends, sectors, derived


# ---- Corners -------------------------------------------------------------------

def corner_marks(corners, outline_x, outline_y, distance, rotation):
    """Each corner's label, where it is along the lap, and which way its label points.

    `corners` are MultiViewer's markers (data.get_corners). The anchor is the
    point on the drawn track line nearest the marker, so the leader line
    starts on the track itself. The direction is MultiViewer's label angle,
    turned with the rest of the map; the player decides how far out (in
    pixels) to put the label, so labels stay the same size on any circuit.
    """
    mx, my = rotate(corners["X"].to_numpy(dtype=float), corners["Y"].to_numpy(dtype=float), rotation)
    angle = corners["Angle"].to_numpy(dtype=float) + rotation
    marks = []
    for label, x, y, a in zip(analysis.corner_labels(corners), mx, my, angle):
        i = int(np.argmin((outline_x - x) ** 2 + (outline_y - y) ** 2))
        marks.append({"label": label, "distance": round(float(distance[i]), 1),
                      "dx": round(float(np.cos(np.radians(a))), 4), "dy": round(float(np.sin(np.radians(a))), 4)})
    return sorted(marks, key=lambda m: m["distance"])


# ---- The payload ---------------------------------------------------------------

def lap_label(lap):
    """'Lap 18 · Q3' (the qualifying part only where there is one)."""
    part = lap.get("part")
    return f"Lap {int(lap['lap_number'])}" + (f" · {part}" if part else "")


def build_payload(run, rotation=0.0):
    """Everything the player needs, as one JSON-ready dict.

    `run` is analysis.load_and_compare()'s result; `rotation` is
    data.get_rotation()'s angle for the circuit. The keys:
      title       "2026 Las Vegas Qualifying"
      length      lap length (m); the grid runs 0..length in `step` m steps
      x, y        the track line on that grid (map units of 0.1 m, turned to
                  the official orientation). It comes from the faster lap.
      drivers     per driver: code, number, team, colour, lap time and label,
                  sector times, and per grid point time (s since the line),
                  speed, throttle, brake (0/1), gear and rpm
      delta       running gap at each grid point: B's time minus A's (s);
                  positive = B behind, the sign used everywhere in this app
      mini        the 25 mini-sectors as [start m, end m, winner 0=A / 1=B]
      sector_ends [S1 end m, S2 end m], or null if unknown
      corners     label, distance along the lap, and label direction
    """
    result, laps = run["result"], run["laps"]
    a, b = result["a"], result["b"]
    times = run["times"]
    drivers = [lap["driver"] for lap in laps]
    names = [d["name_acronym"] for d in drivers]
    colours = charts.driver_colours([d["team_colour"] for d in drivers], [d["team_name"] for d in drivers])

    # The track outline comes from the faster lap (the brief's rule); both laps
    # share the same distance grid, so either car's distance can be drawn on it.
    outline = (a, b)[int(np.argmin(times))]
    x, y = rotate(outline["X"].to_numpy(), outline["Y"].to_numpy(), rotation)
    distance = a["Distance"].to_numpy()

    ends, sectors, derived = sector_table((a, b), [lap["lap"] for lap in laps], times)

    session = run["session"]
    payload = {
        "title": f"{session['year']} {session['location']} {session['session_name']}",
        "length": round(float(distance[-1]), 1),
        "step": round(float(distance[1] - distance[0]), 6),
        "x": numbers(x, 1), "y": numbers(y, 1),
        "delta": numbers(result["delta"], 3),
        "mini": [[round(float(s), 1), round(float(e), 1), int(w)] for s, e, w in result["mini_sectors"]],
        "sector_ends": None if ends is None else [round(e, 1) for e in ends],
        "corners": corner_marks(run["corners"], x, y, distance, rotation),
        "drivers": [],
    }
    for i, (driver, lap, grid) in enumerate(zip(drivers, laps, (a, b))):
        payload["drivers"].append({
            "code": names[i], "number": str(int(driver["driver_number"])), "team": str(driver["team_name"]),
            "colour": colours[i], "lap_time": round(times[i], 3), "lap_label": lap_label(lap["lap"]),
            "sectors": None if sectors[i] is None else [round(s, 3) for s in sectors[i]],
            "sectors_derived": derived[i],
            "time": numbers(grid["Time"], 3), "speed": numbers(grid["Speed"], 1),
            "throttle": numbers(grid["Throttle"], 0), "brake": numbers(grid["Brake"], 0),
            "gear": numbers(grid["Gear"], 0), "rpm": numbers(grid["RPM"], 0),
        })
    return payload


# ---- The player ----------------------------------------------------------------

def font_face():
    """The CSS that loads the car-number font, embedded in the page (base64) so the player needs no files or internet."""
    encoded = base64.b64encode(NUMBER_FONT["file"].read_bytes()).decode()
    return (f'@font-face {{ font-family: "{NUMBER_FONT["family"]}"; font-weight: 100 900; font-stretch: 62% 125%; '
            f'src: url(data:font/woff2;base64,{encoded}) format("woff2"); }}\n'
            f'.car-number {{ font-family: "{NUMBER_FONT["family"]}"; '
            f'font-variation-settings: {NUMBER_FONT["settings"]}; }}')


def player_html(payload):
    """The finished player page: the template with the payload and the font poured in.

    "</" is escaped as "<\\/" so text in the data can never close the
    <script> tag early. The player puts every piece of text from the data on
    the page with textContent, never as HTML, so names can't inject markup.
    """
    data_js = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    template = PLAYER_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("/*__FONT_FACE__*/", font_face()).replace("/*__PAYLOAD__*/null", data_js)


# ---- Terminal check (stage 1 of the brief) --------------------------------------

def check(payload, run):
    """Print the payload's times next to OpenF1's official ones, and stop if they disagree."""
    for driver, lap in zip(payload["drivers"], run["laps"]):
        row = lap["lap"]
        official = official_sectors(row)
        end_time = driver["time"][-1]
        print(f"{driver['code']} #{driver['number']} {driver['lap_label']}: official "
              f"{analysis.lap_time_text(float(row['lap_duration']))}, payload ends at {end_time:.3f} s")
        if official:
            print(f"    sectors official {' / '.join(f'{s:.3f}' for s in official)} "
                  f"(sum {sum(official):.3f}), payload {' / '.join(f'{s:.3f}' for s in driver['sectors'])}")
        else:
            print(f"    no official sector times; derived {driver['sectors']}")
        # The grid's last point is the finish line, so the replay must end exactly on the lap time.
        assert abs(end_time - float(row["lap_duration"])) < 0.002, "payload lap time doesn't match OpenF1"
    delta = payload["delta"][-1]
    gap = payload["drivers"][1]["lap_time"] - payload["drivers"][0]["lap_time"]
    print(f"Gap at the line {delta:+.3f} s, lap-time gap {gap:+.3f} s")
    assert abs(delta - gap) < 0.002, "running gap doesn't end at the lap-time gap"
    print(f"Lap {payload['length']:.0f} m in {len(payload['x'])} grid points; {len(payload['mini'])} mini-sectors; "
          f"sector lines at {payload['sector_ends']} m; {len(payload['corners'])} corners")


def main(year, place, session_name, name_a, name_b, lap_a=None, lap_b=None):
    import fastf1

    import data

    fastf1.set_log_level("ERROR")
    session = data.find_session(year, place, session_name)
    run = analysis.load_and_compare(session, name_a, name_b, lap_a, lap_b)
    payload = build_payload(run, data.get_rotation(int(session["year"]), int(session["circuit_key"])))
    check(payload, run)

    page = player_html(payload)
    out = HERE / "output" / f"{year}_{place}_{session_name}_{name_a}_{name_b}_replay.html".replace(" ", "_")
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Saved {out} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026", "Las Vegas", "Qualifying", "ANT", "VER"]
    laps_given = [int(n) for n in args[5:7]] + [None, None]
    try:
        main(int(args[0]), args[1], args[2], args[3], args[4], laps_given[0], laps_given[1])
    except LookupError as error:
        sys.exit(str(error))
