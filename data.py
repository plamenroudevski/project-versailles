"""
data.py: where every piece of F1 data comes from.

Pipeline position:  data.py -> analysis.py -> charts.py -> app.py
This file fetches raw data and turns it into tidy tables. It does no analysis
and has no Streamlit code, so scripts and tests can use it directly.

Sources:
- OpenF1 (api.openf1.org): sessions, drivers, lap times, car telemetry,
  car position, weather and race-control messages. Free and needs no login,
  but only covers 2023 onwards and allows 3 requests/s and 30 requests/min.
- MultiViewer, through FastF1: corner positions for each circuit.
  (F1's own live-timing archive, which FastF1 normally reads, now requires a
  paid F1 TV login, which is why telemetry comes from OpenF1.)

Caching: every web request goes through FastF1's disk cache, a small database
in FastF1's cache folder. The same request within 12 hours is answered from
disk: instant, works offline, and doesn't use up OpenF1's rate limit.
The app adds Streamlit's in-memory cache on top of this.

Missing data raises LookupError with a message written for the user, so the
app can show it as-is.
"""

import re
import time

import fastf1
import numpy as np
import pandas as pd
from fastf1.mvapi import get_circuit_info

API = "https://api.openf1.org/v1"

# OpenF1's free tier allows 3 requests per second and 30 per minute. Going over
# returns "429 Too Many Requests", so we wait and retry: briefly first (the
# per-second limit), then long enough to clear the per-minute one (~1 min total).
RETRY_WAITS_S = (2, 5, 15, 40)

# Seconds of telemetry fetched either side of a lap. Samples arrive every
# ~0.25 s and never land exactly on the start/finish line, so the padding
# gives us samples on both sides of the line to interpolate between.
LAP_PADDING_S = 2.0

# Race control announces deleted laps as text, for example:
#   "CAR 18 (STR) TIME 1:42.680 DELETED - DOUBLE YELLOW AT TURN 3 LAP 4 16:29:11"
#   "CAR 51 (CAM) LAP DELETED - TRACK LIMITS AT TURN 4 LAP 4 16:29:11 (PIT)"
# This pattern pulls out the driver code and the final hh:mm:ss, which is the
# local clock time of the incident.
DELETED_LAP = re.compile(r"\((?P<driver>[A-Z]{3})\).*\bDELETED\b.*\b(?P<clock>\d{2}:\d{2}:\d{2})\b")


def openf1(endpoint, **params):
    """Fetch one OpenF1 endpoint as a table (a pandas DataFrame).

    OpenF1 answers "404 No results found" when a filter matches nothing, so
    that becomes an empty table rather than an error. "429 Too Many Requests"
    is retried after a pause (see RETRY_WAITS_S). Any other failure (network
    down, server error) raises an exception with the reason.
    """
    for wait_s in (*RETRY_WAITS_S, None):
        response = fastf1.Cache.requests_get(f"{API}/{endpoint}", params=params, timeout=60)
        if response.status_code != 429 or wait_s is None:
            break
        time.sleep(wait_s)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    return pd.DataFrame(response.json())


def find_session(year, place, session_name):
    """Find one session, e.g. (2026, "Brazil", "Qualifying").

    `place` is matched against the country, city and circuit name, so
    "Brazil", "São Paulo" and "Interlagos" all find the same event. That
    matters because F1 renames events (Brazilian GP -> São Paulo GP).
    Returns one row with session_key, circuit_key, location and so on.
    """
    sessions = openf1("sessions", year=year, session_name=session_name)
    if sessions.empty:
        raise LookupError(f"OpenF1 has no '{session_name}' sessions for {year}.")
    names = sessions[["country_name", "location", "circuit_short_name"]].astype(str).agg(" ".join, axis=1)
    matches = sessions[names.str.contains(place, case=False, regex=False)]
    if matches.empty:
        raise LookupError(f"No {year} {session_name} at '{place}'. Try one of: {', '.join(sessions['location'])}.")
    return matches.iloc[0]


def get_session(session_key):
    """One session's details (the same kind of row find_session returns), by its OpenF1 key."""
    sessions = openf1("sessions", session_key=session_key)
    if sessions.empty:
        raise LookupError("OpenF1 doesn't know this session.")
    return sessions.iloc[0]


def started(table):
    """Keep only rows whose date_start is in the past: future sessions have no data yet."""
    return table[pd.to_datetime(table["date_start"], format="ISO8601") <= pd.Timestamp.now(tz="UTC")]


def list_events(year):
    """A season's events (OpenF1 calls them meetings) that have started, newest first.

    Columns include meeting_key, meeting_name, location and date_start.
    """
    meetings = openf1("meetings", year=year)
    if meetings.empty:
        return meetings
    return started(meetings).sort_values("date_start", ascending=False).reset_index(drop=True)


def list_sessions(meeting_key):
    """An event's sessions that have started and weren't cancelled, in running order."""
    sessions = openf1("sessions", meeting_key=meeting_key)
    if sessions.empty:
        return sessions
    if "is_cancelled" in sessions:
        sessions = sessions[sessions["is_cancelled"] != True]  # noqa: E712 (the column can also hold None)
    return started(sessions).sort_values("date_start").reset_index(drop=True)


def list_drivers(session):
    """Everyone in a session with their fastest valid lap, fastest first.

    Columns include driver_number, name_acronym, full_name, team_name,
    team_colour and best_lap (seconds; empty if they set no valid time).
    """
    drivers = openf1("drivers", session_key=int(session["session_key"]))
    if drivers.empty:
        return drivers
    drivers = drivers.drop_duplicates("driver_number").copy()
    best = []
    for _, driver in drivers.iterrows():
        laps = get_laps(session, driver)
        lap = None if laps.empty else fastest_lap(laps)
        best.append(np.nan if lap is None else float(lap["lap_duration"]))
    drivers["best_lap"] = best
    return drivers.sort_values("best_lap", na_position="last").reset_index(drop=True)


def get_driver(session_key, acronym):
    """Return one driver's entry (number, team, team colour) for a session."""
    drivers = openf1("drivers", session_key=session_key, name_acronym=acronym)
    if drivers.empty:
        raise LookupError(f"{acronym} didn't take part in this session.")
    return drivers.iloc[0]  # OpenF1 sometimes lists a driver twice; the rows are identical


def deleted_laps(control, laps, acronym, gmt_offset):
    """Return the set of this driver's lap numbers that race control deleted.

    Deleted laps (track limits, yellow flags) don't count, so they must never
    be picked as a driver's fastest lap.

    Race control's own "LAP n" doesn't always match OpenF1's lap numbers (in
    2026 Brazil it was one higher, after the delayed start), so we use the
    incident's clock time instead and find the lap being driven at that moment.

    control:    race-control messages (columns `date` in UTC and `message`)
    laps:       the driver's timed laps (`lap_number`, `date_start`, `lap_duration`)
    gmt_offset: circuit's local time minus UTC, e.g. -3 h for São Paulo
    """
    found = control["message"].str.extract(DELETED_LAP)
    found["published"] = pd.to_datetime(control["date"], format="ISO8601")
    starts = pd.to_datetime(laps["date_start"], format="ISO8601")
    ends = starts + pd.to_timedelta(laps["lap_duration"], unit="s")

    deleted = set()
    for clock, published in found.loc[found["driver"] == acronym, ["clock", "published"]].itertuples(index=False):
        # The message gives a local clock time with no date: take the date from
        # when the message was published (in local time). If that puts the
        # incident after the message, it happened before midnight: go back a day.
        published_local = published + gmt_offset
        incident_local = pd.Timestamp(f"{published_local.date()} {clock}", tz="UTC")
        if incident_local > published_local:
            incident_local -= pd.to_timedelta(1, unit="D")
        incident = incident_local - gmt_offset
        deleted.update(laps.loc[(starts <= incident) & (incident < ends), "lap_number"])
    return deleted


def get_laps(session, driver):
    """Return a driver's timed laps, sorted by lap number, with a `deleted` column.

    `session` is the row returned by find_session (it holds the time-zone offset).
    The whole session's laps are fetched in one request and filtered here, so
    listing 20 drivers costs one request (then cached), not 20.
    """
    laps = openf1("laps", session_key=int(session["session_key"]))
    if laps.empty:
        return laps
    laps = laps[laps["driver_number"] == int(driver["driver_number"])]
    laps = laps.dropna(subset=["lap_duration", "date_start"]).copy()
    if laps.empty:
        return laps
    control = openf1("race_control", session_key=int(session["session_key"]))
    laps["deleted"] = False
    if not control.empty:
        gone = deleted_laps(control, laps, driver["name_acronym"], pd.to_timedelta(session["gmt_offset"]))
        laps["deleted"] = laps["lap_number"].isin(gone)
    return laps.sort_values("lap_number").reset_index(drop=True)


def fastest_lap(laps):
    """Return the quickest lap that wasn't deleted, or None if there isn't one."""
    valid = laps[~laps["deleted"]]
    if valid.empty:
        return None
    return valid.loc[valid["lap_duration"].idxmin()]


def get_lap_telemetry(session_key, driver_number, lap):
    """Return one lap of telemetry as a table, one row per car-data sample (~4 Hz).

    Columns:
      Time      seconds since the car crossed the start line (padding samples
                are slightly negative, or later than the lap time)
      Distance  metres from the start line, from integrating speed over time
      Speed (km/h), Throttle (%), Brake (True/False), Gear, RPM
      X, Y      position on track (OpenF1/MultiViewer grid, units of 0.1 m)
      DRS       raw DRS value (empty for 2026 cars, which use active aero)
    """
    start = pd.Timestamp(lap["date_start"])
    pad = pd.to_timedelta(LAP_PADDING_S, unit="s")
    end = start + pd.to_timedelta(float(lap["lap_duration"]), unit="s")
    window = {"date>": (start - pad).isoformat(), "date<": (end + pad).isoformat()}
    car = openf1("car_data", session_key=session_key, driver_number=driver_number, **window)
    pos = openf1("location", session_key=session_key, driver_number=driver_number, **window)
    if car.empty or pos.empty:
        raise LookupError(f"No telemetry for lap {lap['lap_number']} yet. OpenF1 may still be processing it.")

    def seconds_since_start(frame):
        return (pd.to_datetime(frame["date"], format="ISO8601") - start).dt.total_seconds().to_numpy()

    car = car.sort_values("date")
    pos = pos.sort_values("date")
    car_t, pos_t = seconds_since_start(car), seconds_since_start(pos)
    tel = pd.DataFrame({
        "Time": car_t,
        "Speed": car["speed"].to_numpy(dtype=float),
        "Throttle": car["throttle"].to_numpy(dtype=float),
        "Brake": car["brake"].to_numpy() > 0,
        "Gear": car["n_gear"].to_numpy(),
        "RPM": car["rpm"].to_numpy(dtype=float),
        "DRS": car["drs"].to_numpy(),
        # Position is sampled at different instants from car data, so interpolate it onto the car-data times.
        "X": np.interp(car_t, pos_t, pos["x"].to_numpy(dtype=float)),
        "Y": np.interp(car_t, pos_t, pos["y"].to_numpy(dtype=float)),
    })

    # Distance = speed integrated over time (trapezium rule): each step adds
    # the average speed of its two samples (km/h / 3.6 = m/s) x the time step.
    step_m = (tel["Speed"].shift() + tel["Speed"]) / 2 / 3.6 * tel["Time"].diff()
    distance = step_m.fillna(0).cumsum().to_numpy()
    # Shift so the moment the car crosses the line (Time = 0) is at 0 m.
    tel["Distance"] = distance - np.interp(0.0, car_t, distance)
    return tel


def load_lap(session, acronym, lap_number=None):
    """Everything about one driver's lap: entry, all laps, the chosen lap, its telemetry.

    `session` is the row returned by find_session. Uses the driver's fastest
    valid lap unless `lap_number` is given. Returns a dict so callers can pick
    what they need.
    """
    session_key = int(session["session_key"])
    driver = get_driver(session_key, acronym)
    laps = get_laps(session, driver)
    if laps.empty:
        raise LookupError(f"{acronym} didn't set a lap time in this session.")
    if lap_number is None:
        lap = fastest_lap(laps)
        if lap is None:
            raise LookupError(f"All of {acronym}'s lap times in this session were deleted.")
    else:
        chosen = laps[laps["lap_number"] == lap_number]
        if chosen.empty:
            raise LookupError(f"{acronym} has no timed lap {lap_number}.")
        lap = chosen.iloc[0]
    telemetry = get_lap_telemetry(session_key, int(driver["driver_number"]), lap)
    return {"driver": driver, "laps": laps, "lap": lap, "telemetry": telemetry}


def get_corners(year, circuit_key):
    """Return corner markers for a circuit, from MultiViewer.

    Columns: X, Y (map position), Number, Letter (e.g. 2 and "A" for T2A) and
    Angle (degrees, the direction to place the corner's label on a map).
    """
    circuit = get_circuit_info(year=year, circuit_key=circuit_key)
    if circuit is None or circuit.corners.empty:
        raise LookupError("No corner positions for this circuit, so the lap can't be split into corners.")
    return circuit.corners[["X", "Y", "Number", "Letter", "Angle"]]


def session_had_rain(session_key):
    """True if the weather station reported rain at any point in the session."""
    weather = openf1("weather", session_key=session_key)
    return bool(not weather.empty and weather["rainfall"].any())
