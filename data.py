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

Caching: OpenF1 answers are kept on disk (a small database in the temp folder)
by requests-cache, the library FastF1's own cache is built on. The same request
within CACHE_EXPIRY_S (10 minutes) is answered from disk: instant, and doesn't
use up OpenF1's rate limit. The app adds Streamlit's in-memory cache on top.
FastF1's request helper isn't used for OpenF1 because it also enforces a hidden
cap of 500 downloads per hour, shared by every visitor of the app, and raises
an error past it. Corner positions still come through FastF1 and its cache.

Your notes: plain Markdown files in notes/, one per session and driver pair,
so you can write them in Obsidian and commit them with the code. No database.

Missing data raises LookupError with a message written for the user, so the
app can show it as-is.
"""

import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests_cache
from fastf1.mvapi import get_circuit_info

API = "https://api.openf1.org/v1"

# OpenF1's free tier allows 3 requests per second and 30 per minute. Going over
# returns "429 Too Many Requests", so we wait and retry: briefly first (the
# per-second limit), then long enough to clear the per-minute one (~1 min total).
RETRY_WAITS_S = (2, 5, 15, 40)

# How long a downloaded OpenF1 answer is reused from disk. Kept short because a
# session that has only just finished may still be filling in on OpenF1, and a
# half-finished answer mustn't stick for hours. Still long enough to absorb
# bursts (listing 20 drivers reads the same laps answer 20 times).
CACHE_EXPIRY_S = 600

# One shared download session, caching to openf1.sqlite in the system's temp folder.
# stale_if_error: if a download fails (OpenF1 down, or a 429/5xx) and an expired copy
# is on disk, use that copy instead of failing. With no copy, errors pass through as before.
HTTP = requests_cache.CachedSession("openf1", use_temp=True, expire_after=CACHE_EXPIRY_S, stale_if_error=True)

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

# A deleted lap can be given back after a review. That message names the lap
# time instead of a clock time, for example (2025 Barcelona):
#   "CAR 44 (HAM) TIME 1:20.755 WILL BE REINSTATED"
REINSTATED_LAP = re.compile(r"\((?P<driver>[A-Z]{3})\) TIME (?P<minutes>\d+):(?P<seconds>\d{2}\.\d{3})\b.*\bREINSTATED\b")


def openf1(endpoint, **params):
    """Fetch one OpenF1 endpoint as a table (a pandas DataFrame).

    `**params` collects any keyword arguments into a dict, which become the
    URL's filters: openf1("laps", session_key=11408) asks for
    .../laps?session_key=11408.

    OpenF1 answers "404 No results found" when a filter matches nothing, so
    that becomes an empty table rather than an error. "429 Too Many Requests"
    is retried after a pause (see RETRY_WAITS_S). Any other failure (network
    down, server error) raises an exception with the reason.
    """
    # One try per wait in RETRY_WAITS_S, plus a last try (None: don't wait again, give up).
    for wait_s in (*RETRY_WAITS_S, None):
        response = HTTP.get(f"{API}/{endpoint}", params=params, timeout=60)
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
    # Different places matched, e.g. "United States" in 2026: Miami, Austin and Las Vegas.
    # Two events at the same place (2026's two Bahrain tests) keep the first, as a city can't split them.
    if matches["location"].nunique() > 1:
        raise LookupError(f"'{place}' matches several {year} events ({', '.join(matches['location'])}). "
                          f"Use the city instead.")
    return matches.iloc[0]


def get_session(session_key):
    """One session's details (the same kind of row find_session returns), by its OpenF1 key."""
    sessions = openf1("sessions", session_key=session_key)
    if sessions.empty:
        raise LookupError("OpenF1 doesn't know this session.")
    return sessions.iloc[0]


def started(table):
    """Keep only rows whose date_start is in the past: future sessions have no data yet."""
    # utc=True: an empty table (an event whose sessions were all cancelled, like 2026's April
    # Bahrain GP) has no time zone to read, and can't be compared with a UTC time without it.
    return table[pd.to_datetime(table["date_start"], format="ISO8601", utc=True) <= pd.Timestamp.now(tz="UTC")]


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
    2026 Brazil qualifying it was one higher), so we use the incident's clock
    time instead and find the lap being driven at that moment. Laps race
    control later reinstates are matched by their lap time and count again.

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

    # shortcut: only "TIME m:ss.sss ... REINSTATED" is understood (9 of the 10 such
    # messages in 2025-26). "LAP 1 WILL BE REINSTATED" (Monza 2026 race) isn't, because
    # race control's lap numbers can be off by one; lap 1 is never a fastest lap, so
    # revisit only if a qualifying lap is announced that way. A reinstatement also
    # always wins, even if the lap were deleted again later (not seen in 2025-26).
    back = control["message"].str.extract(REINSTATED_LAP)
    for minutes, seconds in back.loc[back["driver"] == acronym, ["minutes", "seconds"]].itertuples(index=False):
        lap_time = int(minutes) * 60 + float(seconds)
        deleted -= set(laps.loc[(laps["lap_duration"] - lap_time).abs() < 0.0005, "lap_number"])
    return deleted


def part_starts(control, session_name):
    """When each part of a qualifying session began (UTC), e.g. Q1 18:00, Q2 19:49, Q3 20:14.

    Empty for sessions without parts. Race control tags every message with its
    part (qualifying_phase) and posts "SESSION STARTED" as a part begins, and
    again after each red flag, so a part starts at its first one. Gaps between
    laps can't do this reliably: in 2026 São Paulo, Q1 was red-flagged two
    minutes in and only restarted 84 minutes later.
    """
    if "qualifying_phase" not in control:
        return pd.Series(dtype=object)
    began = control[(control["message"] == "SESSION STARTED") & control["qualifying_phase"].notna()]
    starts = pd.to_datetime(began["date"], format="ISO8601").groupby(began["qualifying_phase"].astype(int)).min()
    prefix = "SQ" if "Sprint" in session_name else "Q"  # sprint qualifying's parts are SQ1-SQ3
    return starts.rename(index=lambda phase: f"{prefix}{phase}").sort_values()


def lap_parts(laps, starts):
    """Each lap's part (from part_starts): the last part to start before the lap began, or None."""
    if starts.empty:
        return [None] * len(laps)
    position = starts.searchsorted(pd.to_datetime(laps["date_start"], format="ISO8601"), side="right") - 1
    return [starts.index[i] if i >= 0 else None for i in position]


def list_parts(session):
    """A session's qualifying parts in order, e.g. ["Q1", "Q2", "Q3"]; empty for other sessions."""
    control = openf1("race_control", session_key=int(session["session_key"]))
    return [] if control.empty else part_starts(control, session["session_name"]).index.tolist()


def get_laps(session, driver):
    """Return a driver's timed laps, sorted by lap number, with `deleted` and `part` columns.

    `part` is "Q1", "Q2" or "Q3" (SQ1-SQ3 in sprint qualifying), None in other
    sessions. `session` is a row from find_session or get_session (it holds the
    time-zone offset). The whole session's laps are fetched in one request and
    filtered here, so listing 20 drivers costs one request (then cached), not 20.
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
    laps["part"] = None
    if not control.empty:
        gone = deleted_laps(control, laps, driver["name_acronym"], pd.to_timedelta(session["gmt_offset"]))
        laps["deleted"] = laps["lap_number"].isin(gone)
        laps["part"] = lap_parts(laps, part_starts(control, session["session_name"]))
    return laps.sort_values("lap_number").reset_index(drop=True)


def fastest_lap(laps):
    """Return the quickest lap that wasn't deleted, or None if there isn't one."""
    valid = laps[~laps["deleted"]]
    if valid.empty:
        return None
    return valid.loc[valid["lap_duration"].idxmin()]


def get_lap_telemetry(session_key, driver_number, lap):
    """Download one lap of telemetry, plus LAP_PADDING_S either side, as a tidy table (see build_telemetry)."""
    start = pd.Timestamp(lap["date_start"])
    pad = pd.to_timedelta(LAP_PADDING_S, unit="s")
    end = start + pd.to_timedelta(float(lap["lap_duration"]), unit="s")
    window = {"date>": (start - pad).isoformat(), "date<": (end + pad).isoformat()}
    car = openf1("car_data", session_key=session_key, driver_number=driver_number, **window)
    pos = openf1("location", session_key=session_key, driver_number=driver_number, **window)
    if car.empty or pos.empty:
        raise LookupError(f"No telemetry for lap {lap['lap_number']} yet. OpenF1 may still be processing it.")
    return build_telemetry(car, pos, start)


def build_telemetry(car, pos, start):
    """Turn OpenF1's raw car-data and position tables into one telemetry table.

    Kept apart from the download so it can be tested without internet.
    `start` is the moment the lap began. One row per car-data sample (~4 Hz):
      Time      seconds since the car crossed the start line (padding samples
                are slightly negative, or later than the lap time)
      Distance  metres from the start line, from integrating speed over time
      Speed (km/h), Throttle (%), Brake (True/False), Gear
      X, Y      position on track (OpenF1/MultiViewer grid, units of 0.1 m)
      DRS       raw DRS value (empty for 2026 cars, which use active aero)
    """
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

    `session` is a row from find_session or get_session. Uses the driver's fastest
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


# ---- Races: the lap table and results for pace.py ------------------------------------

def neutralised_periods(control):
    """When the race was neutralised (safety car, virtual safety car or red flag), as (start, end) times in UTC.

    Race control posts a message when each one starts and when it ends:
      starts: "SAFETY CAR DEPLOYED", "VSC DEPLOYED" (before 2026: "VIRTUAL SAFETY
              CAR DEPLOYED"), "RED FLAG", and "SESSION ABORTED" (2026 red flags
              only post this one)
      ends:   "TRACK CLEAR" or a green light for the whole track (the restart),
              "VSC ENDING", "SESSION STARTED" (racing resumes after a red flag),
              and the chequered flag
    A period runs from the first start message to the first end message after
    it. Starts while a period is already open are part of it (a VSC turned into
    a safety car, a red flag shown during a safety car).

    Why the first end message of any kind: messages go missing. 2026 Monza and
    Mexico City have a "SAFETY CAR DEPLOYED" with no "SAFETY CAR IN THIS LAP",
    and older years have fewer "VSC ENDING" messages than "VSC DEPLOYED". Waiting
    for a matching message would leave the rest of the race neutralised.
    "SAFETY CAR IN THIS LAP" isn't an end: the safety car is still out until
    the restart, which "TRACK CLEAR" marks a minute or two later.

    Known gap: after a red-flag restart, cars can follow the safety car for a
    lap or two with no message at all (2026 Monza, Mexico City, Monaco). Those
    laps aren't marked here. They're 10-130 % slower than racing laps, so
    pace.py's 2-sigma filter removes most of them: 96 of 98 in those three
    races. Two survived (Alonso's and Stroll's lap 6 at Monza, ~126 s against
    ~85 s racing laps).
    """
    if control.empty:
        return []
    message, flag = control["message"].fillna(""), control["flag"].fillna("")
    safety_car = control["category"] == "SafetyCar"
    whole_track = control["scope"] == "Track"
    starts = ((safety_car & message.str.contains("DEPLOYED")) | (flag == "RED")
              | (message == "SESSION ABORTED"))
    ends = ((whole_track & flag.isin(["CLEAR", "GREEN"])) | (safety_car & message.str.contains("ENDING"))
            | (message == "SESSION STARTED") | (flag == "CHEQUERED"))

    events = pd.DataFrame({"date": pd.to_datetime(control["date"], format="ISO8601"), "start": starts, "end": ends})
    periods, opened = [], None
    for date, start, end in events[starts | ends].sort_values("date").itertuples(index=False):
        if start and opened is None:
            opened = date
        elif end and opened is not None:
            periods.append((opened, date))
            opened = None
    if opened is not None:  # still neutralised when the data ends (a race in progress)
        periods.append((opened, pd.Timestamp.max.tz_localize("UTC")))
    return periods


def build_race_laps(laps, drivers, stints, pits, control):
    """Turn OpenF1's race tables into the plain lap table pace.py reads.

    Kept apart from the downloads (race_lap_table) so it can be tested without
    internet. One row per driver per lap:
      driver       three-letter code, e.g. NOR
      lap          lap number (1 = the start)
      lap_time_s   lap time in seconds; empty if the lap has no time
      compound     "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"; empty if
                   OpenF1 has no tyre data for that lap (never guessed)
      tyre_age     laps the set had already done when this lap began (OpenF1
                   counts a stint's tyre age the same way)
      pit_in       the driver came into the pits at the end of this lap
      pit_out      the lap started in the pit lane
      neutralised  any part of the lap was run under a safety car, VSC or red
                   flag (see neutralised_periods)
    """
    code = dict(zip(drivers["driver_number"], drivers["name_acronym"]))
    table = pd.DataFrame({
        "driver_number": laps["driver_number"].astype(int),
        "driver": laps["driver_number"].map(code),
        "lap": laps["lap_number"].astype(int),
        "lap_time_s": laps["lap_duration"].astype(float),
        "pit_out": laps["is_pit_out_lap"].eq(True),  # True/False; a missing value counts as False
    })

    # Tyres: OpenF1 gives one row per stint (laps lap_start to lap_end on one set),
    # so spread each stint over its laps. A stint still running has no lap_end yet.
    last_lap = int(table["lap"].max())
    tyres = pd.DataFrame(
        [(s.driver_number, lap, s.compound, s.tyre_age_at_start + lap - s.lap_start)
         for s in stints.itertuples()
         for lap in range(int(s.lap_start), int(s.lap_end if pd.notna(s.lap_end) else last_lap) + 1)],
        columns=["driver_number", "lap", "compound", "tyre_age"]).astype({"driver_number": int, "lap": int})
    table = table.merge(tyres.drop_duplicates(["driver_number", "lap"], keep="last"),
                        on=["driver_number", "lap"], how="left")

    # In-laps: the pit endpoint names each stop's lap. The lap before a pit out-lap
    # is an in-lap too, which covers a stop the pit endpoint missed.
    stops = set() if pits.empty else set(zip(pits["driver_number"].astype(int), pits["lap_number"].astype(int)))
    before_out = set(zip(table.loc[table["pit_out"], "driver_number"], table.loc[table["pit_out"], "lap"] - 1))
    table["pit_in"] = [key in stops or key in before_out for key in zip(table["driver_number"], table["lap"])]

    # A lap is neutralised if it overlaps a period at all: started before the period
    # ended and ended after it started. Laps with no time have no end, so never match
    # (pace.py drops them anyway).
    starts = pd.to_datetime(laps["date_start"], format="ISO8601").reset_index(drop=True)
    ends = starts + pd.to_timedelta(laps["lap_duration"].to_numpy(), unit="s")
    neutralised = pd.Series(False, index=starts.index)
    for begin, finish in neutralised_periods(control):
        neutralised |= (starts <= finish) & (ends >= begin)
    table["neutralised"] = neutralised.to_numpy()  # same row order as `laps` (a left merge keeps it)

    columns = ["driver", "lap", "lap_time_s", "compound", "tyre_age", "pit_in", "pit_out", "neutralised"]
    return table[columns].sort_values(["driver", "lap"]).reset_index(drop=True)


def race_lap_table(session):
    """Download one race's laps, tyres, pit stops and race control, and build pace.py's lap table.

    Each endpoint is fetched once for the whole field (not once per driver), as
    the free tier allows 30 requests a minute for every visitor together.
    """
    key = int(session["session_key"])
    laps = openf1("laps", session_key=key)
    if laps.empty:
        raise LookupError("OpenF1 has no lap times for this race yet. If it has only just finished, "
                          "try again in about 10 minutes.")
    drivers = openf1("drivers", session_key=key).drop_duplicates("driver_number")
    return build_race_laps(laps, drivers, openf1("stints", session_key=key), openf1("pit", session_key=key),
                           openf1("race_control", session_key=key))


def build_results(results, drivers):
    """One row per driver: driver, team, team_colour, position, status, laps.

    status is "Finished", "DNF" (retired), "DNS" (didn't start), "DSQ"
    (disqualified), or "No result" if the driver is missing from the results.
    position is the official finishing position (empty if not classified);
    laps is how many laps they completed.
    """
    table = drivers[["driver_number", "name_acronym", "team_name", "team_colour"]].merge(
        results[["driver_number", "position", "number_of_laps", "dnf", "dns", "dsq"]], on="driver_number", how="left")
    status = np.select([table["dsq"] == True, table["dns"] == True, table["dnf"] == True,  # noqa: E712 (may be empty)
                        table["position"].notna()], ["DSQ", "DNS", "DNF", "Finished"], default="No result")
    return pd.DataFrame({"driver": table["name_acronym"], "team": table["team_name"],
                         "team_colour": table["team_colour"], "position": table["position"],
                         "status": status, "laps": table["number_of_laps"]})


def race_results(session):
    """Download one race's official result (OpenF1's session_result) with each driver's team.

    session_result exists for every race OpenF1 has since 2023 (checked
    November 2026: the only races without one were cancelled), so there's no
    fallback; a race that has only just finished gets a "try again" message.
    """
    key = int(session["session_key"])
    results = openf1("session_result", session_key=key)
    if results.empty:
        raise LookupError("OpenF1 hasn't published this race's result yet. If it has only just finished, "
                          "try again in about 10 minutes.")
    return build_results(results, openf1("drivers", session_key=key).drop_duplicates("driver_number"))


# ---- Your notes ------------------------------------------------------------------

NOTES_DIR = Path(__file__).parent / "notes"

# Short session codes for note file names (anything else is spelled out).
SESSION_CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q",
                 "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ", "Sprint": "S", "Race": "R"}

NOTES_TEMPLATE = """\
## What the telemetry shows
-

## What I think caused it
-

## To check next time
-
"""


def slug(text):
    """'São Paulo' -> 'sao_paulo': lowercase letters, digits and underscores, safe in any file name."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()  # drop accents
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def notes_path(session, names):
    """Where your notes for a comparison live, e.g. notes/2026_sao_paulo_Q_NOR_RUS.md.

    Uses the city, not the country: the USA alone has three races. The two
    driver codes are sorted, so NOR vs RUS and RUS vs NOR share one file.
    """
    code = SESSION_CODES.get(session["session_name"], slug(session["session_name"]))
    return NOTES_DIR / f"{session['year']}_{slug(session['location'])}_{code}_{'_'.join(sorted(names))}.md"


def read_notes(path):
    """The notes file's text, or None if you haven't written one."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def create_notes(path):
    """Create an empty notes file from NOTES_TEMPLATE.

    Never overwrites: mode "x" means "create a new file", and fails with
    FileExistsError if the file is already there, so your notes can't be wiped.
    """
    path.parent.mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(NOTES_TEMPLATE)
