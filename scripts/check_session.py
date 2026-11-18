"""
check_session.py: sanity check to run before trusting any analysis.

Finds a session on OpenF1 (the free public F1 data API this project uses)
and reports, for two drivers' fastest laps, which telemetry channels exist,
how often they are sampled and what values they take. It also reports rain,
laps deleted for track limits, and whether corner positions exist for the
circuit. New regulations can change or empty a channel (DRS in 2026, for
example), so run this at the start of each season.

Why OpenF1 and not FastF1's own loader: F1's live-timing archive, which
FastF1 reads, now requires a paid F1 TV login. OpenF1 serves the same ~4 Hz
car data for free, from 2023 onwards. FastF1 is still used for corner
positions, which come from MultiViewer and need no login.

It is a standalone helper: the app does not use it.

Usage (from the project folder):
    python scripts/check_session.py                                  # 2026 Brazil Qualifying, NOR vs RUS
    python scripts/check_session.py 2025 Monza Qualifying LEC HAM
"""

import sys
import time

import fastf1
import pandas as pd
import requests
from fastf1.mvapi import get_circuit_info

API = "https://api.openf1.org/v1"

# OpenF1's free tier allows 3 requests per second and 30 per minute. Going over
# returns "429 Too Many Requests", so we wait and retry: briefly first (the
# per-second limit), then long enough to clear the per-minute one (~1 min total).
RETRY_WAITS_S = (2, 5, 15, 40)

# Bookkeeping columns OpenF1 adds to every row; they are not measurements.
ID_COLUMNS = {"date", "session_key", "meeting_key", "driver_number"}


def openf1(endpoint, **params):
    """Fetch one OpenF1 endpoint as a table (a pandas DataFrame).

    OpenF1 answers "404 No results found" when a filter matches nothing, so
    that becomes an empty table rather than an error. "429 Too Many Requests"
    is retried after a pause (see RETRY_WAITS_S). Any other failure (network
    down, server error) stops the script with the reason.
    """
    for wait_s in (*RETRY_WAITS_S, None):
        response = requests.get(f"{API}/{endpoint}", params=params, timeout=60)
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
    """
    sessions = openf1("sessions", year=year, session_name=session_name)
    if sessions.empty:
        sys.exit(f"OpenF1 has no '{session_name}' sessions for {year}.")
    names = sessions[["country_name", "location", "circuit_short_name"]].astype(str).agg(" ".join, axis=1)
    matches = sessions[names.str.contains(place, case=False, regex=False)]
    if matches.empty:
        sys.exit(f"No {year} '{session_name}' at '{place}'. Options: {', '.join(sessions['location'])}")
    return matches.iloc[0]


def fastest_lap(session_key, driver_number):
    """Return the driver's fastest timed lap as one row, or None if they set no time."""
    laps = openf1("laps", session_key=session_key, driver_number=driver_number)
    if laps.empty or laps["lap_duration"].isna().all():
        return None
    # shortcut: may pick a lap later deleted for track limits; Phase 1 filters those out.
    return laps.loc[laps["lap_duration"].idxmin()]


def describe_channel(series):
    """Summarise one telemetry channel in a single line.

    Channels with only a few distinct values (gear, brake, DRS) are listed in
    full, because the set of values tells us what the channel means.
    Continuous channels (speed, throttle) just get min/max, which is enough to
    see whether the numbers are physically plausible.
    """
    values = series.dropna().unique()
    if len(values) == 0:
        return "EMPTY: no data"
    if len(values) <= 12:
        return f"values {sorted(values.tolist())}"
    return f"range {series.min()} to {series.max()}"


def report_samples(name, data):
    """Print every measurement channel in `data` plus its sample rate.

    Sample rate matters: at ~4 Hz and 300 km/h the car travels ~20 m between
    samples, which limits how precisely we can locate a braking point.
    """
    if data.empty:
        print(f"  {name}: NO DATA")
        return
    gap_s = pd.to_datetime(data["date"], format="ISO8601").diff().dt.total_seconds().median()
    print(f"  {name}: {len(data)} samples, ~{1 / gap_s:.1f} Hz (one every {gap_s:.2f} s)")
    for column in sorted(set(data.columns) - ID_COLUMNS):
        print(f"    {column:<9} {describe_channel(data[column])}")


def main(year, place, session_name, acronyms):
    session = find_session(year, place, session_name)
    key = int(session["session_key"])
    print(f"Matched: {session['location']}, {session['country_name']} ({session['circuit_short_name']})"
          f" - {session['session_name']}, {session['date_start']}")

    # Rain changes how much a single-lap comparison can tell us, so report it up front.
    weather = openf1("weather", session_key=key)
    if not weather.empty:
        # Median, not min/max: the sensor occasionally drops out and reports 0 C.
        print(f"Weather: rain {'YES' if weather['rainfall'].any() else 'no'}, "
              f"track ~{weather['track_temperature'].median():.0f} C")

    # Race control announces deleted lap times as text messages, e.g. "CAR 63 (RUS) TIME 1:26.745 DELETED ...".
    control = openf1("race_control", session_key=key)
    deleted = control[control["message"].str.contains("DELETED", na=False)] if not control.empty else control

    for acronym in acronyms:
        print(f"\n=== {acronym} ===")
        drivers = openf1("drivers", session_key=key, name_acronym=acronym)
        if drivers.empty:
            print("  Not entered in this session.")
            continue
        driver = drivers.iloc[0]
        number = int(driver["driver_number"])
        print(f"  #{number}, {driver['team_name']} (team colour #{driver['team_colour']})")

        lap = fastest_lap(key, number)
        if lap is None:
            print("  No timed lap in this session.")
            continue
        print(f"  Fastest lap: {lap['lap_duration']:.3f} s (lap {lap['lap_number']})")
        if not deleted.empty:
            for message in deleted.loc[deleted["message"].str.contains(f"({acronym})", regex=False), "message"]:
                print(f"  Deleted lap: {message}")

        # Ask OpenF1 only for samples inside that lap's start and end times.
        start = pd.Timestamp(lap["date_start"])
        end = start + pd.to_timedelta(float(lap["lap_duration"]), unit="s")
        window = {"date>": start.isoformat(), "date<": end.isoformat()}
        report_samples("Car data", openf1("car_data", session_key=key, driver_number=number, **window))
        report_samples("Position", openf1("location", session_key=key, driver_number=number, **window))

    circuit = get_circuit_info(year=year, circuit_key=int(session["circuit_key"]))
    print("\n=== Corners (MultiViewer, via FastF1) ===")
    if circuit is None:
        print("  None available: corner zones can't be built for this circuit.")
    else:
        labels = (circuit.corners["Number"].astype(str) + circuit.corners["Letter"]).tolist()
        print(f"  {len(labels)} corners: {', '.join(labels)}")


if __name__ == "__main__":
    fastf1.set_log_level("WARNING")  # hide FastF1's download chatter
    args = sys.argv[1:] or ["2026", "Brazil", "Qualifying", "NOR", "RUS"]
    main(int(args[0]), args[1], args[2], args[3:])
