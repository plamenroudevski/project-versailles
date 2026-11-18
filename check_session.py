"""
check_session.py: sanity check to run before trusting any analysis.

Reports, for two drivers' fastest valid laps, which raw telemetry channels
OpenF1 provides, how often they are sampled and what values they take. Also
reports rain, deleted laps and whether corner positions exist. New
regulations can change or empty a channel (DRS in 2026, for example), so run
this at the start of each season.

It uses data.py for fetching, but looks at the raw OpenF1 channels rather
than data.py's tidied telemetry, so nothing is hidden. The app doesn't use it.

Usage (from the project folder):
    python check_session.py                                  # 2026 Brazil Qualifying, NOR vs RUS
    python check_session.py 2025 Monza Qualifying LEC HAM
"""

import sys

import fastf1
import pandas as pd

import data

# Bookkeeping columns OpenF1 adds to every row; they are not measurements.
ID_COLUMNS = {"date", "session_key", "meeting_key", "driver_number"}


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


def report_samples(name, raw):
    """Print every measurement channel in a raw OpenF1 table plus its sample rate.

    Sample rate matters: at ~4 Hz and 300 km/h the car travels ~20 m between
    samples, which limits how precisely we can locate a braking point.
    """
    if raw.empty:
        print(f"  {name}: NO DATA")
        return
    gap_s = pd.to_datetime(raw["date"], format="ISO8601").diff().dt.total_seconds().median()
    print(f"  {name}: {len(raw)} samples, ~{1 / gap_s:.1f} Hz (one every {gap_s:.2f} s)")
    for column in sorted(set(raw.columns) - ID_COLUMNS):
        print(f"    {column:<9} {describe_channel(raw[column])}")


def main(year, place, session_name, acronyms):
    session = data.find_session(year, place, session_name)
    key = int(session["session_key"])
    print(f"Matched: {session['location']}, {session['country_name']} ({session['circuit_short_name']})"
          f" - {session['session_name']}, {session['date_start']}")
    print(f"Rain during session: {'YES' if data.session_had_rain(key) else 'no'}")

    for acronym in acronyms:
        print(f"\n=== {acronym} ===")
        try:
            driver = data.get_driver(key, acronym)
        except LookupError as error:
            print(f"  {error}")
            continue
        number = int(driver["driver_number"])
        print(f"  #{number}, {driver['team_name']} (team colour #{driver['team_colour']})")

        laps = data.get_laps(session, driver)
        lap = None if laps.empty else data.fastest_lap(laps)
        if lap is None:
            print("  No valid timed lap in this session.")
            continue
        deleted = laps.loc[laps["deleted"], "lap_number"].tolist()
        print(f"  Fastest valid lap: {lap['lap_duration']:.3f} s (lap {lap['lap_number']}); deleted laps: {deleted or 'none'}")

        # Ask OpenF1 only for samples inside that lap's start and end times.
        start = pd.Timestamp(lap["date_start"])
        end = start + pd.to_timedelta(float(lap["lap_duration"]), unit="s")
        window = {"date>": start.isoformat(), "date<": end.isoformat()}
        report_samples("Car data", data.openf1("car_data", session_key=key, driver_number=number, **window))
        report_samples("Position", data.openf1("location", session_key=key, driver_number=number, **window))

    print("\n=== Corners (MultiViewer, via FastF1) ===")
    try:
        corners = data.get_corners(year, int(session["circuit_key"]))
        labels = (corners["Number"].astype(str) + corners["Letter"]).tolist()
        print(f"  {len(labels)} corners: {', '.join(labels)}")
    except LookupError as error:
        print(f"  {error}")


if __name__ == "__main__":
    fastf1.set_log_level("WARNING")  # hide FastF1's download chatter
    args = sys.argv[1:] or ["2026", "Brazil", "Qualifying", "NOR", "RUS"]
    try:
        main(int(args[0]), args[1], args[2], args[3:])
    except LookupError as error:
        sys.exit(str(error))
