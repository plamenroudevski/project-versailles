"""
Tests for data.py (no internet needed): deleted-lap matching and notes files.

Race control's "LAP n" didn't match OpenF1's lap numbers in 2026 Brazil, so
data.py matches the incident's local clock time to the lap being driven.
"""

import numpy as np
import pandas as pd
import pytest

import data


def test_find_session_with_several_matches(monkeypatch):
    # monkeypatch swaps data.openf1 for a stand-in during this test, so no internet is needed.
    sessions = pd.DataFrame({"session_key": [1, 2, 3, 4], "country_name": ["Bahrain", "Bahrain", "United States",
                             "United States"], "location": ["Bahrain", "Bahrain", "Miami Gardens", "Austin"],
                             "circuit_short_name": ["Sakhir", "Sakhir", "Miami", "Austin"]})
    monkeypatch.setattr(data, "openf1", lambda endpoint, **params: sessions)
    assert data.find_session(2026, "Bahrain", "Day 1")["session_key"] == 1  # same place twice: the first one
    with pytest.raises(LookupError, match="Miami Gardens, Austin"):
        data.find_session(2026, "United States", "Day 1")                     # different places: ask for the city


def test_event_with_every_session_cancelled_has_no_sessions(monkeypatch):
    # 2026's April Bahrain GP: all five sessions cancelled. Used to crash comparing an empty date column.
    sessions = pd.DataFrame({"session_key": [1, 2], "session_name": ["Qualifying", "Race"], "is_cancelled": True,
                             "date_start": ["2026-04-11T16:00:00+00:00", "2026-04-12T15:00:00+00:00"]})
    monkeypatch.setattr(data, "openf1", lambda endpoint, **params: sessions)
    assert data.list_sessions(1282).empty


def test_laps_get_their_qualifying_part():
    # 2026 São Paulo: Q1 started at 18:00, was red-flagged and restarted at 19:26; Q2 at 19:49; Q3 at 20:14.
    control = pd.DataFrame({
        "date": ["2026-11-07T17:47:43+00:00", "2026-11-07T18:00:00+00:00", "2026-11-07T19:26:00+00:00",
                 "2026-11-07T19:49:00+00:00", "2026-11-07T20:14:00+00:00"],
        "message": ["YELLOW IN TRACK SECTOR 1"] + ["SESSION STARTED"] * 4,
        "qualifying_phase": [1, 1, 1, 2, 3],
    })
    starts = data.part_starts(control, "Qualifying")
    laps = pd.DataFrame({"date_start": ["2026-11-07T18:01:47+00:00", "2026-11-07T19:30:32+00:00",
                                        "2026-11-07T19:50:00+00:00", "2026-11-07T20:20:00+00:00"]})
    assert data.lap_parts(laps, starts) == ["Q1", "Q1", "Q2", "Q3"]  # the 84-minute red flag stays inside Q1
    assert data.part_starts(control, "Sprint Qualifying").index[0] == "SQ1"
    assert data.part_starts(control.assign(qualifying_phase=None), "Practice 2").empty  # no parts outside qualifying


def test_build_telemetry_distance_starts_at_the_line():
    # A car at a steady 180 km/h (50 m/s): car data every 0.25 s from 0.9 s before the line,
    # position at different moments, moving 500 map units (50 m) per second along x.
    start = pd.Timestamp("2026-11-07T19:00:00+00:00")
    car_s, pos_s = np.arange(-0.9, 10, 0.25), np.arange(-1.0, 10.2, 0.27)

    def stamps(seconds):
        return [(start + pd.to_timedelta(s, unit="s")).isoformat() for s in seconds]

    car = pd.DataFrame({"date": stamps(car_s), "speed": 180, "throttle": 100, "brake": 0, "n_gear": 8, "drs": None})
    pos = pd.DataFrame({"date": stamps(pos_s), "x": 500 * pos_s, "y": 0.0})
    tel = data.build_telemetry(car, pos, start)
    assert np.interp(0.0, tel["Time"], tel["Distance"]) == pytest.approx(0.0)    # the line is at 0 m
    assert np.interp(8.0, tel["Time"], tel["Distance"]) == pytest.approx(400.0)  # 8 s at 50 m/s
    assert tel["X"].to_numpy() == pytest.approx(500 * car_s)                      # position moved onto car-data times


def test_notes_file_name():
    # Accents dropped, short session code, driver codes sorted so either order finds the same notes.
    session = pd.Series({"year": 2026, "location": "São Paulo", "session_name": "Qualifying"})
    assert data.notes_path(session, ("RUS", "NOR")).name == "2026_sao_paulo_Q_NOR_RUS.md"
    assert data.notes_path(session, ("NOR", "RUS")) == data.notes_path(session, ("RUS", "NOR"))
    # The race-pace page uses "pace" in place of a driver pair.
    race = session.copy()
    race["session_name"] = "Race"
    assert data.notes_path(race, ("pace",)).name == "2026_sao_paulo_R_pace.md"


def test_creating_notes_never_overwrites(tmp_path):
    # tmp_path is a fresh empty folder pytest makes for this test, so real notes are never touched.
    path = tmp_path / "notes" / "2026_sao_paulo_Q_NOR_RUS.md"
    data.create_notes(path)
    assert path.read_text(encoding="utf-8") == data.NOTES_TEMPLATE
    path.write_text("NOR lifts early into T1.", encoding="utf-8")
    with pytest.raises(FileExistsError):
        data.create_notes(path)
    assert data.read_notes(path) == "NOR lifts early into T1."


def test_deleted_lap_found_by_incident_time():
    # Real 2026 Brazil messages: Stroll's 1:42.680 is OpenF1 lap 3, though race control calls it "LAP 4".
    laps = pd.DataFrame({
        "lap_number": [3, 4],
        "date_start": ["2026-11-07T19:28:50.153+00:00", "2026-11-07T19:30:32.786+00:00"],
        "lap_duration": [102.68, 98.208],
    })
    control = pd.DataFrame({
        "date": ["2026-11-07T19:31:05+00:00", "2026-11-07T19:31:33+00:00"],
        "message": ["CAR 18 (STR) TIME 1:42.680 DELETED - DOUBLE YELLOW AT TURN 3 LAP 4 16:29:11",
                    "CAR 51 (CAM) LAP DELETED - TRACK LIMITS AT TURN 4 LAP 4 16:29:11 (PIT)"],
    })
    sao_paulo = pd.to_timedelta(-3, unit="h")
    assert data.deleted_laps(control, laps, "STR", sao_paulo) == {3}
    assert data.deleted_laps(control, laps, "NOR", sao_paulo) == set()


def test_reinstated_lap_counts_again():
    # Deleted for track limits, then given back after review (real wording from 2025 Barcelona).
    laps = pd.DataFrame({"lap_number": [7, 8], "date_start": ["2025-06-01T13:20:00+00:00", "2025-06-01T13:21:21+00:00"],
                         "lap_duration": [80.755, 81.2]})
    control = pd.DataFrame({
        "date": ["2025-06-01T13:21:30+00:00", "2025-06-01T13:34:43+00:00"],
        "message": ["CAR 44 (HAM) TIME 1:20.755 DELETED - TRACK LIMITS AT TURN 10 LAP 7 15:21:10",
                    "CAR 44 (HAM) TIME 1:20.755 WILL BE REINSTATED"],
    })
    assert data.deleted_laps(control.iloc[:1], laps, "HAM", pd.to_timedelta(2, unit="h")) == {7}  # before review
    assert data.deleted_laps(control, laps, "HAM", pd.to_timedelta(2, unit="h")) == set()          # after review


def race_control(rows):
    """Race-control messages from (UTC time, category, flag, scope, message) tuples."""
    return pd.DataFrame(rows, columns=["date", "category", "flag", "scope", "message"])


def test_neutralised_periods_from_real_messages():
    # 2026 Mexico City race (real messages, plus one sector "clear" during the last safety car).
    control = race_control([
        ("2026-10-31T17:32:52+00:00", "SessionStatus", None, None, "SESSION STARTED"),  # stray, from the day before
        ("2026-11-01T19:20:00+00:00", "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"),
        ("2026-11-01T20:12:05+00:00", "SafetyCar", None, None, "VSC DEPLOYED"),
        ("2026-11-01T20:14:42+00:00", "SafetyCar", None, None, "VSC ENDING"),
        ("2026-11-01T20:14:53+00:00", "Flag", "CLEAR", "Track", "TRACK CLEAR"),
        ("2026-11-01T21:04:06+00:00", "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),  # never "IN THIS LAP"...
        ("2026-11-01T21:06:02+00:00", "SessionStatus", None, None, "SESSION ABORTED"),   # ...it became a red flag
        ("2026-11-01T21:08:04+00:00", "Flag", "CLEAR", "Track", "TRACK CLEAR"),
        ("2026-11-01T21:45:00+00:00", "SessionStatus", None, None, "SESSION STARTED"),
        ("2026-11-01T22:05:02+00:00", "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
        ("2026-11-01T22:08:00+00:00", "Flag", "CLEAR", "Sector", "CLEAR IN TRACK SECTOR 8"),  # one sector: no restart
        ("2026-11-01T22:11:09+00:00", "SafetyCar", None, None, "SAFETY CAR IN THIS LAP"),     # still out until...
        ("2026-11-01T22:11:50+00:00", "Flag", "CLEAR", "Track", "TRACK CLEAR"),               # ...the restart
        ("2026-11-01T22:23:53+00:00", "Flag", "CHEQUERED", "Track", "CHEQUERED FLAG"),
    ])
    assert [(f"{a:%H:%M:%S}", f"{b:%H:%M:%S}") for a, b in data.neutralised_periods(control)] == [
        ("20:12:05", "20:14:42"), ("21:04:06", "21:08:04"), ("22:05:02", "22:11:50")]

    # 2026 Kuala Lumpur: a VSC that became a safety car is one period. And a safety car
    # still out at the chequered flag ends there.
    control = race_control([
        ("2026-10-04T09:53:01+00:00", "SafetyCar", None, None, "VSC DEPLOYED"),
        ("2026-10-04T09:58:26+00:00", "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
        ("2026-10-04T10:11:34+00:00", "SafetyCar", None, None, "SAFETY CAR IN THIS LAP"),
        ("2026-10-04T10:13:20+00:00", "Flag", "CLEAR", "Track", "TRACK CLEAR"),
        ("2026-10-04T10:16:00+00:00", "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),  # made up: out at the flag
        ("2026-10-04T10:20:15+00:00", "Flag", "CHEQUERED", "Track", "CHEQUERED FLAG"),
    ])
    assert [(f"{a:%H:%M:%S}", f"{b:%H:%M:%S}") for a, b in data.neutralised_periods(control)] == [
        ("09:53:01", "10:13:20"), ("10:16:00", "10:20:15")]


def test_build_race_laps():
    # Two drivers, four 100 s laps from 14:00:00; a VSC from 14:02:30 to 14:02:50, during lap 2.
    def stamps(offset_s):
        return [(pd.Timestamp("2026-11-08T14:00:00+00:00") + pd.to_timedelta(offset_s + 100 * i, unit="s")).isoformat()
                for i in range(4)]

    laps = pd.DataFrame({"driver_number": [4] * 4 + [81] * 4, "lap_number": [1, 2, 3, 4] * 2,
                         "date_start": stamps(0) + stamps(1), "lap_duration": 100.0,
                         "is_pit_out_lap": [False, False, False, True, False, False, True, None],
                         "duration_sector_1": [None] + [30.0] * 7, "duration_sector_2": 40.0,
                         "duration_sector_3": 30.0})
    drivers = pd.DataFrame({"driver_number": [4, 81], "name_acronym": ["NOR", "PIA"]})
    stints = pd.DataFrame({"driver_number": [4, 81, 81], "lap_start": [1, 1, 3], "lap_end": [4, 2, None],
                           "compound": ["MEDIUM", "MEDIUM", "HARD"], "tyre_age_at_start": [3, 0, 0]})
    pits = pd.DataFrame({"driver_number": [81], "lap_number": [2]})  # NOR's stop is missing here
    control = race_control([("2026-11-08T14:02:30+00:00", "SafetyCar", None, None, "VSC DEPLOYED"),
                            ("2026-11-08T14:02:50+00:00", "SafetyCar", None, None, "VSC ENDING")])

    table = data.build_race_laps(laps, drivers, stints, pits, control).set_index(["driver", "lap"])
    nor, pia = table.loc["NOR"], table.loc["PIA"]
    assert nor["tyre_age"].tolist() == [3, 4, 5, 6]                   # a used set: 3 laps old at the start
    assert pia["compound"].tolist() == ["MEDIUM", "MEDIUM", "HARD", "HARD"]
    assert pia["tyre_age"].tolist() == [0, 1, 0, 1]                   # the open stint runs to the last lap
    assert pia["pit_in"].tolist() == [False, True, False, False]      # from the pit endpoint
    assert nor["pit_in"].tolist() == [False, False, True, False]      # the lap before NOR's out-lap
    assert pia["pit_out"].tolist() == [False, False, True, False]     # a missing value counts as False
    assert nor["neutralised"].tolist() == pia["neutralised"].tolist() == [False, True, False, False]
    assert nor["sector_1_s"].isna().tolist() == [True, False, False, False]   # lap 1's first sector isn't timed
    assert pia[["sector_1_s", "sector_2_s", "sector_3_s"]].iloc[0].tolist() == [30.0, 40.0, 30.0]


def test_build_results_statuses():
    drivers = pd.DataFrame({"driver_number": [4, 18, 77, 44], "name_acronym": ["NOR", "STR", "BOT", "HAM"],
                            "team_name": ["McLaren", "Aston Martin", "Cadillac", "Ferrari"],
                            "team_colour": ["F47600", "229971", "AAAAAD", "ED1131"]})
    results = pd.DataFrame({"driver_number": [4, 18, 77], "position": [1.0, None, None], "number_of_laps": [71, 30, 0],
                            "dnf": [False, True, False], "dns": [False, False, True], "dsq": [False, False, False]})
    table = data.build_results(results, drivers).set_index("driver")
    assert table["status"].to_dict() == {"NOR": "Finished", "STR": "DNF", "BOT": "DNS", "HAM": "No result"}
    assert table.loc["STR", "laps"] == 30


def test_deleted_lap_just_before_local_midnight():
    # A night race 8 h behind UTC: the incident is at 23:59:50 local time, but
    # the message is published at 00:00:30 local, the next day.
    laps = pd.DataFrame({"lap_number": [5], "date_start": ["2026-11-21T07:59:00+00:00"], "lap_duration": [100.0]})
    control = pd.DataFrame({
        "date": ["2026-11-21T08:00:30+00:00"],
        "message": ["CAR 1 (NOR) TIME 1:40.000 DELETED - TRACK LIMITS AT TURN 12 LAP 5 23:59:50"],
    })
    assert data.deleted_laps(control, laps, "NOR", pd.to_timedelta(-8, unit="h")) == {5}
