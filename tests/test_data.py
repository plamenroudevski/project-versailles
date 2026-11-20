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


def test_deleted_lap_just_before_local_midnight():
    # A night race 8 h behind UTC: the incident is at 23:59:50 local time, but
    # the message is published at 00:00:30 local, the next day.
    laps = pd.DataFrame({"lap_number": [5], "date_start": ["2026-11-21T07:59:00+00:00"], "lap_duration": [100.0]})
    control = pd.DataFrame({
        "date": ["2026-11-21T08:00:30+00:00"],
        "message": ["CAR 1 (NOR) TIME 1:40.000 DELETED - TRACK LIMITS AT TURN 12 LAP 5 23:59:50"],
    })
    assert data.deleted_laps(control, laps, "NOR", pd.to_timedelta(-8, unit="h")) == {5}
