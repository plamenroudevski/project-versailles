"""
Tests for data.py (no internet needed): deleted-lap matching and notes files.

Race control's "LAP n" didn't match OpenF1's lap numbers in 2026 Brazil, so
data.py matches the incident's local clock time to the lap being driven.
"""

import pandas as pd
import pytest

import data


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


def test_deleted_lap_just_before_local_midnight():
    # A night race 8 h behind UTC: the incident is at 23:59:50 local time, but
    # the message is published at 00:00:30 local, the next day.
    laps = pd.DataFrame({"lap_number": [5], "date_start": ["2026-11-21T07:59:00+00:00"], "lap_duration": [100.0]})
    control = pd.DataFrame({
        "date": ["2026-11-21T08:00:30+00:00"],
        "message": ["CAR 1 (NOR) TIME 1:40.000 DELETED - TRACK LIMITS AT TURN 12 LAP 5 23:59:50"],
    })
    assert data.deleted_laps(control, laps, "NOR", pd.to_timedelta(-8, unit="h")) == {5}
