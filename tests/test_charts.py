"""
Tests for charts.py's logic: driver-colour choice, the race-pace timing
tower's HTML, and lap-time axis labels.
"""

import pandas as pd

import charts


def test_distinct_team_colours_are_kept():
    # McLaren orange vs Mercedes teal: clearly different and bright on dark.
    assert charts.driver_colours(["F47600", "00D7B6"], ["McLaren", "Mercedes"]) == ("#F47600", "#00D7B6")


def test_teammates_get_the_fallback_pair():
    assert charts.driver_colours(["F47600", "F47600"], ["McLaren", "McLaren"]) == charts.FALLBACK_COLOURS


def test_dark_or_missing_colours_get_the_fallback_pair():
    # A dark navy line vanishes on the dark background; a missing colour can't be drawn.
    assert charts.driver_colours(["1A1F40", "00D7B6"], ["Williams", "Mercedes"]) == charts.FALLBACK_COLOURS
    assert charts.driver_colours([None, "00D7B6"], ["Unknown", "Mercedes"]) == charts.FALLBACK_COLOURS


def tower_table():
    """Two ranked drivers and one retirement, shaped like pace.classify's table."""
    return pd.DataFrame({
        "driver": ["NOR", "VER", "STR"], "team": ["McLaren", "<b>Red Bull</b>", "Aston Martin"],
        "team_colour": ["F47600", "4781D7", "not a colour; background: url(x)"],
        "position": [1.0, 2.0, None], "status": ["Finished", "Finished", "DNF"], "laps": [71, 71, 30],
        "median": [76.102, 76.689, 87.967], "clean_laps": [64, 64, 27],
        "stints": [["INTERMEDIATE", "SOFT"], ["SOFT", None], ["INTERMEDIATE"]],
        "reason": [None, None, "DNF · lap 31"], "rank": [1.0, 2.0, None], "gap": [0.0, 0.587, 11.865],
    })


def test_timing_tower_rows_and_divider():
    page = charts.classification_html(tower_table())
    ranked, unranked = page.split("Not ranked")  # ranked rows above the divider, the retirement below it
    assert "NOR" in ranked and "+0.587 s" in ranked and "P2" in ranked
    assert "STR" in unranked and "DNF · lap 31" in unranked and "+11.865" not in unranked  # no gap shown
    assert "64 clean laps · I → S" in ranked and "S → ?" in ranked                          # tooltip; unknown tyre


def test_timing_tower_escapes_data():
    # Text and colours from the data can't add HTML or CSS to the page.
    page = charts.classification_html(tower_table())
    assert "&lt;b&gt;Red Bull&lt;/b&gt;" in page and "<b>Red Bull" not in page
    assert "url(x)" not in page and f"background: {charts.MUTED}" in page


def test_lap_time_tick_labels():
    ticks, labels = charts.lap_time_ticks(72.3, 91.0)  # a 19 s range: ticks every 5 s
    assert labels == ["1:15.0", "1:20.0", "1:25.0", "1:30.0"]
    ticks, labels = charts.lap_time_ticks(59.6, 61.4)  # across a whole minute: every 0.5 s
    assert labels == ["1:00.0", "1:00.5", "1:01.0"]


def test_driver_names_link_to_their_laps():
    url = "driver?race=11412&driver={}"
    page = charts.classification_html(tower_table(), driver_url=url)
    assert 'href="driver?race=11412&amp;driver=NOR" target="_blank"' in page  # & escaped inside the attribute
    assert 'tabindex' not in page                                            # the links take keyboard focus
    assert "<a " not in charts.classification_html(tower_table())            # no address, no links
    clean = pd.DataFrame({"driver": ["NOR", "VER"], "lap": [2, 2], "lap_time_s": [76.1, 76.7],
                          "compound": ["SOFT", "SOFT"], "tyre_age": [1, 1]})
    ticks = charts.pace_distribution_figure(tower_table(), clean, driver_url=url).layout.xaxis.ticktext
    assert ticks[0] == '<a href="driver?race=11412&amp;driver=NOR" target="_blank">NOR</a>'


def test_tyre_charts_show_every_team_and_grey_out_missing_data():
    import tyres
    from test_tyres import driver_race, fuel_setting  # made-up laps with a known answer (see test_tyres.py)

    # McLaren has enough laps on softs; Ferrari only 10 (< 15): it must still appear, greyed.
    laps = pd.concat([driver_race("NOR", "McLaren", [("SOFT", 20, 0, 0.05)]),
                      driver_race("PIA", "McLaren", [("SOFT", 20, 0, 0.06)]),
                      driver_race("LEC", "Ferrari", [("SOFT", 11, 0, 0.05)])], ignore_index=True)
    laps["team_colour"] = laps["team"].map({"McLaren": "F47600", "Ferrari": "ED1131"})
    run = tyres.analyse(laps, fuel_setting(0.05, 20))

    teams = charts.tyre_teams_figure(run)
    labels = teams.layout.yaxis.ticktext
    assert "McLaren" in labels and any("Ferrari" in label and charts.MUTED in label for label in labels)
    assert any("insufficient data (10 laps)" in str(trace.text) for trace in teams.data)

    scatter = charts.tyre_scatter_figure(run)
    key = [trace.name for trace in scatter.data if trace.showlegend is not False]
    assert key == ["Ferrari", "McLaren"]  # one plain entry per team, alphabetical
    assert {trace.line.dash for trace in scatter.data if trace.mode == "lines"} == {"solid"}  # all first stints

    ranks = charts.tyre_sensitivity_figure(run)
    assert [trace.y for trace in ranks.data] == [(1, 1, 1)]  # only McLaren has a number: rank 1 at every fuel value
