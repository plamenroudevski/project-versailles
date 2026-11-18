"""
app.py: the Streamlit web app. User interface only.

Pipeline position:  data.py -> analysis.py -> charts.py -> app.py
This file only arranges widgets and output: every number comes from
analysis.py and every chart from charts.py.

Run it from the project folder:
    streamlit run app.py

Layout: sidebar pickers (year -> event -> session -> driver A -> driver B,
plus an optional lap for each), then the findings, then the charts.

How Streamlit works, in short: every time you touch a widget, Streamlit
re-runs this whole script from top to bottom. That's simple, but it would
re-download everything each time. So the functions that fetch or compute are
wrapped in `@st.cache_data`. That line is a decorator: it wraps the function
below it with extra behaviour, here "remember the answer for each set of
inputs". A re-run with the same inputs gets the remembered answer instantly.
Under that, data.py's FastF1 disk cache keeps raw downloads for 12 hours.
"""

from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

import analysis
import charts
import data

CACHE_TTL_S = 3600
# Remembered answers expire after an hour, so new sessions and late-arriving
# data appear during a race weekend without restarting the app.

FIRST_YEAR = 2023  # OpenF1 has no data before 2023

st.set_page_config(page_title="Project Versailles", page_icon="🏁", layout="wide")


# ---- Cached data access --------------------------------------------------------
# Only plain values (numbers, text) go in, so Streamlit can tell inputs apart.
# Errors aren't remembered, so a failed download is retried on the next run.

@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def events(year):
    return data.list_events(year)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def sessions(meeting_key):
    return data.list_sessions(meeting_key)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner="Loading drivers...")
def drivers(session_key):
    return data.list_drivers(data.get_session(session_key))


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def laps(session_key, acronym):
    return data.get_laps(data.get_session(session_key), data.get_driver(session_key, acronym))


@st.cache_data(ttl=CACHE_TTL_S, max_entries=20, show_spinner="Loading telemetry and comparing laps...")
def comparison(session_key, name_a, name_b, lap_a, lap_b):
    # max_entries: keep at most 20 comparisons in memory (free hosting has little).
    return analysis.load_and_compare(data.get_session(session_key), name_a, name_b, lap_a, lap_b)


# ---- Small helpers ----------------------------------------------------------------

def stop(message, show=st.info):
    """Show a message in the main area and end this run of the script."""
    show(message)
    st.stop()


def plain(text):
    """Stop Markdown reading '~0.2 s ... ~15 m' as strikethrough (it treats ~text~ that way)."""
    return text.replace("~", "\\~")


def lap_label(lap):
    label = f"Lap {lap.lap_number} · {analysis.lap_time_text(lap.lap_duration)}"
    return f"{label} (deleted)" if lap.deleted else label


def lap_picker(session_key, acronym):
    """Lap choice for one driver: their fastest valid lap unless another is picked."""
    driver_laps = laps(session_key, acronym)
    if driver_laps.empty:
        return None
    labels = {None: "Fastest valid lap"}
    labels.update({int(lap.lap_number): lap_label(lap) for lap in driver_laps.itertuples()})
    # The key ties the choice to this session and driver, so it resets when either changes.
    return st.sidebar.selectbox(f"{acronym} lap", list(labels), format_func=labels.get,
                                key=f"lap-{session_key}-{acronym}")


# ---- Sidebar: what to compare --------------------------------------------------------

st.sidebar.header("Compare two laps")
try:
    year = st.sidebar.selectbox("Year", range(datetime.now(timezone.utc).year, FIRST_YEAR - 1, -1))
    season = events(year)
    if season.empty:
        stop(f"No {year} events have started yet.")
    event_labels = {int(e.meeting_key): f"{pd.Timestamp(e.date_start):%d %b} · {e.meeting_name}"
                    for e in season.itertuples()}
    meeting_key = st.sidebar.selectbox("Event", list(event_labels), format_func=event_labels.get)

    event_sessions = sessions(meeting_key)
    if event_sessions.empty:
        stop("None of this event's sessions have started yet.")
    session_labels = dict(zip(event_sessions["session_key"].astype(int), event_sessions["session_name"]))
    names = list(session_labels.values())
    default = names.index("Qualifying") if "Qualifying" in names else len(names) - 1
    session_key = st.sidebar.selectbox("Session", list(session_labels), index=default, format_func=session_labels.get)

    field = drivers(session_key)
    if field.empty or field["best_lap"].notna().sum() < 2:
        stop("Fewer than two drivers have a valid lap time in this session so far. If it has only just "
             "finished, OpenF1 may still be processing it: try again in a few minutes.")

    # Drivers are listed fastest first, so the default compares the two quickest.
    driver_labels = {d.name_acronym: f"{d.name_acronym} · {d.full_name} · "
                     + (analysis.lap_time_text(d.best_lap) if d.best_lap == d.best_lap else "no time")  # NaN != NaN
                     for d in field.itertuples()}
    acronyms = list(driver_labels)
    name_a = st.sidebar.selectbox("Driver A", acronyms, index=0, format_func=driver_labels.get)
    name_b = st.sidebar.selectbox("Driver B", acronyms, index=1, format_func=driver_labels.get)
    if name_a == name_b:
        stop("Pick two different drivers.")
    lap_a = lap_picker(session_key, name_a)
    lap_b = lap_picker(session_key, name_b)
except requests.RequestException as error:
    stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)
st.sidebar.caption("Data: OpenF1 (2023 onwards). Corner positions: MultiViewer, via FastF1.")


# ---- Main area: findings, then charts ---------------------------------------------------

try:
    run = comparison(session_key, name_a, name_b, lap_a, lap_b)
except LookupError as error:  # data.py's "missing data" messages are written for you
    stop(str(error), st.warning)
except requests.RequestException as error:
    stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)

pair = (name_a, name_b)
result = run["result"]
st.subheader(analysis.title_text(run, pair))
if run["rain"]:
    st.warning("Rain fell during this session. Wet laps vary more from lap to lap, so read the findings "
               "with extra caution.")
if all(lap["telemetry"]["DRS"].isna().all() for lap in run["laps"]):
    st.caption("No DRS or active-aero channel in this data (2026 cars), so straight-line findings can't "
               "separate aero mode from energy deployment.")

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("#### Where the time goes")
    if result["findings"]:
        st.markdown("\n".join(f"{i}. **{f['cause']}** · {plain(f['text'])}"
                              for i, f in enumerate(result["findings"], 1)))
    else:
        st.markdown(f"No zone differs by more than {analysis.MIN_FINDING_TIME_S} s.")
with right:
    st.markdown("#### Driver vs car")
    st.caption("Heuristic: one lap's telemetry can suggest a cause but can't prove it.")
    st.markdown("\n".join(f"- {plain(line)}" for line in result["read"]))

lap_drivers = [lap["driver"] for lap in run["laps"]]
colours = charts.driver_colours([d["team_colour"] for d in lap_drivers], [d["team_name"] for d in lap_drivers])
plotly_config = {"displaylogo": False}
# theme=None keeps charts.py's own styling instead of Streamlit's default chart theme.
st.plotly_chart(charts.telemetry_figure(result, pair, colours), theme=None, config=plotly_config)
st.caption("Drag across any chart to zoom: they all follow. Double-click to zoom back out.")
st.plotly_chart(charts.track_map_figure(result, run["corners"], pair, colours), theme=None, config=plotly_config)

with st.expander("Zone-by-zone numbers"):
    st.caption(f"Gap s: change in the gap across the zone; + means {name_b} lost time. "
               "Brake and Full thr are distances from the start line (m).")
    table = result["table"]
    for kind in ("corner", "straight"):
        # One table per zone type, keeping only the columns that apply to it.
        rows = table[table["Type"] == kind].drop(columns="Type").dropna(axis=1, how="all")
        st.dataframe(rows.round(0).assign(**{"Gap s": rows["Gap s"].round(3)}), hide_index=True)
