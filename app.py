"""
app.py: the Streamlit web app. User interface only.

Pipeline position:  data.py -> analysis.py / pace.py -> charts.py -> app.py
This file only arranges widgets and output: every number comes from
analysis.py or pace.py and every chart from charts.py.

Run it from the project folder:
    streamlit run app.py

Two pages, listed at the top of the sidebar:
- Qualifying: sidebar pickers (year -> event -> session -> driver A ->
  driver B, plus an optional lap for each), then the findings, your notes,
  the charts.
- Race pace: sidebar pickers (year -> event; the race is picked
  automatically), then a summary, your notes, and the classification or
  the lap-time distribution.

The pages use Streamlit's multipage navigation (st.navigation), not a page
selector widget: each page gets its own address (/ and /race-pace) and its
own sidebar, and only the current page's code runs, so its pickers are the
only ones shown. With a selector, every picker would need an if/else around
it. The pages are functions in this file rather than separate files, so they
share the cached data helpers below, and app.py stays the file Streamlit
Cloud runs.

How Streamlit works, in short: every time you touch a widget, Streamlit
re-runs this whole script from top to bottom. That's simple, but it would
re-download everything each time. So the functions that fetch or compute are
wrapped in `@st.cache_data`. That line is a decorator: it wraps the function
below it with extra behaviour, here "remember the answer for each set of
inputs". A re-run with the same inputs gets the remembered answer instantly.
Under that, data.py's disk cache keeps raw downloads for 10 minutes.
"""

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

import analysis
import charts
import data
import pace

CACHE_TTL_S = 600
# Remembered answers expire after 10 minutes (like data.CACHE_EXPIRY_S), so a
# session opened while OpenF1 is still filling it in corrects itself soon after.

FIRST_YEAR = 2023  # OpenF1 has no data before 2023

# Shown by both pages when the weather station reported rain.
RAIN_WARNING = ("Rain fell during this session. Wet laps vary more from lap to lap, so read the findings "
                "with extra caution.")

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
def parts(session_key):
    return data.list_parts(data.get_session(session_key))


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def laps(session_key, acronym):
    return data.get_laps(data.get_session(session_key), data.get_driver(session_key, acronym))


@st.cache_data(ttl=CACHE_TTL_S, max_entries=20, show_spinner="Loading telemetry and comparing laps...")
def comparison(session_key, name_a, name_b, lap_a, lap_b):
    # max_entries: keep at most 20 comparisons in memory (free hosting has little).
    return analysis.load_and_compare(data.get_session(session_key), name_a, name_b, lap_a, lap_b)


@st.cache_data(ttl=CACHE_TTL_S, max_entries=20, show_spinner="Loading every lap of the race...")
def race_pace(session_key):
    """The race's session details and pace.analyse() result, remembered per race.

    The finished result is cached, not just the downloads: switching between
    the two views, or coming back to a race, costs no requests and no maths.
    """
    session = data.get_session(session_key)
    return {"session": session, "run": pace.analyse(data.race_lap_table(session), data.race_results(session)),
            "rain": data.session_had_rain(session_key)}


# ---- Small helpers ----------------------------------------------------------------

def stop(message, show=st.info):
    """Show a message in the main area and end this run of the script."""
    show(message)
    st.stop()


def plain(text):
    """Stop Markdown reading '~0.2 s ... ~15 m' as strikethrough (it treats ~text~ that way)."""
    return text.replace("~", "\\~")


def running_locally():
    """True when the app is open at localhost (your own computer), not on the hosted site."""
    return urlparse(st.context.url or "").hostname in {"localhost", "127.0.0.1", "::1"}


def show_notes(notes_file):
    """Show your notes file under "My notes", or (on your own computer) a button that creates it.

    Read fresh on every run, so after saving a note in Obsidian, refresh the page.
    Used by both pages; data.notes_path decides the file name.
    """
    notes = data.read_notes(notes_file)
    if notes is not None:
        st.markdown("#### My notes")
        with st.container(border=True):
            # Show the file's headings three levels smaller ("#" -> "####"), so they sit
            # under "My notes" instead of outranking it. The file itself is unchanged.
            st.markdown(re.sub(r"(?m)^(#{1,6}) ", lambda m: "#" * min(6, len(m[1]) + 3) + " ", notes))
        st.caption(f"From notes/{notes_file.name}")
    elif running_locally():
        # Only offered on your computer: the hosted app's disk is wiped on every
        # restart, so there your notes come from what you've pushed to GitHub.
        if st.button(f"Create notes file: notes/{notes_file.name}"):
            try:
                data.create_notes(notes_file)
            except FileExistsError:
                pass  # already created (say, from another tab): the re-run below shows it
            st.rerun()


def lap_label(lap):
    label = f"Lap {lap.lap_number} · {analysis.lap_time_text(lap.lap_duration)}"
    if lap.part:
        label = f"{lap.part} · {label}"
    return f"{label} (deleted)" if lap.deleted else label


def lap_picker(session_key, acronym, part):
    """Lap choice for one driver: their fastest valid lap (within `part`, if one is picked) unless another is picked.

    Returns a lap number, or None for the fastest lap of the whole session.
    """
    driver_laps = laps(session_key, acronym)
    if part and not driver_laps.empty:
        driver_laps = driver_laps[driver_laps["part"] == part]
        if driver_laps.empty:
            stop(f"{acronym} didn't set a lap time in {part}.")
    if driver_laps.empty:
        return None
    labels = {None: f"Fastest valid lap in {part}" if part else "Fastest valid lap"}
    labels.update({int(lap.lap_number): lap_label(lap) for lap in driver_laps.itertuples()})
    # The key ties the choice to this session, driver and part, so it resets when any changes.
    choice = st.sidebar.selectbox(f"{acronym} lap", list(labels), format_func=labels.get,
                                  key=f"lap-{session_key}-{acronym}-{part}")
    if choice is None and part:
        fastest = data.fastest_lap(driver_laps)
        if fastest is None:
            stop(f"All of {acronym}'s lap times in {part} were deleted.")
        return int(fastest["lap_number"])
    return choice


# ---- Sidebar pickers shared by both pages ----------------------------------------------

def pick_event():
    """Year and event pickers in the sidebar. Returns the chosen event's meeting_key."""
    years = list(range(datetime.now(timezone.utc).year, FIRST_YEAR - 1, -1))
    # Until the new season's first event, open on last season instead of an empty page.
    year = st.sidebar.selectbox("Year", years, index=1 if events(years[0]).empty and len(years) > 1 else 0)
    season = events(year)
    if season.empty:
        stop(f"No {year} events have started yet.")
    event_labels = {int(e.meeting_key): f"{pd.Timestamp(e.date_start):%d %b} · {e.meeting_name}"
                    for e in season.itertuples()}
    return st.sidebar.selectbox("Event", list(event_labels), format_func=event_labels.get)


# ---- Page 1: Qualifying (compare two laps) --------------------------------------------

def qualifying_page():
    # Sidebar: what to compare
    st.sidebar.header("Compare two laps")
    try:
        meeting_key = pick_event()

        event_sessions = sessions(meeting_key)
        if event_sessions.empty:
            stop("None of this event's sessions have started yet.")
        session_labels = dict(zip(event_sessions["session_key"].astype(int), event_sessions["session_name"]))
        names = list(session_labels.values())
        default = names.index("Qualifying") if "Qualifying" in names else len(names) - 1
        session_key = st.sidebar.selectbox("Session", list(session_labels), index=default, format_func=session_labels.get)
        # Qualifying only: compare laps from the same part (Q1, Q2 or Q3), as the track changes between them.
        session_parts = parts(session_key)
        part = None
        if session_parts:
            part = st.sidebar.selectbox("Part", [None, *session_parts], format_func=lambda p: p or "Whole session")

        field = drivers(session_key)
        if field.empty or field["best_lap"].notna().sum() < 2:
            stop("Fewer than two drivers have a valid lap time in this session so far. If it has only just "
                 "finished, OpenF1 may still be processing it: try again in about 10 minutes.")

        # Drivers are listed fastest first, so the default compares the two quickest.
        driver_labels = {d.name_acronym: f"{d.name_acronym} · {d.full_name} · "
                         + (analysis.lap_time_text(d.best_lap) if d.best_lap == d.best_lap else "no time")  # NaN != NaN
                         for d in field.itertuples()}
        acronyms = list(driver_labels)
        name_a = st.sidebar.selectbox("Driver A", acronyms, index=0, format_func=driver_labels.get)
        name_b = st.sidebar.selectbox("Driver B", acronyms, index=1, format_func=driver_labels.get)
        if name_a == name_b:
            stop("Pick two different drivers.")
        lap_a = lap_picker(session_key, name_a, part)
        lap_b = lap_picker(session_key, name_b, part)
    except requests.RequestException as error:
        stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)
    st.sidebar.caption("Data: OpenF1 (2023 onwards). Corner positions: MultiViewer, via FastF1.")


    # Main area: findings, then charts
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
        st.warning(RAIN_WARNING)
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

    # Time gained and lost by type of zone, like AWS F1 Insights' car performance breakdown.
    st.markdown("#### By zone type")
    rows = [f"| {row['Type']} | "
            + (f"{row['Faster']} by {abs(row['Gap s']):.2f} s" if row["Faster"] != "-" else "level")
            + f" | {row['Zones'] or '-'} |" for _, row in result["breakdown"].iterrows()]
    st.markdown("| Zone type | Faster | Zones |\n|---|---|---|\n" + "\n".join(rows))
    st.caption(f"Corners are grouped by minimum speed: low-speed under {analysis.LOW_SPEED_MAX_KMH} km/h, "
               f"high-speed from {analysis.HIGH_SPEED_MIN_KMH} km/h, medium in between. "
               "The four rows add up to the lap-time gap.")

    # Your own notes: a plain Markdown file per session and driver pair (see data.notes_path).
    show_notes(data.notes_path(run["session"], pair))

    lap_drivers =[lap["driver"] for lap in run["laps"]]
    colours = charts.driver_colours([d["team_colour"] for d in lap_drivers], [d["team_name"] for d in lap_drivers])
    plotly_config = {"displaylogo": False}
    # theme=None keeps charts.py's own styling instead of Streamlit's default chart theme.
    st.plotly_chart(charts.telemetry_figure(result, pair, colours), theme=None, config=plotly_config)
    st.caption("Drag across any chart to zoom: they all follow. Double-click to zoom back out.")
    st.plotly_chart(charts.track_map_figure(result, run["corners"], pair, colours), theme=None, config=plotly_config)
    st.caption("Map labels: who gained in each corner section and straight, and by how much (seconds).")

    with st.expander("Zone-by-zone numbers"):
        st.caption(f"Gap s: change in the gap across the zone; + means {name_b} lost time. "
                   "Brake and Full thr are distances from the start line (m).")
        table = result["table"]
        for kind in ("corner", "straight"):
            # One table per zone type, keeping only the columns that apply to it.
            rows = table[table["Type"] == kind].drop(columns="Type").dropna(axis=1, how="all")
            st.dataframe(rows.round(0).assign(**{"Gap s": rows["Gap s"].round(3)}), hide_index=True)


# ---- Page 2: Race pace ------------------------------------------------------------------

VIEWS = ("Classification", "Lap time distribution")


def race_pace_page():
    # Sidebar: year and event only. The race is picked automatically (sprints aren't covered).
    st.sidebar.header("Race pace")
    try:
        meeting_key = pick_event()
        event_sessions = sessions(meeting_key)
        names = [] if event_sessions.empty else event_sessions["session_name"].tolist()
        if "Race" not in names:
            stop("This event has no race yet: it hasn't started, or it was cancelled. Pick another event.")
        run = race_pace(int(event_sessions["session_key"].iloc[names.index("Race")]))
    except LookupError as error:  # data.py's "missing data" messages are written for you
        stop(str(error), st.warning)
    except requests.RequestException as error:
        stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)
    st.sidebar.caption("Data: OpenF1 (2023 onwards).")

    # Main area: summary, your notes, then the classification (default) or every clean lap's time.
    session, result = run["session"], run["run"]
    st.subheader(f"{session['year']} {session['location']} race: median race pace")
    if run["rain"]:
        st.warning(RAIN_WARNING)

    st.markdown("#### Summary")
    st.caption("Pace rank ignores strategy, traffic and damage, so treat these as pointers, not verdicts.")
    lines = pace.summary_text(result["summary"])
    st.markdown("\n".join(f"- {line}" for line in lines) if lines else "Too few ranked drivers to compare.")
    st.caption(pace.FILTER_RULE)

    # Your notes for this race, e.g. notes/2026_sao_paulo_R_pace.md ("pace" in place of a driver pair).
    show_notes(data.notes_path(session, ("pace",)))

    # required=True: one view is always chosen (clicking the chosen one again doesn't clear it).
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], required=True, label_visibility="collapsed")
    if view == VIEWS[0]:
        # charts.py escapes every piece of text from the data before it reaches this HTML.
        st.html(charts.classification_html(result["table"]))
        st.caption("Hover over a driver (or tab to them) for their clean-lap count and tyre strategy.")
    else:
        st.plotly_chart(charts.pace_distribution_figure(result["table"], result["clean"]), theme=None,
                        config={"displaylogo": False})
        st.caption("Each dot is one clean lap: hover for its details. Click a compound in the key to hide or show it.")


# ---- Navigation -------------------------------------------------------------------------
# Lists the pages at the top of the sidebar and runs the one that's open.
st.navigation([
    st.Page(qualifying_page, title="Qualifying", icon=":material/timer:", default=True),
    st.Page(race_pace_page, title="Race pace", icon=":material/leaderboard:", url_path="race-pace"),
]).run()
