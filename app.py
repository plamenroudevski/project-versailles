"""
app.py: the Streamlit web app. User interface only.

Pipeline position:  data.py -> analysis.py / pace.py / tyres.py -> charts.py -> app.py
This file only arranges widgets and output: every number comes from
analysis.py, pace.py or tyres.py and every chart from charts.py.

Run it from the project folder:
    streamlit run app.py

Three pages, listed at the top of the sidebar:
- Qualifying: sidebar pickers (year -> event -> session -> driver A ->
  driver B, plus an optional lap for each), then the findings, your notes,
  the charts.
- Race pace: sidebar pickers (year -> event; the race is picked
  automatically), then a summary, your notes, and the classification or
  the lap-time distribution.
- Tyre degradation: the same pickers, then summary cards, how the numbers
  are made, your notes, the team chart, every counted lap, the fuel
  sensitivity and the full tables.
- Lap replay: the same pickers as Qualifying, then both laps replayed side
  by side on the track map with synced telemetry (replay.py builds it).
- Driver laps (not in the sidebar): every lap of one driver's race with
  sector times, opened by clicking a driver's name on the Race pace page.

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
import streamlit.components.v1 as components

import analysis
import charts
import data
import pace
import replay
import tyres

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


@st.cache_data(ttl=CACHE_TTL_S, max_entries=20, show_spinner="Building the replay...")
def replay_player(session_key, name_a, name_b, lap_a, lap_b):
    """The finished replay player page (HTML), remembered per session, driver pair and laps.

    It starts from comparison(), the same remembered result the Qualifying
    page uses, so a pair you've already compared costs no downloads.
    """
    run = comparison(session_key, name_a, name_b, lap_a, lap_b)
    session = run["session"]
    rotation = data.get_rotation(int(session["year"]), int(session["circuit_key"]))
    return replay.player_html(replay.build_payload(run, rotation))


@st.cache_data(ttl=CACHE_TTL_S, max_entries=20, show_spinner="Loading every lap of the race...")
def race_pace(session_key):
    """The race's session details, every lap, and pace.analyse() result, remembered per race.

    The finished result is cached, not just the downloads: switching between
    the two views, coming back to a race, or opening a driver's laps costs no
    requests and no maths.
    """
    session = data.get_session(session_key)
    laps = data.race_lap_table(session)
    return {"session": session, "laps": laps, "run": pace.analyse(laps, data.race_results(session)),
            "rain": data.session_had_rain(session_key)}


@st.cache_data(ttl=CACHE_TTL_S, max_entries=20, show_spinner="Loading every lap and fitting tyre degradation...")
def tyre_model(session_key):
    """The race's session details and the finished tyres.analyse() result, remembered per race.

    The fits (bootstraps included) take a second or two, so the result is
    cached, not just the downloads: changing nothing but re-running the page,
    or coming back to a race, costs no requests and no maths. The downloads
    are the same five as the race pace page's, so after that page they come
    from the disk cache.
    """
    session = data.get_session(session_key)
    return {"session": session, "run": tyres.analyse(data.tyre_lap_table(session)),
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


def pick_race():
    """Year and event pickers, then the event's race (picked automatically; sprints aren't covered).

    Returns the race's session_key, or stops the page if the event has no race yet.
    """
    meeting_key = pick_event()
    event_sessions = sessions(meeting_key)
    names = [] if event_sessions.empty else event_sessions["session_name"].tolist()
    if "Race" not in names:
        stop("This event has no race yet: it hasn't started, or it was cancelled. Pick another event.")
    return int(event_sessions["session_key"].iloc[names.index("Race")])


def pick_pair():
    """Sidebar pickers for two laps: event, session, qualifying part, two drivers and a lap each.

    Shared by the Qualifying and Lap replay pages. Opens on the event's
    qualifying and its two fastest drivers' fastest laps; any session and
    any lap (race laps included) can be picked.
    Returns (session_key, name_a, name_b, lap_a, lap_b); a lap is None for "fastest valid lap".
    """
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
    return session_key, name_a, name_b, lap_a, lap_b


# ---- Page 1: Qualifying (compare two laps) --------------------------------------------

def qualifying_page():
    # Sidebar: what to compare
    st.sidebar.header("Compare two laps")
    session_key, name_a, name_b, lap_a, lap_b = pick_pair()

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
        session_key = pick_race()
        run = race_pace(session_key)
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

    # Driver names link to the driver laps page for this race ({} becomes the driver's code).
    driver_url = f"driver?race={session_key}&driver={{}}"
    # required=True: one view is always chosen (clicking the chosen one again doesn't clear it).
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], required=True, label_visibility="collapsed")
    if view == VIEWS[0]:
        # charts.py escapes every piece of text from the data before it reaches this HTML.
        st.html(charts.classification_html(result["table"], driver_url=driver_url))
        st.caption("Hover over a driver (or tab to them) for their clean-lap count and tyre strategy. "
                   "Click a driver's name to open every lap of their race in a new tab.")
    else:
        st.plotly_chart(charts.pace_distribution_figure(result["table"], result["clean"], driver_url=driver_url),
                        theme=None, config={"displaylogo": False})
        st.caption("Each dot is one clean lap: hover for its details. Click a compound in the key to hide or show "
                   "it, or a driver's name to open every lap of their race in a new tab.")


# ---- Page 3 (hidden): every lap of one driver's race -------------------------------------

def driver_laps_page():
    """Every lap of one driver's race, counted or not, with its sector times.

    Opened from a driver's name on the Race pace page, in a new tab, so that
    page keeps its choices. The address says which race and driver
    (driver?race=11412&driver=NOR), so the page stands on its own. Not listed
    in the sidebar.
    """
    try:
        session_key, driver = int(st.query_params["race"]), st.query_params["driver"]
    except (KeyError, ValueError):  # address typed by hand, missing a part or with a non-number race
        stop("Open this page by clicking a driver's name on the Race pace page.")
    try:
        run = race_pace(session_key)  # already remembered from the Race pace page: no new downloads
    except LookupError as error:
        stop(str(error), st.warning)
    except requests.RequestException as error:
        stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)

    laps, table = run["laps"], run["run"]["table"]
    mine = laps[laps["driver"] == driver]
    if mine.empty:  # the driver code comes from the address, so it's only shown once it's known to be real
        stop("No laps for that driver in this race. Open this page by clicking a driver's name on the Race pace page.")
    reasons = pace.why_not_counted(laps).loc[mine.index]  # over the whole field: the 2σ filter needs each driver's laps
    row = table[table["driver"] == driver].iloc[0]

    session = run["session"]
    st.subheader(f"{driver} · {session['year']} {session['location']} race: every lap")
    median = f" (median {analysis.lap_time_text(row['median'])})" if pd.notna(row["median"]) else ""
    st.caption(f"{row['team']} · {pace.finished_text(row)} · {int(reasons.isna().sum())} of {len(mine)} laps "
               f"count towards the race pace{median}.")

    shown = pd.DataFrame({
        "Lap": mine["lap"],
        "Lap time": mine["lap_time_s"].map(lambda t: analysis.lap_time_text(t) if pd.notna(t) else None),
        **{f"Sector {n}": mine[f"sector_{n}_s"] for n in (1, 2, 3)},
        "Counted": reasons.map(lambda reason: "Yes" if reason is None else f"No: {reason}"),
    })
    sector = st.column_config.NumberColumn(format="%.3f")  # seconds, to the thousandth like the timing screens
    st.dataframe(shown, hide_index=True, height=35 * (len(shown) + 1) + 3,  # tall enough to show every lap
                 column_config={f"Sector {n}": sector for n in (1, 2, 3)})
    st.caption(f"Counted: used for the median race pace. {pace.FILTER_RULE}")


# ---- Page 4: Tyre degradation -----------------------------------------------------------

def tyres_page():
    """How much lap time each car lost per lap of tyre age, per compound: an indicator, not a verdict.

    Order: summary cards (the answer first), how the numbers are made, your
    notes, then the team chart (which the cards summarise), every lap with
    each stint's line, the fuel sensitivity, and the full tables.
    """
    st.sidebar.header("Tyre degradation")
    try:
        session_key = pick_race()
        model = tyre_model(session_key)
    except LookupError as error:  # data.py's "missing data" messages are written for you
        stop(str(error), st.warning)
    except requests.RequestException as error:
        stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)
    st.sidebar.caption("Data: OpenF1 (2023 onwards).")

    session, run = model["session"], model["run"]
    st.subheader(f"{session['year']} {session['location']} race: tyre degradation")
    if model["rain"]:
        st.warning(RAIN_WARNING + " Intermediate and wet-tyre laps are excluded; slick laps on a drying track "
                                  "can look like they get faster with age.")
    if not run["traffic"]:
        st.warning("No gap-to-car-ahead data for this race, so laps in traffic weren't removed.")
    if run["counts"]["kept"] == 0:
        stop("No laps were left to fit in this race after the filters below.")

    # Answer-first summary: two cards per row, each a bordered box.
    st.markdown("#### Summary")
    cards = tyres.summary_cards(run)
    for start in range(0, len(cards), 2):
        for column, (heading, text) in zip(st.columns(2), cards[start:start + 2]):
            with column.container(border=True):
                st.markdown(f"**{heading}**  \n{plain(text)}")

    # How the numbers are made: normal text size (not a grey caption), because the caveats matter.
    st.markdown("#### How these numbers are made")
    st.markdown("\n".join(f"- {plain(line)}" for line in tyres.method_text(run)))

    # Your notes for this race, e.g. notes/2026_sao_paulo_R_tyres.md ("tyres" in place of a driver pair).
    show_notes(data.notes_path(session, ("tyres",)))

    plotly_config = {"displaylogo": False}
    st.markdown("#### Degradation by team")
    st.plotly_chart(charts.tyre_teams_figure(run), theme=None, config=plotly_config)
    st.caption("Lowest degradation at the top. A hollow dot: the interval includes zero (not distinguishable "
               "from flat). Grey teams didn't have enough laps. The faint band, where shown, is Method B's "
               "whole-field estimate, which needs no fuel figure.")
    st.markdown("#### Every counted lap")
    st.plotly_chart(charts.tyre_scatter_figure(run), theme=None, config=plotly_config)
    st.caption("One dot per lap; teammates have different shapes. Lines are each stint's fit: solid for a "
               "driver's first stint, dashed for the second, dotted for the third. Click a team in the key to "
               "hide or show it.")
    st.markdown("#### Does the fuel assumption change the ranking?")
    st.plotly_chart(charts.tyre_sensitivity_figure(run), theme=None, config=plotly_config)
    st.caption("Flat lines: the same order at every fuel value. Hover for each team's slope.")

    with st.expander("Tables: every stint, every team, and the laps left out"):
        order = run["stints"]["compound"].map(tyres.SLICKS.index)  # softs, mediums, hards: not alphabetical
        stints = run["stints"].assign(order=order).sort_values(["order", "slope"])
        st.markdown("**Every fitted stint** (fuel-corrected, default fuel value)")
        st.dataframe(pd.DataFrame({
            "Driver": stints["driver"], "Team": stints["team"], "Tyre": stints["compound"].str.title(),
            "Stint": stints["stint"], "Laps": stints["first_lap"].astype(str) + "–" + stints["last_lap"].astype(str),
            "Tyre age": stints["ages"], "Counted": stints["laps"], "s/lap": stints["slope"],
            "Low": stints["low"], "High": stints["high"],
            "Flags": [", ".join(f for f, on in (("flat?", flat), ("negative", neg), ("rebuilt from pit stops", rec))
                                if on) for flat, neg, rec in zip(stints["flat"], stints["negative"],
                                                                    stints["reconstructed"])],
        }), hide_index=True, column_config={c: st.column_config.NumberColumn(format="%+.3f")
                                            for c in ("s/lap", "Low", "High")})
        st.markdown("**Every team** (both drivers pooled)")
        teams = run["teams"]
        st.dataframe(pd.DataFrame({
            "Tyre": teams["compound"].str.title(), "Team": teams["team"], "Counted laps": teams["laps"].astype(int),
            "Stints": teams["stints"], "s/lap": teams["slope"], "Low": teams["low"], "High": teams["high"],
            "Over 10 laps (s)": teams["loss_10"], "Status": teams["status"],
        }), hide_index=True, column_config={c: st.column_config.NumberColumn(format="%+.3f")
                                            for c in ("s/lap", "Low", "High", "Over 10 laps (s)")})
        counts = run["counts"]
        st.markdown(f"**Laps left out** ({counts['laps'] - counts['kept']:,} of {counts['laps']:,}). Stints: "
                    f"{counts['stints_fitted']} fitted, {counts['stints_short']} too short, "
                    f"{counts['stints_no_laps']} with no laps left.")
        st.dataframe(pd.Series(counts["dropped"], name="Laps").rename_axis("Reason").reset_index(), hide_index=True)


# ---- Page 5: Lap replay -----------------------------------------------------------------

def replay_page():
    """Two laps replayed together on the track map, with the telemetry underneath.

    Python builds the player once (replay.py); playing, pausing, scrubbing
    and hovering then run in your browser, inside the embedded player, so
    they never re-run this script.
    """
    st.sidebar.header("Lap replay")
    session_key, name_a, name_b, lap_a, lap_b = pick_pair()
    try:
        page = replay_player(session_key, name_a, name_b, lap_a, lap_b)
    except LookupError as error:  # data.py's "missing data" messages are written for you
        stop(str(error), st.warning)
    except requests.RequestException as error:
        stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)
    # components.html puts the page in its own frame, where its JavaScript runs. It can't size
    # itself to its content, so the height comes from replay.py (matched to the player's layout).
    components.html(page, height=replay.PLAYER_HEIGHT)
    st.caption("Space bar: play or pause. Hover the map or the traces to find the same spot in the other. "
               "Mini-sectors take the faster driver's colour once both cars have passed them. "
               "Gap: positive means the second driver is behind.")


# ---- Navigation -------------------------------------------------------------------------
# Lists the pages at the top of the sidebar and runs the one that's open.
st.navigation([
    st.Page(qualifying_page, title="Qualifying", icon=":material/timer:", default=True),
    st.Page(race_pace_page, title="Race pace", icon=":material/leaderboard:", url_path="race-pace"),
    st.Page(tyres_page, title="Tyre degradation", icon=":material/tire_repair:", url_path="tyres"),
    st.Page(replay_page, title="Lap replay", icon=":material/play_circle:", url_path="replay"),
    st.Page(driver_laps_page, title="Driver laps", icon=":material/table:", url_path="driver", visibility="hidden"),
]).run()
