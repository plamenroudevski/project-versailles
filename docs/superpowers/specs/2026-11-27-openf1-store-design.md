# OpenF1 stored copy — design

Date: 2026-11-27. Comes before Project 2 (owner: "this is critical").

## Intent

Since late 2026, OpenF1 returns `401 Unauthorized` to **every** free request,
past seasons included, while any F1 session is live:

> "Live F1 session in progress. Global API access (including past sessions)
> is restricted to authenticated users until the session ends."

Every Versailles page except Season downloads from OpenF1 on demand, so
during each practice, qualifying and race session the app shows
"Couldn't reach OpenF1 (401 …)". Seen on the live app on 2026-11-27 (Lusail
weekend).

Versailles should keep its own copy of the data people open by default, take
new data from OpenF1 only when there is something new, and keep working while
OpenF1 is locked. Side benefits: faster pages, far fewer requests against
OpenF1's 30-per-minute limit.

## Goals

1. A **stored copy** in the public repo (`store/`) of OpenF1's answers for
   every finished session from 2023 onwards ("A: what people open by
   default", owner's choice):
   - the season's meetings and sessions lists;
   - per session: drivers, laps, race control, weather; for races and
     sprints also stints, pit stops and results;
   - per session: **each driver's fastest valid lap telemetry**
     (`car_data` + `location`, the exact window `data.get_lap_telemetry`
     asks for);
   - MultiViewer corner markers and map rotation per (year, circuit)
     (`data.get_corners`, `data.get_rotation`), which come via FastF1, not
     OpenF1, and would otherwise still fail.
2. `data.openf1()` answers from the store when it has the answer, and goes
   to OpenF1 only when it doesn't (approach A below). Pages and analysis code
   don't change.
3. A **build script** that fills and updates the store, run offline after
   each race weekend (same rhythm as `season.py`: run, check, commit, push).
   Resumable, rate-limit-safe, and pauses cleanly through live-session locks.
4. A **clear message** when something isn't stored and OpenF1 is locked:
   e.g. "OpenF1 is closed to free users while an F1 session is live. This
   lap isn't in Versailles' stored copy yet: try again after the session."
   Replaces the misleading "Try again in a minute" for that case.
5. Tests for the new logic; README + HOW_IT_WORKS sections; the update
   routine documented.

## Non-goals

- **Every lap of every driver** (option B, rejected): only fastest laps
  are stored. Picking another lap, or another qualifying part's fastest lap
  (the Part picker), needs OpenF1 and shows the clear message during a lock.
- Live or in-progress sessions. A session is stored only once it has
  ended (provisionally after 1 hour, settled after 24; see section 2).
- A private store, cloud bucket, or login (owner chose the public repo).
- Automatic/scheduled updates (GitHub Actions). Offline, like `season.py`.
- Paying for an OpenF1 key.
- Seasons before 2023 (OpenF1 has none).

## Anti-goals

- **Stale data that silently sticks:** a session stored while OpenF1 was
  still filling it in (partial laps, missing telemetry) must not be frozen.
- **The app going dark during a lock for something it has stored.** If the
  store has the answer, OpenF1 must not be asked at all.
- **The event list freezing:** a new weekend must appear in the pickers
  without waiting for a store update, whenever OpenF1 is reachable.
- **Repository bloat from rewriting big files:** updates add new files.
  A provisional session file is rewritten exactly once, when it settles;
  settled files only change on a deliberate rebuild (`--force`).

## Constraints

- One choke point: all OpenF1 traffic goes through `data.openf1(endpoint,
  **params)` (11 endpoints: meetings, sessions, drivers, laps, race_control,
  weather, stints, pit, session_result, car_data, location).
  `check_session.py` also calls it.
- Pages call `data` helpers (`list_events`, `list_sessions`,
  `list_drivers`, `get_laps`, `get_lap_telemetry` via `load_lap`,
  `race_lap_table`, `race_results`, `tyre_lap_table`, `session_had_rain`,
  `get_corners`, `get_rotation`, `get_session`, `find_session`). These must
  keep their signatures and behaviour.
- `data.list_events`/`list_sessions` filter by "already started" against the
  current clock (`data.started`): stored lists must go through the same
  filter.
- OpenF1 free limits: 3 req/s, 30 req/min; `openf1()` retries 429s with
  waits (2, 5, 15, 40 s) then raises. During a live session: 401 on
  everything.
- **Sizes, measured 2026-11-27 (Las Vegas Q):** one fastest lap's
  telemetry ≈ 9 KB gzip CSV (373 rows); a qualifying lap table ≈ 14 KB gzip;
  race control ≈ 1.5 KB. Raw OpenF1 JSON carries more fields, so estimate
  ≈ 1.5–2 MB per weekend, ≈ 40–50 MB per season, ≈ 150–200 MB for 2023–26,
  growing ≈ 40–50 MB a year. Fine for GitHub (no file near 50 MB) and
  Streamlit Cloud (clones the repo on deploy).
- **First fill:** ≈ 480 sessions × ≈ 50 requests ≈ 24,000 requests ≈
  13–14 hours at 30/min, and impossible during live sessions. Must be
  resumable and safe to stop at any point. Later updates: ≈ 5 sessions ×
  50 ≈ 10 minutes per weekend.
- Hosted on Streamlit Community Cloud; the existing `requests_cache`
  (10 min, temp folder) stays as the short-term layer for anything not in
  the store.
- Python 3.12 venv; no new dependencies (gzip and json are standard
  library).

## Approaches considered

**A. Store OpenF1's answers as received (chosen by the owner).** The store
sits inside `data.openf1()`: look up `(endpoint, params)` in the store first,
fall back to OpenF1. A build script fills it by running the same `data`
helpers the pages use, recording every answer. Nothing above `openf1()`
changes, and fixes to the maths apply to stored data without a rebuild.
Costs: raw answers are bigger than our tables (~150–200 MB vs ~100 MB);
if a request's parameters ever change in code (e.g. `LAP_PADDING_S`), old
stored answers stop matching and the app falls back to OpenF1 until a
rebuild.

**B. Store our own processed tables** (lap table, race and tyre tables,
results, best-lap telemetry as CSV), checked by each data helper. Smaller and
readable, but ~10 functions each need store logic and tests, and any change
to how a table is built forces a full rebuild.

**C. Commit the requests_cache SQLite file.** Almost no code, but one large
binary that git re-stores in full on every update (~100 MB per update).
Dropped.

## Design

### 1. Store layout

```
store/
  manifest.json.gz              # which file holds which answer (see 3)
  2026/
    meetings.json.gz            # meetings?year=2026 and sessions?year=2026[&session_name=…]
    11416.json.gz               # everything recorded for session_key 11416
    11412.json.gz
    corners_<circuit_key>.json.gz   # MultiViewer corners + rotation for that year
  2025/ …
```

- One gzip JSON file per **session**, holding a dict `{request_key:
  answer_rows}` for every request recorded while building that session
  (sessions/drivers/laps/race_control/weather/stints/pit/session_result and
  each fastest lap's car_data + location).
- `request_key` = endpoint + parameters in a canonical form (sorted, values
  as strings), e.g. `laps?session_key=11416`. Canonical so the same request
  always maps to the same key regardless of argument order.
- Answers are stored as OpenF1 returned them (list of row dicts), so
  `openf1()` can rebuild the identical DataFrame.

### 2. Provisional and settled sessions (owner: same-weekend coverage)

The main lock case is Sunday's race, when people want Saturday's
qualifying. So sessions are stored in two steps:

- **Provisional:** from `PROVISIONAL_AFTER_H` (1 hour, the same wait as
  `season.QUALI_SETTLE_H`) after a session ends, a run stores it marked
  `provisional` in the manifest. The app serves it exactly like a settled one.
- **Settled:** a run at least `STORE_SETTLE_H` (24 hours) after the session
  ended re-fetches every provisional session, rewrites its file once, and
  marks it `settled`. That picks up OpenF1's late fills (deleted laps, stint
  fixes, missing telemetry).
- **Settled files are never rewritten** except with `--force` (deliberate
  rebuild, e.g. after changing request parameters, or a rare late OpenF1
  correction).
- **Incomplete:** a session whose fastest-lap telemetry was missing for some
  drivers is still stored (its tables are useful) and the manifest lists the
  missing pieces; the next run retries only those (and it settles as usual).
- Same-weekend coverage only happens if the owner runs the script (and
  pushes) between sessions on a race weekend, e.g. Saturday evening for
  qualifying. If not, the app simply behaves as "past weekends only".

### 3. Lookup in `data.openf1()`

- Requests are split into two kinds:
  - **Session-scoped** (any request with `session_key`): look up the
    manifest's session file; if the key is there, return it — **never ask
    OpenF1**. If not, ask OpenF1 (current behaviour, incl. requests_cache).
  - **Lists** (`meetings?year=…`, `sessions?year=…`,
    `sessions?meeting_key=…`): **ask OpenF1 first** (so new weekends appear),
    and fall back to the stored copy if OpenF1 fails (401, 429 after retries,
    network). For past seasons, the stored copy can be used directly (a
    finished season's list doesn't change).
- The manifest is small and loaded once per process; session files are
  loaded lazily and kept in memory with a bounded cache (most-recently-used
  sessions), so the app never loads the whole store.
- Telemetry keys include the exact `date>`/`date<` window strings that
  `get_lap_telemetry` builds from the lap's `date_start` and duration; since
  the lap table also comes from the store, the window is reproduced exactly.

### 4. The lock message

- `openf1()` recognises OpenF1's live-session 401 (status 401 and a body
  containing "Live F1 session in progress") and raises a dedicated error,
  e.g. `OpenF1Locked`, subclassing `requests.RequestException` so every
  existing `except requests.RequestException` in `app.py` still catches it.
- The pages show, for that error: "OpenF1 is closed to free users while an
  F1 session is live, and this isn't in Versailles' stored copy yet. Try again
  after the session ends." Other errors keep their current message.

- The existing requests_cache keeps `stale_if_error=True`: on a 401 it may
  hand back an expired copy instead of raising. That is correct data and is
  fine to use; the lock error only has to fire when there is no copy at all.
  The plan must check this interplay with a test (fake 401 with and without a
  cached copy).

### 5. The build script

`python store.py 2026` (and `2023`…`2025` for the first fill;
`--force` rebuilds):

1. Record the year's meetings and sessions lists.
2. For every non-cancelled session that ended ≥ `PROVISIONAL_AFTER_H` ago
   and is not yet stored, or is provisional and now ≥ `STORE_SETTLE_H` old,
   or is incomplete: run the `data` helpers the pages use, in
   record mode (every `openf1()` answer is captured into that session's
   file): `get_session`, `list_drivers` (which reads laps and race control
   per driver), `session_had_rain`, for each driver their fastest valid lap
   via `load_lap` (so `get_driver`'s `drivers?session_key=…&name_acronym=…`
   is recorded too: `load_lap` and the app's `laps()` both call it) and
   `get_lap_telemetry`; for Race and Sprint
   sessions also `race_lap_table`, `race_results`, `tyre_lap_table`.
3. Record corners and rotation for each (year, circuit_key) once.
4. Write the session file and update the manifest **after each session**,
   so stopping at any point loses at most one session's work.
5. On `OpenF1Locked`: stop cleanly with "OpenF1 is locked for a live
   session; run again afterwards" (the next run resumes).
6. Print a summary: sessions stored, incomplete, skipped, total store size.

`season.py` keeps working unchanged: its downloads now come from the store
when present.

### 6. Tests

- Canonical request keys (argument order doesn't matter).
- Session-scoped request found in store → OpenF1 never called (fake HTTP).
- Session-scoped request missing → OpenF1 called.
- List request: OpenF1 answer used when reachable; stored list used when
  OpenF1 raises the lock error.
- Live-session 401 → `OpenF1Locked`, which is a `RequestException`.
- Record mode writes every answer into the right session file; manifest
  updated per session; a rerun skips complete sessions and retries
  incomplete ones; `--force` rewrites.
- Provisional rule: a session that ended 30 min ago isn't stored; one that
  ended 2 h ago is stored as provisional; a provisional one is re-fetched and
  marked settled once 24 h have passed; a settled one isn't touched without
  `--force`.
- Round-trip: a DataFrame from the store equals the one OpenF1 returned.

### 7. Docs

README "Stored data" section (what's stored, what still needs OpenF1, the
update routine including "on a race weekend, run it after qualifying and
push, so the race-day lock doesn't hide qualifying", the size) and HOW_IT_WORKS section 13; the
"Rate limits" limitation updated; `.gitignore` unchanged (`store/` is
committed).

## Decided by the owner

- Fix this before Project 2; critical.
- Scope A: every session's tables + each driver's fastest lap, 2023 onwards.
- Store in the public GitHub repo.
- Approach A: store OpenF1's answers as received, inside `data.openf1()`.
- Same-weekend coverage: sessions are stored provisionally soon after they
  end and replaced once settled.

## My calls (owner can override)

- `PROVISIONAL_AFTER_H = 1`, `STORE_SETTLE_H = 24`.
- Lists: OpenF1 first, store as fallback (so new weekends appear).
- One gzip JSON file per session plus a manifest; corners per circuit.
- Incomplete sessions stored and retried later.
- `OpenF1Locked` error and the page message wording.
- Script name `store.py`, folder `store/`.
- First fill run in the background by Claude, in chunks, after the Lusail
  weekend (OpenF1 blocks everything during sessions).

## Left to the builder

- Exact manifest format and in-memory cache size.
- How record mode is switched on (context manager, module flag, …).
- Whether old requests_cache stays at 10 min (expected: yes).
- How the corner/rotation data is serialised.
- Phasing and commit points (the owner likes a stop for review after the
  first real build).
