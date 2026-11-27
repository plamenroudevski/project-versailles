# OpenF1 Stored Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a committed copy of OpenF1's answers for every finished session (tables + each driver's fastest lap) so Versailles works while OpenF1 locks free users during live sessions.

**Architecture:** New `store.py` owns the on-disk store (manifest + one gzip JSON per session + per-year lists + per-circuit corner files) and a recording switch. `data.openf1()` looks session-scoped requests up in the store before going live, asks OpenF1 first for list requests and falls back to the store, records every live answer while recording is on, and turns OpenF1's live-session 401 into `OpenF1Locked`. `store.py`'s build section runs the same `data` helpers the pages use with recording on, and writes the files.

**Tech Stack:** Python 3.12, pandas, requests / requests_cache 1.3.3, FastF1 3.9.2 (MultiViewer), pytest. Standard-library `gzip`, `json`, `contextlib`, `functools`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-27-openf1-store-design.md`

## Global Constraints

- Python: `C:\Users\proud\.venvs\project-versailles\Scripts\python.exe` (Bash: `/c/Users/proud/.venvs/project-versailles/Scripts/python.exe`); tests `<python> -m pytest -q` from the project folder.
- No new dependencies. Store files are gzip JSON; `store/` is committed (public repo, owner's choice).
- Scope A: every session's tables + each driver's fastest valid lap telemetry, 2023 onwards. Other laps and the qualifying Part picker's laps are not stored.
- `PROVISIONAL_AFTER_H = 1`, `STORE_SETTLE_H = 24` (named CONFIG constants with why-comments, house style).
- Session-scoped requests (any request with `session_key`) found in the store never touch OpenF1. List requests (`meetings?year=…`, `sessions?meeting_key=…`, `sessions?year=…`) ask OpenF1 first and fall back to the store on any `requests.RequestException`.
- Lock message, exact copy: `OpenF1 is closed to free users while an F1 session is live, and this isn't in Versailles' stored copy yet. Try again after the session ends.`
- `data` helpers keep their signatures and behaviour; pages and analysis code don't change except the error message in `app.py`.
- Writes are atomic (write `*.tmp`, then `os.replace`), and the manifest is saved after every session.
- Commits: `git commit -F <file>`, message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `openf1-store`. Never push without the owner's yes.
- **OpenF1 is locked during live sessions** (Lusail weekend 2026-11-27..29): Tasks 1-3 use fakes only. Task 4 (first real build) waits until `curl -s -o /dev/null -w "%{http_code}" "https://api.openf1.org/v1/meetings?year=2026"` returns 200.
- **Stop point:** after Task 4, stop and review the first real build with the owner before the long backfill (Task 7).

## Review Focus

1. **A stored session that doesn't hold the request** (another lap, a Part picker lap): falls through to OpenF1; during a lock the page shows the lock message, not a crash. Test in Task 2.
2. **Manifest points at a missing or corrupt file** (bad checkout, half-pulled repo): treated as not stored, falls through to OpenF1, no crash. Test in Task 1.
3. **A session without `date_end`** (cancelled, or OpenF1 hasn't filled it): the build skips it without crashing. Test in Task 3.
4. **The builder writing while the app reads** (both running locally): readers never see a half-written file; no `*.tmp` left behind after a write. Test in Task 1.
5. **A list request during a lock for a year not in the store** (e.g. 2027 before its first build): raises `OpenF1Locked`, so the page shows the lock message. Test in Task 2.

---

### Task 1: The store module

**Files:**
- Create: `store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces (all in `store.py`):
  - `STORE_DIR = Path(__file__).parent / "store"`
  - `request_key(endpoint: str, params: dict) -> str` — `"laps?driver_number=1&session_key=11416"`: params sorted by name, values via `str()`, joined with `&`; no `?` suffix when params is empty.
  - `class Store(root: Path = STORE_DIR)`:
    - `session_rows(session_key: int, key: str) -> list[dict] | None` — the stored answer, or None (not stored, key missing, or file missing/corrupt).
    - `list_rows(key: str) -> list[dict] | None`
    - `circuit(year: int, circuit_key: int) -> dict | None` — `{"corners": [row dicts with X, Y, Number, Letter, Angle], "rotation": float}`
    - `session_entry(session_key: int) -> dict | None` — manifest entry `{"file": str, "status": "provisional"|"settled", "missing": [acronyms], "ended": iso str}`
    - `save_session(session_key: int, year: int, answers: dict[str, list[dict]], status: str, missing: list[str], ended: str) -> None` — writes `<root>/<year>/<session_key>.json.gz`, updates and saves the manifest.
    - `save_lists(year: int, answers: dict[str, list[dict]]) -> None` — merges into `<root>/<year>/lists.json.gz`, manifest `lists` maps each key to that file.
    - `save_circuit(year: int, circuit_key: int, info: dict) -> None` — `<root>/<year>/circuit_<circuit_key>.json.gz`.
  - Manifest: `<root>/manifest.json.gz` = `{"sessions": {str(session_key): entry}, "lists": {key: relative file}, "circuits": {f"{year}_{circuit_key}": relative file}}`. Loaded lazily once per `Store`; session files loaded lazily and kept in an LRU of `SESSION_CACHE_SIZE = 32` files.
  - Recording: `recording() -> contextmanager` yielding a dict that collects `{key: rows}`; `remember(key: str, rows: list[dict]) -> None` (no-op unless recording); `is_recording() -> bool`.

- [ ] **Step 1: Write the failing tests** (all on `tmp_path`)

```python
def test_request_key_is_canonical():
    assert store.request_key("laps", {"session_key": 11416, "driver_number": 1}) == \
        store.request_key("laps", {"driver_number": 1, "session_key": 11416}) == "laps?driver_number=1&session_key=11416"
    assert store.request_key("meetings", {}) == "meetings"

def test_session_round_trip(tmp_path):
    s = store.Store(tmp_path)
    s.save_session(11416, 2026, {"laps?session_key=11416": [{"lap_number": 1}]}, "provisional", [], "2026-11-21T06:00:00+00:00")
    fresh = store.Store(tmp_path)                    # a new process reads it back from disk
    assert fresh.session_rows(11416, "laps?session_key=11416") == [{"lap_number": 1}]
    assert fresh.session_rows(11416, "pit?session_key=11416") is None
    assert fresh.session_rows(99999, "laps?session_key=99999") is None
    assert fresh.session_entry(11416)["status"] == "provisional"

def test_missing_or_corrupt_file_counts_as_not_stored(tmp_path):
    s = store.Store(tmp_path)
    s.save_session(1, 2026, {"laps?session_key=1": []}, "settled", [], "2026-01-01T00:00:00+00:00")
    (tmp_path / "2026" / "1.json.gz").write_bytes(b"not gzip")
    assert store.Store(tmp_path).session_rows(1, "laps?session_key=1") is None
    (tmp_path / "2026" / "1.json.gz").unlink()
    assert store.Store(tmp_path).session_rows(1, "laps?session_key=1") is None

def test_lists_and_circuits_round_trip(tmp_path):
    s = store.Store(tmp_path)
    s.save_lists(2026, {"meetings?year=2026": [{"meeting_key": 1}]})
    s.save_lists(2026, {"sessions?meeting_key=1": [{"session_key": 10}]})   # merges, doesn't replace
    s.save_circuit(2026, 63, {"corners": [{"X": 1.0, "Y": 2.0, "Number": 1, "Letter": "", "Angle": 0.0}], "rotation": 90.0})
    fresh = store.Store(tmp_path)
    assert fresh.list_rows("meetings?year=2026") == [{"meeting_key": 1}]
    assert fresh.list_rows("sessions?meeting_key=1") == [{"session_key": 10}]
    assert fresh.circuit(2026, 63)["rotation"] == 90.0 and fresh.circuit(2026, 1) is None

def test_writes_leave_no_temp_files(tmp_path):
    s = store.Store(tmp_path)
    s.save_session(1, 2026, {"laps?session_key=1": []}, "settled", [], "2026-01-01T00:00:00+00:00")
    s.save_lists(2026, {"meetings?year=2026": []})
    assert not list(tmp_path.rglob("*.tmp"))

def test_recording_collects_only_while_on():
    store.remember("laps?session_key=1", [{"a": 1}])
    with store.recording() as answers:
        assert store.is_recording()
        store.remember("laps?session_key=2", [{"b": 2}])
    assert answers == {"laps?session_key=2": [{"b": 2}]} and not store.is_recording()
```

- [ ] **Step 2: Run to verify they fail** — `<python> -m pytest tests/test_store.py -q`; expected `ModuleNotFoundError: No module named 'store'`.
- [ ] **Step 3: Implement `store.py`** (module docstring in the house style: what it's for, pipeline position `OpenF1 -> store.py <- data.py`, the run command added in Task 3). No imports of `data` at module level.
- [ ] **Step 4: Run** `<python> -m pytest -q` — all pass (114 existing + 6).
- [ ] **Step 5: Commit** — `store.py tests/test_store.py`, "Store: on-disk copy of OpenF1 answers".

---

### Task 2: `data.openf1()` uses the store; the lock error and message

**Files:**
- Modify: `data.py` (`openf1` ~line 77, `get_corners`/`get_rotation` ~line 373, imports)
- Modify: `app.py` (the six `except requests.RequestException` handlers at lines ~299, 317, 387, 440, 483, 635)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: Task 1's `store.request_key`, `Store`, `remember`, `is_recording`.
- Produces:
  - `data.STORE = store.Store()` (module level; tests monkeypatch it with `store.Store(tmp_path)`).
  - `class data.OpenF1Locked(requests.RequestException)` whose `str()` is the exact lock message (Global Constraints).
  - `data.openf1(endpoint, **params)`, same signature and return:
    1. `key = store.request_key(endpoint, params)`.
    2. If `session_key` in params and not recording: stored rows → `pd.DataFrame(rows)`, no HTTP.
    3. Otherwise ask OpenF1 as today. A 401 whose body contains `"Live F1 session in progress"` raises `OpenF1Locked`.
    4. List requests (no `session_key`): on any `requests.RequestException` (including `OpenF1Locked`), return `pd.DataFrame(STORE.list_rows(key))` if stored, else re-raise.
    5. Every live answer is passed to `store.remember(key, rows)` (rows = the JSON list; `[]` for a 404).
  - `data.circuit_info(year, circuit_key) -> dict | None` — stored `{"corners", "rotation"}` unless recording; else from `get_circuit_info`, converted to that dict and `remember`ed under key `f"circuit?circuit_key={circuit_key}&year={year}"`. `get_corners`/`get_rotation` use it and keep their behaviour (LookupError / 0.0).
  - `app.download_failed(error) -> None` — `stop(str(error), st.warning)` for `OpenF1Locked`, else the current `stop(f"Couldn't reach OpenF1 ({error}). Try again in a minute.", st.error)`. All six handlers call it.

- [ ] **Step 1: Write the failing tests** (a `FakeHTTP` in the test file: `.get(url, params, timeout)` returns an object with `status_code`, `text`, `json()`, `raise_for_status()`; counts calls; monkeypatch `data.HTTP` and `data.STORE`)

```python
def test_stored_session_answer_never_calls_openf1(monkeypatch, tmp_path):
    s = store.Store(tmp_path); s.save_session(5, 2026, {"laps?session_key=5": [{"lap_number": 3}]}, "settled", [], "2026-01-01T00:00:00+00:00")
    http = FakeHTTP(status=500); monkeypatch.setattr(data, "HTTP", http); monkeypatch.setattr(data, "STORE", s)
    assert data.openf1("laps", session_key=5)["lap_number"].tolist() == [3] and http.calls == 0

def test_unstored_session_answer_goes_live(monkeypatch, tmp_path):
    http = FakeHTTP(rows=[{"lap_number": 7}]); monkeypatch.setattr(data, "HTTP", http); monkeypatch.setattr(data, "STORE", store.Store(tmp_path))
    assert data.openf1("laps", session_key=5)["lap_number"].tolist() == [7] and http.calls == 1

def test_live_session_401_raises_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "HTTP", FakeHTTP(status=401, text='{"detail":"Live F1 session in progress. Global API access ..."}'))
    monkeypatch.setattr(data, "STORE", store.Store(tmp_path))
    with pytest.raises(data.OpenF1Locked) as error:
        data.openf1("laps", session_key=5)
    assert isinstance(error.value, requests.RequestException)
    assert str(error.value).startswith("OpenF1 is closed to free users while an F1 session is live")

def test_list_uses_openf1_first_then_store_when_locked(monkeypatch, tmp_path):
    s = store.Store(tmp_path); s.save_lists(2026, {"meetings?year=2026": [{"meeting_key": 1}]}); monkeypatch.setattr(data, "STORE", s)
    monkeypatch.setattr(data, "HTTP", FakeHTTP(rows=[{"meeting_key": 1}, {"meeting_key": 2}]))
    assert len(data.openf1("meetings", year=2026)) == 2                       # new weekend visible
    monkeypatch.setattr(data, "HTTP", FakeHTTP(status=401, text="Live F1 session in progress"))
    assert data.openf1("meetings", year=2026)["meeting_key"].tolist() == [1]  # stored copy during a lock

def test_list_not_stored_during_lock_raises_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "STORE", store.Store(tmp_path))
    monkeypatch.setattr(data, "HTTP", FakeHTTP(status=401, text="Live F1 session in progress"))
    with pytest.raises(data.OpenF1Locked):
        data.openf1("meetings", year=2027)

def test_recording_bypasses_store_and_remembers(monkeypatch, tmp_path):
    s = store.Store(tmp_path); s.save_session(5, 2026, {"laps?session_key=5": [{"lap_number": 3}]}, "provisional", [], "2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(data, "STORE", s); monkeypatch.setattr(data, "HTTP", FakeHTTP(rows=[{"lap_number": 9}]))
    with store.recording() as answers:
        assert data.openf1("laps", session_key=5)["lap_number"].tolist() == [9]
    assert answers == {"laps?session_key=5": [{"lap_number": 9}]}

def test_stale_copy_beats_lock(...):
    # Real requests_cache.CachedSession(backend="memory", stale_if_error=True) with a fake transport
    # adapter mounted for https://: first call 200 caches; make the cached copy expired; second call
    # 401 "Live F1 session in progress" -> openf1 returns the stale rows (no OpenF1Locked).
    # Without a cached copy the same 401 -> OpenF1Locked.

def test_stored_circuit_used_for_corners_and_rotation(monkeypatch, tmp_path):
    s = store.Store(tmp_path); s.save_circuit(2026, 63, {"corners": [{"X": 1.0, "Y": 2.0, "Number": 4, "Letter": "", "Angle": 0.0}], "rotation": 90.0})
    monkeypatch.setattr(data, "STORE", s)
    monkeypatch.setattr(data, "get_circuit_info", lambda **kw: pytest.fail("MultiViewer called"))
    assert data.get_corners(2026, 63)["Number"].tolist() == [4] and data.get_rotation(2026, 63) == 90.0
```

- [ ] **Step 2: Run to verify they fail** — `<python> -m pytest tests/test_data.py -q`.
- [ ] **Step 3: Implement** the `data.py` changes. Extend the `openf1` docstring with the store rules; update the `HTTP` comment to say the 10-minute cache stays as the short-term layer for anything not stored.
- [ ] **Step 4: Add `download_failed` to `app.py`** and replace the six handlers' bodies with `download_failed(error)`.
- [ ] **Step 5: Run** `<python> -m pytest -q` — all pass.
- [ ] **Step 6: Commit** — `data.py app.py tests/test_data.py`, "OpenF1: answer from the stored copy first; clear message during live-session locks".

---

### Task 3: The build script

**Files:**
- Modify: `store.py` (download section + `if __name__ == "__main__"`)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: Tasks 1-2; `data.list_events`, `data.list_sessions`, `data.get_session`, `data.list_drivers`, `data.session_had_rain`, `data.load_lap`, `data.race_lap_table`, `data.race_results`, `data.tyre_lap_table`, `data.get_corners`, `data.get_rotation`, `data.OpenF1Locked`. Import `data` inside the functions (avoids the `data` ↔ `store` import cycle).
- Produces:
  - `PROVISIONAL_AFTER_H = 1`, `STORE_SETTLE_H = 24`.
  - `session_action(entry: dict | None, ended: pd.Timestamp | None, now: pd.Timestamp, force: bool = False) -> str | None` — returns `None` (skip), `"provisional"` or `"settled"` (store with that status). Rules: no `ended` → None; age < 1 h → None; `force` → status by age; no entry → status by age (`"settled"` if age ≥ 24 h); entry provisional and age ≥ 24 h → `"settled"`; entry with non-empty `missing` → status by age; otherwise None.
  - `record_session(year: int, session: pd.Series) -> tuple[dict, list[str]]` — `(answers, missing acronyms)`. Inside `store.recording()`: `data.get_session`, `data.list_drivers`, `data.session_had_rain`; for each driver with a `best_lap`: `data.load_lap(session, acronym)` (LookupError → add to missing); for `session_name` in `("Race", "Sprint")`: `race_lap_table`, `race_results`, `tyre_lap_table` (LookupError ignored). Also `data.get_corners` + `data.get_rotation` (LookupError ignored); circuit answers are split out of `answers` and saved with `save_circuit` by the caller.
  - `build_store(year: int, root: Path = STORE_DIR, force: bool = False, now: pd.Timestamp | None = None) -> None` — records the year's lists (`meetings?year=…` via `data.list_events`, `sessions?meeting_key=…` via `data.list_sessions`) with `save_lists`; then, for every listed session in date order, `session_action(...)` and, when it says so, `record_session` → `save_session` (and `save_circuit` once per circuit). Saves after **each** session. On `data.OpenF1Locked`: print `"OpenF1 is locked for a live session; run again afterwards."` and return. On another `requests.RequestException`: print it, skip that session, continue. Ends with a summary: sessions stored (provisional/settled), incomplete (with missing count), skipped, total size of `root` in MB.
  - CLI: `python store.py 2026 [--force]`; several years allowed: `python store.py 2023 2024 2025`.

- [ ] **Step 1: Write the failing tests**

```python
NOW = pd.Timestamp("2026-11-28T12:00:00Z")

@pytest.mark.parametrize("entry, hours_ago, force, expected", [
    (None, None, False, None),                                  # no end time: skip
    (None, 0.5, False, None),                                   # just ended: too early
    (None, 2, False, "provisional"),
    (None, 30, False, "settled"),
    ({"status": "provisional", "missing": []}, 5, False, None),
    ({"status": "provisional", "missing": []}, 25, False, "settled"),
    ({"status": "settled", "missing": []}, 100, False, None),
    ({"status": "settled", "missing": ["VER"]}, 100, False, "settled"),   # retry missing laps
    ({"status": "settled", "missing": []}, 100, True, "settled"),
])
def test_session_action(entry, hours_ago, force, expected):
    ended = None if hours_ago is None else NOW - pd.Timedelta(hours=hours_ago)
    assert store.session_action(entry, ended, NOW, force) == expected

def test_record_session_captures_every_answer(monkeypatch):
    # FakeHTTP keyed by endpoint (sessions, drivers, laps, race_control, weather, car_data, location) with two drivers;
    # data.get_circuit_info faked. Expect: answers has keys for drivers?session_key=…, laps?session_key=…,
    # race_control?session_key=…, weather?session_key=…, drivers?name_acronym=…&session_key=… for each driver,
    # and a car_data and a location key per driver; missing == [].

def test_record_session_lists_driver_without_telemetry_as_missing(monkeypatch): ...  # car_data empty for one driver -> missing == ["BBB"]

def test_build_store_saves_after_each_session_and_stops_on_lock(monkeypatch, tmp_path):
    # Fake data.list_events/list_sessions with 3 sessions; record_session faked to return answers for the
    # first, raise data.OpenF1Locked for the second. Expect: session 1 in the manifest, session 2 and 3 not,
    # no exception, lists saved.

def test_build_store_skips_sessions_without_end_time(monkeypatch, tmp_path): ...  # date_end None -> not recorded

def test_build_store_rerun_skips_settled(monkeypatch, tmp_path): ...  # second run records nothing; --force records again
```

(The three `...` bodies follow the comment beside each; reuse one fake-data helper.)

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the build section and CLI; update the module docstring's run commands.
- [ ] **Step 4: Run** `<python> -m pytest -q` — all pass.
- [ ] **Step 5: Commit** — `store.py tests/test_store.py`, "Store: build script with provisional and settled sessions".

---

### Task 4: First real build (2026) — then STOP

- [ ] **Step 1: Wait for OpenF1 to unlock** — the curl in Global Constraints returns 200. If it returns 401, stop and tell the owner when to resume (after the Lusail session).
- [ ] **Step 2: Run in the background** — `<python> store.py 2026` (≈ 2,500-3,500 requests ≈ 1.5-2 h). If it stops on a lock, run again later; it resumes.
- [ ] **Step 3: Check** — print: sessions stored by status, incomplete sessions and missing counts, `store/` size, largest file. Then with OpenF1 **blocked** (monkeypatch `data.HTTP` to always 401 in a scratch script): `data.race_lap_table`, `data.tyre_lap_table`, `data.list_drivers` and `analysis.load_and_compare` for 2026 Las Vegas qualifying (fastest laps of two drivers) and the race all return data. `python season.py 2026 --force` with OpenF1 blocked rebuilds the season from the store with identical CSVs (`git diff --stat seasons/` empty).
- [ ] **Step 4: Run the app** (preview "app") with OpenF1 blocked by an env-free method (e.g. a temporary `HTTP` monkeypatch in a scratch launcher), open Qualifying, Race pace, Tyres, Replay for a 2026 event: all render. Then pick a non-fastest lap: the lock message appears.
- [ ] **Step 5: Commit** — `store/`, "Store: the 2026 season".
- [ ] **Step 6: STOP.** Report to the owner: size, coverage, incomplete sessions, what works during a lock, backfill time estimate. Wait for review before Task 6.

---

### Task 5: Docs

**Files:**
- Modify: `README.md` (new "Stored data" section; "Rate limits" limitation; project layout rows for `store.py` and `store/`; run-it-locally note)
- Modify: `docs/HOW_IT_WORKS.md` (new "## 13. The stored copy (`store.py`)" before "Honest weaknesses", house sections: 30-second version, layout, lookup rules, provisional/settled, build, tests, honest weaknesses)

- [ ] **Step 1: Write both sections.** Must include: what's stored and what still needs OpenF1 (other laps, Part picker), the update routine ("after each weekend: `python store.py 2026`, check, commit `store/`, push; on a race weekend also run it after qualifying and push, so the race-day lock doesn't hide qualifying"), sizes measured in Task 4, that the repo republishes OpenF1/MultiViewer data (owner's choice), and the lock message.
- [ ] **Step 2: Run** `<python> -m pytest -q` — all pass.
- [ ] **Step 3: Commit** — `README.md docs/HOW_IT_WORKS.md`, "Docs: the stored copy of OpenF1 data".

---

### Task 6: Backfill 2023-2025 (after the owner's review)

- [ ] **Step 1: Run in the background, one season at a time** — `<python> store.py 2025`, then 2024, then 2023 (≈ 4 h each at 30 requests/minute). Resume after any lock.
- [ ] **Step 2: Check each season** as in Task 4 Step 3 (one race and one qualifying per season), and that `store/` stays under the size the owner approved.
- [ ] **Step 3: Commit each season separately** — `store/`, "Store: the 2025 season" (etc.).
- [ ] **Step 4: Ask the owner before pushing** (`main` redeploys the live app; the push adds ≈ 150-200 MB).
