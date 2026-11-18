# How it works

Your revision guide: the whole pipeline from raw data to findings, the reasoning behind each choice, and the questions you're likely to be asked. Numbers come from the acceptance test: **Norris vs Russell, 2026 São Paulo qualifying** (wet).

## The 30-second version

> "It takes two qualifying laps, puts them on a common distance axis and computes the time gap metre by metre. It cuts the lap into corners (from lift to full throttle) and straights, and measures each driver in each zone: braking point, minimum speed, throttle pickup, exit speed, top speed. Then it ranks where the time went and tags a likely cause (braking, mid-corner, traction, or straight-line/deployment) with the evidence. A last heuristic asks whether the deficits look like the car (repeating everywhere) or the driver (varying corner to corner). It's all hedged, because one lap can't prove cause."

## The pipeline

```
OpenF1 (laps, telemetry, position, weather, race control)    MultiViewer via FastF1 (corner positions)
                         \                                    /
                          data.py      fetch, cache, tidy into tables
                             |
                          analysis.py  align -> corners -> zones -> metrics -> findings -> car/driver read
                             |
                          charts.py    Plotly figures
                             |
                          app.py       Streamlit page (UI only)
```

Each file only talks to the one before it. `analysis.py` has no downloads and no Streamlit, which is what makes it testable with made-up laps.

## 1. Getting the data (`data.py`)

- **Why OpenF1, not FastF1's loader:** F1's live-timing archive, which FastF1 reads, now returns *401 Unauthorized* without a paid F1 TV login. OpenF1 serves the same ~4 Hz car data for free, from 2023 onwards. FastF1 is still used for corner positions (from MultiViewer).
- **Channels:** speed, throttle (0–100 %), brake (**on/off only**), gear, RPM, plus X/Y position. For 2026 cars, **DRS is empty**, because active aero replaced it and isn't in the feed.
- **Deleted laps:** race control posts text such as `CAR 18 (STR) TIME 1:42.680 DELETED ... LAP 4 16:29:11`. Its "LAP 4" was OpenF1's lap 3, an off-by-one in this session. So the code ignores lap numbers and uses the **incident's clock time** (16:29:11 local → UTC) to find the lap being driven at that moment. It also handles sessions that run past midnight.
- **Rate limits and caching:** OpenF1's free tier allows 3 requests/s and 30/min. A "429 Too Many Requests" reply is retried after 2, 5, 15, then 40 s. Every request goes through FastF1's disk cache for 10 minutes: short enough that a just-finished session corrects itself, long enough to absorb bursts.

### Building the telemetry table
- **Padding:** fetch 2 s either side of the lap. Samples never land exactly on the line, so we need one on each side to interpolate between.
- **Position** is sampled at different moments from car data, so it's interpolated onto the car-data timestamps.
- **Distance = speed integrated over time** with the trapezium rule: each step adds (average of two speeds) × (time step). Then the whole column is shifted so that the moment the lap starts (time 0) is exactly 0 m.

## 2. Aligning the two laps (`align_laps`)

The two cars' samples are taken at different places, so you can't subtract row by row. The question becomes: *how long had each car been driving when it reached each metre?*

1. **Stretch.** Integrated distance drifts: 4250 m (NOR) vs 4241 m (RUS). Both laps are scaled to the average, so T1 is at the same metre for both. Without this, a 9 m drift by the end of the lap would look like a braking-point difference.
2. **Grid.** One point every 1 m from 0 m to the finish.
3. **Interpolate** each driver's time (and speed, throttle, position) onto the grid with `np.interp`.
4. **Delta = B's time − A's time** at every metre. Because time 0 = 0 m and lap time = finish line, the delta starts at exactly 0 and ends at exactly the lap-time gap (−0.725 s here).

**The brake trick:** brake is 0/1. Interpolated, it ramps from 0 to 1 between the last "off" and first "on" sample, and the 0.5 crossing is the halfway point. That halves the location error compared with "first sample that says on". At 300 km/h a sample comes every ~20 m, so the estimate is good to about ±10 m.

**Gear** takes the last sample's value instead (no "4.5th gear").

## 3. Corners and zones (`locate_corners`, `split_zones`)

- **Corner distances:** MultiViewer gives each corner's X/Y. The code finds the grid point where the car passed closest to it. One lap is used, because averaging two laps could put a marker near the start line in the middle of the lap (5 m and 4245 m average to 2125 m).
- **Flat-out kinks** (both drivers at full throttle, off the brake within 50 m of the marker) are dropped. They're part of the straight (T15 at Interlagos).
- **A corner zone = lift to full throttle.** It runs from the earlier driver's last full-throttle point before the apex to the later driver's first full-throttle point after it, so it contains both drivers' entire braking zone and exit.
  - *Why not a fixed window?* The first version used "200 m before the marker". On this wet lap, braking for T1 started 215 m before the marker, so the braking point was cut off. Fixed windows also pull straight-line time into corners.
- **Linked corners** (no full throttle between them, e.g. T7→T8→T9) **split at the fastest point between them**, where one exit ends and the next braking begins. Splitting halfway between markers had handed part of Russell's stronger T8 exit to "Norris loses time braking into T9", even though they braked at the same point.
- **Short gaps** (< 150 m) between corners are shared; longer gaps are straights.
- **Self-check:** the zones cover every metre exactly once, so their gaps sum to the lap gap. `analysis.py` prints this check.

## 4. Metrics (`corner_metrics`, `straight_metrics`)

| Corners | Straights |
|---|---|
| Braking point (first brake-on, if the zone started off the brake) | Entry speed (carried from the previous exit) |
| Minimum speed, its position (apex) and gear | Top speed (measured up to the next corner marker) |
| Full-throttle point after the apex (throttle ≥ 95 %) | Deployment pattern: speed *drops* ≥ 3 km/h, or *flattens* (< 1 km/h gain over the last 150 m), at full throttle |
| Exit speed (at the zone end) | Time gained or lost |
| Time gained or lost, split before and after the apex | |

**Deployment is always "possible"**: a drag-limited car also flattens, and there's no energy channel to confirm it.

## 5. Findings and causes (`rank_findings`)

Zones are ranked by time lost (≥ 0.02 s, top 5). For a corner, the loss is split at the apex (average of both drivers' minimum-speed points):

- **Braking**: most of the loss came before the apex (braking, turn-in).
- **Mid-corner**: the loss came after the apex, the loser was already slower at the apex, and the exit-speed gap is no bigger than the apex gap. The time was lost in the middle and carried down the exit.
- **Traction/exit**: the loss came after the apex and the speed gap *grew* on the way out.

**Straights:** time on a straight is mostly decided by the speed you enter with. If the loser entered slower but wasn't slower at the top, it's really the previous corner's **Traction/exit**; otherwise it's **Straight-line/deployment**.

**Evidence** lists only differences above the noise thresholds (15 m, 3 km/h). When nothing crosses them, which is common in the wet because nobody reaches full throttle, the finding says where the speed went instead: "up to 11 km/h slower after the apex" (Norris at T9, where Russell was on 40–57 % throttle against Norris's 19–39 %).

## 6. Driver vs car (`driver_vs_car`), a heuristic

- **Repeats everywhere → car.** A lower top speed on *every* straight, or losing in *every* slow corner (< 150 km/h): a driver's mistakes rarely repeat that neatly.
  - Straights are judged on **top speed, not time**. Norris "lost time on all straights" here but had the *higher* top speed on two of them; the time came from weaker corner exits.
- **Swings corner to corner → driver.** The car is the same car in every corner. Brazil: NOR quicker in 5 corners, RUS in 9.
- **One-sided (e.g. 11 to 1) → can't tell:** either overall grip or a consistent driving difference. A "swing" needs the less successful driver to win at least 25 % of the corners.

## 7. Charts and app

- **One Plotly figure with five rows sharing an x-axis.** That's the only way zoom stays in sync, because separate Streamlit charts can't talk to each other.
- **Brake lanes:** on/off signals for two drivers would hide each other as lines, so each driver gets a lane with bars.
- **Colours:** team colours unless they look too alike or too dark. "Alike" is measured as a perceptual colour difference (CIE Lab ΔE ≥ 30) for normal vision *and* simulated deuteranopia. "Too dark" means WCAG contrast below 3:1 against the background. Otherwise a colour-blind-safe orange/blue pair is used.
- **Streamlit re-runs the whole script** on every click. `@st.cache_data` remembers results per input for 10 minutes, so re-runs are instant.
- **Notes:** plain Markdown in `notes/`, named by session and sorted driver pair. The create button only appears on localhost, because the hosted app's disk is wiped on restart.

## 8. Testing

`tests/` uses **made-up laps with hand-checkable answers**, generated at 4 Hz like the real data:

| Test | What it proves |
|---|---|
| Constant speeds 180 vs 144 km/h | Gap is 2.5 s at 500 m and 5.0 s at the line |
| Same lap with distance 1 % short | Stretching removes integration drift (gap = 0) |
| Brake switches on between samples at 592.5 and 605 m | Braking point lands halfway (598.75 m) |
| Hand-built lap with linked corners and a kink | Zone boundaries exactly as worked out on paper |
| Car B brakes 20 m earlier | End-to-end: "B loses at T1 [Braking], brakes earlier", and zone gaps sum to the lap gap |
| Real race-control messages, and a lap past midnight | Deleted laps found by clock time |

## Honest weaknesses (say these before they're asked)

- **One lap, wet session:** the thresholds were tuned on it, and dry running may need different values.
- **4 Hz and on/off brake:** no trail-braking or brake-pressure analysis. Sub-15 m differences are invisible.
- **No energy data for 2026:** deployment can only be inferred from the speed shape.
- **Cause tags are rules, not models:** they explain where on the corner time went, not why the car or driver did it.
- **Corner positions are approximate**, so a zone can include a bit of the neighbouring straight.

## Questions you might be asked

- **Why not subtract the two traces sample by sample?** The samples are at different places on track. You need a common axis, distance, and interpolation.
- **Why distance and not time on the x-axis?** Corners are fixed in distance. On a time axis, the faster car reaches T1 earlier and the traces drift apart.
- **How accurate is a braking point?** About ±half a sample gap: ±10 m at 300 km/h. Hence the 15 m reporting threshold.
- **How do you know the delta is right?** It ends at exactly the official lap-time gap, the zones sum to it, and a synthetic test checks it against hand calculation.
- **Why might "time lost on a straight" mislead?** It's mostly exit speed carried from the previous corner. Judge the car on top speed and on the speed *shape* at full throttle.
- **What would you add with more data?** Brake pressure for trail-braking, steering angle for understeer vs oversteer, energy state for 2026 deployment, and several laps per driver to separate repeatable traits from one-offs.
- **What surprised you?** F1 put its timing archive behind a login, race control's lap numbers were off by one, and in the wet nobody reached full throttle in some corners, so throttle-point metrics went blank and I needed a fallback.

## Python worth being able to explain

- **`np.interp(x, xp, fp)`**: linear interpolation. "Given known points (xp, fp), what's the value at x?" It's used for every resample.
- **`DataFrame.loc[...]` and `idxmin()`**: select rows by label or condition; `idxmin` gives the row label of the minimum (the apex).
- **Decorators (`@st.cache_data`)**: a line above a function that wraps it with extra behaviour, here memoising results.
- **`**params`**: collects keyword arguments into a dictionary (turned into URL filters in `data.openf1`).
- **Opening a file with mode `"x"`**: create the file, and fail if it exists, so notes can't be overwritten.
- **Regular expressions** (`data.DELETED_LAP`): a pattern that pulls the driver code and clock time out of race-control text.
