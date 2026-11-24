"""
charts.py: Plotly figures for a two-driver comparison, the race-pace
classification and lap-time distribution, and the tyre degradation charts.

Pipeline position:  data.py -> analysis.py / pace.py / tyres.py -> charts.py -> app.py
Draws analysis.compare()'s, pace.analyse()'s and tyres.analyse()'s results and calculates
nothing new. It has no Streamlit code, so the same figures work in the app
and as standalone HTML:

    python charts.py                                    # 2026 Brazil Qualifying, NOR vs RUS
    python charts.py 2025 Monza Qualifying LEC HAM

writes two files into output/:
    ..._telemetry.html  speed (with corner numbers), time gap, throttle, brake
                        and gear, stacked on ONE shared lap-distance axis, so
                        zooming any of them zooms them all
    ..._track_map.html  the circuit, coloured by the faster driver in each
                        mini-sector

    python charts.py 2026 Brazil                        # race pace (year and place only)

writes two more:
    ..._Race_pace_classification.html  a TV-style timing tower ranked by median pace
    ..._Race_pace_distribution.html    every driver's clean laps as a box plot

    python charts.py 2026 Brazil tyres                  # tyre degradation

writes three more:
    ..._Race_tyres_scatter.html      fuel-corrected lap time against tyre age, with each stint's line
    ..._Race_tyres_teams.html        each team's degradation per compound, with intervals
    ..._Race_tyres_sensitivity.html  the team ranking at the low, default and high fuel effect

The look is a pit-wall screen, not a busy dashboard: dark, two high-contrast
driver colours, large labels, almost no gridlines.
"""

import html
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import analysis
import pace

# ---- Look and feel ----------------------------------------------------------
BACKGROUND = "#0E1117"  # Streamlit's dark-theme background, so charts blend into the app
TEXT = "#FAFAFA"
MUTED = "#8B8F98"       # corner labels and the zero line
GRID = "#262730"        # faint horizontal gridlines only
FONT_SIZE = 15
LABEL_OFFSET = 350      # corner labels sit this far off the track (map units of 0.1 m, so 35 m);
                        # further out, they land on nearby track at tight circuits like Interlagos
LABEL_GAP_M = 150       # corner labels closer than this along the lap are staggered on two heights

FALLBACK_COLOURS = ("#E69F00", "#56B4E9")
# Okabe-Ito orange and sky blue: told apart with the common forms of colour
# blindness, and bright on a dark background. Used when team colours aren't.

MIN_COLOUR_DIFFERENCE = 30
# How different two line colours must look (CIE76 "delta E": 0 = identical,
# ~2 = just noticeable, 30+ = clearly different at a glance).

MIN_BACKGROUND_CONTRAST = 3.0
# WCAG's minimum contrast for graphics: below it a line (a dark navy team
# colour, say) is hard to see on the dark background.

DEUTERANOPIA = np.array([[0.29031, 0.70969, 0.0],
                         [0.29031, 0.70969, 0.0],
                         [-0.02197, 0.02197, 1.0]])
# How a deuteranope (the commonest colour blindness, ~5 % of men) sees a
# colour, as a matrix on linear RGB (Vienot et al., 1999). Red-green team
# pairs can look alike to them, so colours are compared both ways.


# ---- Choosing driver colours -------------------------------------------------

def linear_rgb(hex_colour):
    """'#F47600' -> red, green, blue from 0 to 1 with the screen's gamma curve
    undone. Colour maths (mixing, brightness) only works on these values."""
    srgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def lab(rgb):
    """Linear RGB -> CIE Lab, a colour space where the distance between two
    colours matches how different they look to people."""
    xyz = np.array([[0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]]) @ np.clip(rgb, 0, 1)
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])  # relative to daylight white
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def colour_difference(colour_1, colour_2):
    """How different two colours look (delta E): the smaller of the
    difference for normal vision and for deuteranopia."""
    rgb_1, rgb_2 = linear_rgb(colour_1), linear_rgb(colour_2)
    normal = np.linalg.norm(lab(rgb_1) - lab(rgb_2))
    deutan = np.linalg.norm(lab(DEUTERANOPIA @ rgb_1) - lab(DEUTERANOPIA @ rgb_2))
    return float(min(normal, deutan))


def contrast_ratio(colour_1, colour_2):
    """WCAG contrast ratio: 1 = no contrast, 21 = black on white."""
    darker, lighter = sorted(float(np.dot([0.2126, 0.7152, 0.0722], linear_rgb(c))) for c in (colour_1, colour_2))
    return (lighter + 0.05) / (darker + 0.05)


def driver_colours(team_colours, teams):
    """Pick one line colour per driver.

    Team colours are used when both are valid, clearly different (for
    colour-blind viewers too) and visible on the dark background. Otherwise,
    and always for teammates, the colour-blind-safe fallback pair is used.
    `team_colours` are OpenF1 hex strings like "F47600" (may be missing).
    """
    colours = ["#" + c.lstrip("#") for c in team_colours
               if isinstance(c, str) and re.fullmatch(r"#?[0-9A-Fa-f]{6}", c)]
    if (len(colours) == 2 and teams[0] != teams[1]
            and colour_difference(*colours) >= MIN_COLOUR_DIFFERENCE
            and all(contrast_ratio(c, BACKGROUND) >= MIN_BACKGROUND_CONTRAST for c in colours)):
        return tuple(colours)
    return FALLBACK_COLOURS


# ---- Figures -----------------------------------------------------------------

def style(fig, title, height):
    """Apply the shared pit-wall look: dark, large text, legend on top, no clutter."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=FONT_SIZE + 2), x=0, xanchor="left"),
        height=height, paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, size=FONT_SIZE),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom", font=dict(size=FONT_SIZE)),
        margin=dict(l=90, r=30, t=100, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, tickfont=dict(size=FONT_SIZE - 1))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(size=FONT_SIZE - 1))
    return fig


# (axis title, share of the figure's height), top to bottom
ROWS = [("Speed (km/h)", 0.34), ("Gap (s)", 0.18), ("Throttle (%)", 0.18), ("Brake", 0.10), ("Gear", 0.20)]


def telemetry_figure(result, names, colours, title=""):
    """Speed, time gap, throttle, brake and gear, stacked on one lap-distance axis.

    One figure with five rows, not five figures, is deliberate: rows that
    share an x-axis zoom and pan together, so zooming into T1 on the speed
    trace zooms every other row to T1 as well.
    `result` comes from analysis.compare(); `names` and `colours` are (A, B).
    """
    a, b, delta = result["a"], result["b"], result["delta"]
    fig = make_subplots(rows=len(ROWS), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[share for _, share in ROWS])

    for lap, name, colour in ((a, names[0], colours[0]), (b, names[1], colours[1])):
        x = lap["Distance"]
        fig.add_trace(go.Scatter(x=x, y=lap["Speed"], name=name, legendgroup=name,
                                 line=dict(color=colour, width=2), hovertemplate="%{y:.0f} km/h"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=lap["Throttle"], name=name, legendgroup=name, showlegend=False,
                                 line=dict(color=colour, width=2), hovertemplate="%{y:.0f} %"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=lap["Gear"], name=name, legendgroup=name, showlegend=False,
                                 line=dict(color=colour, width=2, shape="hv"), hovertemplate="gear %{y}"), row=5, col=1)

    # Brake is only on/off (OpenF1 has no pressure channel), so each driver gets a
    # lane (A on top) with a thick bar wherever the brake is pressed; overlaid on/off
    # lines would hide each other. The bars have gaps, and hover would borrow the
    # nearest bar point a few metres away, saying "braking" before a driver brakes.
    # So the bars don't hover; an invisible line with a value at every metre does.
    for lane, lap, name, colour in ((1, a, names[0], colours[0]), (0, b, names[1], colours[1])):
        fig.add_trace(go.Scatter(x=lap["Distance"], y=np.where(lap["Brake"], lane, np.nan), name=name,
                                 legendgroup=name, showlegend=False, line=dict(color=colour, width=10),
                                 hoverinfo="skip"), row=4, col=1)
        fig.add_trace(go.Scatter(x=lap["Distance"], y=np.full(len(lap), lane), name=name, legendgroup=name,
                                 showlegend=False, line=dict(color=colour, width=0),
                                 customdata=np.where(lap["Brake"], "on", "off"),
                                 hovertemplate="brake %{customdata}"), row=4, col=1)

    # Gap = B's time minus A's: above zero, B is behind.
    fig.add_trace(go.Scatter(x=a["Distance"], y=delta, name="Gap", showlegend=False,
                             line=dict(color=TEXT, width=2), hovertemplate="%{y:+.3f} s"), row=2, col=1)

    # Numbered corner markers: a faint line through every row, the label above
    # the speed trace. A label closer than LABEL_GAP_M to the previous one goes
    # on the other of two heights, so clusters (T6-T9 at Interlagos) stay readable.
    height_row, previous = 0, -np.inf
    for label, distance in zip(result["corners"]["Label"], result["corners"]["Distance"]):
        height_row = 1 - height_row if distance - previous < LABEL_GAP_M else 0
        previous = distance
        fig.add_shape(type="line", x0=distance, x1=distance, xref="x", y0=0, y1=1, yref="paper",
                      line=dict(color=GRID, width=1, dash="dot"), layer="below")
        fig.add_annotation(x=distance, xref="x", y=1.0, yref="y domain", yanchor="bottom", text=label,
                           yshift=15 * height_row, showarrow=False, font=dict(color=MUTED, size=FONT_SIZE - 3))

    style(fig, title, height=950)
    fig.update_layout(hovermode="x unified", legend=dict(y=1.06))  # legend above both rows of corner labels
    for row, (axis_title, _) in enumerate(ROWS, start=1):
        fig.update_yaxes(title=dict(text=axis_title, font=dict(size=FONT_SIZE)), row=row, col=1)
    fig.update_yaxes(title=dict(text=f"Gap (s)<br>+ = {names[1]} behind"), zeroline=True,
                     zerolinecolor=MUTED, row=2, col=1)
    fig.update_yaxes(range=[-5, 105], row=3, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=[names[1], names[0]], range=[-0.6, 1.6], showgrid=False, row=4, col=1)
    fig.update_yaxes(dtick=1, row=5, col=1)
    fig.update_xaxes(title=dict(text="Lap distance (m)", font=dict(size=FONT_SIZE)), row=len(ROWS), col=1)
    fig.update_xaxes(range=[0, float(a["Distance"].iloc[-1])])
    return fig


def track_map_figure(result, corners, names, colours, title=""):
    """The circuit, each mini-sector coloured by the driver who was faster through it.

    Each corner section and straight is also labelled with who gained there and
    by how much (its zone time from the analysis), in that driver's colour.
    The track line is driver A's position data. `corners` are the raw markers
    from data.get_corners(); their labels are pushed LABEL_OFFSET off the
    track in the direction MultiViewer gives, so they don't sit on the line.
    """
    a = result["a"]
    xs, ys = {0: [], 1: []}, {0: [], 1: []}
    for start, end, winner in result["mini_sectors"]:
        piece = a[(a["Distance"] >= start) & (a["Distance"] <= end)]
        xs[winner] += piece["X"].tolist() + [None]  # None breaks the line between pieces
        ys[winner] += piece["Y"].tolist() + [None]

    fig = go.Figure()
    for winner in (0, 1):
        fig.add_trace(go.Scatter(x=xs[winner], y=ys[winner], mode="lines", name=f"{names[winner]} faster",
                                 line=dict(color=colours[winner], width=7), hoverinfo="skip"))

    # Labels get a see-through dark box: where a corner sits right next to
    # other track (T15 on Interlagos's main straight), the label stays readable.
    def label_at(x, y, text, colour):
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, bgcolor="rgba(14, 17, 23, 0.75)",
                           font=dict(color=colour, size=FONT_SIZE - 2))

    def gain(time):
        """'NOR 0.21' and NOR's colour, for whoever gained `time` (B minus A: positive = B lost, A gained)."""
        winner = 0 if time > 0 else 1
        return f"{names[winner]} {abs(time):.2f}", colours[winner]

    # Corners: "T4 RUS 0.21". Flat-out kinks have no zone of their own, so they stay plain.
    corner_times = {z["name"]: z["time"] for z in result["zones"] if z["kind"] == "corner"}
    angle = np.radians(corners["Angle"].to_numpy(dtype=float))
    labels = analysis.corner_labels(corners)
    for label, x, y in zip(labels, corners["X"] + LABEL_OFFSET * np.cos(angle),
                           corners["Y"] + LABEL_OFFSET * np.sin(angle)):
        if label in corner_times:
            note, colour = gain(corner_times[label])
            label_at(x, y, f"{label} {note}", colour)
        else:
            label_at(x, y, label, MUTED)

    # Straights: the same note, halfway along.
    for zone in result["zones"]:
        if zone["kind"] == "straight":
            middle = int(np.argmin(np.abs(a["Distance"] - (zone["start"] + zone["end"]) / 2)))
            label_at(a["X"].iloc[middle], a["Y"].iloc[middle], *gain(zone["time"]))
    fig.add_trace(go.Scatter(x=[a["X"].iloc[0]], y=[a["Y"].iloc[0]], mode="markers", name="Start/finish",
                             marker=dict(color=TEXT, size=11, symbol="square"), hoverinfo="skip"))

    style(fig, title, height=750)
    fig.update_layout(hovermode=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)  # 1 m across = 1 m up, so the shape is true
    return fig


# ---- Race pace -----------------------------------------------------------------

COMPOUNDS = {
    "SOFT": ("Soft", "#E8002D", "circle"),
    "MEDIUM": ("Medium", "#FFD12E", "square"),
    "HARD": ("Hard", "#F0F0F0", "diamond"),
    "INTERMEDIATE": ("Intermediate", "#43B02A", "triangle-up"),
    "WET": ("Wet", "#1E8FFF", "triangle-down"),
    "UNKNOWN": ("Unknown", MUTED, "x"),
}
# Pirelli's tyre colours (soft red, medium yellow, hard white, intermediate
# green, wet blue), plus grey for laps OpenF1 has no tyre data for. Each also
# gets its own marker shape: soft red and intermediate green look alike to
# deuteranopes (see DEUTERANOPIA), and a wet race can have both.

JITTER = 0.2
# Dots are spread up to this far either side of their box's centre (boxes are
# 1 apart), so laps with similar times don't hide each other.


def team_colour(value):
    """OpenF1's 'F47600' -> '#F47600'. Anything else (missing, malformed) -> MUTED.

    Checked strictly because the colour is written into HTML and CSS.
    """
    if isinstance(value, str) and re.fullmatch(r"#?[0-9A-Fa-f]{6}", value):
        return "#" + value.lstrip("#")
    return MUTED


def see_through(colour, alpha):
    """'#F47600', 0.25 -> 'rgba(244, 118, 0, 0.25)': the same colour, partly transparent."""
    return f"rgba({int(colour[1:3], 16)}, {int(colour[3:5], 16)}, {int(colour[5:7], 16)}, {alpha})"


TOWER_CSS = """
.pace-tower { font-family: "Source Sans Pro", system-ui, sans-serif; color: #FAFAFA; max-width: 780px;
  font-variant-numeric: tabular-nums; overflow-x: auto; padding-top: 4px; }
.pace-title { font-size: 19px; font-weight: 600; margin: 0 0 12px; }
.pace-head, .pace-row { display: grid; align-items: center; column-gap: 12px; padding: 0 14px;
  grid-template-columns: 2em 6px 3.4em minmax(7em, 1fr) 5.6em 6.4em 9em; }
.pace-head { height: 26px; font-size: 12px; color: #8B8F98; text-transform: uppercase; letter-spacing: .08em; }
.pace-head span:first-child, .pace-head span:nth-child(n+5) { text-align: right; }  /* line up with the numbers */
.pace-row { position: relative; height: 38px; margin-bottom: 2px; border-radius: 3px; font-size: 16px;
  background: #161B24; }
.pace-row:hover, .pace-row:focus-within { background: #222937; outline: none; }
.pace-row:focus-visible { outline: 2px solid #FAFAFA; }
.pace-rank { font-weight: 700; text-align: right; }
.pace-bar { height: 22px; border-radius: 2px; }
.pace-code { font-weight: 800; letter-spacing: .04em; }
.pace-tower a.pace-code { color: inherit; text-decoration: none; }
.pace-tower a.pace-code:hover, .pace-tower a.pace-code:focus-visible { text-decoration: underline; }
.pace-team { color: #B5B9C1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pace-time { font-weight: 600; text-align: right; }
.pace-gap { color: #B5B9C1; text-align: right; }
.pace-finish { text-align: right; white-space: nowrap; }
.pace-unranked > :not(.pace-tip) { opacity: .45; }
.pace-divider { display: flex; align-items: center; gap: 10px; margin: 12px 0 6px; font-size: 12px;
  color: #8B8F98; text-transform: uppercase; letter-spacing: .08em; }
.pace-divider::after { content: ""; flex: 1; border-top: 1px solid #3A3F4B; }
/* The tooltip is always in the page (so screen readers read it) but squeezed out
   of sight, and shown above the row on hover or keyboard focus. */
.pace-tip { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%);
  white-space: nowrap; }
.pace-row:hover .pace-tip, .pace-row:focus-within .pace-tip { width: auto; height: auto; clip-path: none;
  right: 14px; bottom: calc(100% + 4px); z-index: 2; padding: 4px 10px; border-radius: 4px;
  background: #2E3542; font-size: 13px; box-shadow: 0 2px 8px rgba(0, 0, 0, .5); }
"""


def classification_html(table, title="", driver_url=None):
    """The race-pace classification as a TV-style timing tower (HTML and CSS).

    One row per driver: rank, team-colour bar, driver code, team, median lap,
    gap to the fastest median, and where they finished. Clean laps and tyre
    strategy show in a tooltip on hover (or keyboard focus: every row, or its
    driver link, can be tabbed to), keeping the row itself uncluttered.
    Unranked drivers sit below a divider, greyed out, with the reason in the
    last column.

    `driver_url`, e.g. "driver?race=11412&driver={}", turns each driver code
    into a link to that address (opened in a new tab, so this page keeps its
    choices); {} becomes the code. Without it the codes are plain text.

    HTML rather than a Plotly table: a Plotly table can't draw a coloured bar
    in a row or show a tooltip per row, and looks like a spreadsheet. Every
    piece of text from the data goes through html.escape, so a team name
    can't inject HTML into the page. `table` comes from pace.classify.
    """
    def row_html(row):
        ranked = pd.isna(row["reason"])
        median = analysis.lap_time_text(row["median"]) if pd.notna(row["median"]) else "–"
        tip = f"{row['clean_laps']} clean laps"
        if isinstance(row["stints"], list):
            tip += f" · {pace.stint_text(row['stints'])}"
        cells = [
            ("rank", f"{int(row['rank'])}" if ranked else "–"),
            ("code", row["driver"]),
            ("team", row["team"]),
            ("time", median),
            ("gap", f"+{row['gap']:.3f} s" if ranked else ""),
            ("finish", pace.finished_text(row)),
        ]
        spans = [f'<span class="pace-{name}">{html.escape(str(text))}</span>' for name, text in cells]
        if driver_url:
            href = html.escape(driver_url.format(row["driver"]))
            spans[1] = (f'<a class="pace-code" href="{href}" target="_blank" rel="noopener">'
                        f'{html.escape(row["driver"])}</a>')
        bar = f'<span class="pace-bar" style="background: {team_colour(row["team_colour"])}"></span>'
        spans.insert(1, bar)
        # Without links, the row itself takes keyboard focus (to show its tooltip); with links, the link does.
        focus = "" if driver_url else ' tabindex="0"'
        return (f'<div class="pace-row{"" if ranked else " pace-unranked"}"{focus}>'
                + "".join(spans) + f'<span class="pace-tip">{html.escape(tip)}</span></div>')

    ranked = table["reason"].isna()
    head = "".join(f"<span>{label}</span>" for label in ("Pace", "", "Driver", "Team", "Median", "Gap", "Finished"))
    parts = [f"<style>{TOWER_CSS}</style>", '<div class="pace-tower">']
    if title:
        parts.append(f'<div class="pace-title">{html.escape(title)}</div>')
    parts.append(f'<div class="pace-head">{head}</div>')
    parts += [row_html(row) for _, row in table[ranked].iterrows()]
    if (~ranked).any():
        parts.append('<div class="pace-divider">Not ranked</div>')
        parts += [row_html(row) for _, row in table[~ranked].iterrows()]
    parts.append("</div>")
    return "\n".join(parts)


def lap_time_ticks(low, high, most=9):
    """Y-axis ticks for lap times: positions (s) and 'm:ss.s' labels from `low` to `high`.

    Plotly can only format numbers, not seconds as minutes, so the labels are
    made here. The step is the smallest of 0.5, 1, 2, 5... s that gives at most
    `most` ticks.
    """
    step = next((s for s in (0.5, 1, 2, 5, 10, 20, 30) if (high - low) / s <= most), 60)
    ticks = np.arange(np.ceil(low / step) * step, high + 1e-9, step)  # + 1e-9: keep a tick exactly at `high`
    return ticks, [f"{int(t // 60)}:{t % 60:04.1f}" for t in ticks]


def pace_distribution_figure(table, clean, title="", driver_url=None):
    """Every driver's clean laps: one box per driver, every lap a dot coloured by tyre compound.

    Boxes are in classification order (fastest median on the left) in team
    colours; unranked drivers come last, faded, after a dotted line. A box
    spans the middle half of the laps (25th to 75th percentile) with a line at
    the median, which is the pace in the classification. Box-plot outlier
    markers are off, because every clean lap is drawn as a dot anyway. Dots
    are jittered sideways with a fixed random seed, so they don't jump around
    each time the page re-runs. `table` and `clean` come from pace.analyse.
    `driver_url` links the driver codes under the boxes, as in classification_html.
    """
    order = table[table["driver"].isin(clean["driver"])].reset_index(drop=True)  # drivers with clean laps
    position = {driver: i for i, driver in enumerate(order["driver"])}
    fig = go.Figure()

    for i, row in order.iterrows():
        colour = team_colour(row["team_colour"])
        times = clean.loc[clean["driver"] == row["driver"], "lap_time_s"]
        # Unranked drivers are faded, like the greyed-out rows of the timing tower
        # (plain grey would look like Haas's or Cadillac's team colour).
        fig.add_trace(go.Box(x=np.full(len(times), i), y=times, name=row["driver"], width=0.6, boxpoints=False,
                             line=dict(color=colour, width=2), fillcolor=see_through(colour, 0.25),
                             opacity=1 if pd.isna(row["reason"]) else 0.4, showlegend=False, hoverinfo="skip"))

    rng = np.random.default_rng(0)  # fixed seed: the same "random" spread every time
    x = clean["driver"].map(position).to_numpy() + rng.uniform(-JITTER, JITTER, len(clean))
    compounds = clean["compound"].where(clean["compound"].isin(list(COMPOUNDS)), "UNKNOWN")  # missing -> UNKNOWN
    for compound, (label, colour, symbol) in COMPOUNDS.items():
        mine = (compounds == compound).to_numpy()
        if not mine.any():
            continue  # the key lists only compounds that were used
        laps = clean[mine]
        age = laps["tyre_age"].map(lambda a: f"{a:.0f} laps old" if pd.notna(a) else "tyre age unknown")
        details = np.column_stack([laps["driver"], laps["lap"], laps["lap_time_s"].map(analysis.lap_time_text),
                                   age])
        fig.add_trace(go.Scatter(
            x=x[mine], y=laps["lap_time_s"], mode="markers", name=label, legendgroup=compound, showlegend=False,
            marker=dict(color=colour, symbol=symbol, size=7, line=dict(color=BACKGROUND, width=0.6)),
            customdata=details,
            hovertemplate=f"<b>%{{customdata[0]}}</b> · lap %{{customdata[1]}}<br>%{{customdata[2]}}<br>"
                          f"{label}, %{{customdata[3]}}<extra></extra>"))
        # The key: a larger copy of the marker with no data, in the same legend group,
        # so clicking it in the key hides or shows that compound's dots.
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, legendgroup=compound,
                                 marker=dict(color=colour, symbol=symbol, size=13), hoverinfo="skip"))

    unranked = order["reason"].notna()
    if unranked.any() and not unranked.all():
        split = int((~unranked).sum()) - 0.5
        fig.add_vline(x=split, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=split, xanchor="left", y=0, yref="paper", yanchor="bottom", text=" Not ranked",
                           showarrow=False, font=dict(color=MUTED, size=FONT_SIZE - 3))  # bottom: the key is at the top

    style(fig, title, height=650)
    ticks, labels = lap_time_ticks(clean["lap_time_s"].min(), clean["lap_time_s"].max())
    fig.update_layout(hovermode="closest", legend=dict(title=dict(text="Tyre compound ")))
    codes = order["driver"].tolist()
    if driver_url:  # Plotly draws <a href> in tick labels as links
        codes = [f'<a href="{html.escape(driver_url.format(c))}" target="_blank">{html.escape(c)}</a>' for c in codes]
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=codes, range=[-0.7, len(order) - 0.3])
    fig.update_yaxes(tickvals=ticks, ticktext=labels, title=dict(text="Lap time", font=dict(size=FONT_SIZE)))
    return fig


# ---- Tyre degradation ------------------------------------------------------------
# All three figures draw tyres.analyse()'s result (`run`) and calculate nothing new.

STINT_DASHES = ("solid", "dash", "dot", "dashdot")
# Each stint's fitted line gets a line style by stint number (first stint
# solid, second dashed...), so stints aren't told apart by colour alone.

DRIVER_SYMBOLS = ("circle", "diamond")
# Teammates share a team colour, so the two drivers of a team (in alphabetical
# order) get different marker shapes.


def team_colours(laps):
    """{team: '#RRGGBB'} from the lap table's team_colour column (MUTED if it's missing or malformed)."""
    teams = laps.drop_duplicates("team")
    colours = teams["team_colour"] if "team_colour" in teams else pd.Series(None, index=teams.index)
    return {team: team_colour(colour) for team, colour in zip(teams["team"], colours)}


def compound_order(compounds):
    """Compounds present, softest first (not alphabetical)."""
    present = set(compounds)
    return [c for c in ("SOFT", "MEDIUM", "HARD") if c in present]


def no_data_figure(text, title, height=300):
    """An empty figure with a message, for a race with nothing to draw."""
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(color=MUTED, size=FONT_SIZE))
    style(fig, title, height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def tyre_scatter_figure(run, title=""):
    """Chart 1: every counted lap's fuel-corrected time against its tyre age, one panel per compound.

    One dot per lap in its team's colour (marker shape tells teammates apart),
    and each stint's Theil-Sen line over the ages it ran, in the same colour,
    its line style set by stint number. Clicking a team in the key hides or
    shows its laps and lines. Hover gives driver, lap, tyre age and the
    corrected time (with the real lap time beside it).
    The panels share the time axis, so compounds can be compared by eye.
    """
    kept, stints = run["kept"], run["stints"]
    compounds = compound_order(kept["compound"])
    if not compounds:
        return no_data_figure("No laps left to fit in this race.", title)
    colours = team_colours(run["laps"])
    symbol = {driver: DRIVER_SYMBOLS[i % len(DRIVER_SYMBOLS)]
              for _, team in kept.groupby("team") for i, driver in enumerate(sorted(team["driver"].unique()))}
    fits = stints.set_index(["driver", "stint"])

    fig = make_subplots(rows=1, cols=len(compounds), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[COMPOUNDS[c][0] for c in compounds])
    # The key: one plain circle per team, alphabetical, with no data. It's in the same legend
    # group as the team's dots and lines, so clicking it hides or shows them all.
    for team in sorted(kept["team"].unique()):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=team, legendgroup=team, hoverinfo="skip",
                                 marker=dict(color=colours.get(team, MUTED), size=11)), row=1, col=1)
    for col, compound in enumerate(compounds, start=1):
        for (team, driver, stint), laps in kept[kept["compound"] == compound].groupby(["team", "driver", "stint"]):
            colour = colours.get(team, MUTED)
            details = np.column_stack([laps["driver"], laps["lap"], laps["lap_time_s"].map(analysis.lap_time_text),
                                       laps["corrected_s"].map(analysis.lap_time_text)])
            fig.add_trace(go.Scatter(
                x=laps["tyre_age"], y=laps["corrected_s"], mode="markers", name=team, legendgroup=team,
                showlegend=False, customdata=details,
                marker=dict(color=colour, symbol=symbol[driver], size=7, line=dict(color=BACKGROUND, width=0.6)),
                hovertemplate="<b>%{customdata[0]}</b> · lap %{customdata[1]} · tyre age %{x}<br>"
                              "corrected %{customdata[3]} (actual %{customdata[2]})<extra></extra>"), row=1, col=col)
            fit = fits.loc[(driver, stint)]
            ages = np.array([laps["tyre_age"].min(), laps["tyre_age"].max()], dtype=float)
            fig.add_trace(go.Scatter(
                x=ages, y=fit["intercept"] + fit["slope"] * ages, mode="lines", legendgroup=team, showlegend=False,
                line=dict(color=colour, width=2, dash=STINT_DASHES[(int(stint) - 1) % len(STINT_DASHES)]),
                hovertemplate=f"<b>{html.escape(driver)}</b> stint {int(stint)}: {fit['slope']:+.3f} s per lap of age"
                              "<extra></extra>"), row=1, col=col)

    style(fig, title, height=680)
    # The key goes under the chart: with 11 teams it wraps, and on top it would cover the panel titles.
    fig.update_layout(hovermode="closest", margin=dict(b=150),
                      legend=dict(title=dict(text="Team "), x=0, xanchor="left", y=-0.16, yanchor="top"))
    ticks, labels = lap_time_ticks(kept["corrected_s"].min(), kept["corrected_s"].max())
    fig.update_yaxes(tickvals=ticks, ticktext=labels)
    fig.update_yaxes(title=dict(text="Lap time, corrected to full fuel", font=dict(size=FONT_SIZE)), row=1, col=1)
    fig.update_xaxes(title=dict(text="Tyre age (laps)", font=dict(size=FONT_SIZE)))
    return fig


def tyre_teams_figure(run, title=""):
    """Chart 2: each team's degradation (s per lap of tyre age) with its 90 % interval, one panel per compound.

    Teams are sorted within each compound, lowest degradation at the top. A
    filled marker means the interval excludes zero; a hollow one means "not
    distinguishable from flat" (shape, not just colour, carries it). Teams
    without enough laps are listed in grey with "insufficient data", not
    hidden. Where Method B could be fitted, its whole-field interval is a
    faint band: a check that needs no fuel figure.
    """
    teams = run["teams"]
    if teams is None or teams.empty:
        return no_data_figure("No laps left to fit in this race.", title)
    compounds = compound_order(teams["compound"])
    colours = team_colours(run["laps"])
    method_b = run["method_b"].set_index("compound") if run.get("method_b") is not None else pd.DataFrame()
    sizes = [int((teams["compound"] == c).sum()) for c in compounds]
    fig = make_subplots(rows=len(compounds), cols=1, shared_xaxes=True, row_heights=sizes, vertical_spacing=0.08,
                        subplot_titles=[COMPOUNDS[c][0] for c in compounds])

    for row, compound in enumerate(compounds, start=1):
        rows = teams[teams["compound"] == compound]  # tyres.team_slopes sorts lowest first, no-data teams last
        for team in rows[rows["status"] == "OK"].itertuples():
            colour = colours.get(team.team, MUTED)
            fig.add_trace(go.Scatter(
                x=[team.slope], y=[team.team], mode="markers", showlegend=False,
                error_x=dict(type="data", symmetric=False, array=[team.high - team.slope],
                             arrayminus=[team.slope - team.low], color=colour, thickness=2, width=6),
                marker=dict(color=BACKGROUND if team.flat else colour, size=12, line=dict(color=colour, width=2),
                            symbol="circle"),
                hovertemplate=f"<b>{html.escape(team.team)}</b> · {COMPOUNDS[compound][0]}<br>"
                              f"{team.slope:+.3f} s/lap [{team.low:+.3f}, {team.high:+.3f}]<br>"
                              f"≈ {team.loss_10:+.2f} s over 10 laps · {int(team.laps)} laps, {team.stints} stints"
                              + ("<br>not distinguishable from flat" if team.flat else "") + "<extra></extra>"),
                row=row, col=1)
        missing = rows[rows["status"] != "OK"]
        if len(missing):
            fig.add_trace(go.Scatter(
                x=[0] * len(missing), y=missing["team"], mode="text", showlegend=False, hoverinfo="skip",
                text=[f"  insufficient data ({int(n)} laps)" for n in missing["laps"]], textposition="middle right",
                textfont=dict(color=MUTED, size=FONT_SIZE - 2), cliponaxis=False), row=row, col=1)  # cliponaxis: not cut at the plot edge
        # Lowest at the top: Plotly draws the first category at the bottom, so reverse the order.
        # Teams without a number get a grey label (Plotly reads a <span style> in tick text).
        order = rows["team"].tolist()[::-1]
        labels = [name if name in set(rows.loc[rows["status"] == "OK", "team"])
                  else f'<span style="color:{MUTED}">{html.escape(name)}</span>' for name in order]
        fig.update_yaxes(categoryorder="array", categoryarray=order, tickvals=order, ticktext=labels, showgrid=False,
                         row=row, col=1)
        fig.add_vline(x=0, line=dict(color=MUTED, width=1), row=row, col=1)
        if compound in method_b.index and method_b.loc[compound, "status"] == "OK":
            b = method_b.loc[compound]
            fig.add_vrect(x0=b["low"], x1=b["high"], fillcolor=see_through(TEXT, 0.08), line_width=0, layer="below",
                          row=row, col=1)
            fig.add_annotation(x=b["slope"], y=1, yref=f"y{row if row > 1 else ''} domain", yanchor="bottom",
                               text="Method B, whole field", showarrow=False,
                               font=dict(color=MUTED, size=FONT_SIZE - 4))

    style(fig, title, height=140 + 34 * sum(sizes) + 60 * len(compounds))
    fig.update_layout(hovermode="closest", margin=dict(l=150))
    fig.update_xaxes(zeroline=False, title=dict(text="s lost per lap of tyre age", font=dict(size=FONT_SIZE)),
                     row=len(compounds), col=1)
    return fig


def tyre_sensitivity_figure(run, title=""):
    """Chart 3 (small): each team's rank (1 = lowest degradation) at the low, default and high fuel effect.

    Flat lines mean the ranking doesn't depend on the fuel assumption. Hover
    gives the slope at each value. One panel per compound, stacked (side by
    side, the team names had no room in the app's column); only teams with a
    number on that compound.
    """
    table = run["sensitivity"]
    if table is None or table.empty:
        return no_data_figure("No team has enough laps for a ranking.", title, height=240)
    names = [n for n in ("low", "default", "high") if f"rank_{n}" in table]
    steps = [f"{n.capitalize()} · {run['fuel'][n]:.3f} s/lap" for n in names]
    compounds = compound_order(table["compound"])
    colours = team_colours(run["laps"])
    sizes = [int((table["compound"] == c).sum()) for c in compounds]
    # shared_xaxes: the fuel steps are the same in every panel, so they're labelled once, at the bottom.
    fig = make_subplots(rows=len(compounds), cols=1, row_heights=sizes, vertical_spacing=0.12, shared_xaxes=True,
                        subplot_titles=[COMPOUNDS[c][0] for c in compounds])
    for row, compound in enumerate(compounds, start=1):
        for team in table[table["compound"] == compound].to_dict("records"):
            colour = colours.get(team["team"], MUTED)
            fig.add_trace(go.Scatter(
                x=steps, y=[team[f"rank_{n}"] for n in names], mode="lines+markers+text", showlegend=False,
                line=dict(color=colour, width=2), marker=dict(color=colour, size=8),
                text=[""] * (len(names) - 1) + [f" {team['team']}"], textposition="middle right",
                textfont=dict(color=colour, size=FONT_SIZE - 3),
                customdata=[team[f"slope_{n}"] for n in names],
                hovertemplate=f"<b>{html.escape(team['team'])}</b> · %{{x}}<br>rank %{{y}} · %{{customdata:+.3f}} s/lap"
                              "<extra></extra>"), row=row, col=1)
    style(fig, title, height=120 + 34 * sum(sizes) + 90 * len(compounds))
    fig.update_layout(hovermode="closest", margin=dict(l=60))
    fig.update_xaxes(range=[-0.2, len(names)])  # a spare step on the right for the team names
    # Ranks run 1 (top) downwards; a little padding so the first and last lines don't touch the edges.
    for row, size in enumerate(sizes, start=1):
        fig.update_yaxes(range=[size + 0.5, 0.5], dtick=1, showgrid=False, row=row, col=1)
    return fig


def tyres_main(year, place):
    import data
    import tyres

    session = data.find_session(year, place, "Race")
    run = tyres.analyse(data.tyre_lap_table(session))
    title = f"{session['year']} {session['location']} race · tyre degradation"

    os.makedirs("output", exist_ok=True)
    stem = os.path.join("output", f"{year}_{place}_Race_tyres".replace(" ", "_"))
    figures = {"scatter": tyre_scatter_figure(run, f"{title} · fuel-corrected laps"),
               "teams": tyre_teams_figure(run, f"{title} · by team"),
               "sensitivity": tyre_sensitivity_figure(run, f"{title} · ranking vs fuel assumption")}
    for suffix, fig in figures.items():
        fig.write_html(f"{stem}_{suffix}.html", include_plotlyjs="cdn", config={"displaylogo": False})
        print(f"Saved {stem}_{suffix}.html")


def main(year, place, session_name, name_a, name_b):
    import fastf1

    import data

    fastf1.set_log_level("ERROR")
    names = (name_a, name_b)
    run = analysis.load_and_compare(data.find_session(year, place, session_name), name_a, name_b)
    drivers = [lap["driver"] for lap in run["laps"]]
    colours = driver_colours([d["team_colour"] for d in drivers], [d["team_name"] for d in drivers])
    title = analysis.title_text(run, names)

    os.makedirs("output", exist_ok=True)
    stem = os.path.join("output", f"{year}_{place}_{session_name}_{name_a}_{name_b}".replace(" ", "_"))
    figures = {"telemetry": telemetry_figure(run["result"], names, colours, title),
               "track_map": track_map_figure(run["result"], run["corners"], names, colours, title)}
    for suffix, fig in figures.items():
        path = f"{stem}_{suffix}.html"
        fig.write_html(path, include_plotlyjs="cdn", config={"displaylogo": False})
        print(f"Saved {path}")
    print(f"Colours: {name_a} {colours[0]}, {name_b} {colours[1]}")


def race_pace_main(year, place):
    import data

    session = data.find_session(year, place, "Race")
    run = pace.analyse(data.race_lap_table(session), data.race_results(session))
    title = f"{session['year']} {session['location']} race · median pace"

    os.makedirs("output", exist_ok=True)
    stem = os.path.join("output", f"{year}_{place}_Race_pace".replace(" ", "_"))
    # The tower is a piece of HTML for the app; on its own it needs a page around it.
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title></head>'
            f'<body style="background: {BACKGROUND}; margin: 32px;">{classification_html(run["table"], title)}'
            f"</body></html>")
    with open(f"{stem}_classification.html", "w", encoding="utf-8") as file:
        file.write(page)
    pace_distribution_figure(run["table"], run["clean"], title).write_html(
        f"{stem}_distribution.html", include_plotlyjs="cdn", config={"displaylogo": False})
    print(f"Saved {stem}_classification.html and {stem}_distribution.html")


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026", "Brazil", "Qualifying", "NOR", "RUS"]
    try:
        if len(args) == 2:  # year and place only: race pace
            race_pace_main(int(args[0]), args[1])
        elif len(args) == 3 and args[2].lower() == "tyres":  # year, place, "tyres": tyre degradation
            tyres_main(int(args[0]), args[1])
        else:
            main(int(args[0]), args[1], args[2], args[3], args[4])
    except LookupError as error:
        sys.exit(str(error))
