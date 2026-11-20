"""
charts.py: Plotly figures for a two-driver comparison.

Pipeline position:  data.py -> analysis.py -> charts.py -> app.py
Draws analysis.compare()'s result and calculates nothing new. It has no
Streamlit code, so the same figures work in the app and as standalone HTML:

    python charts.py                                    # 2026 Brazil Qualifying, NOR vs RUS
    python charts.py 2025 Monza Qualifying LEC HAM

writes two files into output/:
    ..._telemetry.html  speed (with corner numbers), time gap, throttle, brake
                        and gear, stacked on ONE shared lap-distance axis, so
                        zooming any of them zooms them all
    ..._track_map.html  the circuit, coloured by the faster driver in each
                        mini-sector

The look is a pit-wall screen, not a busy dashboard: dark, two high-contrast
driver colours, large labels, almost no gridlines.
"""

import os
import re
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import analysis

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


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026", "Brazil", "Qualifying", "NOR", "RUS"]
    try:
        main(int(args[0]), args[1], args[2], args[3], args[4])
    except LookupError as error:
        sys.exit(str(error))
