#!/usr/bin/env python3
"""
dash_wards_app.py

Interactive Dash app to visualize ward placements with hover tooltips.
- Load folder with match_*_wards.csv or a single CSV
- Provide map image, grid size and filters
- Hover shows match_id, time, account_id, source file
- Export current view as interactive HTML

Usage:
  pip install dash pandas pillow plotly
  python3 dash_wards_app.py
  Open http://127.0.0.1:8050
"""
from pathlib import Path
import json
import math
from typing import List

import pandas as pd
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ---------- Helpers ----------
def read_points_from_path(in_path: str, ward_types: List[str]):
    p = Path(in_path)
    rows = []
    if not p.exists():
        return pd.DataFrame(rows)
    files = []
    if p.is_dir():
        files = sorted([f for f in p.glob("match_*_wards.csv")])
    else:
        files = [p]
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        for _, r in df.iterrows():
            # try x,y extraction, fallback to key parsing
            x = None; y = None
            for k in ("x","pos_x","posx","position_x"):
                if k in r and pd.notna(r[k]):
                    try:
                        x = float(r[k]); break
                    except Exception:
                        x = None
            for k in ("y","pos_y","posy","position_y"):
                if k in r and pd.notna(r[k]):
                    try:
                        y = float(r[k]); break
                    except Exception:
                        y = None
            if x is None or y is None:
                key = r.get("key") or r.get("pos_key") or ""
                if isinstance(key, str) and key:
                    import re
                    m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", key)
                    if m:
                        try:
                            x = float(m.group(1)); y = float(m.group(2))
                        except Exception:
                            pass
            if x is None or y is None:
                continue
            wtype = str(r.get("ward_type") or r.get("type") or "").lower()
            if ward_types and wtype not in ward_types:
                continue
            rows.append({
                "match_id": str(r.get("match_id") or r.get("match") or ""),
                "time": r.get("time"),
                "x": x,
                "y": y,
                "ward_type": wtype,
                "account_id": str(r.get("account_id") or ""),
                "player_slot": str(r.get("player_slot") or ""),
                "team_name": str(r.get("team_name") or ""),
                "radiant_or_dire": str(r.get("radiant_or_dire") or r.get("team") or "").lower(),
                "src_file": f.name
            })
    if not rows:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    # coerce numeric time
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df

def map_to_image_px(x: float, y: float, grid_size: int, img_w: int, img_h: int, flip_y: bool=False):
    max_index = grid_size - 1
    gx = max(0.0, min(float(x), float(max_index)))
    gy = max(0.0, min(float(y), float(max_index)))
    px = (gx / max_index) * (img_w - 1)
    py = (gy / max_index) * (img_h - 1)
    if flip_y:
        py = img_h - py
    return px, py

def build_figure(df: pd.DataFrame, map_image_path: str, grid_size: int, flip_y: bool,
                 color_by: str = "window", time_window: List[float]=None, team_side: str = None,
                 players: List[str]=None, max_points:int=5000):
    """
    df: DataFrame with columns x,y,time,match_id,account_id,src_file,radiant_or_dire
    color_by: "window" or "player" or "none"
    time_window: [start_min, end_min] in minutes (can be None)
    team_side: "radiant" or "dire" or None
    players: list of account_id strings to filter (None = all)
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        return fig

    # filter by team
    if team_side:
        df = df[df["radiant_or_dire"] == team_side.lower()]

    # filter by players
    if players:
        df = df[df["account_id"].isin(players)]

    # filter by time window (in minutes)
    if time_window:
        a_min, b_min = time_window
        def in_window(t):
            try:
                if pd.isna(t):
                    return False
                t_min = float(t) / 60.0
                if a_min is not None and t_min < a_min:
                    return False
                if b_min is not None and t_min >= b_min:
                    return False
                return True
            except Exception:
                return False
        df = df[df["time"].apply(in_window)]

    # load image to get size
    img = Image.open(map_image_path).convert("RGBA")
    img_w, img_h = img.size

    # map pixel coords
    pxs = []; pys = []; texts = []; colors = []
    # determine coloring groups
    if color_by == "player":
        groups = df["account_id"].fillna("unknown").astype(str).unique().tolist()
    else:
        # window: single group (color per input slice) -> we'll use hover group labels
        groups = ["all"]

    # Cap total points to avoid browser slowdown
    total = len(df)
    if total > max_points:
        df = df.sample(n=max_points, random_state=1).reset_index(drop=True)

    for _, r in df.iterrows():
        px, py = map_to_image_px(r["x"], r["y"], grid_size, img_w, img_h, flip_y=flip_y)
        pxs.append(px); pys.append(py)
        texts.append(f"match: {r['match_id']}<br>time(s): {r['time']}<br>acc: {r['account_id']}<br>file: {r['src_file']}")
        if color_by == "player":
            colors.append(str(r.get("account_id") or "unknown"))
        elif color_by == "none":
            colors.append("black")
        else:
            # simple single color; windows coloring could be implemented per-window if needed
            colors.append("red")

    fig = go.Figure()

    # background image
    fig.add_layout_image(
        dict(
            source=Path(map_image_path).absolute().as_uri(),
            xref="x", yref="y",
            x=0, y=0,
            sizex=img_w, sizey=img_h,
            sizing="stretch",
            opacity=1.0,
            layer="below"
        )
    )

    # scatter
    marker_kwargs = dict(size=10, line=dict(width=1, color="black"))
    if color_by == "player":
        # color by account_id category
        fig.add_trace(go.Scatter(
            x=pxs, y=pys, mode="markers",
            marker=dict(symbol="circle", sizemode="area", sizeref=1, **marker_kwargs),
            hoverinfo="text",
            hovertext=texts,
            marker_color=colors,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=pxs, y=pys, mode="markers",
            marker=dict(color=colors, **marker_kwargs),
            hoverinfo="text", hovertext=texts
        ))

    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[0, img_w])
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, range=[img_h, 0])  # invert y
    fig.update_layout(width=img_w, height=img_h, margin=dict(l=0,r=0,t=0,b=0))
    return fig

# ---------- Dash App ----------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H3("Wards Interactive Viewer (Dash + Plotly)"),
        html.Label("CSV folder or single CSV path:"),
        dcc.Input(id="input-folder", type="text", value="./data/obs_logs", style={"width":"100%"}),
        html.Br(), html.Br(),
        html.Label("Map image path:"),
        dcc.Input(id="input-map", type="text", value="images/map_738.jpg", style={"width":"100%"}),
        html.Br(), html.Br(),
        html.Label("Grid size:"),
        dcc.Input(id="input-grid", type="number", value=128, min=8, step=1),
        html.Br(), html.Br(),
        html.Label("Ward types (comma separated, empty=all):"),
        dcc.Input(id="input-ward-types", type="text", value="obs", style={"width":"100%"}),
        html.Br(), html.Br(),
        html.Label("Team side:"),
        dcc.Dropdown(id="input-team-side", options=[
            {"label":"All","value":""},{"label":"Radiant","value":"radiant"},{"label":"Dire","value":"dire"}], value=""),
        html.Br(),
        html.Label("Color by:"),
        dcc.RadioItems(id="input-color-by", options=[
            {"label":"Player (account_id)","value":"player"},
            {"label":"Window","value":"window"},
            {"label":"None","value":"none"}], value="player"),
        html.Br(),
        html.Label("Time window (minutes):"),
        dcc.RangeSlider(id="input-time-window", min=-5, max=60, step=0.5, value=[-1, 2],
                        marks={-5:"-5",0:"0",2:"2",5:"5",10:"10",20:"20",40:"40",60:"60"}),
        html.Div(id="time-window-label"),
        html.Br(),
        html.Button("Load data", id="btn-load", n_clicks=0),
        html.Button("Export HTML", id="btn-export", n_clicks=0, style={"marginLeft":"10px"}),
        html.Div(id="load-status", style={"marginTop":"8px","color":"green"}),
    ], style={"width":"320px","padding":"12px","display":"inline-block","verticalAlign":"top","borderRight":"1px solid #ddd"}),

    html.Div([
        dcc.Graph(id="ward-graph", style={"width":"100%", "height":"calc(100vh - 40px)"}),
    ], style={"display":"inline-block","width":"calc(100% - 360px)", "padding":"6px"}),

    # Store loaded data as JSON
    dcc.Store(id="store-points", data=""),
    dcc.Download(id="download-html"),
])

# ---------- Callbacks ----------
@app.callback(
    Output("load-status", "children"),
    Output("store-points", "data"),
    Input("btn-load", "n_clicks"),
    State("input-folder", "value"),
    State("input-ward-types", "value"),
)
def load_data(n_clicks, folder, ward_types_str):
    if n_clicks is None or n_clicks == 0:
        return "", ""
    wt = [w.strip().lower() for w in (ward_types_str or "").split(",") if w.strip()]
    df = read_points_from_path(folder, wt)
    if df.empty:
        return f"No points found in {folder}", ""
    # convert DataFrame to JSON for client-side storage
    data = df.to_json(orient="records", date_format="iso")
    return f"Loaded {len(df)} points from {folder}", data

@app.callback(
    Output("ward-graph", "figure"),
    Input("store-points", "data"),
    Input("input-map", "value"),
    Input("input-grid", "value"),
    Input("input-color-by", "value"),
    Input("input-time-window", "value"),
    Input("input-team-side", "value"),
)
def update_figure(store_data, map_image, grid_size, color_by, time_window, team_side):
    if not store_data:
        return go.Figure()
    df = pd.read_json(store_data, orient="records")
    # time_window is list [a,b] from slider
    a_min, b_min = (None, None)
    if isinstance(time_window, (list,tuple)) and len(time_window) == 2:
        a_min, b_min = float(time_window[0]), float(time_window[1])
    fig = build_figure(df, map_image, int(grid_size or 128), flip_y=False,
                       color_by=color_by, time_window=[a_min,b_min], team_side=team_side or None)
    return fig

@app.callback(
    Output("download-html", "data"),
    Input("btn-export", "n_clicks"),
    State("store-points", "data"),
    State("input-map", "value"),
    State("input-grid", "value"),
    State("input-color-by", "value"),
    State("input-time-window", "value"),
    State("input-team-side", "value"),
    prevent_initial_call=True
)
def export_html(n_clicks, store_data, map_image, grid_size, color_by, time_window, team_side):
    if not store_data:
        return None
    df = pd.read_json(store_data, orient="records")
    a_min, b_min = (None, None)
    if isinstance(time_window, (list,tuple)) and len(time_window) == 2:
        a_min, b_min = float(time_window[0]), float(time_window[1])
    fig = build_figure(df, map_image, int(grid_size or 128), flip_y=False,
                       color_by=color_by, time_window=[a_min,b_min], team_side=team_side or None)
    # write to HTML string
    html_str = fig.to_html(include_plotlyjs="cdn")
    filename = f"wards_view_export.html"
    return dcc.send_string(html_str, filename)

@app.callback(
    Output("time-window-label", "children"),
    Input("input-time-window", "value")
)
def show_time_window_label(val):
    if not val or len(val) != 2:
        return ""
    return f"Window: {val[0]} to {val[1]} minutes"

# ---------- Run ----------
if __name__ == "__main__":
    # start dev server; you can also set host/port, e.g. host='0.0.0.0', port=8050
    app.run(debug=True)