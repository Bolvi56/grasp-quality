# -*- coding: utf-8 -*-
"""
dashboard.py — Interactive Grasp Quality Dashboard
Reads from all_metrics.csv, no file generation.
Run: python dashboard.py  →  open http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import sample_colorscale
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
from pathlib import Path
import io
import base64

from config import (
    SUMMARY_DIR, MAIN_METRICS, ASCENDING_METRICS, FULL_NAMES, UNITS,
    OBLIQUE_NAMES, DIAMETERS, DEV_CONFIG_NAMES,
)

BASELINE_OBL = "Non"
BASELINE_DEV = "dev_p00"

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
df = pd.read_csv(SUMMARY_DIR / "all_metrics.csv")
df["diameter"] = df["diameter"].astype(int)

ALL_OBLIQUES  = sorted(df["oblique"].unique().tolist())
ALL_DIAMETERS = sorted(df["diameter"].unique().tolist())
ALL_DEVS      = sorted(df["dev"].unique().tolist())

METRIC_OPTIONS = [
    {"label": f"{m} — {FULL_NAMES.get(m, m)} [{UNITS.get(m,'—')}]", "value": m}
    for m in MAIN_METRICS if m in df.columns
]

COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.D3

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def make_label(row):
    return f"{row['oblique']} d{int(row['diameter']):02d}mm {row['dev']}"

def pct_vs_baseline(df_in, baseline_obl=BASELINE_OBL):
    cols = [m for m in MAIN_METRICS if m in df_in.columns]
    result = df_in.copy()
    for _, row in df_in.iterrows():
        ref = df_in[
            (df_in["oblique"]  == baseline_obl) &
            (df_in["diameter"] == row["diameter"]) &
            (df_in["dev"]      == row["dev"])
        ]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        for col in cols:
            bv = ref_row[col]
            if abs(bv) > 1e-12:
                result.loc[row.name, col] = (row[col] - bv) / abs(bv) * 100
            else:
                result.loc[row.name, col] = 0.0
    return result

PLOTLY_CMAP = "RdYlGn"
RANK_CMAP   = "Blues"

# ══════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Grasp Quality Dashboard",
)

# ── Sidebar ───────────────────────────────────────────────────
sidebar = dbc.Card([
    html.H5("⚙️ Controls", className="text-info fw-bold mb-3"),

    html.Label("Chart type", className="text-light fw-bold"),
    dcc.Dropdown(
        id="chart-type",
        options=[
            {"label": "📊 Heatmap — raw values",        "value": "heatmap_raw"},
            {"label": "📊 Heatmap — % vs Non",          "value": "heatmap_pct"},
            {"label": "📈 Line chart",                   "value": "line"},
            {"label": "🏆 Top-N vs Non",                "value": "topn"},
            {"label": "🥇 Best config per row vs Non",  "value": "best_per_row"},
            {"label": "🔢 Data table",                  "value": "table"},
            {"label": "📦 Box plot",                    "value": "box"},
            {"label": "🔵 Scatter plot",                "value": "scatter"},
        ],
        value="heatmap_raw",
        clearable=False,
        className="mb-3",
    ),

    html.Hr(className="border-secondary"),
    html.Label("Metrics (Y axis / columns)", className="text-light fw-bold"),
    dcc.Dropdown(
        id="sel-metrics",
        options=METRIC_OPTIONS,
        value=MAIN_METRICS[:6],
        multi=True,
        className="mb-3",
    ),

    html.Label("Group rows by", className="text-light fw-bold"),
    dcc.RadioItems(
        id="group-by",
        options=[
            {"label": " Oblique",           "value": "oblique"},
            {"label": " Diameter",          "value": "diameter"},
            {"label": " Deviation",         "value": "dev"},
            {"label": " Oblique × Diam",    "value": "obl_diam"},
            {"label": " Oblique × Dev",     "value": "obl_dev"},
            {"label": " All combinations",  "value": "all"},
        ],
        value="oblique",
        className="text-light mb-3",
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "block"},
    ),

    html.Hr(className="border-secondary"),
    html.Label("Filter — Oblique", className="text-light fw-bold"),
    dcc.Dropdown(
        id="filter-oblique",
        options=[{"label": o, "value": o} for o in ALL_OBLIQUES],
        value=ALL_OBLIQUES,
        multi=True,
        className="mb-2",
    ),

    html.Label("Filter — Diameter (mm)", className="text-light fw-bold"),
    dcc.Dropdown(
        id="filter-diam",
        options=[{"label": str(d), "value": d} for d in ALL_DIAMETERS],
        value=ALL_DIAMETERS,
        multi=True,
        className="mb-2",
    ),

    html.Label("Filter — Deviation", className="text-light fw-bold"),
    dcc.Dropdown(
        id="filter-dev",
        options=[{"label": d, "value": d} for d in ALL_DEVS],
        value=ALL_DEVS,
        multi=True,
        className="mb-3",
    ),

    html.Hr(className="border-secondary"),

    # Top-N options (visible only for topn)
    html.Div(id="topn-options", children=[
        html.Label("Top-N configurations", className="text-light fw-bold"),
        dcc.Slider(id="top-n", min=3, max=20, step=1, value=5,
                   marks={i: str(i) for i in range(3, 21, 2)},
                   className="mb-2"),
        html.Label("Group top-N by", className="text-light fw-bold"),
        dcc.RadioItems(
            id="topn-group",
            options=[
                {"label": " Oblique",  "value": "oblique"},
                {"label": " Diameter", "value": "diameter"},
                {"label": " Deviation","value": "dev"},
            ],
            value="oblique",
            className="text-light mb-3",
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block"},
        ),
    ]),

    # Scatter options
    html.Div(id="scatter-options", children=[
        html.Label("Scatter X metric", className="text-light fw-bold"),
        dcc.Dropdown(
            id="scatter-x",
            options=METRIC_OPTIONS,
            value=MAIN_METRICS[0] if MAIN_METRICS else None,
            clearable=False,
            className="mb-2",
        ),
        html.Label("Scatter Y metric", className="text-light fw-bold"),
        dcc.Dropdown(
            id="scatter-y",
            options=METRIC_OPTIONS,
            value=MAIN_METRICS[1] if len(MAIN_METRICS) > 1 else None,
            clearable=False,
            className="mb-2",
        ),
        html.Label("Color by", className="text-light fw-bold"),
        dcc.Dropdown(
            id="scatter-color",
            options=[
                {"label": "Oblique",  "value": "oblique"},
                {"label": "Diameter", "value": "diameter"},
                {"label": "Deviation","value": "dev"},
            ],
            value="oblique",
            clearable=False,
            className="mb-3",
        ),
    ]),

    # Best-per-row options
    html.Div(id="bpr-options", children=[
        html.Label("Rows fixed factor (Non shows best Non per row)",
                   className="text-light fw-bold"),
        dcc.RadioItems(
            id="bpr-row-factor",
            options=[
                {"label": " Diameter  → best oblique+dev per row", "value": "diameter"},
                {"label": " Oblique   → best diam+dev per row",    "value": "oblique"},
                {"label": " Deviation → best oblique+diam per row","value": "dev"},
            ],
            value="diameter",
            className="text-light mb-2",
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block"},
        ),
        html.Label("Fix Deviation (or All)", className="text-light fw-bold"),
        dcc.Dropdown(
            id="bpr-fix-dev",
            options=[{"label": "All deviations", "value": "__all__"}] +
                    [{"label": d, "value": d} for d in ALL_DEVS],
            value="__all__",
            clearable=False,
            className="mb-2",
        ),
        html.Label("Fix Oblique (or All)", className="text-light fw-bold"),
        dcc.Dropdown(
            id="bpr-fix-oblique",
            options=[{"label": "All obliques", "value": "__all__"}] +
                    [{"label": o, "value": o} for o in ALL_OBLIQUES],
            value="__all__",
            clearable=False,
            className="mb-2",
        ),
        html.Label("Fix Diameter (or All)", className="text-light fw-bold"),
        dcc.Dropdown(
            id="bpr-fix-diam",
            options=[{"label": "All diameters", "value": "__all__"}] +
                    [{"label": str(d), "value": d} for d in ALL_DIAMETERS],
            value="__all__",
            clearable=False,
            className="mb-2",
        ),
        html.Label("Cell font size", className="text-light fw-bold"),
        dcc.Slider(id="bpr-cell-fs", min=6, max=16, step=1, value=9,
                   marks={6:"6", 9:"9", 12:"12", 16:"16"},
                   className="mb-3"),
    ]),

    html.Hr(className="border-secondary"),
    html.Label("Sort rows", className="text-light fw-bold"),
    dcc.RadioItems(
        id="sort-by",
        options=[
            {"label": " Default order", "value": "default"},
            {"label": " Sort by first metric ↑", "value": "asc"},
            {"label": " Sort by first metric ↓", "value": "desc"},
        ],
        value="default",
        className="text-light mb-3",
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "block"},
    ),

    html.Hr(className="border-secondary"),
    html.Label("Appearance", className="text-light fw-bold"),

    html.Div([
        html.Span("Font size", className="text-light me-2"),
        dcc.Slider(id="font-size", min=8, max=24, step=1, value=12,
                   marks={8:"8", 12:"12", 16:"16", 20:"20", 24:"24"}),
    ], className="mb-2"),

    html.Div([
        html.Span("Figure width", className="text-light me-2"),
        dcc.Slider(id="fig-width", min=600, max=2400, step=100, value=1200,
                   marks={600:"600", 1200:"1200", 1800:"1800", 2400:"2400"}),
    ], className="mb-2"),

    html.Div([
        html.Span("Figure height", className="text-light me-2"),
        dcc.Slider(id="fig-height", min=400, max=1800, step=100, value=700,
                   marks={400:"400", 700:"700", 1200:"1200", 1800:"1800"}),
    ], className="mb-3"),

], body=True, color="dark", className="h-100 overflow-auto",
   style={"minWidth": "280px", "maxWidth": "320px"})

# ── Export panel ──────────────────────────────────────────────
export_panel = dbc.Card([
    html.H6("💾 Export", className="text-info fw-bold"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="export-format",
            options=[
                {"label": "PNG",  "value": "png"},
                {"label": "SVG",  "value": "svg"},
                {"label": "PDF",  "value": "pdf"},
                {"label": "CSV",  "value": "csv"},
            ],
            value="png", clearable=False,
        ), width=3),
        dbc.Col(dcc.Dropdown(
            id="export-scale",
            options=[{"label": f"{s}x DPI", "value": s} for s in [1, 2, 3, 4]],
            value=2, clearable=False,
        ), width=2),
        dbc.Col(dbc.Button("⬇ Download", id="btn-export", color="info",
                           size="sm", className="w-100"), width=2),
        dbc.Col(html.Div(id="export-status", className="text-light small pt-1"), width=5),
    ], align="center"),
    dcc.Download(id="download"),
], body=True, color="dark", className="mt-2 py-2")

# ── Main layout ───────────────────────────────────────────────
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H3("🖐 Grasp Quality Dashboard",
                        className="text-info fw-bold my-2"), width=8),
        dbc.Col(html.Div(id="data-info",
                         className="text-secondary small text-end pt-3"), width=4),
    ]),
    html.Hr(className="border-secondary mt-0"),

    # Body
    dbc.Row([
        # Sidebar
        dbc.Col(sidebar, width="auto"),

        # Main content
        dbc.Col([
            export_panel,
            dbc.Card([
                dcc.Graph(
                    id="main-graph",
                    config={"displayModeBar": True,
                            "toImageButtonOptions": {"format": "png", "scale": 3},
                            "modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                    style={"height": "75vh"},
                ),
            ], body=True, color="dark", className="mt-2"),

            # Data table (shown only for table view)
            html.Div(id="table-container", className="mt-2"),
        ], className="flex-grow-1"),
    ], className="flex-nowrap g-2"),

], fluid=True, className="bg-dark min-vh-100 pb-4")


# ══════════════════════════════════════════════════════════════
#  CALLBACKS — show/hide panels
# ══════════════════════════════════════════════════════════════

@app.callback(
    Output("topn-options",    "style"),
    Output("scatter-options", "style"),
    Output("bpr-options",     "style"),
    Input("chart-type", "value"),
)
def toggle_panels(chart_type):
    show   = {"display": "block"}
    hide   = {"display": "none"}
    topn    = show if chart_type == "topn"         else hide
    scatter = show if chart_type == "scatter"      else hide
    bpr     = show if chart_type == "best_per_row" else hide
    return topn, scatter, bpr


@app.callback(
    Output("data-info", "children"),
    Input("filter-oblique", "value"),
    Input("filter-diam",    "value"),
    Input("filter-dev",     "value"),
)
def update_info(obls, diams, devs):
    n = len(df[
        df["oblique"].isin(obls or []) &
        df["diameter"].isin(diams or []) &
        df["dev"].isin(devs or [])
    ])
    return f"{n} configurations | {len(df)} total"


# ══════════════════════════════════════════════════════════════
#  MAIN GRAPH CALLBACK
# ══════════════════════════════════════════════════════════════

@app.callback(
    Output("main-graph",      "figure"),
    Output("table-container", "children"),
    Input("chart-type",     "value"),
    Input("sel-metrics",    "value"),
    Input("group-by",       "value"),
    Input("filter-oblique", "value"),
    Input("filter-diam",    "value"),
    Input("filter-dev",     "value"),
    Input("sort-by",        "value"),
    Input("top-n",          "value"),
    Input("topn-group",     "value"),
    Input("scatter-x",      "value"),
    Input("scatter-y",      "value"),
    Input("scatter-color",  "value"),
    Input("font-size",      "value"),
    Input("fig-width",      "value"),
    Input("fig-height",     "value"),
    Input("bpr-row-factor", "value"),
    Input("bpr-fix-dev",    "value"),
    Input("bpr-fix-oblique","value"),
    Input("bpr-fix-diam",   "value"),
    Input("bpr-cell-fs",    "value"),
)
def update_graph(chart_type, metrics, group_by,
                 f_obl, f_diam, f_dev, sort_by,
                 top_n, topn_group,
                 sc_x, sc_y, sc_color,
                 font_size, fig_w, fig_h,
                 bpr_row_factor, bpr_fix_dev, bpr_fix_oblique,
                 bpr_fix_diam, bpr_cell_fs):

    # ── Filter ────────────────────────────────────────────────
    dff = df.copy()
    if f_obl:  dff = dff[dff["oblique"].isin(f_obl)]
    if f_diam: dff = dff[dff["diameter"].isin(f_diam)]
    if f_dev:  dff = dff[dff["dev"].isin(f_dev)]

    metrics = metrics or MAIN_METRICS[:4]
    metrics = [m for m in metrics if m in dff.columns]

    if dff.empty or not metrics:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark",
                          title="No data for current filters")
        return fig, None

    layout_kw = dict(
        template="plotly_dark",
        font=dict(size=font_size),
        width=fig_w,
        height=fig_h,
        margin=dict(l=120, r=80, t=60, b=100),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
    )

    # ── Build row labels ──────────────────────────────────────
    def get_grouped(dff, group_by):
        if group_by == "oblique":
            g = dff.groupby("oblique")[metrics].mean().reset_index()
            g["_label"] = g["oblique"].astype(str)
        elif group_by == "diameter":
            g = dff.groupby("diameter")[metrics].mean().reset_index()
            g["_label"] = "d" + g["diameter"].astype(str) + "mm"
        elif group_by == "dev":
            g = dff.groupby("dev")[metrics].mean().reset_index()
            g["_label"] = g["dev"].astype(str)
        elif group_by == "obl_diam":
            g = dff.groupby(["oblique","diameter"])[metrics].mean().reset_index()
            g["_label"] = g["oblique"] + " d" + g["diameter"].astype(str) + "mm"
        elif group_by == "obl_dev":
            g = dff.groupby(["oblique","dev"])[metrics].mean().reset_index()
            g["_label"] = g["oblique"] + " | " + g["dev"]
        else:  # all
            g = dff.copy()
            g["_label"] = (g["oblique"] + " d" + g["diameter"].astype(str)
                           + "mm " + g["dev"])
        return g

    # ── HEATMAP RAW ───────────────────────────────────────────
    if chart_type == "heatmap_raw":
        g = get_grouped(dff, group_by)
        if sort_by == "asc":
            g = g.sort_values(metrics[0], ascending=True)
        elif sort_by == "desc":
            g = g.sort_values(metrics[0], ascending=False)

        mat = g[metrics].values.astype(float)

        # Normalize per column for color (green = better)
        norm = np.zeros_like(mat)
        for j, m in enumerate(metrics):
            col = mat[:, j]
            vmin, vmax = np.nanmin(col), np.nanmax(col)
            if vmax - vmin > 1e-12:
                n_ = (col - vmin) / (vmax - vmin)
                norm[:, j] = 1 - n_ if m in ASCENDING_METRICS else n_
            else:
                norm[:, j] = 0.5

        text = [[f"{mat[i,j]:.3g}" for j in range(len(metrics))]
                for i in range(len(g))]

        fig = go.Figure(go.Heatmap(
            z=norm, x=metrics, y=g["_label"].tolist(),
            text=text, texttemplate="%{text}",
            colorscale=PLOTLY_CMAP, zmin=0, zmax=1,
            colorbar=dict(title="Better →",
                          tickvals=[0, 0.5, 1],
                          ticktext=["Worse", "Mid", "Better"]),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title="Metrics Heatmap — Raw Values",
            xaxis_title="Metric", yaxis_title="Configuration",
        )

    # ── HEATMAP PCT ───────────────────────────────────────────
    elif chart_type == "heatmap_pct":
        g = get_grouped(dff, group_by)
        g_non = get_grouped(
            df[df["oblique"] == BASELINE_OBL], group_by
        )

        # Calculate pct change vs Non
        pct_mat = np.zeros((len(g), len(metrics)))
        for j, m in enumerate(metrics):
            for i, row in g.iterrows():
                lbl = g.loc[i, "_label"]
                ref_rows = g_non[g_non["_label"] == lbl]
                if ref_rows.empty:
                    ref_rows = g_non
                base = float(ref_rows[m].mean())
                val  = float(row[m])
                pct_mat[g.index.get_loc(i), j] = (
                    (val - base) / abs(base) * 100 if abs(base) > 1e-12 else 0.0
                )

        # Color: positive = better (respecting direction)
        color_mat = np.zeros_like(pct_mat)
        for j, m in enumerate(metrics):
            for i in range(len(g)):
                v = pct_mat[i, j]
                color_mat[i, j] = -v if m in ASCENDING_METRICS else v

        norm_mat = np.zeros_like(color_mat)
        for j in range(len(metrics)):
            mx = float(np.nanmax(np.abs(color_mat[:, j])))
            if mx > 1e-9:
                norm_mat[:, j] = np.clip(color_mat[:, j] / mx, -1, 1)

        if sort_by == "asc":
            order = np.argsort(pct_mat[:, 0])
            g = g.iloc[order]
            pct_mat   = pct_mat[order]
            norm_mat  = norm_mat[order]
        elif sort_by == "desc":
            order = np.argsort(-pct_mat[:, 0])
            g = g.iloc[order]
            pct_mat   = pct_mat[order]
            norm_mat  = norm_mat[order]

        text = [[f"{pct_mat[i,j]:+.1f}%" for j in range(len(metrics))]
                for i in range(len(g))]

        fig = go.Figure(go.Heatmap(
            z=norm_mat, x=metrics, y=g["_label"].tolist(),
            text=text, texttemplate="%{text}",
            colorscale=PLOTLY_CMAP, zmin=-1, zmax=1,
            colorbar=dict(title="vs Non",
                          tickvals=[-1, 0, 1],
                          ticktext=["Worse", "Equal", "Better"]),
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"% Change vs {BASELINE_OBL}",
            xaxis_title="Metric", yaxis_title="Configuration",
        )

    # ── LINE CHART ────────────────────────────────────────────
    elif chart_type == "line":
        g = get_grouped(dff, group_by)
        if sort_by == "asc":
            g = g.sort_values(metrics[0], ascending=True)
        elif sort_by == "desc":
            g = g.sort_values(metrics[0], ascending=False)

        fig = go.Figure()
        for ci, m in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=g["_label"], y=g[m],
                mode="lines+markers",
                name=f"{m} [{UNITS.get(m,'—')}]",
                line=dict(color=COLORS[ci % len(COLORS)], width=2),
                marker=dict(size=7),
                hovertemplate=f"<b>%{{x}}</b><br>{m}: %{{y:.4g}}<extra></extra>",
            ))
        fig.update_layout(
            title="Metric Values by Configuration",
            xaxis_title="Configuration",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )

    # ── TOP-N VS NON ──────────────────────────────────────────
    elif chart_type == "topn":
        cols = [m for m in metrics if m in dff.columns]

        # Non reference row per diameter+dev
        non_global = df[df["oblique"] == BASELINE_OBL][cols].mean()

        # Build improvement table
        rows_imp = []
        for _, row in dff.iterrows():
            if row["oblique"] == BASELINE_OBL:
                continue
            ref = df[
                (df["oblique"]  == BASELINE_OBL) &
                (df["diameter"] == row["diameter"]) &
                (df["dev"]      == row["dev"])
            ]
            if ref.empty:
                continue
            ref_row = ref.iloc[0]
            entry = {"_label": make_label(row),
                     "oblique": row["oblique"],
                     "diameter": row["diameter"],
                     "dev": row["dev"]}
            for col in cols:
                bv = ref_row[col]
                rv = row[col]
                pct = (rv - bv) / abs(bv) * 100 if abs(bv) > 1e-12 else 0.0
                imp = -pct if col in ASCENDING_METRICS else pct
                entry[f"imp_{col}"] = imp
                entry[f"pct_{col}"] = pct
                entry[f"val_{col}"] = rv
            rows_imp.append(entry)

        if not rows_imp:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              title="No data for Top-N")
            return fig, None

        df_imp = pd.DataFrame(rows_imp)

        # Select top_n per metric, then take most frequent
        from collections import Counter
        per_metric_top = {}
        for col in cols:
            per_metric_top[col] = (
                df_imp.nlargest(top_n, f"imp_{col}")["_label"].tolist()
            )
        all_lbls   = [l for lbls in per_metric_top.values() for l in lbls]
        counts     = Counter(all_lbls)
        selected   = sorted(set(all_lbls),
                            key=lambda l: (-counts[l], l))[:top_n]

        # Build matrix
        rows_plot  = [f"{BASELINE_OBL} (ref)"] + selected
        pct_matrix = np.zeros((len(rows_plot), len(cols)))
        val_matrix = np.zeros_like(pct_matrix)
        col_matrix = np.zeros_like(pct_matrix)

        # Baseline row
        for j, col in enumerate(cols):
            pct_matrix[0, j] = 0.0
            val_matrix[0, j] = float(non_global[col])

        for i, lbl in enumerate(selected, 1):
            row_data = df_imp[df_imp["_label"] == lbl].iloc[0]
            for j, col in enumerate(cols):
                pct_matrix[i, j] = row_data[f"pct_{col}"]
                val_matrix[i, j] = row_data[f"val_{col}"]
                imp = row_data[f"imp_{col}"]
                col_matrix[i, j] = imp

        # Normalize color
        norm_mat = np.full_like(col_matrix, 0.5)
        for j in range(len(cols)):
            mx = float(np.nanmax(np.abs(col_matrix[1:, j])))
            if mx > 1e-9:
                norm_mat[1:, j] = np.clip(col_matrix[1:, j] / mx * 0.5 + 0.5, 0, 1)

        text = []
        for i in range(len(rows_plot)):
            row_t = []
            for j in range(len(cols)):
                if i == 0:
                    row_t.append(f"{val_matrix[i,j]:.3g}<br>(ref)")
                else:
                    sign = "+" if pct_matrix[i,j] >= 0 else ""
                    row_t.append(f"{val_matrix[i,j]:.3g}<br>({sign}{pct_matrix[i,j]:.1f}%)")
            text.append(row_t)

        fig = go.Figure(go.Heatmap(
            z=norm_mat, x=cols, y=rows_plot,
            text=text, texttemplate="%{text}",
            colorscale=PLOTLY_CMAP, zmin=0, zmax=1,
            colorbar=dict(title="vs Non",
                          tickvals=[0, 0.5, 1],
                          ticktext=["Worse", "Ref", "Better"]),
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Top-{top_n} configurations vs {BASELINE_OBL}",
            xaxis_title="Metric", yaxis_title="Configuration",
        )

    # ── BOX PLOT ──────────────────────────────────────────────
    elif chart_type == "box":
        fig = go.Figure()
        group_col = (group_by if group_by in ["oblique","diameter","dev"]
                     else "oblique")
        for ci, m in enumerate(metrics):
            for gi, gval in enumerate(dff[group_col].unique()):
                sub = dff[dff[group_col] == gval][m].dropna()
                fig.add_trace(go.Box(
                    y=sub, name=f"{gval} — {m}",
                    marker_color=COLORS[(ci * 7 + gi) % len(COLORS)],
                    boxmean=True,
                    hovertemplate=f"<b>{gval}</b><br>{m}: %{{y:.4g}}<extra></extra>",
                ))
        fig.update_layout(
            title="Distribution by Group",
            yaxis_title="Value",
            boxmode="group",
        )

    # ── SCATTER ───────────────────────────────────────────────
    elif chart_type == "scatter":
        sc_x = sc_x or metrics[0]
        sc_y = sc_y or metrics[min(1, len(metrics)-1)]
        dff2 = dff[[sc_x, sc_y, sc_color, "oblique",
                    "diameter", "dev"]].dropna()
        dff2["_label"] = dff2.apply(make_label, axis=1)

        unique_vals = dff2[sc_color].unique()
        color_map   = {v: COLORS[i % len(COLORS)]
                       for i, v in enumerate(unique_vals)}

        fig = go.Figure()
        for gval in unique_vals:
            sub = dff2[dff2[sc_color] == gval]
            fig.add_trace(go.Scatter(
                x=sub[sc_x], y=sub[sc_y],
                mode="markers",
                name=str(gval),
                marker=dict(color=color_map[gval], size=9, opacity=0.85),
                text=sub["_label"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{sc_x}: %{{x:.4g}}<br>"
                    f"{sc_y}: %{{y:.4g}}<extra></extra>"
                ),
            ))
        fig.update_layout(
            title=f"Scatter: {sc_x} vs {sc_y}",
            xaxis_title=f"{sc_x} [{UNITS.get(sc_x,'—')}]",
            yaxis_title=f"{sc_y} [{UNITS.get(sc_y,'—')}]",
        )

    # ── TABLE ─────────────────────────────────────────────────
    elif chart_type == "table":
        g = get_grouped(dff, group_by)
        if sort_by == "asc":
            g = g.sort_values(metrics[0], ascending=True)
        elif sort_by == "desc":
            g = g.sort_values(metrics[0], ascending=False)

        display_cols = ["_label"] + metrics
        g_show = g[display_cols].rename(columns={"_label": "Configuration"})
        for m in metrics:
            g_show[m] = g_show[m].apply(lambda x: f"{x:.4g}")

        table = dash_table.DataTable(
            data=g_show.to_dict("records"),
            columns=[{"name": c, "id": c} for c in g_show.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#2D2B6B", "color": "white",
                          "fontWeight": "bold", "fontSize": f"{font_size}px"},
            style_cell={"backgroundColor": "#1e1e2e", "color": "white",
                        "fontSize": f"{font_size}px",
                        "textAlign": "center", "padding": "6px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": "#252535"}
            ],
            sort_action="native",
            filter_action="native",
            page_size=30,
        )

        fig = go.Figure()
        fig.update_layout(template="plotly_dark",
                          title="Use the table below ↓")
        return fig, table

    # ── BEST CONFIG PER ROW VS NON ────────────────────────────
    elif chart_type == "best_per_row":
        cols = metrics

        # Apply fixed filters to working dataset (full df, not filtered dff)
        dw = df.copy()
        if bpr_fix_dev     != "__all__": dw = dw[dw["dev"]      == bpr_fix_dev]
        if bpr_fix_oblique != "__all__": dw = dw[dw["oblique"]  == bpr_fix_oblique]
        if bpr_fix_diam    != "__all__": dw = dw[dw["diameter"] == bpr_fix_diam]

        # Row setup
        if bpr_row_factor == "diameter":
            row_vals   = sorted(dw["diameter"].unique())
            row_labels = [f"d{int(v):02d}mm" for v in row_vals]
            def cfg_lbl(r): return f"{r['oblique']} {r['dev']}"
            def get_non_for_row(dw_, rv):
                return dw_[(dw_["oblique"] == BASELINE_OBL) & (dw_["diameter"] == rv)]
            def get_cands_for_row(dw_, rv):
                return dw_[(dw_["oblique"] != BASELINE_OBL) & (dw_["diameter"] == rv)]

        elif bpr_row_factor == "oblique":
            row_vals   = [o for o in OBLIQUE_NAMES
                          if o in dw["oblique"].unique() and o != BASELINE_OBL]
            row_labels = list(row_vals)
            def cfg_lbl(r): return f"d{int(r['diameter']):02d}mm {r['dev']}"
            def get_non_for_row(dw_, rv):
                return dw_[dw_["oblique"] == BASELINE_OBL]
            def get_cands_for_row(dw_, rv):
                return dw_[(dw_["oblique"] == rv) & (dw_["oblique"] != BASELINE_OBL)]

        else:  # dev
            row_vals   = [d for d in DEV_CONFIG_NAMES if d in dw["dev"].unique()]
            row_labels = list(row_vals)
            def cfg_lbl(r): return f"{r['oblique']} d{int(r['diameter']):02d}mm"
            def get_non_for_row(dw_, rv):
                return dw_[(dw_["oblique"] == BASELINE_OBL) & (dw_["dev"] == rv)]
            def get_cands_for_row(dw_, rv):
                return dw_[(dw_["oblique"] != BASELINE_OBL) & (dw_["dev"] == rv)]

        all_rows  = [f"{BASELINE_OBL} (best)"] + row_labels
        n_r, n_c  = len(all_rows), len(cols)

        val_mat      = np.full((n_r, n_c), np.nan)
        pct_mat      = np.zeros((n_r, n_c))
        imp_mat      = np.zeros((n_r, n_c))
        text_mat     = [[""] * n_c for _ in range(n_r)]
        non_best_val = np.full(n_c, np.nan)

        # ── Row 0: best Non per metric (global best Non) ──────
        non_all = dw[dw["oblique"] == BASELINE_OBL]
        for j, col in enumerate(cols):
            if non_all.empty or col not in non_all.columns:
                text_mat[0][j] = "—"
                continue
            ascending = col in ASCENDING_METRICS
            if ascending:
                best_idx = non_all[col].idxmin()
            else:
                best_idx = non_all[col].idxmax()
            best_row  = non_all.loc[best_idx]
            best_val  = float(best_row[col])
            non_best_val[j] = best_val
            val_mat[0, j]   = best_val
            if bpr_row_factor == "dev":
                ctx = f"{best_row['oblique']} d{int(best_row['diameter']):02d}mm"
            else:
                ctx = f"d{int(best_row['diameter']):02d}mm {best_row['dev']}"
            text_mat[0][j] = f"<b>{best_val:.3g}</b><br>(best Non)<br>({ctx})"

        # ── Rows 1..N: best non-Non config per row value ──────
        for i, (rv, rl) in enumerate(zip(row_vals, row_labels), 1):
            non_rv = get_non_for_row(dw, rv)
            cands  = get_cands_for_row(dw, rv)

            for j, col in enumerate(cols):
                ascending = col in ASCENDING_METRICS

                if not non_rv.empty and col in non_rv.columns:
                    if ascending:
                        ref_idx = non_rv[col].idxmin()
                    else:
                        ref_idx = non_rv[col].idxmax()
                    base_val = float(non_rv.loc[ref_idx, col])
                else:
                    base_val = float(non_best_val[j]) if not np.isnan(non_best_val[j]) else np.nan

                if cands.empty or col not in cands.columns or np.isnan(base_val):
                    text_mat[i][j] = "—"
                    continue

                if abs(base_val) > 1e-12:
                    raw_pct = (cands[col] - base_val) / abs(base_val) * 100
                    imp_ser = -raw_pct if ascending else raw_pct
                else:
                    raw_pct = pd.Series(np.zeros(len(cands)), index=cands.index)
                    imp_ser = raw_pct.copy()

                best_idx = imp_ser.idxmax()
                best_row = cands.loc[best_idx]
                best_val = float(best_row[col])
                best_pct = float(raw_pct[best_idx])
                best_imp = float(imp_ser[best_idx])
                cfg      = cfg_lbl(best_row)

                val_mat[i, j]  = best_val
                pct_mat[i, j]  = best_pct
                imp_mat[i, j]  = best_imp
                sign = "+" if best_pct >= 0 else ""
                text_mat[i][j] = (
                    f"<b>{best_val:.3g}</b><br>"
                    f"({sign}{best_pct:.1f}%)<br>"
                    f"({cfg})"
                )

        # ── Normalize color ───────────────────────────────────
        norm_mat = np.full((n_r, n_c), 0.5)

        # Non ref row: color based on Non quality across row_vals
        non_row_vals = np.zeros((len(row_vals), n_c))
        for i, rv in enumerate(row_vals):
            non_rv = get_non_for_row(dw, rv)
            for j, col in enumerate(cols):
                if not non_rv.empty and col in non_rv.columns:
                    non_row_vals[i, j] = float(non_rv[col].mean())

        for j, col in enumerate(cols):
            col_vals = non_row_vals[:, j]
            vmin, vmax = np.nanmin(col_vals), np.nanmax(col_vals)
            if vmax - vmin > 1e-12:
                nc = (col_vals - vmin) / (vmax - vmin)
                norm_mat[0, j] = float(np.nanmax(1 - nc if col in ASCENDING_METRICS else nc))
            else:
                norm_mat[0, j] = 0.5

        # Candidate rows: based on improvement score
        for j in range(n_c):
            mx = float(np.nanmax(np.abs(imp_mat[1:, j])))
            if mx > 1e-9:
                norm_mat[1:, j] = np.clip(imp_mat[1:, j] / mx * 0.5 + 0.5, 0.0, 1.0)

        # ── Gold borders for cells better than Non ────────────
        shapes = [dict(
            type="line",
            x0=-0.5, x1=n_c - 0.5,
            y0=0.5,  y1=0.5,
            line=dict(color="white", width=3, dash="dash"),
        )]
        for i_r in range(len(row_vals)):
            for j_c in range(n_c):
                if imp_mat[i_r + 1, j_c] > 0:
                    shapes.append(dict(
                        type="rect",
                        x0=j_c - 0.48, x1=j_c + 0.48,
                        y0=(i_r + 1) - 0.48, y1=(i_r + 1) + 0.48,
                        line=dict(color="#F5D547", width=2.5),
                        fillcolor="rgba(0,0,0,0)",
                    ))

        fix_parts = []
        if bpr_fix_dev     != "__all__": fix_parts.append(f"dev={bpr_fix_dev}")
        if bpr_fix_oblique != "__all__": fix_parts.append(f"obl={bpr_fix_oblique}")
        if bpr_fix_diam    != "__all__": fix_parts.append(f"d={bpr_fix_diam}mm")
        fix_str = " | " + ", ".join(fix_parts) if fix_parts else ""

        cell_fs = bpr_cell_fs or 9

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=norm_mat,
            x=cols,
            y=all_rows,
            text=text_mat,
            texttemplate="%{text}",
            textfont=dict(size=cell_fs),
            colorscale=PLOTLY_CMAP,
            zmin=0, zmax=1,
            colorbar=dict(
                title="Score",
                tickvals=[0, 0.5, 1],
                ticktext=["Worse", "Mid", "Better"],
                tickfont=dict(size=font_size),
            ),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            shapes=shapes,
            title=dict(
                text=(f"Best config vs Non (best Non shown) | "
                      f"rows={bpr_row_factor}{fix_str}"),
                font=dict(size=font_size + 4, color="white"),
            ),
            xaxis=dict(title="Metric", tickfont=dict(size=font_size), side="bottom"),
            yaxis=dict(title=bpr_row_factor.capitalize(),
                       tickfont=dict(size=font_size), autorange="reversed"),
            **layout_kw,
        )
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=max(font_size-2, 8)))
        return fig, None

    else:
        fig = go.Figure()

    # ── Common layout ─────────────────────────────────────────
    fig.update_layout(**layout_kw)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=max(font_size-2, 8)))
    fig.update_yaxes(tickfont=dict(size=max(font_size-2, 8)))

    return fig, None


# ══════════════════════════════════════════════════════════════
#  EXPORT CALLBACK
# ══════════════════════════════════════════════════════════════

@app.callback(
    Output("download",      "data"),
    Output("export-status", "children"),
    Input("btn-export", "n_clicks"),
    State("main-graph",    "figure"),
    State("export-format", "value"),
    State("export-scale",  "value"),
    State("chart-type",    "value"),
    State("sel-metrics",   "value"),
    State("group-by",      "value"),
    State("filter-oblique","value"),
    State("filter-diam",   "value"),
    State("filter-dev",    "value"),
    prevent_initial_call=True,
)
def export_figure(n_clicks, figure, fmt, scale,
                  chart_type, metrics, group_by,
                  f_obl, f_diam, f_dev):
    if not n_clicks:
        return dash.no_update, ""

    fname = f"grasp_dashboard_{chart_type}"

    if fmt == "csv":
        dff = df.copy()
        if f_obl:  dff = dff[dff["oblique"].isin(f_obl)]
        if f_diam: dff = dff[dff["diameter"].isin(f_diam)]
        if f_dev:  dff = dff[dff["dev"].isin(f_dev)]
        metrics = metrics or MAIN_METRICS
        cols    = ["oblique","diameter","dev"] + [m for m in metrics
                                                   if m in dff.columns]
        csv_str = dff[cols].to_csv(index=False)
        return (
            dict(content=csv_str, filename=f"{fname}.csv", type="text/csv"),
            "✅ CSV ready"
        )

    # Image export via kaleido
    try:
        import plotly.io as pio
        fig_obj = go.Figure(figure)
        img_bytes = pio.to_image(fig_obj, format=fmt, scale=scale)
        b64 = base64.b64encode(img_bytes).decode()
        mime = {"png": "image/png", "svg": "image/svg+xml",
                "pdf": "application/pdf"}.get(fmt, "image/png")
        return (
            dict(content=b64, filename=f"{fname}.{fmt}",
                 type=mime, base64=True),
            f"✅ {fmt.upper()} ready"
        )
    except Exception as e:
        return dash.no_update, f"⚠ {e}"


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=False, port=8050)