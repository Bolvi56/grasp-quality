# -*- coding: utf-8 -*-
"""
grasp_quality_analysis_full.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

N_WORKERS = 36

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config import (
    RESULTS_DIR, FIGURES_DIR, SUMMARY_DIR,
    OBLIQUE_NAMES, DIAMETERS, DEV_CONFIGS, DEV_CONFIG_NAMES,
    NAME_MAPPING, R_GROUND_TO_LOCAL,
    MAIN_METRICS, ASCENDING_METRICS, FULL_NAMES, UNITS,
)

# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

MU           = 0.29
BASELINE_OBL = "Non"
BASELINE_DEV = "dev_p00"

POSTER = {
    "navy":       "#2D2B6B",
    "blue_mid":   "#4B5BA6",
    "blue_light": "#6ECFDF",
    "teal":       "#3DBFA8",
    "yellow":     "#F5D547",
    "bg":         "#F7F7F7",
    "white":      "#FFFFFF",
    "dark_text":  "#1A1A2E",
}

POSTER_CMAP = LinearSegmentedColormap.from_list(
    "poster_rg",
    ["#8B0000","#D73027","#FC8D59","#FEE08B","#91CF60","#1A9850","#004529"],
    N=256,
)
POSTER_RANK_CMAP = LinearSegmentedColormap.from_list(
    "poster_rank",
    [POSTER["navy"], POSTER["blue_mid"], POSTER["blue_light"],
     POSTER["teal"], POSTER["yellow"]],
    N=256,
)

IEEE_W   = 88 / 25.4
FS       = 12
FS_ANN   = 8
FS_TITLE = 16
DPI      = 300

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": FS,
    "axes.titlesize": FS_TITLE, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS_ANN, "figure.dpi": 150,
    "savefig.dpi": DPI, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "axes.linewidth": 0.6,
    "axes.facecolor": POSTER["bg"], "figure.facecolor": POSTER["white"],
    "lines.linewidth": 1.2, "patch.linewidth": 0.4,
    "text.usetex": False,
    "axes.titlecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "legend.edgecolor": "white",
})

SAVE_KW = dict(dpi=DPI, bbox_inches="tight", pad_inches=0.03)

COLORS_OBL = plt.cm.tab20(np.linspace(0, 1, len(OBLIQUE_NAMES)))
COLORS_DEV = plt.cm.Set2(np.linspace(0, 1, len(DEV_CONFIGS)))

SCORING_SYSTEMS = {
    "borda":     lambda r, n: 100 * (n - r) / (n - 1),
    "linear":    lambda r, n: 100 * (n - r + 1) / n,
    "power_2.0": lambda r, n: 100 * (1 - ((r-1)/(n-1))**2.0),
    "power_3.0": lambda r, n: 100 * (1 - ((r-1)/(n-1))**3.0),
    "exp_0.9":   lambda r, n: 100 * 0.9**(r-1),
}


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def plot_top_vs_non(df_master, top_n, group_by, out_dir, fname_suffix):
    """
    Eje Y = valores únicos de group_by (ej: oblicuidades, diámetros, deviations)
    Para cada fila (valor de group_by) y cada métrica:
      - Busca la mejor combinación de los otros 2 factores vs Non del mismo contexto
      - Celda: valor_real\n(±%)\n(mejor_config)
    Primera fila siempre = Non oblique reference.
    """
    cols = [m for m in MAIN_METRICS if m in df_master.columns]

    # Definir qué varia en Y y qué son los "otros" factores libres
    if group_by == "oblique":
        y_vals      = OBLIQUE_NAMES
        free_cols   = ["diameter", "dev"]
        ref_y_val   = BASELINE_OBL
        def y_label(v):      return str(v)
        def config_label(r): return f"d{int(r['diameter']):02d}mm {r['dev']}"
        def ref_filter(df, yv):
            # Non oblique con mismo diam+dev → promedio sobre todos
            return df[df["oblique"] == BASELINE_OBL][cols].mean()

    elif group_by == "diameter":
        y_vals      = DIAMETERS
        free_cols   = ["oblique", "dev"]
        ref_y_val   = None   # no hay "Non diameter", usamos Non oblique
        def y_label(v):      return f"d{int(v):02d}mm"
        def config_label(r): return f"{r['oblique']} {r['dev']}"
        def ref_filter(df, yv):
            return df[
                (df["oblique"]  == BASELINE_OBL) &
                (df["diameter"] == yv)
            ][cols].mean()

    elif group_by == "dev":
        y_vals      = DEV_CONFIG_NAMES
        free_cols   = ["oblique", "diameter"]
        ref_y_val   = BASELINE_DEV
        def y_label(v):      return str(v)
        def config_label(r): return f"{r['oblique']} d{int(r['diameter']):02d}mm"
        def ref_filter(df, yv):
            return df[
                (df["oblique"] == BASELINE_OBL) &
                (df["dev"]     == yv)
            ][cols].mean()

    # ── Construir filas ───────────────────────────────────────────────────────
    # Fila 0: Non reference (promedio global del Non oblique)
    non_global = df_master[df_master["oblique"] == BASELINE_OBL][cols].mean()

    annot_matrix  = []
    color_matrix  = []
    display_index = []

    # Fila baseline
    annot_matrix.append([f"{non_global[col]:.2f}\n(ref)" for col in cols])
    color_matrix.append([0.5] * len(cols))
    display_index.append(f"{BASELINE_OBL} (ref)")

    # Una fila por cada valor del eje Y
    for yv in y_vals:
        # Subset de datos para este valor del eje Y
        df_yv = df_master[df_master[group_by] == yv]
        if df_yv.empty:
            continue

        # Referencia Non para este contexto
        ref_vals = ref_filter(df_master, yv)
        if ref_vals.isna().all():
            continue

        # Para cada métrica: buscar la fila con mejor improvement
        row_annot = []
        row_color = []

        for col in cols:
            base_val = ref_vals[col]

            # Candidatos: todas las filas de este yv, excluyendo Non oblique
            candidates = df_yv[df_yv["oblique"] != BASELINE_OBL].copy()
            if candidates.empty:
                # Si solo hay Non, mostrar el propio Non
                val     = ref_vals[col]
                row_annot.append(f"{val:.2f}\n(+0.0%)\n(Non ref)")
                row_color.append(0.5)
                continue

            # Calcular improvement para cada candidato
            if abs(base_val) < 1e-12:
                candidates = candidates.copy()
                candidates["_imp"] = 0.0
                candidates["_pct"] = 0.0
            else:
                raw_pct = (candidates[col] - base_val) / abs(base_val) * 100
                improvement = -raw_pct if col in ASCENDING_METRICS else raw_pct
                candidates = candidates.copy()
                candidates["_imp"] = improvement.values
                candidates["_pct"] = raw_pct.values

            # Mejor candidato
            best_idx = candidates["_imp"].idxmax()
            best_row = candidates.loc[best_idx]
            val      = best_row[col]
            raw_pct  = best_row["_pct"]
            imp      = best_row["_imp"]
            sign     = "+" if raw_pct >= 0 else ""
            cfg_lbl  = config_label(best_row)

            row_annot.append(f"{val:.2f}\n({sign}{raw_pct:.1f}%)\n({cfg_lbl})")
            row_color.append(float(imp))

        annot_matrix.append(row_annot)
        color_matrix.append(row_color)
        display_index.append(y_label(yv))

    n_r = len(display_index)
    n_c = len(cols)

    # Normalizar color por columna
    color_arr = np.array(color_matrix, dtype=float)
    norm_arr  = np.zeros_like(color_arr)
    norm_arr[0, :] = 0.5

    for j in range(n_c):
        col_vals = color_arr[1:, j]
        mx = float(np.nanmax(np.abs(col_vals))) if len(col_vals) > 0 else 1.0
        if mx > 1e-9:
            norm_arr[1:, j] = np.clip(col_vals / mx * 0.5 + 0.5, 0.0, 1.0)
        else:
            norm_arr[1:, j] = 0.5

    # ── Figura ────────────────────────────────────────────────────────────────
    cell_h = 1.2
    cell_w = 1.3
    fig, ax = plt.subplots(figsize=(max(IEEE_W*2.2, n_c*cell_w),
                                    max(IEEE_W*2.0, n_r*cell_h)))
    fig.patch.set_facecolor(POSTER["navy"])
    ax.set_facecolor(POSTER["navy"])

    im = ax.imshow(norm_arr, cmap=POSTER_CMAP, vmin=0, vmax=1, aspect="auto")

    for i in range(n_r):
        for j in range(n_c):
            c = auto_text_color(norm_arr[i, j], 0, 1, POSTER_CMAP)
            ax.text(j, i, annot_matrix[i][j],
                    ha="center", va="center",
                    fontsize=max(FS_ANN - 1, 5.5), color=c,
                    fontweight="bold", linespacing=1.25)

    # Borde dorado en celdas con improvement > 0 (mejor que Non)
    for i in range(1, n_r):
        for j in range(n_c):
            if color_arr[i, j] > 0:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                      fill=False,
                                      edgecolor=POSTER["yellow"],
                                      lw=1.8, zorder=12)
                ax.add_patch(rect)

    # Línea separadora tras baseline
    ax.axhline(0.5, color="white", lw=2.5, ls="--")

    styled_heatmap_axes(
        ax, cols, display_index,
        title=f"Best config vs {BASELINE_OBL} | by {group_by}",
        xtick_fs=8, ytick_fs=9,
    )
    styled_colorbar(im, ax, "",
                    ticks=[0, 0.5, 1],
                    ticklabels=["Worse", "Ref/Equal", "Better"])
    draw_grid(ax, n_r, n_c)
    plt.tight_layout()
    savefig(fig, ensure_dir(out_dir), f"top{top_n}_vs_non_{fname_suffix}")
    print(f"  ✅ top{top_n}_vs_non_{fname_suffix}")

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def contacts_csv_path(oblique: str, diameter_mm: int, dev_name: str) -> Path:
    stem = f"Oblique_{oblique}_{diameter_mm:02d}mm_p{dev_name}"
    return RESULTS_DIR / f"d{diameter_mm:02d}mm" / oblique / dev_name / f"{stem}_contacts.csv"

def auto_text_color(val, vmin, vmax, cmap):
    norm = (val - vmin) / (vmax - vmin + 1e-12)
    rgba = cmap(norm) if callable(cmap) else mpl.colormaps[cmap](norm)
    lum  = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
    return "white" if lum < 0.45 else POSTER["dark_text"]

def styled_colorbar(im, ax, label, ticks=None, ticklabels=None, fontsize=None):
    fs = fontsize or FS_ANN
    cb = ax.get_figure().colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.ax.tick_params(labelsize=fs, colors="white")
    cb.set_label(label, fontsize=fs, color="white", fontweight="bold")
    cb.outline.set_edgecolor("white")
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight("bold"); t.set_fontsize(fs); t.set_color("white")
    return cb

def draw_grid(ax, n_rows, n_cols):
    for x in np.arange(-0.5, n_cols, 1): ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, n_rows, 1): ax.axhline(y, color="white", lw=0.8)

def styled_heatmap_axes(ax, xlabels, ylabels, title="", xtick_fs=7, ytick_fs=9):
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right",
                       color="white", fontweight="bold", fontsize=xtick_fs)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, rotation=0, color="white",
                       fontweight="bold", fontsize=ytick_fs)
    ax.set_xlabel("Metric",        color="white", fontweight="bold", fontsize=12)
    ax.set_ylabel("Configuration", color="white", fontweight="bold", fontsize=12)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, color="white",
                     fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("white"); sp.set_linewidth(1.0)

def savefig(fig, path: Path, name: str):
    for ext in ("png"):
        fig.savefig(path / f"{name}.{ext}", **SAVE_KW)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  GRASP MATRIX Y MÉTRICAS
# ══════════════════════════════════════════════════════════════

def build_grasp_matrix(df: pd.DataFrame) -> np.ndarray:
    G = np.zeros((6, len(df)))
    for i, row in enumerate(df.itertuples(index=False)):
        n_g = np.array([row.geom_normal_x, row.geom_normal_y, row.geom_normal_z])
        r_g = np.array([row.point_x, row.point_y, row.point_z])
        n_l = R_GROUND_TO_LOCAL @ n_g
        r_l = R_GROUND_TO_LOCAL @ r_g
        G[:3, i] = n_l
        G[3:, i] = np.cross(r_l, n_l)
    return G

def compute_metrics(df: pd.DataFrame, G: np.ndarray) -> dict:
    R      = R_GROUND_TO_LOCAL
    r_vecs = (R @ df[["point_x","point_y","point_z"]].values.T).T
    forces = (R @ df[["force_x","force_y","force_z"]].values.T).T
    y_loc  = r_vecs[:, 1]
    z_loc  = r_vecs[:, 2]
    pts_radial = np.column_stack([y_loc, z_loc])

    metrics = {"QDCC": float(np.linalg.norm(pts_radial.mean(axis=0)) * 1000)}

    f_res            = forces.sum(axis=0)
    metrics["RFD"]   = float(np.linalg.norm(f_res))
    metrics["RFD_x"] = float(f_res[0])
    metrics["RFD_y"] = float(f_res[1])
    metrics["RFD_z"] = float(f_res[2])

    # ── FAD ──────────────────────────────────────────────────────────────────
    # FAD_resultante — ángulo de la fuerza neta vs dirección al COM
    centroid_2d = pts_radial.mean(axis=0)           # (cy, cz)
    fy_res = float(np.sum(forces[:, 1]))
    fz_res = float(np.sum(forces[:, 2]))
    f_res_2d  = np.array([fy_res, fz_res])
    r_ref     = -centroid_2d                        # dirección centroide → COM

    r_ref_norm = np.linalg.norm(r_ref)
    f_res_norm = np.linalg.norm(f_res_2d)

    if r_ref_norm > 1e-9 and f_res_norm > 1e-9:
        cos_t = np.clip(np.dot(r_ref / r_ref_norm,
                                f_res_2d / f_res_norm), -1.0, 1.0)
        fad_res = float(np.degrees(np.arccos(cos_t)))
        cross   = float(np.cross(r_ref / r_ref_norm,
                                  f_res_2d / f_res_norm))
        metrics["FAD"] = fad_res if cross < 0 else -fad_res
    else:
        metrics["FAD"] = float("nan")

    # ── FAD ──────────────────────────────────────────────────────────────────
    fy_res = float(np.sum(forces[:, 1]))
    fz_res = float(np.sum(forces[:, 2]))
    f_res_2d   = np.array([fy_res, fz_res])
    f_res_norm = np.linalg.norm(f_res_2d)

    # Centro de Presión — promedio ponderado por magnitud de fuerza
    f_mags_2d = np.sqrt(forces[:, 1]**2 + forces[:, 2]**2)
    f_total   = float(np.sum(f_mags_2d))

    if f_total > 1e-9:
        cop_y = float(np.sum(f_mags_2d * y_loc) / f_total)
        cop_z = float(np.sum(f_mags_2d * z_loc) / f_total)
    else:
        cop_y, cop_z = float(np.mean(y_loc)), float(np.mean(z_loc))

    cop_2d   = np.array([cop_y, cop_z])
    r_ideal  = -cop_2d / np.linalg.norm(cop_2d)   # CoP → COM

    if f_res_norm > 1e-9:
        f_dir  = f_res_2d / f_res_norm
        cos_t  = np.clip(np.dot(r_ideal, f_dir), -1.0, 1.0)
        angle  = float(np.degrees(np.arccos(cos_t)))
        cross  = float(np.cross(r_ideal, f_dir))
        metrics["FAD"] = angle if cross >= 0 else -angle
    else:
        metrics["FAD"] = float("nan")

    f_mags      = np.linalg.norm(forces, axis=1)
    f_total_mag = float(f_mags.sum())
    metrics["RE"]  = float(np.linalg.norm(f_res[1:])) / f_total_mag if f_total_mag>1e-9 else 0.0

    if len(pts_radial) >= 3:
        try:    metrics["QAGP"] = float(ConvexHull(pts_radial).volume * 1e6)
        except: metrics["QAGP"] = float("nan")
    else:
        metrics["QAGP"] = float("nan")

    angles_sorted = np.sort(np.arctan2(z_loc, y_loc))
    n = len(angles_sorted)
    if n >= 2:
        gaps = np.diff(angles_sorted, append=angles_sorted[0]+2*np.pi)
        metrics["QSGP"] = float(np.std(gaps - 2*np.pi/n))
    else:
        metrics["QSGP"] = float("nan")

    metrics["FSD"] = float(np.std(f_mags))
    metrics["AXB"] = float(abs(f_res[0]))

    tau_axis       = np.array([0.,0.,0.,1.,0.,0.])
    metrics["CTR"] = float(np.linalg.norm(tau_axis @ G)) * 1e6

    try:
        sv = np.linalg.svd(G, compute_uv=False)
        metrics["QVEW"] = float(np.prod(sv)) * 1e18
    except:
        metrics["QVEW"] = float("nan")

    return metrics


# ══════════════════════════════════════════════════════════════
#  FUNCIONES DE PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_metrics_heatmap_real(df_sub, row_col, row_vals, title, out_dir, fname):
    cols   = [m for m in MAIN_METRICS if m in df_sub.columns]
    matrix = df_sub.set_index(row_col)[cols]
    matrix = matrix.loc[[r for r in row_vals if r in matrix.index]]
    if matrix.empty: return

    rows_plot = list(matrix.index)
    n_r, n_c  = matrix.shape
    norm_mat  = np.zeros_like(matrix.values, dtype=float)
    for j in range(n_c):
        col_vals = matrix.values[:, j].astype(float)
        vmin_c, vmax_c = np.nanmin(col_vals), np.nanmax(col_vals)
        if vmax_c - vmin_c > 1e-12:
            nc = (col_vals - vmin_c) / (vmax_c - vmin_c)
            norm_mat[:, j] = 1 - nc if cols[j] in ASCENDING_METRICS else nc
        else:
            norm_mat[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(max(IEEE_W*1.6, n_c*0.9),
                                    max(IEEE_W*1.2, n_r*0.55)))
    fig.patch.set_facecolor(POSTER["navy"])
    ax.set_facecolor(POSTER["navy"])
    im = ax.imshow(norm_mat, cmap=POSTER_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(n_r):
        for j in range(n_c):
            val = matrix.values[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "—"
            c   = auto_text_color(norm_mat[i,j], 0, 1, POSTER_CMAP)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FS_ANN, color=c, fontweight="bold")
    styled_heatmap_axes(ax, cols, rows_plot, title)
    styled_colorbar(im, ax, "Normalized score",
                    ticks=[0,0.5,1], ticklabels=["Worse","Mid","Better"])
    draw_grid(ax, n_r, n_c)
    plt.tight_layout()
    savefig(fig, ensure_dir(out_dir), fname)


def plot_pct_heatmap(df_target, df_baseline, row_col, row_vals,
                     title, out_dir, fname):
    cols   = [m for m in MAIN_METRICS if m in df_target.columns]
    tgt    = df_target.set_index(row_col)[cols]
    bas    = df_baseline.set_index(row_col)[cols]
    common = [r for r in row_vals if r in tgt.index and r in bas.index]
    if not common: return
    tgt = tgt.loc[common]; bas = bas.loc[common]
    pct = (tgt.div(bas.replace(0, np.nan)) - 1) * 100

    color_mat = np.zeros(pct.shape)
    for j, col in enumerate(cols):
        for i in range(len(common)):
            val = pct.values[i,j]
            if not np.isnan(val):
                color_mat[i,j] = -val if col in ASCENDING_METRICS else val

    norm_mat = np.zeros_like(color_mat)
    for j in range(len(cols)):
        mx = float(np.nanmax(np.abs(color_mat[:,j])))
        if mx > 1e-9: norm_mat[:,j] = color_mat[:,j] / mx

    annot = np.empty(pct.shape, dtype=object)
    for i in range(len(common)):
        for j in range(len(cols)):
            v = pct.values[i,j]
            annot[i,j] = "—" if np.isnan(v) else f"{v:+.1f}%"

    n_r, n_c = len(common), len(cols)
    fig, ax  = plt.subplots(figsize=(max(IEEE_W*1.6, n_c*0.9),
                                      max(IEEE_W*1.2, n_r*0.55)))
    fig.patch.set_facecolor(POSTER["navy"])
    ax.set_facecolor(POSTER["navy"])
    im = ax.imshow(norm_mat, cmap=POSTER_CMAP, vmin=-1, vmax=1, aspect="auto")
    for i in range(n_r):
        for j in range(n_c):
            c = auto_text_color(norm_mat[i,j], -1, 1, POSTER_CMAP)
            ax.text(j, i, annot[i,j], ha="center", va="center",
                    fontsize=FS_ANN, color=c, fontweight="bold")
    styled_heatmap_axes(ax, cols, common, title)
    styled_colorbar(im, ax, "", ticks=[-1,0,1],
                    ticklabels=["Worse","Equal","Better"])
    draw_grid(ax, n_r, n_c)
    plt.tight_layout()
    savefig(fig, ensure_dir(out_dir), fname)


def plot_contact_distribution(key, out_dir, contact_dfs_local):
    oblique, diameter_mm, dev = key
    if key not in contact_dfs_local: return
    df = contact_dfs_local[key]

    R      = R_GROUND_TO_LOCAL
    r_vecs = (R @ df[["point_x","point_y","point_z"]].values.T).T
    forces = (R @ df[["force_x","force_y","force_z"]].values.T).T
    y_loc, z_loc   = r_vecs[:,1], r_vecs[:,2]
    fy_pts, fz_pts = forces[:,1], forces[:,2]

    cyl_r     = np.mean(np.sqrt(y_loc**2 + z_loc**2))
    theta     = np.linspace(0, 2*np.pi, 300)
    lim       = cyl_r * 1.65
    f_mags_2d = np.sqrt(fy_pts**2 + fz_pts**2)
    f_tot     = float(np.sum(f_mags_2d))
    cop_y = float(np.sum(f_mags_2d*y_loc)/f_tot) if f_tot>1e-9 else float(np.mean(y_loc))
    cop_z = float(np.sum(f_mags_2d*z_loc)/f_tot) if f_tot>1e-9 else float(np.mean(z_loc))
    fy_res    = float(np.sum(fy_pts))
    fz_res    = float(np.sum(fz_pts))
    f_res_mag = np.sqrt(fy_res**2 + fz_res**2)
    f_scale   = (cyl_r * 0.55) / (np.max(f_mags_2d) + 1e-9)
    res_scale = (cyl_r * 0.85) / (f_res_mag + 1e-9)

    key_non = (BASELINE_OBL, diameter_mm, dev)
    has_non = (key_non in contact_dfs_local) and (oblique != BASELINE_OBL)

    fig, ax = plt.subplots(figsize=(IEEE_W*2, IEEE_W*2))
    fig.patch.set_facecolor(POSTER["white"])
    ax.set_facecolor(POSTER["white"])
    ax.set_aspect("equal")
    ax.plot(cyl_r*np.cos(theta), cyl_r*np.sin(theta),
            color=POSTER["blue_mid"], lw=1.0, ls="--", zorder=1)
    ax.scatter(0, 0, marker="+", s=80, color=POSTER["navy"],
               linewidths=1.4, zorder=5, label="COM")

    if has_non:
        df_non = contact_dfs_local[key_non]
        rv_non = (R @ df_non[["point_x","point_y","point_z"]].values.T).T
        yn, zn = rv_non[:,1], rv_non[:,2]
        pts_non = np.column_stack([yn, zn])
        if len(pts_non) >= 3:
            try:
                hull  = ConvexHull(pts_non)
                verts = np.vstack([pts_non[hull.vertices], pts_non[hull.vertices[0]]])
                ax.add_patch(MplPolygon(verts, closed=True,
                    facecolor=POSTER["yellow"], edgecolor=POSTER["yellow"],
                    alpha=0.15, lw=1.0, zorder=2))
            except: pass
        ax.scatter(yn, zn, color=POSTER["yellow"], s=18, marker="^",
                   alpha=0.7, zorder=3, label=f"{BASELINE_OBL} (ref)")

    pts = np.column_stack([y_loc, z_loc])
    if len(pts) >= 3:
        try:
            hull  = ConvexHull(pts)
            verts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
            ax.add_patch(MplPolygon(verts, closed=True,
                facecolor=POSTER["blue_mid"], edgecolor=POSTER["blue_mid"],
                alpha=0.18, lw=1.2, zorder=4))
        except: pass

    for yi, zi, fyi, fzi in zip(y_loc, z_loc, fy_pts, fz_pts):
        ax.annotate("", xy=(0,0), xytext=(yi,zi),
            arrowprops=dict(arrowstyle="-|>", color=POSTER["blue_mid"],
                            lw=0.6, mutation_scale=5, alpha=0.6), zorder=5)
        ax.annotate("", xy=(yi+fyi*f_scale, zi+fzi*f_scale), xytext=(yi,zi),
            arrowprops=dict(arrowstyle="-|>", color=POSTER["teal"],
                            lw=0.9, mutation_scale=6), zorder=7)

    ax.annotate("",
        xy=(cop_y+fy_res*res_scale, cop_z+fz_res*res_scale),
        xytext=(cop_y, cop_z),
        arrowprops=dict(arrowstyle="-|>", color=POSTER["navy"],
                        lw=2.2, mutation_scale=12), zorder=10)
    ax.text(cop_y+fy_res*res_scale*1.12, cop_z+fz_res*res_scale*1.12,
            f"{f_res_mag:.1f} N", fontsize=FS_ANN, color=POSTER["navy"],
            ha="center", va="center", fontweight="bold", zorder=11)

    ax.scatter(y_loc, z_loc, color=POSTER["teal"], s=24, zorder=6, label="Contacts")
    ax.scatter(np.mean(y_loc), np.mean(z_loc), color=POSTER["teal"],
               s=45, marker="D", alpha=0.5, zorder=8, label="Centroid")
    ax.scatter(cop_y, cop_z, color=POSTER["navy"], s=70, marker="*",
               zorder=9, label="CoP")

    qdcc_val = np.sqrt(cop_y**2 + cop_z**2) * 1000
    ax.set_title(f"{oblique} | d{diameter_mm}mm | {dev}  —  QDCC={qdcc_val:.2f} mm",
                 fontsize=FS_TITLE*0.8, color=POSTER["navy"], fontweight="bold")
    ax.set_xlabel("Y$_{local}$ (m)", color=POSTER["navy"])
    ax.set_ylabel("Z$_{local}$ (m)", color=POSTER["navy"])
    ax.tick_params(colors=POSTER["navy"])
    for sp in ax.spines.values():
        sp.set_edgecolor(POSTER["navy"]); sp.set_linewidth(0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc="upper right", fontsize=FS_ANN, framealpha=0.9,
              edgecolor=POSTER["navy"], labelcolor=POSTER["navy"])
    ax.grid(True, alpha=0.2, lw=0.4, ls="--", color=POSTER["navy"])
    savefig(fig, ensure_dir(out_dir),
            f"contacts_{oblique}_{diameter_mm:02d}mm_p{dev}")


def plot_line_by_group(df_sub, x_col, x_vals, group_col, group_vals,
                        colors, metric, title, out_dir, fname):
    fig, ax = plt.subplots(figsize=(IEEE_W*1.8, IEEE_W*1.1))
    fig.patch.set_facecolor(POSTER["navy"])
    ax.set_facecolor(POSTER["bg"])
    for gi, gval in enumerate(group_vals):
        sub = df_sub[df_sub[group_col] == gval]
        ys  = [float(sub[sub[x_col]==xv][metric].values[0])
               if len(sub[sub[x_col]==xv]) > 0 else np.nan
               for xv in x_vals]
        ax.plot(range(len(x_vals)), ys, color=colors[gi],
                marker="o", ms=4, lw=1.4, label=str(gval))
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45,
                       ha="right", color="white")
    ax.yaxis.set_tick_params(colors="white")
    ax.set_xlabel(x_col, color="white")
    ax.set_ylabel(f"{metric} [{UNITS.get(metric,'—')}]", color="white")
    ax.set_title(title, color="white", fontweight="bold")
    for sp in ax.spines.values(): sp.set_edgecolor("white")
    ax.legend(loc="best", fontsize=FS_ANN-1, ncol=2,
              framealpha=0.3, edgecolor="white", labelcolor="white")
    ax.grid(True, alpha=0.2, lw=0.4, ls="--")
    plt.tight_layout()
    savefig(fig, ensure_dir(out_dir), fname)


def compute_ranking(df_sub, label_col):
    cv       = (df_sub[MAIN_METRICS].std() / df_sub[MAIN_METRICS].mean().abs()) * 100
    rel_cols = [c for c in MAIN_METRICS if c in cv.index and cv[c] > 1.0] or MAIN_METRICS
    labels   = df_sub[label_col].tolist()
    n        = len(labels)

    df_ranks = pd.DataFrame(index=labels)
    for col in rel_cols:
        df_ranks[col] = (
            df_sub.set_index(label_col)[col]
            .rank(method="dense", ascending=(col in ASCENDING_METRICS))
            .astype(int)
        )

    df_scores = pd.DataFrame(index=labels)
    for sname, sfunc in SCORING_SYSTEMS.items():
        df_scores[sname] = [
            sum(sfunc(df_ranks.loc[lbl, m], n) for m in rel_cols)
            for lbl in labels
        ]

    df_sys_ranks = df_scores.rank(method="dense", ascending=False).astype(int)
    sys_names    = list(SCORING_SYSTEMS.keys())
    corr_mat     = np.array([[spearmanr(df_sys_ranks[s1], df_sys_ranks[s2])[0]
                               for s2 in sys_names] for s1 in sys_names])
    df_corr      = pd.DataFrame(corr_mat, index=sys_names, columns=sys_names)
    remaining    = list(sys_names)
    selected     = []
    while remaining:
        ref = remaining.pop(0); selected.append(ref)
        remaining = [s for s in remaining if abs(df_corr.loc[ref, s]) <= 0.94]

    return df_ranks, df_scores, df_sys_ranks[selected], rel_cols


def plot_ranking_heatmap(df_ranks, title, out_dir, fname):
    n_r, n_c   = df_ranks.shape
    vmin, vmax = 1, int(df_ranks.values.max())
    fig, ax    = plt.subplots(figsize=(max(IEEE_W*1.4, n_c*0.8),
                                       max(IEEE_W*1.1, n_r*0.5)))
    fig.patch.set_facecolor(POSTER["navy"])
    ax.set_facecolor(POSTER["navy"])
    im = ax.imshow(df_ranks.values, cmap=POSTER_RANK_CMAP,
                   vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(n_r):
        for j in range(n_c):
            val = df_ranks.values[i,j]
            c   = auto_text_color(val, vmin, vmax, POSTER_RANK_CMAP)
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=FS_ANN, color=c, fontweight="bold")
    styled_heatmap_axes(ax, list(df_ranks.columns), list(df_ranks.index), title)
    styled_colorbar(im, ax, "Rank",
                    ticks=list(range(1, vmax+1, max(1, vmax//5))))
    draw_grid(ax, n_r, n_c)
    plt.tight_layout()
    savefig(fig, ensure_dir(out_dir), fname)


# ══════════════════════════════════════════════════════════════
#  WORKERS — top-level para que sean pickleable
# ══════════════════════════════════════════════════════════════

def load_single(args):
    obl, diam, dev = args
    p = contacts_csv_path(obl, diam, dev)
    if not p.exists(): return None
    df = pd.read_csv(p)
    if df.empty: return None
    G = build_grasp_matrix(df)
    m = compute_metrics(df, G)
    return (obl, diam, dev, m, df)

def contact_worker(args):
    key, out_dir, contact_dfs_local = args
    plot_contact_distribution(key, out_dir, contact_dfs_local)

def heatmap_real_worker(args):
    df_sub, row_col, row_vals, title, out_dir, fname = args
    plot_metrics_heatmap_real(df_sub, row_col, row_vals, title, out_dir, fname)

def pct_heatmap_worker(args):
    df_target, df_baseline, row_col, row_vals, title, out_dir, fname = args
    plot_pct_heatmap(df_target, df_baseline, row_col, row_vals, title, out_dir, fname)

def ranking_worker(args):
    df_sub, label_col, title_ranks, title_final, out_dir, fname_ranks, fname_final = args
    if len(df_sub) < 3: return
    df_ranks, _, df_final_r, _ = compute_ranking(df_sub, label_col)
    plot_ranking_heatmap(df_ranks, title_ranks, out_dir, fname_ranks)
    plot_ranking_heatmap(df_final_r, title_final, out_dir, fname_final)

def line_worker(args):
    df_sub, x_col, x_vals, group_col, group_vals, colors, metric, title, out_dir, fname = args
    plot_line_by_group(df_sub, x_col, x_vals, group_col, group_vals,
                       colors, metric, title, out_dir, fname)

def mean_line_worker(args):
    diameters, ys, metric, out_dir = args
    fig, ax = plt.subplots(figsize=(IEEE_W*1.6, IEEE_W*0.9))
    fig.patch.set_facecolor(POSTER["navy"])
    ax.set_facecolor(POSTER["bg"])
    ax.plot(diameters, ys, color=POSTER["teal"], marker="o", ms=6, lw=2.0)
    ax.set_xlabel("Diameter (mm)", color="white")
    ax.set_ylabel(f"{metric} [{UNITS.get(metric,'—')}]", color="white")
    ax.set_title(f"{metric} vs Diameter (mean all configs)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("white")
    ax.grid(True, alpha=0.2, lw=0.4, ls="--")
    plt.tight_layout()
    savefig(fig, ensure_dir(out_dir), f"mean_{metric}_vs_diameter")

def run_pool(tasks: list, desc: str, workers: int = N_WORKERS):
    if not tasks: return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, arg): (fn, arg) for fn, arg in tasks}
        with tqdm(total=len(futures), desc=f"  {desc}",
                  bar_format="{desc}: {percentage:3.0f}%|{bar:25}| "
                             "{n}/{total} [{elapsed}<{remaining}]") as bar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    fn, arg = futures[future]
                    print(f"\n  ⚠ Error in {fn.__name__}: {e}")
                bar.update(1)


# ══════════════════════════════════════════════════════════════
#  CARGA DE DATOS — PARALELO
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # ══════════════════════════════════════════════════════════════
    #  CARGA DE DATOS — PARALELO
    # ══════════════════════════════════════════════════════════════

    load_tasks = [
        (obl, diam, dev_cfg["name"])
        for diam in DIAMETERS
        for obl in OBLIQUE_NAMES
        for dev_cfg in DEV_CONFIGS
    ]

    records:     dict[tuple, dict]          = {}
    contact_dfs: dict[tuple, pd.DataFrame] = {}

    print(f"\n{'='*65}")
    print(f"  LOADING & COMPUTING METRICS — {N_WORKERS} workers")
    print(f"{'='*65}")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(load_single, t): t for t in load_tasks}
        with tqdm(total=len(load_tasks), desc="Loading",
                  bar_format="{desc}: {percentage:3.0f}%|{bar:30}| "
                             "{n}/{total} [{elapsed}<{remaining}]") as bar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    obl, diam, dev, m, df = result
                    records[(obl, diam, dev)]     = m
                    contact_dfs[(obl, diam, dev)] = df
                    bar.set_postfix_str(f"{obl}|d{diam}|{dev}")
                bar.update(1)

    print(f"\n  Total loaded: {len(records)} combinations\n")

    df_master = pd.DataFrame([
        {"oblique": k[0], "diameter": k[1], "dev": k[2], **v}
        for k, v in records.items()
    ])
    df_master.to_csv(SUMMARY_DIR / "all_metrics.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    #  CONSTRUIR TODAS LAS TAREAS DE PLOTTING
    # ══════════════════════════════════════════════════════════════

    all_plot_tasks: list[tuple] = []

    for diam in DIAMETERS:
        df_d = df_master[df_master["diameter"] == diam].copy()
        if df_d.empty: continue

        fig_base = FIGURES_DIR / f"d{diam:02d}mm"
        dir_obl  = fig_base / "by_oblique"
        dir_dev  = fig_base / "by_deviation"
        dir_cnt  = fig_base / "contacts"
        dir_glob = fig_base / "global"

        for obl in OBLIQUE_NAMES:
            for dev_cfg in DEV_CONFIGS:
                dev = dev_cfg["name"]
                key = (obl, diam, dev)
                if key in contact_dfs:
                    all_plot_tasks.append((contact_worker,
                        (key, dir_cnt, contact_dfs)))

        for dev_cfg in DEV_CONFIGS:
            dev    = dev_cfg["name"]
            df_sub = df_d[df_d["dev"] == dev].copy()
            if df_sub.empty: continue

            all_plot_tasks.append((heatmap_real_worker, (
                df_sub, "oblique", OBLIQUE_NAMES,
                f"Metrics | d{diam}mm | {dev}", dir_obl,
                f"metrics_real_obl_{dev}")))

            baseline_rows = df_sub[df_sub["oblique"] == BASELINE_OBL]
            if not baseline_rows.empty:
                df_base = pd.concat([baseline_rows]*len(OBLIQUE_NAMES), ignore_index=True)
                df_base["oblique"] = OBLIQUE_NAMES[:len(df_base)]
                all_plot_tasks.append((pct_heatmap_worker, (
                    df_sub, df_base, "oblique", OBLIQUE_NAMES,
                    f"% vs {BASELINE_OBL} | d{diam}mm | {dev}",
                    dir_obl, f"metrics_pct_obl_{dev}")))

            all_plot_tasks.append((ranking_worker, (
                df_sub, "oblique",
                f"Metric Ranks | d{diam}mm | {dev}",
                f"Final Ranks | d{diam}mm | {dev}",
                dir_obl, f"ranks_obl_{dev}", f"final_ranks_obl_{dev}")))

            for mi, metric in enumerate(MAIN_METRICS):
                if metric not in df_sub.columns: continue
                all_plot_tasks.append((line_worker, (
                    df_sub, "oblique", OBLIQUE_NAMES,
                    "dev", [dev], [COLORS_DEV[mi % len(COLORS_DEV)]],
                    metric, f"{metric} by Oblique | d{diam}mm | {dev}",
                    dir_obl / "lines", f"line_{metric}_obl_{dev}")))

        for obl in OBLIQUE_NAMES:
            df_sub = df_d[df_d["oblique"] == obl].copy()
            if df_sub.empty: continue

            all_plot_tasks.append((heatmap_real_worker, (
                df_sub, "dev", DEV_CONFIG_NAMES,
                f"Metrics | d{diam}mm | {obl}", dir_dev,
                f"metrics_real_dev_{obl}")))

            baseline_rows = df_sub[df_sub["dev"] == BASELINE_DEV]
            if not baseline_rows.empty:
                df_base = pd.concat([baseline_rows]*len(DEV_CONFIG_NAMES), ignore_index=True)
                df_base["dev"] = DEV_CONFIG_NAMES[:len(df_base)]
                all_plot_tasks.append((pct_heatmap_worker, (
                    df_sub, df_base, "dev", DEV_CONFIG_NAMES,
                    f"% vs {BASELINE_DEV} | d{diam}mm | {obl}",
                    dir_dev, f"metrics_pct_dev_{obl}")))

            all_plot_tasks.append((ranking_worker, (
                df_sub, "dev",
                f"Metric Ranks | d{diam}mm | {obl}",
                f"Final Ranks | d{diam}mm | {obl}",
                dir_dev, f"ranks_dev_{obl}", f"final_ranks_dev_{obl}")))

            for mi, metric in enumerate(MAIN_METRICS):
                if metric not in df_sub.columns: continue
                all_plot_tasks.append((line_worker, (
                    df_sub, "dev", DEV_CONFIG_NAMES,
                    "oblique", [obl], [COLORS_OBL[mi % len(COLORS_OBL)]],
                    metric, f"{metric} by Deviation | d{diam}mm | {obl}",
                    dir_dev / "lines", f"line_{metric}_dev_{obl}")))

        df_mean_obl = df_d.groupby("oblique")[MAIN_METRICS].mean().reset_index()
        df_mean_dev = df_d.groupby("dev")[MAIN_METRICS].mean().reset_index()

        all_plot_tasks.append((heatmap_real_worker, (
            df_mean_obl, "oblique", OBLIQUE_NAMES,
            f"Mean Metrics by Oblique | d{diam}mm",
            dir_glob, "metrics_mean_by_oblique")))
        all_plot_tasks.append((heatmap_real_worker, (
            df_mean_dev, "dev", DEV_CONFIG_NAMES,
            f"Mean Metrics by Deviation | d{diam}mm",
            dir_glob, "metrics_mean_by_deviation")))

        if BASELINE_OBL in df_mean_obl["oblique"].values:
            base_row = df_mean_obl[df_mean_obl["oblique"] == BASELINE_OBL]
            df_base  = pd.concat([base_row]*len(OBLIQUE_NAMES), ignore_index=True)
            df_base["oblique"] = OBLIQUE_NAMES[:len(df_base)]
            all_plot_tasks.append((pct_heatmap_worker, (
                df_mean_obl, df_base, "oblique", OBLIQUE_NAMES,
                f"Mean % vs {BASELINE_OBL} | d{diam}mm",
                dir_glob, "pct_mean_by_oblique")))

        if BASELINE_DEV in df_mean_dev["dev"].values:
            base_row = df_mean_dev[df_mean_dev["dev"] == BASELINE_DEV]
            df_base  = pd.concat([base_row]*len(DEV_CONFIG_NAMES), ignore_index=True)
            df_base["dev"] = DEV_CONFIG_NAMES[:len(df_base)]
            all_plot_tasks.append((pct_heatmap_worker, (
                df_mean_dev, df_base, "dev", DEV_CONFIG_NAMES,
                f"Mean % vs {BASELINE_DEV} | d{diam}mm",
                dir_glob, "pct_mean_by_deviation")))

    # By diameter
    dir_diam = FIGURES_DIR / "by_diameter"

    for obl in OBLIQUE_NAMES:
        for dev_cfg in DEV_CONFIGS:
            dev    = dev_cfg["name"]
            df_sub = df_master[(df_master["oblique"]==obl) &
                                (df_master["dev"]==dev)].copy()
            if df_sub.empty: continue

            all_plot_tasks.append((heatmap_real_worker, (
                df_sub, "diameter", DIAMETERS,
                f"Metrics by Diameter | {obl} | {dev}",
                dir_diam, f"metrics_real_diam_{obl}_{dev}")))

            for mi, metric in enumerate(MAIN_METRICS):
                if metric not in df_sub.columns: continue
                all_plot_tasks.append((line_worker, (
                    df_sub, "diameter", DIAMETERS,
                    "oblique", [obl], [COLORS_OBL[mi % len(COLORS_OBL)]],
                    metric, f"{metric} by Diameter | {obl} | {dev}",
                    dir_diam / "lines", f"line_{metric}_diam_{obl}_{dev}")))

    df_mean_diam = df_master.groupby("diameter")[MAIN_METRICS].mean().reset_index()
    all_plot_tasks.append((heatmap_real_worker, (
        df_mean_diam, "diameter", DIAMETERS,
        "Mean Metrics by Diameter (all configs)",
        dir_diam, "metrics_mean_all")))

    for metric in MAIN_METRICS:
        ys = [float(df_mean_diam[df_mean_diam["diameter"]==d][metric].values[0])
              if len(df_mean_diam[df_mean_diam["diameter"]==d]) > 0 else np.nan
              for d in DIAMETERS]
        all_plot_tasks.append((mean_line_worker,
            (DIAMETERS, ys, metric, dir_diam / "mean_lines")))

    # Global
    dir_global = FIGURES_DIR / "global"
    df_g_obl   = df_master.groupby("oblique")[MAIN_METRICS].mean().reset_index()
    df_g_dev   = df_master.groupby("dev")[MAIN_METRICS].mean().reset_index()
    df_g_diam  = df_master.groupby("diameter")[MAIN_METRICS].mean().reset_index()

    all_plot_tasks.append((heatmap_real_worker, (
        df_g_obl, "oblique", OBLIQUE_NAMES,
        "Global Mean Metrics by Oblique", dir_global, "global_metrics_by_oblique")))
    all_plot_tasks.append((heatmap_real_worker, (
        df_g_dev, "dev", DEV_CONFIG_NAMES,
        "Global Mean Metrics by Deviation", dir_global, "global_metrics_by_deviation")))
    all_plot_tasks.append((heatmap_real_worker, (
        df_g_diam, "diameter", DIAMETERS,
        "Global Mean Metrics by Diameter", dir_global, "global_metrics_by_diameter")))

    if BASELINE_OBL in df_g_obl["oblique"].values:
        base_row = df_g_obl[df_g_obl["oblique"] == BASELINE_OBL]
        df_base  = pd.concat([base_row]*len(OBLIQUE_NAMES), ignore_index=True)
        df_base["oblique"] = OBLIQUE_NAMES[:len(df_base)]
        all_plot_tasks.append((pct_heatmap_worker, (
            df_g_obl, df_base, "oblique", OBLIQUE_NAMES,
            f"Global % vs {BASELINE_OBL}", dir_global, "global_pct_by_oblique")))

    if BASELINE_DEV in df_g_dev["dev"].values:
        base_row = df_g_dev[df_g_dev["dev"] == BASELINE_DEV]
        df_base  = pd.concat([base_row]*len(DEV_CONFIG_NAMES), ignore_index=True)
        df_base["dev"] = DEV_CONFIG_NAMES[:len(df_base)]
        all_plot_tasks.append((pct_heatmap_worker, (
            df_g_dev, df_base, "dev", DEV_CONFIG_NAMES,
            f"Global % vs {BASELINE_DEV}", dir_global, "global_pct_by_deviation")))

    df_g_all = df_master.groupby(["oblique","dev"])[MAIN_METRICS].mean().reset_index()
    df_g_all["label"] = df_g_all["oblique"] + "|" + df_g_all["dev"]
    all_plot_tasks.append((ranking_worker, (
        df_g_all, "label",
        "Global Metric Rankings", "Global Final Rankings",
        dir_global, "global_metric_ranks", "global_final_ranks")))
    

    # ══════════════════════════════════════════════════════════════
    #  TOP-N VS NON OBLIQUE — generados aquí (no en pool, usan df_master)
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*65}")
    print("  TOP-N vs Non Oblique comparison plots")
    print(f"{'='*65}")

    dir_topn = FIGURES_DIR / "global" / "top_vs_non"

    for top_n in [5, 10]:
        plot_top_vs_non(df_master, top_n,
                        group_by="oblique",
                        out_dir=dir_topn,
                        fname_suffix="by_oblique")
        plot_top_vs_non(df_master, top_n,
                        group_by="diameter",
                        out_dir=dir_topn,
                        fname_suffix="by_diameter")
        plot_top_vs_non(df_master, top_n,
                        group_by="dev",
                        out_dir=dir_topn,
                        fname_suffix="by_deviation")

    # ══════════════════════════════════════════════════════════════
    #  EJECUTAR TODO EN PARALELO
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*65}")
    print(f"  PLOTTING — {len(all_plot_tasks)} tasks — {N_WORKERS} workers")
    print(f"{'='*65}\n")

    run_pool(all_plot_tasks, "Plotting", workers=N_WORKERS)

    # ══════════════════════════════════════════════════════════════
    #  EXPORTAR CSVs GLOBALES
    # ══════════════════════════════════════════════════════════════

    df_g_obl.to_csv(SUMMARY_DIR  / "global_metrics_by_oblique.csv",   index=False)
    df_g_dev.to_csv(SUMMARY_DIR  / "global_metrics_by_deviation.csv", index=False)
    df_g_diam.to_csv(SUMMARY_DIR / "global_metrics_by_diameter.csv",  index=False)

    print(f"\n{'='*65}")
    print("  ✅ ANALYSIS COMPLETE")
    print(f"{'='*65}\n")