# -*- coding: utf-8 -*-
"""
Shared configuration for all grip simulation experiments.
Author: joserojas
"""

from pathlib import Path
import numpy as np

# ==============================
#  Rutas base
# ==============================
BASE_DIR    = Path("C:/Users/joserojas/Documents/Coding/Opensim/ARMS/Geometry_flatted")
MODELS_DIR  = BASE_DIR / "Models"
BASE_MODELS = MODELS_DIR / "Base"
FULL_MODELS = MODELS_DIR / "Full"
RESULTS_DIR = BASE_DIR / "Results"
FIGURES_DIR = BASE_DIR / "Figures"
SUMMARY_DIR = BASE_DIR / "Summary"
SOURCE_MODEL = BASE_DIR / "Grip_Model_initial_3.2.osim"

for _d in [BASE_MODELS, FULL_MODELS, RESULTS_DIR, FIGURES_DIR, SUMMARY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ==============================
#  Simulación
# ==============================
STEPS_DEG       = [1.0, 0.1, 0.01, 0.001, 0.0001]
RETROCESO_MULT  = 5
FORCE_TOLERANCE = 0.03
FINGER_IDS      = [2, 3, 4, 5]
JOINT_SEQUENCE  = ["mcp", "pm", "md"]
DOF_TYPES       = ["flexion", "axial", "deviation"]
OBJECT_BODY     = "jamar_body"
CYLINDER_RADIUS = 0.024   # metros (diámetro base 48 mm)
MU              = 0.29

# ==============================
#  Cilindro jamar — ejes en ground
# ==============================
JAMAR_AXIS_X = np.array([ 0.1312,  0.9912,  0.0155])
JAMAR_AXIS_Y = np.array([-0.8818,  0.1238, -0.4551])
JAMAR_AXIS_Z = np.array([-0.4531,  0.0460,  0.8903])

R_GROUND_TO_LOCAL = np.array([
    JAMAR_AXIS_X / np.linalg.norm(JAMAR_AXIS_X),
    JAMAR_AXIS_Y / np.linalg.norm(JAMAR_AXIS_Y),
    JAMAR_AXIS_Z / np.linalg.norm(JAMAR_AXIS_Z),
])

# ==============================
#  Mapeo articulación → cuerpo contacto
# ==============================
NAME_MAPPING: dict[str, str] = {}
for _x in FINGER_IDS:
    NAME_MAPPING[f"{_x}mcp_flexion"] = f"{_x}proxph"
    NAME_MAPPING[f"{_x}pm_flexion"]  = f"{_x}midph"
    NAME_MAPPING[f"{_x}md_flexion"]  = f"{_x}distph"

# ==============================
#  Fuerzas por diámetro — Tabla 4 (sin Meta)
# ==============================
# Fuerza total por dedo por diámetro (N), columna Index/Middle/Ring/Little sin Meta
FORCE_TABLE: dict[int, dict[str, float]] = {
    25: {"2": 166.6, "3": 173.7, "4": 197.5, "5": 125.0},
    30: {"2": 136.4, "3": 199.3, "4": 171.5, "5":  78.8},
    35: {"2": 117.8, "3": 171.3, "4": 126.0, "5":  57.1},
    40: {"2":  92.7, "3": 149.5, "4": 100.3, "5":  48.4},
    45: {"2":  75.5, "3": 115.3, "4":  72.2, "5":  36.4},
    50: {"2": 67.0, "3": 88.5, "4": 58.4, "5":  33.9},
}

# Fuerzas brutas por falange por dedo por diámetro (N)
# [proximal_N, middle_N, distal_N]  — sin Meta
RAW_PHALANGE_FORCES: dict[int, dict[str, list[float]]] = {
    25: {
        "2": [40.657946, 57.40902, 68.533034],
        "3": [42.390668	,59.855623, 71.453709],
        "4": [48.198946, 68.056911, 81.244143],
        "5": [30.505662, 43.073995, 51.420344]
    },
    30: {
        "2": [32.267913, 41.160416, 62.971671],
        "3": [47.148057, 60.141283, 92.01066],
        "4": [40.571459, 51.752283, 79.176258],
        "5": [18.64158, 23.778891, 36.379529]
    },
    35: {
        "2": [27.844172, 32.470135, 57.485693],
        "3": [40.48987, 47.216758, 83.593372],
        "4": [29.782391, 34.730365, 61.487244],
        "5": [13.496623, 15.738919, 27.864457]
    },
    40: {
        "2": [21.658116, 24.814323, 46.227561],
        "3": [34.928676, 40.018785, 74.552539],
        "4": [23.433754, 26.848723, 50.017523],
        "5": [11.308013, 12.955914, 24.136073]
    },
    45: {
        "2": [16.420467, 19.949303, 39.13023],
        "3": [25.076555, 30.465624, 59.757821],
        "4": [15.702752, 19.077346, 37.419902],
        "5": [7.916623, 9.617942, 18.865435]
    },
    50: {
        "2": [12.59908, 16.08, 38.32092],
        "3": [16.642069, 21.24, 50.617931],
        "4": [10.981885, 14.016, 33.402115],
        "5": [6.374759, 8.136, 19.389241]
    },
}


def compute_phalange_percentages(diameter: int) -> dict[str, list[float]]:
    """
    Calcula porcentajes [prox%, mid%, dist%] por dedo para un diámetro dado,
    normalizados sin Meta.
    """
    raw = RAW_PHALANGE_FORCES[diameter]
    result: dict[str, list[float]] = {}
    for finger, vals in raw.items():
        total = sum(vals)
        result[finger] = [v / total * 100 for v in vals]
    return result


def compute_force_thresholds(
    diameter: int,
    tolerance: float = FORCE_TOLERANCE,
) -> dict[str, tuple[float, float]]:
    """
    Retorna {contacto: (f_min, f_max)} para todos los contactos del modelo,
    dado el diámetro del cilindro.
    """
    pct_map   = compute_phalange_percentages(diameter)
    ft        = FORCE_TABLE[diameter]
    phalanges = ["proxph", "midph", "distph"]
    thresholds: dict[str, tuple[float, float]] = {}

    for finger_str, pcts in pct_map.items():
        total_force = ft[finger_str]
        for i, suffix in enumerate(phalanges):
            name   = f"{finger_str}{suffix}"
            force  = total_force * (pcts[i] / 100.0)
            thresholds[name] = (
                force * (1 - tolerance),
                force * (1 + tolerance),
            )
    return thresholds


# ==============================
#  Diámetros de experimento
# ==============================
DIAMETERS: list[int] = [25, 30, 35, 40, 45, 50]

# ==============================
#  Configuraciones de MCP deviation (C)
# ==============================
# Cada entrada: {finger_id (str): grados}
# Dedos: "2"=index, "3"=middle, "4"=ring, "5"=little
# D3 fijo en 0 salvo configs perp y max_dev
# D2 abre positivo, D4 y D5 negativos
DEV_CONFIGS: list[dict] = [
    {
        "name": "dev_p00",
        "mcp_dev": {"2":  0.0, "3":  0.0, "4":  0.0, "5":  0.0},
        "desc": "0.0 deg spread",
    },
    {
        "name": "dev_p02",
        "mcp_dev": {"2":  2.5, "3":  0.0, "4": -0.625, "5": -2.5},
        "desc": "2.5 deg spread",
    },
    {
        "name": "dev_p05",
        "mcp_dev": {"2":  5.0, "3":  0.0, "4": -1.25, "5": -5.0},
        "desc": "5 deg spread",
    },
    {
        "name": "dev_p07",
        "mcp_dev": {"2":  7.5, "3":  0.0, "4": -2.5, "5": -7.5},
        "desc": "7.5 deg spread",
    },
    {
        "name": "dev_p10",
        "mcp_dev": {"2": 10.0, "3":  0.0, "4": -5.0, "5": -10.0},
        "desc": "10 deg spread",
    },
    {
        "name": "dev_p12",
        "mcp_dev": {"2": 12.5, "3":  0.0, "4": -7.5, "5": -12.5},
        "desc": "12.5 deg spread",
    },
    {
        "name": "dev_p15",
        "mcp_dev": {"2": 15.0, "3":  0.0, "4": -10.0, "5": -15.0},
        "desc": "15 deg spread",
    }
]

DEV_CONFIG_NAMES: list[str] = [c["name"] for c in DEV_CONFIGS]

# ==============================
#  Configuraciones oblicuas (A)
# ==============================
OBLIQUE_NAMES: list[str] = (
    ["Non"]
    + [f"{i:03d}" for i in range(10, 100, 10)]
    + ["Mean", "Min", "Max"]
)

ALL_LIMITS: dict[str, dict[str, float]] = {
    "Non": {
        "2mcp_axial": 0, "3mcp_axial": 0, "4mcp_axial": 0, "5mcp_axial": 0,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": 0, "3pm_axial": 0, "4pm_axial": 0, "5pm_axial": 0,
        "2pm_deviation": 0, "3pm_deviation": 0, "4pm_deviation": 0, "5pm_deviation": 0,
        "2md_axial": 0, "3md_axial": 0, "4md_axial": 0, "5md_axial": 0,
        "2md_deviation": 0, "3md_deviation": 0, "4md_deviation": 0, "5md_deviation": 0,
    },
    "010": {
        "2mcp_axial": -6.00, "3mcp_axial": -10.80, "4mcp_axial": -13.60, "5mcp_axial": -23.90,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -9.30, "3pm_axial": 0.10, "4pm_axial": -14.80, "5pm_axial": -12.80,
        "2pm_deviation": 7.50, "3pm_deviation": 7.50, "4pm_deviation": 8.10, "5pm_deviation": 11.40,
        "2md_axial": -1.80, "3md_axial": 1.20, "4md_axial": -12.40, "5md_axial": -0.90,
        "2md_deviation": 7.50, "3md_deviation": 5.80, "4md_deviation": 2.80, "5md_deviation": 5.80,
    },
    "020": {
        "2mcp_axial": -5.00, "3mcp_axial": -9.60, "4mcp_axial": -12.20, "5mcp_axial": -21.80,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -7.60, "3pm_axial": 1.20, "4pm_axial": -13.60, "5pm_axial": -11.60,
        "2pm_deviation": 10.00, "3pm_deviation": 11.00, "4pm_deviation": 11.20, "5pm_deviation": 13.80,
        "2md_axial": -0.60, "3md_axial": 2.40, "4md_axial": -9.80, "5md_axial": 0.20,
        "2md_deviation": 10.00, "3md_deviation": 6.60, "4md_deviation": 3.60, "5md_deviation": 7.60,
    },
    "030": {
        "2mcp_axial": -4.00, "3mcp_axial": -8.40, "4mcp_axial": -10.80, "5mcp_axial": -19.70,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -5.90, "3pm_axial": 2.30, "4pm_axial": -12.40, "5pm_axial": -10.40,
        "2pm_deviation": 12.50, "3pm_deviation": 14.50, "4pm_deviation": 14.30, "5pm_deviation": 16.20,
        "2md_axial": 0.60, "3md_axial": 3.60, "4md_axial": -7.20, "5md_axial": 1.30,
        "2md_deviation": 12.50, "3md_deviation": 7.40, "4md_deviation": 4.40, "5md_deviation": 9.40,
    },
    "040": {
        "2mcp_axial": -3.00, "3mcp_axial": -7.20, "4mcp_axial": -9.40, "5mcp_axial": -17.60,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -4.20, "3pm_axial": 3.40, "4pm_axial": -11.20, "5pm_axial": -9.20,
        "2pm_deviation": 15.00, "3pm_deviation": 16.75, "4pm_deviation": 16.40, "5pm_deviation": 18.60,
        "2md_axial": 1.80, "3md_axial": 4.80, "4md_axial": -4.80, "5md_axial": 2.60,
        "2md_deviation": 15.00, "3md_deviation": 8.20, "4md_deviation": 5.20, "5md_deviation": 11.20,
    },
    "050": {
        "2mcp_axial": -2.00, "3mcp_axial": -6.00, "4mcp_axial": -8.00, "5mcp_axial": -15.50,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -2.50, "3pm_axial": 4.50, "4pm_axial": -10.00, "5pm_axial": -8.00,
        "2pm_deviation": 17.50, "3pm_deviation": 20.25, "4pm_deviation": 19.50, "5pm_deviation": 21.00,
        "2md_axial": 3.00, "3md_axial": 6.00, "4md_axial": -2.20, "5md_axial": 3.70,
        "2md_deviation": 17.50, "3md_deviation": 9.00, "4md_deviation": 6.00, "5md_deviation": 13.00,
    },
    "060": {
        "2mcp_axial": -1.0, "3mcp_axial": -4.8, "4mcp_axial": -6.7, "5mcp_axial": -13.4,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -4.4, "3pm_axial": 5.6, "4pm_axial": -8.8, "5pm_axial": -6.8,
        "2pm_deviation": 20.0, "3pm_deviation": 25.0, "4pm_deviation": 17.6, "5pm_deviation": 23.4,
        "2md_axial": 4.2, "3md_axial": 7.2, "4md_axial": 1.2, "5md_axial": 5.8,
        "2md_deviation": 20.0, "3md_deviation": 12.8, "4md_deviation": 6.8, "5md_deviation": 18.4,
    },
    "070": {
        "2mcp_axial": 0.0, "3mcp_axial": -3.6, "4mcp_axial": -5.3, "5mcp_axial": -11.3,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -3.3, "3pm_axial": 6.7, "4pm_axial": -7.6, "5pm_axial": -5.6,
        "2pm_deviation": 22.5, "3pm_deviation": 28.5, "4pm_deviation": 19.7, "5pm_deviation": 25.8,
        "2md_axial": 5.4, "3md_axial": 8.4, "4md_axial": 3.9, "5md_axial": 7.1,
        "2md_deviation": 22.5, "3md_deviation": 14.1, "4md_deviation": 7.6, "5md_deviation": 20.8,
    },
    "080": {
        "2mcp_axial": 1.0, "3mcp_axial": -2.4, "4mcp_axial": -4.0, "5mcp_axial": -9.2,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -2.2, "3pm_axial": 7.8, "4pm_axial": -6.4, "5pm_axial": -4.4,
        "2pm_deviation": 25.0, "3pm_deviation": 32.0, "4pm_deviation": 21.8, "5pm_deviation": 28.2,
        "2md_axial": 6.6, "3md_axial": 9.6, "4md_axial": 6.6, "5md_axial": 8.4,
        "2md_deviation": 25.0, "3md_deviation": 15.4, "4md_deviation": 8.4, "5md_deviation": 23.2,
    },
    "090": {
        "2mcp_axial": 2.0, "3mcp_axial": -1.2, "4mcp_axial": -2.5, "5mcp_axial": -7.1,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -1.1, "3pm_axial": 8.9, "4pm_axial": -5.2, "5pm_axial": -3.2,
        "2pm_deviation": 27.5, "3pm_deviation": 35.5, "4pm_deviation": 23.9, "5pm_deviation": 30.6,
        "2md_axial": 7.8, "3md_axial": 10.8, "4md_axial": 9.3, "5md_axial": 9.7,
        "2md_deviation": 27.5, "3md_deviation": 16.7, "4md_deviation": 9.2, "5md_deviation": 25.6,
    },
    "Mean": {
        "2mcp_axial": -1, "3mcp_axial": -5, "4mcp_axial": -8, "5mcp_axial": -13,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": 1, "3pm_axial": 6, "4pm_axial": -9, "5pm_axial": -8,
        "2pm_deviation": 11, "3pm_deviation": 19, "4pm_deviation": 16, "5pm_deviation": 17,
        "2md_axial": 4, "3md_axial": 5, "4md_axial": 1, "5md_axial": 5,
        "2md_deviation": 10, "3md_deviation": 8, "4md_deviation": 6, "5md_deviation": 9,
    },
    "Min": {
        "2mcp_axial": -7, "3mcp_axial": -12, "4mcp_axial": -15, "5mcp_axial": -26,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": -11, "3pm_axial": -1, "4pm_axial": -16, "5pm_axial": -14,
        "2pm_deviation": 5, "3pm_deviation": 4, "4pm_deviation": 5, "5pm_deviation": 9,
        "2md_axial": -3, "3md_axial": 0, "4md_axial": -15, "5md_axial": -2,
        "2md_deviation": 5, "3md_deviation": 5, "4md_deviation": 2, "5md_deviation": 4,
    },
    "Max": {
        "2mcp_axial": 3, "3mcp_axial": 0, "4mcp_axial": -1, "5mcp_axial": -5,
        "2mcp_deviation": 0, "3mcp_deviation": 0, "4mcp_deviation": 0, "5mcp_deviation": 0,
        "2pm_axial": 6, "3pm_axial": 10, "4pm_axial": -4, "5pm_axial": -2,
        "2pm_deviation": 30, "3pm_deviation": 39, "4pm_deviation": 36, "5pm_deviation": 33,
        "2md_axial": 9, "3md_axial": 12, "4md_axial": 11, "5md_axial": 9,
        "2md_deviation": 30, "3md_deviation": 13, "4md_deviation": 10, "5md_deviation": 22,
    },
}

# ==============================
#  Columnas de exportación
# ==============================
COLUMNS_ORDER = ["time"]
for _x in FINGER_IDS:
    for _j in JOINT_SEQUENCE:
        for _d in DOF_TYPES:
            COLUMNS_ORDER.append(f"{_x}{_j}_{_d}")

FORCE_COLS_VX  = [f"{n}_force_vx" for n in NAME_MAPPING.values()]
FORCE_COLS_VY  = [f"{n}_force_vy" for n in NAME_MAPPING.values()]
FORCE_COLS_VZ  = [f"{n}_force_vz" for n in NAME_MAPPING.values()]
ALL_FORCE_COLS = FORCE_COLS_VX + FORCE_COLS_VY + FORCE_COLS_VZ
ALL_COLUMNS    = COLUMNS_ORDER + ALL_FORCE_COLS

# ==============================
#  Métricas de calidad de agarre
# ==============================
MAIN_METRICS      = ["QDCC", "FAD", "RFD", "RFD_x", "RFD_y", "RFD_z",
                     "RE", "QAGP", "QSGP", "FSD", "AXB", "CTR", "QVEW"]
ASCENDING_METRICS = {"QDCC", "FAD", "AXB", "QSGP", "FSD"}

FULL_NAMES = {
    "QDCC":  "Centroid Distance to COM",
    "RFD":   "Resultant Force Magnitude",
    "RFD_x": "Resultant Force X (axial)",
    "RFD_y": "Resultant Force Y (radial horiz.)",
    "RFD_z": "Resultant Force Z (radial vert.)",
    "FAD":   "Force Angular Deviation",
    "RE":    "Radial Efficiency",
    "QAGP":  "Area of Grasping Polygon",
    "QSGP":  "Regularity of Grasping Polygon",
    "FSD":   "Force Standard Deviation",
    "AXB":   "Axial Balance",
    "CTR":   "Cylindrical Torque Resistance",
    "QVEW":  "Volume of Grasp Wrench Space",
}

UNITS = {
    "QDCC":  "mm",
    "RFD":   "N",
    "RFD_x": "N",
    "RFD_y": "N",
    "RFD_z": "N",
    "FAD":   "deg",
    "RE":    "—",
    "QAGP":  "mm²",
    "QSGP":  "rad",
    "FSD":   "N",
    "AXB":   "N",
    "CTR":   "N·m",
    "QVEW":  "—",
}