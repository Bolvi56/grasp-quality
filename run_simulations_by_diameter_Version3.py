# -*- coding: utf-8 -*-
"""
run_simulations_by_diameter_Version3.py  —  Paralelo con ProcessPoolExecutor
"""

import sys
import os
import opensim as osim
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    FULL_MODELS, RESULTS_DIR, SUMMARY_DIR,
    OBLIQUE_NAMES, DIAMETERS, DEV_CONFIGS, DEV_CONFIG_NAMES,
    FINGER_IDS, JOINT_SEQUENCE,
    STEPS_DEG, RETROCESO_MULT, FORCE_TOLERANCE, OBJECT_BODY,
    NAME_MAPPING, COLUMNS_ORDER,
    FORCE_COLS_VX, FORCE_COLS_VY, FORCE_COLS_VZ, ALL_COLUMNS,
    R_GROUND_TO_LOCAL,
    compute_force_thresholds,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ╔══════════════════════════════════════════════════════════════╗
# ║              CONFIGURACIÓN MANUAL — EDITAR AQUÍ             ║
# ╠══════════════════════════════════════════════════════════════╣
DIAMETERS_TO_RUN = [25, 30, 35, 40, 45, 50]  # ← quita los que no quieras
RESUME           = False
N_WORKERS        = 36
# ╚══════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════
#  Helpers de nombres
# ══════════════════════════════════════════════════════════════

def model_filename(oblique: str, dev_name: str, diameter_mm: int) -> str:
    return f"Oblique_{oblique}_{diameter_mm:02d}mm_p{dev_name}.osim"

def result_dir(oblique: str, diameter_mm: int, dev_name: str) -> Path:
    return RESULTS_DIR / f"d{diameter_mm:02d}mm" / oblique / dev_name

def result_stem(oblique: str, diameter_mm: int, dev_name: str) -> str:
    return f"Oblique_{oblique}_{diameter_mm:02d}mm_p{dev_name}"


# ══════════════════════════════════════════════════════════════
#  Helpers de fuerza
# ════════════════════════════════════════════════════════════��═

def compute_expected_force(f_min: float, f_max: float) -> float:
    return ((f_min / (1 - FORCE_TOLERANCE)) + (f_max / (1 + FORCE_TOLERANCE))) / 2


# ══════════════════════════════════════════════════════════════
#  DOFs secundarios
# ══════════════════════════════════════════════════════════════

def scale_secondary_dofs(joint_prefix, coords, state, model) -> None:
    flex_name = f"{joint_prefix}_flexion"
    if flex_name not in coords:
        return
    max_flex  = coords[flex_name].getRangeMax()
    flex_val  = coords[flex_name].getValue(state)
    flex_norm = flex_val / max_flex if max_flex != 0 else 0.0
    for suffix in ["_axial", "_deviation"]:
        dof_name = f"{joint_prefix}{suffix}"
        if dof_name in coords:
            val = flex_norm * coords[dof_name].getRangeMax()
            coords[dof_name].setValue(state, val)
            coords[dof_name].setDefaultValue(val)
    model.realizeVelocity(state)


# ══════════════════════════════════════════════════════════════
#  Helpers de contacto
# ══════════════════════════════════════════════════════════════

def get_contact_full(contacto, contact_forces, state, model) -> tuple:
    force = contact_forces.get(contacto)
    if force:
        try:
            model.realizeDynamics(state)
            vals = force.getRecordValues(state)
            n    = vals.size()
            def v(i): return float(vals.get(i)) if i < n else 0.0
            return (v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8))
        except Exception as e:
            logging.warning(f"Error leyendo contacto {contacto}: {e}")
    return (0.0,) * 9

def get_contact_force(contacto, contact_forces, state, model) -> tuple:
    fx, fy, fz, *_ = get_contact_full(contacto, contact_forces, state, model)
    return fx, fy, fz


# ══════════════════════════════════════════════════════════════
#  Posición jamar
# ══════════════════════════════════════════════════════════════

def get_jamar_position(model, state) -> np.ndarray:
    try:
        model.realizeDynamics(state)
        body      = model.getBodySet().get(OBJECT_BODY)
        com_local = body.getMassCenter()
        com_g     = body.findStationLocationInGround(state, com_local)
        return np.array([com_g.get(0), com_g.get(1), com_g.get(2)])
    except Exception as e:
        logging.warning(f"Error obteniendo posición de {OBJECT_BODY}: {e}")
        return np.zeros(3)


# ══════════════════════════════════════════════════════════════
#  Helpers de estado de dedos
# ══════════════════════════════════════════════════════════════

def active_joint_name(fid: int, finger_state: dict) -> str:
    jtype = JOINT_SEQUENCE[finger_state[fid]["joint_idx"]]
    return f"{fid}{jtype}_flexion"

def advance_to_next_joint(fid: int, finger_state: dict) -> None:
    fs = finger_state[fid]
    fs["joint_idx"] += 1
    fs["step_idx"]   = 0
    fs["angle"]      = 0.0
    fs["resolved"]   = False
    fs["failed"]     = False
    if fs["joint_idx"] >= len(JOINT_SEQUENCE):
        fs["done"] = True


# ══════════════════════════════════════════════════════════════
#  Resumen por articulación
# ══════════════════════════════════════════════════════════════

def build_summary_row(oblique, diameter_mm, dev_name, joint, angle,
                      fx, fy, fz, umbral) -> dict:
    contacto     = NAME_MAPPING[joint]
    f_min, f_max = umbral[contacto]
    expected     = compute_expected_force(f_min, f_max)
    f_total      = (fx**2 + fy**2 + fz**2) ** 0.5
    error        = f_total - expected
    error_pct    = (error / expected) * 100 if expected != 0 else 0.0
    return {
        "Oblique":         oblique,
        "Diameter_mm":     diameter_mm,
        "Dev_config":      dev_name,
        "Joint":           joint,
        "Angle (°)":       round(np.rad2deg(angle), 4),
        "Force X (N)":     round(fx, 4),
        "Force Y (N)":     round(fy, 4),
        "Force Z (N)":     round(fz, 4),
        "Force Total (N)": round(f_total, 4),
        "Expected (N)":    round(expected, 4),
        "Error (N)":       round(error, 4),
        "Error (%)":       round(error_pct, 3),
    }


# ══════════════════════════════════════════════════════════════
#  Export helpers
# ══════════════════════════════════════════════════════════════

def export_mot_file(df: pd.DataFrame, filepath: Path) -> None:
    force_cols = [col for col in df.columns if "_force_v" in col]
    df_export  = df[["time"] + force_cols]
    header = (
        f"{filepath.name}\nversion=1\n"
        f"nRows={len(df_export)}\nnColumns={len(df_export.columns)}\n"
        "inDegrees=no\nendheader\n"
    )
    with open(filepath, "w") as f:
        f.write(header)
        df_export.to_csv(f, sep="\t", index=False, float_format="%.6f")

def export_sto_file(df: pd.DataFrame, filepath: Path) -> None:
    header = (
        "Coordinates\nversion=1\n"
        f"nRows={df.shape[0]}\nnColumns={df.shape[1]}\n"
        "inDegrees=yes\nendheader\n"
    )
    with open(filepath, "w") as f:
        f.write(header)
        df.to_csv(f, sep="\t", index=False, float_format="%.6f")

def generate_external_loads_xml(output_path: Path, mapping: dict, stem: str) -> None:
    forces_xml = ""
    for contacto in mapping.values():
        forces_xml += (
            f'\n      <ExternalForce name="{contacto}_force">'
            f'\n        <isDisabled>false</isDisabled>'
            f'\n        <applied_to_body>{contacto}</applied_to_body>'
            f'\n        <force_expressed_in_body>ground</force_expressed_in_body>'
            f'\n        <point_expressed_in_body>ground</point_expressed_in_body>'
            f'\n        <force_identifier>{contacto}_force_v</force_identifier>'
            f'\n        <data_source_name>{stem}</data_source_name>'
            f'\n      </ExternalForce>'
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OpenSimDocument Version="40000">\n'
        '  <ExternalLoads name="HandForces">\n'
        f'    <objects>{forces_xml}\n    </objects>\n'
        f'    <datafile>{stem}.mot</datafile>\n'
        '  </ExternalLoads>\n'
        '</OpenSimDocument>'
    )
    with open(output_path, "w") as f:
        f.write(xml)

def export_contacts_csv(fixed_angles, contact_forces, state, model,
                        filepath, cylinder_radius) -> None:
    jamar_com = get_jamar_position(model, state)
    R_g2l     = R_GROUND_TO_LOCAL
    rows      = []

    for joint, angle in fixed_angles.items():
        contacto                            = NAME_MAPPING[joint]
        fx, fy, fz, tx, ty, tz, px, py, pz = get_contact_full(
            contacto, contact_forces, state, model
        )
        point_local_body = np.array([px, py, pz]) / 1000.0
        try:
            model.realizeDynamics(state)
            body        = model.getBodySet().get(contacto)
            origin_g    = body.findStationLocationInGround(state, osim.Vec3(0, 0, 0))
            origin_g_np = np.array([origin_g.get(0), origin_g.get(1), origin_g.get(2)])
            R_body      = np.zeros((3, 3))
            for i in range(3):
                col = body.expressVectorInGround(
                    state,
                    osim.Vec3(1 if i == 0 else 0, 1 if i == 1 else 0, 1 if i == 2 else 0),
                )
                R_body[:, i] = [col.get(0), col.get(1), col.get(2)]
            point_ground = origin_g_np + R_body @ point_local_body
        except Exception:
            point_ground = jamar_com + point_local_body

        r_ground    = point_ground - jamar_com
        r_local     = R_g2l @ r_ground
        r_axial     = r_local[0]
        r_radial_yz = r_local[1:]
        radial_norm = np.linalg.norm(r_radial_yz)
        r_contact_yz = (
            (r_radial_yz / radial_norm) * cylinder_radius
            if radial_norm > 1e-9 else r_radial_yz
        )
        r_contact_local  = np.array([r_axial, r_contact_yz[0], r_contact_yz[1]])
        r_contact_ground = R_g2l.T @ r_contact_local

        f_vec        = np.array([fx, fy, fz])
        f_norm_val   = np.linalg.norm(f_vec)
        force_normal = f_vec / f_norm_val if f_norm_val > 1e-9 else np.array([0.0, 1.0, 0.0])

        r_local2   = R_g2l @ r_contact_ground
        r_radial   = np.array([0.0, r_local2[1], r_local2[2]])
        r_rad_norm = np.linalg.norm(r_radial)
        geom_normal_local  = (
            r_radial / r_rad_norm if r_rad_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        )
        geom_normal_ground = R_g2l.T @ geom_normal_local

        rows.append({
            "joint":         joint,          "contact":       contacto,
            "angle_deg":     round(np.rad2deg(angle), 5),
            "force_x":       round(fx, 9),   
            "force_y":       round(fy, 9),
            "force_z":       round(fz, 9),   
            "torque_x":      round(tx, 9),
            "torque_y":      round(ty, 9),   
            "torque_z":      round(tz, 9),
            "point_x":       round(float(r_contact_ground[0]), 9),
            "point_y":       round(float(r_contact_ground[1]), 9),
            "point_z":       round(float(r_contact_ground[2]), 9),
            "jamar_cx":      round(float(jamar_com[0]), 9),
            "jamar_cy":      round(float(jamar_com[1]), 9),
            "jamar_cz":      round(float(jamar_com[2]), 9),
            "r_x":           round(float(r_contact_ground[0]), 9),
            "r_y":           round(float(r_contact_ground[1]), 9),
            "r_z":           round(float(r_contact_ground[2]), 9),
            "normal_x":      round(float(force_normal[0]), 9),
            "normal_y":      round(float(force_normal[1]), 9),
            "normal_z":      round(float(force_normal[2]), 9),
            "geom_normal_x": round(float(geom_normal_ground[0]), 9),
            "geom_normal_y": round(float(geom_normal_ground[1]), 9),
            "geom_normal_z": round(float(geom_normal_ground[2]), 9),
        })

    pd.DataFrame(rows).to_csv(filepath, index=False)


# ══════════════════════════════════════════════════════════════
#  Función de simulación — corre en un proceso hijo
#  (sin tqdm de dedos, para no mezclar salida entre procesos)
# ══════════════════════════════════════════════════════════════

def run_simulation_worker(args: tuple) -> list[dict]:
    """
    Punto de entrada para cada proceso hijo.
    Recibe (oblique, diameter_mm, dev_cfg, resume) como tupla
    porque ProcessPoolExecutor solo admite un argumento.
    """
    oblique, diameter_mm, dev_cfg, resume = args

    dev_name   = dev_cfg["name"]
    umbral     = compute_force_thresholds(diameter_mm)
    cylinder_r = diameter_mm / 2000.0

    model_path = FULL_MODELS / model_filename(oblique, dev_name, diameter_mm)
    out_dir    = result_dir(oblique, diameter_mm, dev_name)
    stem       = result_stem(oblique, diameter_mm, dev_name)
    tag        = f"{oblique}|d{diameter_mm}|{dev_name}"

    if not model_path.exists():
        return []

    if resume and (out_dir / f"{stem}_contacts.csv").exists():
        return []   # señal de "saltado"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar modelo ────────────────────────────────────────────────────────
    model = osim.Model(str(model_path))
    state = model.initSystem()

    coords: dict = {c.getName(): c for c in model.getCoordinateSet()}

    for c in coords.values():
        c.setLocked(state, False)
        c.setSpeedValue(state, 0.0)

    for fid in FINGER_IDS:
        dev_coord = f"{fid}mcp_deviation"
        if dev_coord in coords:
            coords[dev_coord].setValue(state, coords[dev_coord].getDefaultValue())

    for name, c in coords.items():
        if "mcp_deviation" not in name:
            c.setValue(state, 0.0)

    model.realizeVelocity(state)

    contact_forces = {}
    for i in range(model.getForceSet().getSize()):
        f = model.getForceSet().get(i)
        if f.getName() in NAME_MAPPING.values():
            contact_forces[f.getName()] = f

    finger_state = {
        fid: {"joint_idx": 0, "step_idx": 0, "angle": 0.0,
              "resolved": False, "failed": False, "done": False}
        for fid in FINGER_IDS
    }

    total_time               = 0.0
    time_series_data         = []
    fixed_angles             = {}
    contact_force_x_detected = {}
    contact_force_y_detected = {}
    contact_force_z_detected = {}

    def apply_angle(joint_name, angle):
        coords[joint_name].setValue(state, angle)
        coords[joint_name].setDefaultValue(angle)
        prefix = joint_name.replace("_flexion", "")
        if "mcp" in prefix:
            max_flex  = coords[joint_name].getRangeMax()
            flex_val  = coords[joint_name].getValue(state)
            flex_norm = flex_val / max_flex if max_flex != 0 else 0.0
            axial_name = f"{prefix}_axial"
            if axial_name in coords:
                val = flex_norm * coords[axial_name].getRangeMax()
                coords[axial_name].setValue(state, val)
                coords[axial_name].setDefaultValue(val)
            model.realizeVelocity(state)
        else:
            scale_secondary_dofs(prefix, coords, state, model)

    def record_state():
        nonlocal total_time
        model.realizeDynamics(state)
        row = {"time": round(total_time, 4)}
        for c in model.getCoordinateSet():
            row[c.getName()] = np.rad2deg(c.getValue(state))
        for contacto in NAME_MAPPING.values():
            fx, fy, fz = get_contact_force(contacto, contact_forces, state, model)
            row[f"{contacto}_force_vx"] = fx
            row[f"{contacto}_force_vy"] = fy
            row[f"{contacto}_force_vz"] = fz
        for col in ALL_COLUMNS:
            row.setdefault(col, 0.0)
        time_series_data.append(row)
        total_time += 0.005

    # ── Loop principal ───────────────────────────────────────────────────────
    while not all(finger_state[fid]["done"] for fid in FINGER_IDS):
        any_active = False

        for fid in FINGER_IDS:
            fs = finger_state[fid]
            if fs["done"]:
                continue
            any_active  = True
            joint_name  = active_joint_name(fid, finger_state)
            joint_label = JOINT_SEQUENCE[fs["joint_idx"]].upper()
            step_deg    = STEPS_DEG[fs["step_idx"]]
            new_angle   = fs["angle"] + np.deg2rad(step_deg)
            max_range   = coords[joint_name].getRangeMax()

            if new_angle > max_range:
                fs["failed"] = True
                advance_to_next_joint(fid, finger_state)
                continue

            fs["angle"] = new_angle
            apply_angle(joint_name, new_angle)

        if not any_active:
            break

        try:
            model.realizeDynamics(state)
        except Exception as e:
            logging.error(f"[{tag}] realizeDynamics: {e}")
            break

        record_state()

        for fid in FINGER_IDS:
            fs = finger_state[fid]
            if fs["done"] or fs["failed"]:
                continue

            joint_name   = active_joint_name(fid, finger_state)
            joint_label  = JOINT_SEQUENCE[fs["joint_idx"]].upper()
            contacto     = NAME_MAPPING[joint_name]
            f_min, f_max = umbral[contacto]
            step_deg     = STEPS_DEG[fs["step_idx"]]

            fx, fy, fz = get_contact_force(contacto, contact_forces, state, model)
            f_total    = (fx**2 + fy**2 + fz**2) ** 0.5

            if f_total >= f_min:
                expected  = compute_expected_force(f_min, f_max)
                error_pct = abs((f_total - expected) / expected) * 100

                if error_pct <= FORCE_TOLERANCE * 100:
                    fixed_angles[joint_name]             = fs["angle"]
                    contact_force_x_detected[joint_name] = fx
                    contact_force_y_detected[joint_name] = fy
                    contact_force_z_detected[joint_name] = fz
                    advance_to_next_joint(fid, finger_state)
                else:
                    if fs["step_idx"] >= len(STEPS_DEG) - 1:
                        fixed_angles[joint_name]             = fs["angle"]
                        contact_force_x_detected[joint_name] = fx
                        contact_force_y_detected[joint_name] = fy
                        contact_force_z_detected[joint_name] = fz
                        advance_to_next_joint(fid, finger_state)
                    else:
                        retroceso_rad = RETROCESO_MULT * np.deg2rad(step_deg)
                        new_a = max(0.0, fs["angle"] - retroceso_rad)
                        fs["step_idx"] += 1
                        fs["angle"]     = new_a
                        apply_angle(joint_name, new_a)

    # ── Exportar ─────────────────────────────────────────────────────────────
    try:
        model.printToXML(str(FULL_MODELS / f"{stem}_Final.osim"))
    except Exception as e:
        logging.error(f"[{tag}] Error guardando modelo: {e}")

    df_all = pd.DataFrame(time_series_data)

    mot_cols = ["time"] + FORCE_COLS_VX + FORCE_COLS_VY + FORCE_COLS_VZ
    df_mot   = df_all[mot_cols].copy().fillna(0.0)
    df_mot["time"] = df_mot["time"].round(3)
    export_mot_file(df_mot, out_dir / f"{stem}.mot")

    df_angles = (
        df_all[COLUMNS_ORDER]
        .drop_duplicates(subset=["time"])
        .sort_values("time")
    )
    df_angles["time"] = df_angles["time"].round(3)
    export_sto_file(df_angles, out_dir / f"{stem}.sto")

    generate_external_loads_xml(out_dir / f"{stem}.xml", NAME_MAPPING, stem)

    export_contacts_csv(
        fixed_angles, contact_forces, state, model,
        out_dir / f"{stem}_contacts.csv",
        cylinder_radius=cylinder_r,
    )

    return [
        build_summary_row(
            oblique, diameter_mm, dev_name, joint, angle,
            contact_force_x_detected.get(joint, 0.0),
            contact_force_y_detected.get(joint, 0.0),
            contact_force_z_detected.get(joint, 0.0),
            umbral,
        )
        for joint, angle in fixed_angles.items()
    ]


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Validar diámetros
    invalid = [d for d in DIAMETERS_TO_RUN if d not in DIAMETERS]
    if invalid:
        print(f"❌ Diámetros no válidos: {invalid}. Opciones: {DIAMETERS}")
        sys.exit(1)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Construir todas las tareas (todos los diámetros × obliques × devs)
    tasks = [
        (oblique, diameter_mm, dev_cfg, RESUME)
        for diameter_mm in DIAMETERS_TO_RUN
        for oblique    in OBLIQUE_NAMES
        for dev_cfg    in DEV_CONFIGS
    ]
    total_runs    = len(tasks)
    overall_start = time.time()
    all_summary_rows: list[dict] = []

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SIMULACIONES PARALELAS — Todos los diámetros")
    print(f"{'='*65}")
    print(f"  Diámetros    : {DIAMETERS_TO_RUN}")
    print(f"  Obliques     : {len(OBLIQUE_NAMES)}")
    print(f"  Dev configs  : {len(DEV_CONFIGS)}")
    print(f"  Total runs   : {total_runs}")
    print(f"  Núcleos      : {N_WORKERS} / {os.cpu_count()}")
    if RESUME:
        print("  Modo RESUME  : activado")
    print(f"{'='*65}\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_simulation_worker, t): t for t in tasks}

        with tqdm(
            total=total_runs,
            desc="Simulaciones completadas",
            position=0,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n}/{total} [{elapsed}<{remaining}]",
        ) as bar:
            for future in as_completed(futures):
                oblique, diameter_mm, dev_cfg, _ = futures[future]
                dev_name = dev_cfg["name"]
                try:
                    rows = future.result()
                    all_summary_rows.extend(rows)
                    bar.set_postfix_str(f"{oblique}|d{diameter_mm}|{dev_name}")
                except Exception as e:
                    logging.error(f"Error en {oblique}|d{diameter_mm}|{dev_name}: {e}")
                bar.update(1)

    overall_end = time.time()

    # ── Resumen en consola ───────────────────────────────────────────────────
    sep = "=" * 125
    print(f"\n\n{sep}")
    print(f"  RESUMEN GLOBAL — {DIAMETERS_TO_RUN}")
    print(sep)
    print(
        f"{'Diameter':<10} {'Oblique':<10} {'Dev':<12} {'Joint':<18} "
        f"{'Angle(°)':<10} {'F_total(N)':<14} {'Expected(N)':<14} {'Error(%)'}"
    )
    print("-" * 125)

    all_summary_rows.sort(key=lambda r: (r["Diameter_mm"], r["Oblique"], r["Dev_config"], r["Joint"]))
    current_key = None
    for row in all_summary_rows:
        key = f"{row['Diameter_mm']}|{row['Oblique']}|{row['Dev_config']}"
        if key != current_key:
            if current_key is not None:
                print()
            current_key = key
        print(
            f"{row['Diameter_mm']:<10} {row['Oblique']:<10} {row['Dev_config']:<12} "
            f"{row['Joint']:<18} {row['Angle (°)']:<10} "
            f"{row['Force Total (N)']:<14} {row['Expected (N)']:<14} "
            f"{row['Error (%)']:.3f}%"
        )

    print(sep)
    print(f"Tiempo total: {overall_end - overall_start:.2f} s\n")

    # ── Excel por diámetro + global ──────────────────────────────────────────
    if all_summary_rows:
        df_all = pd.DataFrame(all_summary_rows)

        # Un Excel por diámetro
        for diameter_mm in DIAMETERS_TO_RUN:
            df_d = df_all[df_all["Diameter_mm"] == diameter_mm]
            if df_d.empty:
                continue
            excel_path = SUMMARY_DIR / f"results_d{diameter_mm:02d}mm.xlsx"
            try:
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    df_d.to_excel(writer, sheet_name="Global", index=False)
                    for obl in OBLIQUE_NAMES:
                        df_obl = df_d[df_d["Oblique"] == obl]
                        if not df_obl.empty:
                            df_obl.to_excel(writer, sheet_name=obl[:31], index=False)
                    for dev_cfg in DEV_CONFIGS:
                        df_dev = df_d[df_d["Dev_config"] == dev_cfg["name"]]
                        if not df_dev.empty:
                            df_dev.to_excel(writer, sheet_name=dev_cfg["name"][:31], index=False)
                print(f"  ✅ Excel: {excel_path.name}")
            except Exception as e:
                logging.error(f"Error Excel d{diameter_mm}: {e}")

        # Excel global con todos los diámetros
        global_path = SUMMARY_DIR / "results_all_diameters.xlsx"
        try:
            with pd.ExcelWriter(global_path, engine="openpyxl") as writer:
                df_all.to_excel(writer, sheet_name="Global", index=False)
                for diameter_mm in DIAMETERS_TO_RUN:
                    df_d = df_all[df_all["Diameter_mm"] == diameter_mm]
                    if not df_d.empty:
                        df_d.to_excel(writer, sheet_name=f"d{diameter_mm:02d}mm", index=False)
            print(f"  ✅ Excel global: {global_path.name}")
        except Exception as e:
            logging.error(f"Error Excel global: {e}")