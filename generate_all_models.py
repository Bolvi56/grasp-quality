# -*- coding: utf-8 -*-
"""
generate_all_models.py
Genera todos los modelos .osim combinando oblicuidades × deviations MCP.

  - Oblicuidad      → setRangeMax() en axial/deviation de pm, md y mcp_axial
  - MCP deviation   → setRangeMax() Y default_value (valor fijo)
  - Cilindro        → regex post-save en filename del ContactMesh jamar

CONFIGURACIÓN MANUAL — EDITAR AQUÍ:
  BASE_MODEL   : ruta al .osim base
  DIAMETER_MM  : diámetro del cilindro que pusiste en el .osim base
  OUTPUT_DIR   : carpeta de salida
"""

import os
import re
import numpy as np
import opensim as osim
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════╗
# ║              CONFIGURACIÓN MANUAL — EDITAR AQUÍ             ║
# ╠══════════════════════════════════════════════════════════════╣
BASE_DIR    = r"C:\Users\joserojas\Documents\Coding\Opensim\ARMS\Geometry_flatted"
DIAMETERS   = [25, 30, 35, 40, 45, 50]   # ← quita los que no quieras
OUTPUT_DIR  = r"C:\Users\joserojas\Documents\Coding\Opensim\ARMS\Geometry_flatted\Models\Full"
N_WORKERS   = 36
# ╚══════════════════════════════════════════════════════════════╝

from config import ALL_LIMITS, DEV_CONFIGS


# ══════════════════════════════════════════════════════════════
#  Patch filename del ContactMesh jamar (post-save)
# ══════════════════════════════════════════════════════════════

def patch_jamar_filename(osim_path: str, new_filename: str) -> None:
    with open(osim_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r'(<ContactMesh\b[^>]*\bname\s*=\s*["\']jamar["\'][^>]*>)'
        r'(.*?)'
        r'(</ContactMesh>)',
        re.DOTALL,
    )

    def replacer(m):
        inner = re.sub(
            r'(<filename\s*>)[^<]*(</\s*filename\s*>)',
            r'\g<1>' + new_filename + r'\2',
            m.group(2),
        )
        return m.group(1) + inner + m.group(3)

    new_text, n = pattern.subn(replacer, text)
    if n == 0:
        print(f"    ⚠  ContactMesh 'jamar' no encontrado en {osim_path}")
        return

    with open(osim_path, "w", encoding="utf-8") as f:
        f.write(new_text)


# ══════════════════════════════════════════════════════════════
#  Generador de un modelo
# ══════════════════════════════════════════════════════════════

def generate_model_worker(args: tuple) -> tuple:
    """Corre en proceso hijo. Retorna (model_name, ok, error_msg)"""
    base_model_path, output_path, obl_name, dev_cfg, diameter_mm = args

    try:
        model  = osim.Model(base_model_path)
        coords = model.getCoordinateSet()
        limits = ALL_LIMITS[obl_name]

        for coord_name, max_deg in limits.items():
            if coords.contains(coord_name):
                coords.get(coord_name).setRangeMax(np.deg2rad(max_deg))

        for finger, deg in dev_cfg["mcp_dev"].items():
            coord_name = f"{finger}mcp_deviation"
            if coords.contains(coord_name):
                coord = coords.get(coord_name)
                coord.setRangeMax(np.deg2rad(deg))
                coord.setDefaultValue(np.deg2rad(deg))

        model.printToXML(output_path)
        patch_jamar_filename(output_path, f"jamar_{diameter_mm}mm.obj")

        return (Path(output_path).stem, True, "")
    except Exception as exc:
        return (Path(output_path).stem, False, str(exc))


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    obliquities    = list(ALL_LIMITS.keys())
    deviations     = DEV_CONFIGS
    total_expected = len(DIAMETERS) * len(obliquities) * len(deviations)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Construir todas las tareas
    tasks = []
    for diameter_mm in DIAMETERS:
        base_model = os.path.join(BASE_DIR, f"Base_Model_{diameter_mm}mm.osim")
        if not os.path.exists(base_model):
            print(f"  ⚠  Modelo base no encontrado, saltando: {base_model}")
            continue
        for obl_name in obliquities:
            for dev_cfg in deviations:
                dev_name   = dev_cfg["name"]
                model_name = f"Oblique_{obl_name}_{diameter_mm:02d}mm_p{dev_name}"
                out_path   = os.path.join(OUTPUT_DIR, f"{model_name}.osim")
                tasks.append((base_model, out_path, obl_name, dev_cfg, diameter_mm))

    print()
    print("=" * 70)
    print("  GENERATE ALL MODELS — PARALELO")
    print("=" * 70)
    print(f"  Diámetros    : {DIAMETERS}")
    print(f"  Obliquities  : {len(obliquities)}")
    print(f"  Dev configs  : {len(deviations)}")
    print(f"  Total models : {len(tasks)}")
    print(f"  Núcleos      : {N_WORKERS}")
    print("=" * 70)
    print()

    generated = 0
    errors    = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(generate_model_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Generando modelos",
                  bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n}/{total} [{elapsed}<{remaining}]") as bar:
            for future in as_completed(futures):
                model_name, ok, err = future.result()
                if ok:
                    generated += 1
                else:
                    errors.append((model_name, err))
                bar.set_postfix_str(model_name[-30:])
                bar.update(1)

    print()
    print("=" * 70)
    print(f"  Generados : {generated} / {len(tasks)}")
    if errors:
        print(f"  Errores   : {len(errors)}")
        for name, err in errors:
            print(f"    ✗  {name}: {err}")
    else:
        print("  Errores   : 0  ✅")
    print("=" * 70)
    print()