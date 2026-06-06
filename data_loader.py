"""
src/data_loader.py
==================
Carga y preprocesamiento de historiales JSONL.
Incluye lógica de fases del curriculum learning.

Fases (1800 episodios totales):
  Fase 1: ep   1– 270  Radio fijo ★        ga0=3, sa0=200
  Fase 2: ep 271– 540  Radio ±(1.0, 50)    vecindad de ★
  Fase 3: ep 541– 810  Radio ±(3.0, 150)   zona media
  Fase 4: ep 811–1800  Todo el mapa        política madura
"""

import json, glob
import numpy as np
import pandas as pd
from pathlib import Path

# ── Definición de fases ───────────────────────────────────────────────────────
FASES = {1: (1, 270), 2: (271, 540), 3: (541, 810), 4: (811, 1800)}
COLORES_FASE = {1: "#D85A30", 2: "#BA7517", 3: "#185FA5", 4: "#1D9E75"}
LABELS_FASE  = {
    1: "Fase 1\n(fijo ★)",
    2: "Fase 2\n(vecindad)",
    3: "Fase 3\n(zona media)",
    4: "Fase 4\n(mapa completo)",
}

def asignar_fase(ep: int) -> int:
    for f, (ini, fin) in FASES.items():
        if ini <= ep <= fin:
            return f
    return 4


# ── Nivel episodio ────────────────────────────────────────────────────────────

def cargar_semilla(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            row = {k: v for k, v in d.items()
                   if k not in ("trayectoria", "acciones", "neck_por_paso")}
            row["neck_por_paso"] = d.get("neck_por_paso", [])
            rows.append(row)
    df = pd.DataFrame(rows)
    # Expandir vector w
    ws = np.array(df["w"].tolist())
    for i, name in enumerate(["w_bif", "w_dist", "w_rob", "w_vir"]):
        df[name] = ws[:, i]
    df.drop(columns=["w"], inplace=True)
    # Añadir fase
    df["fase"] = df["ep"].apply(asignar_fase)
    return df


def cargar_pasos(path: str, solo_fase: int = None,
                 min_pasos: int = 2) -> pd.DataFrame:
    """
    DataFrame de nivel PASO. Filtra episodios triviales (pasos < min_pasos).
    solo_fase: si se especifica, solo carga episodios de esa fase.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ep    = d["ep"]
            fase  = asignar_fase(ep)
            if solo_fase and fase != solo_fase:
                continue
            if d["pasos"] < min_pasos:
                continue
            exito  = d["exito"]
            n_pasos = d["pasos"]
            ws      = d.get("w", [0,0,0,0])
            for t in d.get("trayectoria", []):
                if t.get("accion") is None:
                    continue
                rows.append({
                    "ep": ep, "fase": fase, "paso": t["paso"],
                    "ga0": t["ga0"], "sa0": t["sa0"],
                    "neck_gap": t.get("neck_gap"),
                    "tipo": t.get("tipo"),
                    "r_bif":  t.get("r_bif"),
                    "r_dist": t.get("r_dist"),
                    "r_rob":  t.get("r_rob"),
                    "accion_ga0": t["accion"][0],
                    "accion_sa0": t["accion"][1],
                    "exito": exito, "pasos_ep": n_pasos,
                    "w_bif": ws[0], "w_dist": ws[1],
                    "w_rob": ws[2], "w_vir": ws[3],
                })
    return pd.DataFrame(rows)


# ── Nivel semilla ─────────────────────────────────────────────────────────────

def semilla_id(path: str) -> str:
    return Path(path).stem.replace("historial_", "")


def agregar_semilla(df: pd.DataFrame, seed_id: str) -> dict:
    ep_f4   = df[df["fase"] == 4]
    ep_fin50 = df[df["ep"] >= df["ep"].max() - 49]
    ep_f1   = df[df["fase"] == 1]

    row = {"semilla": seed_id, "n_episodios": len(df)}
    # Por fase
    for f in [1, 2, 3, 4]:
        sub = df[df["fase"] == f]
        row[f"exito_f{f}"]  = sub["exito"].mean() if len(sub) > 0 else np.nan
        row[f"pasos_f{f}"]  = sub["pasos"].mean() if len(sub) > 0 else np.nan
    # Globales
    row["tasa_exito_global"]   = df["exito"].mean()
    row["tasa_exito_final50"]  = ep_fin50["exito"].mean()
    row["tasa_exito_f4"]       = ep_f4["exito"].mean() if len(ep_f4) > 0 else np.nan
    row["pasos_media"]         = df["pasos"].mean()
    row["pasos_f4_media"]      = ep_f4["pasos"].mean() if len(ep_f4) > 0 else np.nan
    row["r_rob_final50"]       = ep_fin50["r_rob"].mean() if "r_rob" in ep_fin50 else np.nan
    row["r_dist_final50"]      = ep_fin50["r_dist"].mean() if "r_dist" in ep_fin50 else np.nan
    row["pct_triviales"]       = (df["pasos"] <= 1).mean()
    row["pct_isola"]           = (df["tipo_final"] == "Isola").mean()
    row["neck_final_media"]    = df["neck_final"].mean() if "neck_final" in df else np.nan
    # Costo del curriculum: caída de éxito al entrar fase 2
    if len(ep_f1) > 0 and len(df[df["fase"] == 2]) > 0:
        f1_last30  = df[(df["fase"]==1) & (df["ep"] >= FASES[1][1]-29)]["exito"].mean()
        f2_first30 = df[(df["fase"]==2) & (df["ep"] <= FASES[2][0]+29)]["exito"].mean()
        row["costo_curriculum"] = float(f1_last30 - f2_first30)
    else:
        row["costo_curriculum"] = np.nan
    return row


def cargar_todas_semillas(patron: str):
    paths = sorted(glob.glob(patron))
    if not paths:
        raise FileNotFoundError(f"No hay archivos: {patron}")
    dfs, metas = [], []
    for p in paths:
        sid = semilla_id(p)
        df  = cargar_semilla(p)
        df["semilla"] = sid
        dfs.append(df)
        metas.append(agregar_semilla(df, sid))
        print(f"  {sid}: éxito_global={df['exito'].mean():.3f}  "
              f"éxito_f4={df[df['fase']==4]['exito'].mean():.3f}")
    return dfs, pd.DataFrame(metas)


# ── Trayectorias neck rellenas ─────────────────────────────────────────────────

def necks_rellenos(df: pd.DataFrame, max_pasos: int = 80) -> np.ndarray:
    necks_raw = df["neck_por_paso"].tolist()
    mat = np.full((len(necks_raw), max_pasos), np.nan)
    for i, n in enumerate(necks_raw):
        if len(n) == 0:
            mat[i, :] = 0.0
        else:
            mat[i, : len(n)] = n
            mat[i, len(n):] = n[-1]
    return mat
