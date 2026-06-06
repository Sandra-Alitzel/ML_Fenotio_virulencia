"""
src/analisis_pca.py
====================
Tema 5 — Modelos de variables latentes
PCA con análisis por fase del curriculum learning.

Figuras generadas:
  fig_pca_varianza_scatter_*     : varianza + scatter PC1-PC2 (todas las fases)
  fig_pca_trayectorias_*         : prototipos por fase
  fig_pca_evolucion_temporal_*   : PC1 medio a lo largo del entrenamiento
  fig_pca_fases_comparacion_*    : scatter PC1-PC2 separado por fase (2x2)
  fig_espacio_control_*          : trayectorias en (ga0, sa0) coloreadas por neck
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from data_loader import necks_rellenos, FASES, COLORES_FASE, LABELS_FASE

C = {"directo": "#185FA5", "explorador": "#BA7517",
     "fallido": "#D85A30", "neutral": "#888780"}


@dataclass
class ResultadoPCA:
    Z: np.ndarray
    varianza: np.ndarray
    componentes: np.ndarray
    pca: PCA
    necks: np.ndarray
    exitos: np.ndarray
    pasos: np.ndarray
    eps: np.ndarray
    tipos: np.ndarray
    fases: np.ndarray
    # Prototipos globales
    tray_directo: np.ndarray
    tray_explorador: np.ndarray
    tray_fallido: np.ndarray
    # Prototipos por fase
    tray_por_fase: dict = field(default_factory=dict)


def ejecutar_pca(df: pd.DataFrame, n_components: int = 10) -> ResultadoPCA:
    necks  = necks_rellenos(df)
    exitos = df["exito"].values
    pasos  = df["pasos"].values
    eps    = df["ep"].values
    tipos  = df["tipo_final"].values
    fases  = df["fase"].values

    X = StandardScaler().fit_transform(necks)
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)

    def _proto(mask, n=20):
        rows = necks[mask][:n]
        return np.nan_to_num(rows.mean(axis=0), nan=0.0) if mask.sum() > 0 \
               else np.zeros(80)

    tray_d = _proto((pasos <= 5)  & exitos)
    tray_e = _proto((pasos >= 20) & (pasos <= 60) & exitos)
    tray_f = _proto(~exitos)

    # Prototipos por fase (exitosos de cada fase)
    tray_por_fase = {}
    for f in [1, 2, 3, 4]:
        mask_f = (fases == f) & exitos & (pasos > 1)
        tray_por_fase[f] = _proto(mask_f)

    return ResultadoPCA(
        Z=Z, varianza=pca.explained_variance_ratio_,
        componentes=pca.components_, pca=pca,
        necks=necks, exitos=exitos, pasos=pasos,
        eps=eps, tipos=tipos, fases=fases,
        tray_directo=tray_d, tray_explorador=tray_e, tray_fallido=tray_f,
        tray_por_fase=tray_por_fase,
    )


# ── Figuras ────────────────────────────────────────────────────────────────────

def fig_varianza_y_scatter(res: ResultadoPCA, out: Path):
    """Varianza explicada + scatter PC1-PC2 coloreado por fase."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel a: varianza
    ax = axes[0]
    var = res.varianza
    x   = np.arange(1, len(var) + 1)
    ax.bar(x, var * 100, color=C["directo"], alpha=0.8, width=0.6)
    ax.step(x, np.cumsum(var) * 100, color=C["fallido"],
            lw=2, where="mid", label="Acumulada")
    ax.axhline(85, color=C["neutral"], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Componente principal", fontsize=11)
    ax.set_ylabel("Varianza explicada (%)", fontsize=11)
    ax.set_xticks(x); ax.legend(fontsize=9)
    ax.set_title("(a) Varianza explicada", fontsize=11)

    # Panel b: scatter coloreado por fase
    ax = axes[1]
    np.random.seed(42)
    idx = np.random.choice(len(res.Z), min(800, len(res.Z)), replace=False)
    for f in [1, 2, 3, 4]:
        mask = res.fases[idx] == f
        ax.scatter(res.Z[idx][mask, 0], res.Z[idx][mask, 1],
                   c=COLORES_FASE[f], s=10, alpha=0.5,
                   label=f"Fase {f}", zorder=f)
    # Marcar fracasos con ×
    mask_fail = ~res.exitos[idx]
    ax.scatter(res.Z[idx][mask_fail, 0], res.Z[idx][mask_fail, 1],
               marker="x", color="black", s=20, lw=0.7,
               alpha=0.6, label="Fracaso", zorder=5)
    ax.set_xlabel(f"PC1 ({res.varianza[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({res.varianza[1]:.1%})", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("(b) PC1–PC2 coloreado por fase", fontsize=11)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_trayectorias_prototipo(res: ResultadoPCA, df: pd.DataFrame, out: Path):
    """Prototipos globales (directo/explorador/fallido)."""
    n_d = int(((res.pasos <= 5)  & res.exitos).sum())
    n_e = int(((res.pasos >= 20) & (res.pasos <= 60) & res.exitos).sum())
    n_f = int((~res.exitos).sum())

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, 81)
    ax.plot(x, res.tray_directo,    color=C["directo"],    lw=2.2,
            label=f"Directo ($\\leq$5 pasos, $n$={n_d})")
    ax.plot(x, res.tray_explorador, color=C["explorador"], lw=2.2,
            label=f"Explorador (20–60 pasos, $n$={n_e})")
    ax.plot(x, res.tray_fallido,    color=C["fallido"],    lw=2.2,
            ls="--", label=f"Fallido ($n$={n_f})")
    ax.axhline(0, color=C["neutral"], lw=0.7, ls=":")
    ax.set_xlabel("Paso dentro del episodio", fontsize=11)
    ax.set_ylabel("neck gap (media de grupo)", fontsize=11)
    ax.set_xlim(1, 80); ax.legend(fontsize=9)
    ax.set_title("Trayectorias prototipo identificadas por PCA", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_evolucion_temporal(res: ResultadoPCA, out: Path):
    """
    PC1 medio por ventana de 20 episodios a lo largo del entrenamiento.
    Muestra cómo cambia la estructura de las trayectorias con el aprendizaje.
    """
    ventana = 20
    ep_vals  = res.eps
    pc1_vals = res.Z[:, 0]
    exito_vals = res.exitos.astype(float)

    # Ventanas rodantes
    n = len(ep_vals)
    ep_mid, pc1_med, exito_med = [], [], []
    for i in range(0, n - ventana, ventana // 2):
        ep_mid.append(ep_vals[i:i+ventana].mean())
        pc1_med.append(pc1_vals[i:i+ventana].mean())
        exito_med.append(exito_vals[i:i+ventana].mean())

    ep_mid    = np.array(ep_mid)
    pc1_med   = np.array(pc1_med)
    exito_med = np.array(exito_med)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # Panel superior: PC1 medio
    ax = axes[0]
    ax.plot(ep_mid, pc1_med, color=C["directo"], lw=1.5)
    ax.fill_between(ep_mid, pc1_med, alpha=0.15, color=C["directo"])
    ax.set_ylabel("PC1 medio\n(↓ = trayectorias más cortas)", fontsize=10)
    ax.set_title("Evolución temporal del PC1 — refleja el aprendizaje del agente",
                 fontsize=11)

    # Panel inferior: tasa de éxito
    ax2 = axes[1]
    ax2.plot(ep_mid, exito_med * 100, color=C["explorador"], lw=1.5)
    ax2.fill_between(ep_mid, exito_med * 100, alpha=0.15, color=C["explorador"])
    ax2.set_ylabel("Tasa de éxito (%)", fontsize=10)
    ax2.set_xlabel("Episodio", fontsize=11)
    ax2.set_ylim(0, 105)

    # Líneas de fase
    for ax in axes:
        for f, (ini, fin) in FASES.items():
            ax.axvline(ini, color=COLORES_FASE[f], lw=1.2,
                       ls="--", alpha=0.7)
            ax.text(ini + 5, ax.get_ylim()[1] * 0.92,
                    f"F{f}", color=COLORES_FASE[f],
                    fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_fases_comparacion(res: ResultadoPCA, out: Path):
    """
    Scatter PC1-PC2 en panel 2×2, uno por fase.
    Muestra cómo cambia la distribución de trayectorias en cada fase.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for i, f in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        mask_f = res.fases == f
        Z_f    = res.Z[mask_f]
        ex_f   = res.exitos[mask_f]
        ep_f   = res.eps[mask_f]

        sc = ax.scatter(Z_f[ex_f, 0], Z_f[ex_f, 1],
                        c=ep_f[ex_f], cmap="viridis",
                        s=12, alpha=0.6)
        ax.scatter(Z_f[~ex_f, 0], Z_f[~ex_f, 1],
                   marker="x", color=C["fallido"],
                   s=30, lw=1, alpha=0.9, label="Fracaso")

        plt.colorbar(sc, ax=ax, label="Episodio")
        tasa = ex_f.mean()
        n    = mask_f.sum()
        ax.set_title(f"{LABELS_FASE[f].replace(chr(10),' ')}  "
                     f"(éxito={tasa:.2f}, $n$={n})",
                     fontsize=10, color=COLORES_FASE[f])
        ax.set_xlabel(f"PC1 ({res.varianza[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({res.varianza[1]:.1%})", fontsize=9)
        if (~ex_f).sum() > 0:
            ax.legend(fontsize=8)

    fig.suptitle("Proyección PC1–PC2 separada por fase del curriculum",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_trayectorias_por_fase(res: ResultadoPCA, out: Path):
    """
    Trayectoria media de episodios exitosos en cada fase.
    Muestra cómo el agente aprende rutas más cortas con el tiempo.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(1, 81)

    for f in [1, 2, 3, 4]:
        tray = res.tray_por_fase.get(f, np.zeros(80))
        n_f  = int(((res.fases == f) & res.exitos & (res.pasos > 1)).sum())
        ax.plot(x, tray, color=COLORES_FASE[f], lw=2,
                label=f"Fase {f} ($n$={n_f})")

    ax.axhline(0, color=C["neutral"], lw=0.7, ls=":")
    ax.set_xlabel("Paso dentro del episodio", fontsize=11)
    ax.set_ylabel("neck gap (media episodios exitosos)", fontsize=11)
    ax.set_xlim(1, 80)
    ax.legend(fontsize=9)
    ax.set_title("Evolución de las trayectorias exitosas por fase\n"
                 "Fase 1: rutas largas  →  Fase 4: rutas directas", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_trayectoria_espacio_control(df_pasos: pd.DataFrame,
                                    out: Path, solo_fase: int = 4):
    """
    Trayectorias en espacio (ga0, sa0) coloreadas por neck gap.
    Por defecto solo fase 4 (política madura).
    """
    df = df_pasos[df_pasos["fase"] == solo_fase] if solo_fase else df_pasos

    exitosos_rapidos = df[(df["pasos_ep"] <= 5)  & df["exito"]]["ep"].unique()[:2]
    exitosos_lentos  = df[(df["pasos_ep"].between(20,60)) & df["exito"]]["ep"].unique()[:2]
    fallidos         = df[~df["exito"]]["ep"].unique()[:2]
    ep_sel = np.concatenate([exitosos_rapidos, exitosos_lentos, fallidos])

    col_ep  = {**{ep: C["directo"]    for ep in exitosos_rapidos},
               **{ep: C["explorador"] for ep in exitosos_lentos},
               **{ep: C["fallido"]    for ep in fallidos}}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.cm.RdYlGn_r

    for ep in ep_sel:
        seg = df[df["ep"] == ep].sort_values("paso")
        if len(seg) < 2: continue
        neck_norm = (seg["neck_gap"].fillna(0) / 160.0).clip(0, 1)
        for i in range(len(seg) - 1):
            ax.plot(seg["ga0"].iloc[i:i+2], seg["sa0"].iloc[i:i+2],
                    color=cmap(neck_norm.iloc[i]), lw=1.8, alpha=0.85)
        ax.scatter(seg["ga0"].iloc[0],  seg["sa0"].iloc[0],
                   marker="o", s=70, color=col_ep[ep], zorder=5,
                   edgecolors="white", lw=0.5)
        ax.scatter(seg["ga0"].iloc[-1], seg["sa0"].iloc[-1],
                   marker="*", s=120, color=col_ep[ep], zorder=5,
                   edgecolors="white", lw=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 160))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="neck gap")

    parches = [mpatches.Patch(color=C["directo"],    label="Directo"),
               mpatches.Patch(color=C["explorador"], label="Explorador"),
               mpatches.Patch(color=C["fallido"],    label="Fallido")]
    ax.legend(handles=parches, fontsize=9)
    ax.set_xlabel("$ga_0$", fontsize=12)
    ax.set_ylabel("$sa_0$", fontsize=12)
    fase_label = f"Fase {solo_fase}" if solo_fase else "Todas las fases"
    ax.set_title(f"Trayectorias en espacio de control ($ga_0$, $sa_0$) — {fase_label}\n"
                 "Círculo=inicio · Estrella=fin · Color=neck gap", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")
