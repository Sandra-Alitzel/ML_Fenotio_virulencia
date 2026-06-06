"""
src/analisis_lasso.py
======================
Tema 2 — Regresión lineal y clasificación
Lasso por fase + análisis de acciones en fase 4.

Figuras generadas:
  fig_lasso_coef_*              : coeficientes + predicción vs real (fase 4)
  fig_lasso_r2_por_fase_*       : R² del Lasso en cada fase
  fig_acciones_recompensas_*    : correlación Δ(ga0,sa0) vs recompensas
  fig_distribucion_acciones_*   : histograma de acciones por tipo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import pearsonr

from data_loader import FASES, COLORES_FASE, LABELS_FASE

FEATURE_NAMES = ["ga0_inicio", "sa0_inicio",
                 "w_bif", "w_dist", "w_rob", "w_vir",
                 "pasos", "PC1", "PC2", "PC3"]

C = {"pos": "#1D9E75", "neg": "#D85A30", "neutro": "#888780",
     "directo": "#185FA5", "explorador": "#BA7517"}


@dataclass
class ResultadoLasso:
    coefs: dict
    alpha: float
    r2_train: float
    cv_r2_mean: float
    cv_r2_std: float
    y_true: np.ndarray
    y_pred: np.ndarray
    residuos: np.ndarray
    ranking: list
    fase: int = 4
    r2_por_fase: dict = field(default_factory=dict)


def _ajustar_lasso(df_fase, Z_fase, cv_folds):
    """Ajusta Lasso para un subconjunto de datos."""
    ga0s  = df_fase["ga0_inicio"].values
    sa0s  = df_fase["sa0_inicio"].values
    ws    = df_fase[["w_bif","w_dist","w_rob","w_vir"]].values
    pasos = df_fase["pasos"].values
    robs  = df_fase["r_rob_sum"].values

    X_raw = np.column_stack([ga0s, sa0s, ws, pasos, Z_fase[:, :3]])
    sc    = StandardScaler()
    X     = sc.fit_transform(X_raw)

    lcv = LassoCV(cv=cv_folds, max_iter=10_000, random_state=42)
    lcv.fit(X, robs)

    lasso = Lasso(alpha=lcv.alpha_, max_iter=10_000)
    kf    = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(lasso, X, robs, cv=kf, scoring="r2")
    lasso.fit(X, robs)

    coefs   = dict(zip(FEATURE_NAMES, lasso.coef_.tolist()))
    ranking = sorted(FEATURE_NAMES, key=lambda n: abs(coefs[n]), reverse=True)
    return coefs, float(lcv.alpha_), float(lasso.score(X, robs)), \
           float(cv_r2.mean()), float(cv_r2.std()), \
           robs, lasso.predict(X), ranking


def ejecutar_lasso(df: pd.DataFrame, Z: np.ndarray,
                   cv_folds: int = 5,
                   fase_principal: int = 4) -> ResultadoLasso:
    """
    Ajusta Lasso en fase_principal y calcula R² para todas las fases.
    """
    fases_arr = df["fase"].values

    # Lasso principal
    mask_p = fases_arr == fase_principal
    idx_p  = np.where(mask_p)[0]
    coefs, alpha, r2_train, cv_mean, cv_std, y_true, y_pred, ranking = \
        _ajustar_lasso(df.iloc[idx_p].reset_index(drop=True),
                       Z[idx_p], cv_folds)
    residuos = y_true - y_pred
    print(f"  Lasso Fase {fase_principal}: R²={r2_train:.4f}  "
          f"CV R²={cv_mean:.4f}  alpha={alpha:.5f}")

    # R² por cada fase
    r2_por_fase = {}
    for f in [1, 2, 3, 4]:
        mask_f = fases_arr == f
        if mask_f.sum() < 20:
            continue
        idx_f = np.where(mask_f)[0]
        try:
            _, _, r2_f, cv_f, _, _, _, _ = _ajustar_lasso(
                df.iloc[idx_f].reset_index(drop=True),
                Z[idx_f], min(cv_folds, mask_f.sum()//2))
            r2_por_fase[f] = {"r2_train": r2_f, "cv_r2": cv_f}
            print(f"  Lasso Fase {f}: R²={r2_f:.4f}")
        except Exception as e:
            print(f"  Lasso Fase {f}: error — {e}")

    return ResultadoLasso(
        coefs=coefs, alpha=alpha, r2_train=r2_train,
        cv_r2_mean=cv_mean, cv_r2_std=cv_std,
        y_true=y_true, y_pred=y_pred, residuos=residuos,
        ranking=ranking, fase=fase_principal,
        r2_por_fase=r2_por_fase,
    )


# ── Figuras ────────────────────────────────────────────────────────────────────

def fig_lasso_coeficientes(res: ResultadoLasso, out: Path):
    """Coeficientes Lasso + predicción vs real (fase principal)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Coeficientes
    ax = axes[0]
    names  = FEATURE_NAMES
    coefs  = [res.coefs[n] for n in names]
    colors = [C["pos"] if c >= 0 else C["neg"] for c in coefs]
    y_pos  = np.arange(len(names))
    ax.barh(y_pos, coefs, color=colors, alpha=0.85)
    ax.axvline(0, color=C["neutro"], lw=0.8)
    etiq = {"ga0_inicio": "$ga_0$ inicio", "sa0_inicio": "$sa_0$ inicio",
             "w_bif": "$w_{bif}$", "w_dist": "$w_{dist}$",
             "w_rob": "$w_{rob}$", "w_vir": "$w_{vir}$",
             "pasos": "pasos", "PC1": "PC1", "PC2": "PC2", "PC3": "PC3"}
    ax.set_yticks(y_pos)
    ax.set_yticklabels([etiq[n] for n in names], fontsize=10)
    ax.set_xlabel("Coeficiente (estandarizado)", fontsize=11)
    ax.set_title(f"(a) Coeficientes Lasso — Fase {res.fase}\n"
                 f"$R^2$={res.r2_train:.3f}  CV $R^2$={res.cv_r2_mean:.3f}",
                 fontsize=11)
    parches = [mpatches.Patch(color=C["pos"],  label="Aumenta $r_{{rob}}$"),
               mpatches.Patch(color=C["neg"],  label="Reduce $r_{{rob}}$")]
    ax.legend(handles=parches, fontsize=9)

    # Predicción vs real
    ax2 = axes[1]
    ax2.scatter(res.y_true, res.y_pred, alpha=0.2, s=8, color=C["directo"])
    lims = [max(0, min(res.y_true.min(), res.y_pred.min()) - 0.5),
            max(res.y_true.max(), res.y_pred.max()) + 0.5]
    ax2.plot(lims, lims, "k--", lw=1, alpha=0.6, label="$y=\\hat{y}$")
    ax2.set_xlim(left=lims[0]); ax2.set_ylim(bottom=lims[0])
    ax2.set_xlabel("$r_{rob,sum}$ real", fontsize=11)
    ax2.set_ylabel("$r_{rob,sum}$ predicho", fontsize=11)
    ax2.set_title(f"(b) Predicción vs real — Fase {res.fase}", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_r2_por_fase(res: ResultadoLasso, out: Path):
    """
    R² del Lasso en cada fase del curriculum.
    Muestra que el modelo lineal mejora conforme el agente aprende.
    """
    fases_disp = sorted(res.r2_por_fase.keys())
    r2_train   = [res.r2_por_fase[f]["r2_train"] for f in fases_disp]
    r2_cv      = [res.r2_por_fase[f]["cv_r2"]    for f in fases_disp]
    cols       = [COLORES_FASE[f] for f in fases_disp]
    labs       = [LABELS_FASE[f].replace("\n", " ") for f in fases_disp]

    x    = np.arange(len(fases_disp))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - ancho/2, r2_train, ancho, color=cols, alpha=0.85,
                   label="$R^2$ entrenamiento")
    bars2 = ax.bar(x + ancho/2, r2_cv, ancho, color=cols, alpha=0.45,
                   label="$R^2$ CV", hatch="//")
    ax.axhline(0, color=C["neutro"], lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("$R^2$ Lasso", fontsize=11)
    ax.set_title("$R^2$ del modelo Lasso por fase del curriculum\n"
                 "Un $R^2$ más alto indica que el comportamiento del agente\n"
                 "es más predecible (política más consolidada)", fontsize=10)
    ax.legend(fontsize=9)
    for bar, val in zip(list(bars1)+list(bars2), r2_train+r2_cv):
        if val > 0.05:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_acciones_vs_recompensas(df_pasos: pd.DataFrame, out: Path,
                                 solo_fase: int = 4):
    """Correlación Δ(ga0,sa0) vs recompensas — fase 4 por defecto."""
    df = df_pasos[df_pasos["fase"] == solo_fase].copy() if solo_fase \
         else df_pasos.copy()
    df = df.dropna(subset=["accion_ga0","accion_sa0","r_bif","r_dist","r_rob"])

    grupos = {
        "Directo\n($\\leq$5 pasos)":    df[(df["pasos_ep"]<=5) & df["exito"]],
        "Explorador\n(20-60 pasos)":     df[(df["pasos_ep"].between(20,60)) & df["exito"]],
        "Fallido":                        df[~df["exito"]],
    }
    acciones    = ["accion_ga0", "accion_sa0"]
    recompensas = ["r_bif", "r_dist", "r_rob"]
    etiq_a = {"accion_ga0": "$\\Delta ga_0$", "accion_sa0": "$\\Delta sa_0$"}
    etiq_r = {"r_bif": "$r_{bif}$", "r_dist": "$r_{dist}$", "r_rob": "$r_{rob}$"}
    colores_g = {list(grupos.keys())[0]: C["directo"],
                 list(grupos.keys())[1]: C["explorador"],
                 list(grupos.keys())[2]: C["neg"]}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    grupo_names = list(grupos.keys())

    for i, acc in enumerate(acciones):
        for j, rew in enumerate(recompensas):
            ax = axes[i][j]
            corrs = []
            for nombre, sub in grupos.items():
                s = sub[[acc, rew]].dropna()
                if len(s) < 10:
                    corrs.append(0)
                    continue
                r, _ = pearsonr(s[acc], s[rew])
                corrs.append(r)
            bars = ax.bar(range(len(grupo_names)), corrs,
                          color=[colores_g[n] for n in grupo_names], alpha=0.78)
            for bi, (bar, val) in enumerate(zip(bars, corrs)):
                ypos = val + 0.02 if val >= 0 else val - 0.07
                ax.text(bar.get_x()+bar.get_width()/2, ypos,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
            ax.axhline(0, color=C["neutro"], lw=0.7, ls="--")
            ax.set_ylim(-1, 1)
            ax.set_xticks(range(len(grupo_names)))
            ax.set_xticklabels(grupo_names, fontsize=7)
            if i == 0: ax.set_title(etiq_r[rew], fontsize=11)
            if j == 0: ax.set_ylabel(f"Corr. con\n{etiq_a[acc]}", fontsize=10)

    fase_label = f"Fase {solo_fase}" if solo_fase else "Todas las fases"
    fig.suptitle(f"Correlación Pearson: acciones vs recompensas inmediatas — {fase_label}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_distribucion_acciones(df_pasos: pd.DataFrame, out: Path,
                               solo_fase: int = 4):
    """Distribución de acciones por tipo de episodio — fase 4."""
    df = df_pasos[df_pasos["fase"] == solo_fase].copy() if solo_fase \
         else df_pasos.copy()
    df = df.dropna(subset=["accion_ga0","accion_sa0"])

    grupos = {
        "Directo":    df[(df["pasos_ep"]<=5)  & df["exito"]],
        "Explorador": df[(df["pasos_ep"].between(20,60)) & df["exito"]],
        "Fallido":    df[~df["exito"]],
    }
    colores = {"Directo": C["directo"], "Explorador": C["explorador"],
               "Fallido": C["neg"]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, label in zip(axes,
                               ["accion_ga0", "accion_sa0"],
                               ["$\\Delta ga_0$ (acción sobre represor)",
                                "$\\Delta sa_0$ (acción sobre umbral)"]):
        for nombre, sub in grupos.items():
            if len(sub) < 5: continue
            ax.hist(sub[col], bins=40, alpha=0.5, density=True,
                    color=colores[nombre], label=nombre)
        ax.axvline(0, color=C["neutro"], lw=0.8, ls="--")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Densidad", fontsize=11)
        ax.legend(fontsize=9)

    fase_label = f"Fase {solo_fase}" if solo_fase else "Todas las fases"
    fig.suptitle(f"Distribución de acciones por tipo de episodio — {fase_label}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")
