"""
src/analisis_svm.py
====================
Tema 6 — SVM y kernels
Frontera de decisión por fase del curriculum learning.

Figuras generadas:
  fig_svm_frontera_*         : frontera fase 4 (política madura) + CV scores
  fig_svm_fronteras_fases_*  : comparación de fronteras en panel 2×2 por fase
  fig_svm_pesos_w_*          : efecto del peso dominante w (fase 4)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from data_loader import FASES, COLORES_FASE, LABELS_FASE

C_FALLO   = "#D85A30"
C_DIRECTO = "#185FA5"
C_NEUTRAL = "#888780"


@dataclass
class ResultadoSVM:
    modelo: Pipeline
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    grid_ga0: np.ndarray
    grid_sa0: np.ndarray
    proba_grid: np.ndarray
    ga0s: np.ndarray
    sa0s: np.ndarray
    exitos: np.ndarray
    eps: np.ndarray
    fases: np.ndarray
    reporte: str
    kernel: str
    C_param: float
    fase_entrenamiento: Optional[int] = None
    # Resultados por fase
    por_fase: dict = field(default_factory=dict)


def _entrenar_svm(X, y, kernel, C, n_grid, cv_folds):
    """Entrena un SVM y calcula la grilla de probabilidad."""
    _svc = CalibratedClassifierCV(
        SVC(kernel=kernel, C=C, gamma="scale",
            class_weight="balanced", random_state=42),
        ensemble=False,
    )
    pipe = Pipeline([("sc", StandardScaler()), ("svm", _svc)])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Necesitamos al menos 2 clases para CV
    if len(np.unique(y)) < 2:
        scores = np.array([np.nan] * cv_folds)
    else:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    pipe.fit(X, y)
    reporte = classification_report(y, pipe.predict(X),
                                    target_names=["Fracaso", "Éxito"],
                                    zero_division=0)

    ga0_r = np.linspace(X[:,0].min(), X[:,0].max(), n_grid)
    sa0_r = np.linspace(X[:,1].min(), X[:,1].max(), n_grid)
    gg, ss = np.meshgrid(ga0_r, sa0_r)
    proba = pipe.predict_proba(
        np.column_stack([gg.ravel(), ss.ravel()])
    )[:, 1].reshape(n_grid, n_grid)

    return pipe, scores, reporte, ga0_r, sa0_r, proba


def ejecutar_svm(df: pd.DataFrame, kernel="rbf", C=1.0,
                 n_grid=60, cv_folds=5,
                 fase_principal: int = 4) -> ResultadoSVM:
    """
    Entrena SVM principal en fase_principal (default=4, política madura).
    También calcula fronteras para las otras 3 fases para comparación.
    """
    ga0s   = df["ga0_inicio"].values
    sa0s   = df["sa0_inicio"].values
    exitos = df["exito"].values
    eps    = df["ep"].values
    fases  = df["fase"].values

    # ── SVM principal (fase_principal) ───────────────────────────────────────
    mask_p = fases == fase_principal
    X_p    = np.column_stack([ga0s[mask_p], sa0s[mask_p]])
    y_p    = exitos[mask_p].astype(int)

    pipe, scores, reporte, g_r, s_r, proba = _entrenar_svm(
        X_p, y_p, kernel, C, n_grid, cv_folds)

    print(f"  SVM Fase {fase_principal}: "
          f"accuracy={scores.mean():.3f} ± {scores.std():.3f}")

    # ── SVM por cada fase ─────────────────────────────────────────────────────
    por_fase = {}
    for f in [1, 2, 3, 4]:
        mask_f = fases == f
        if mask_f.sum() < 20:
            continue
        X_f = np.column_stack([ga0s[mask_f], sa0s[mask_f]])
        y_f = exitos[mask_f].astype(int)
        try:
            pf, sc_f, rep_f, g_f, s_f, proba_f = _entrenar_svm(
                X_f, y_f, kernel, C, n_grid, min(cv_folds, mask_f.sum()//2))
            por_fase[f] = {
                "modelo": pf, "scores": sc_f,
                "grid_g": g_f, "grid_s": s_f, "proba": proba_f,
                "ga0s": ga0s[mask_f], "sa0s": sa0s[mask_f],
                "exitos": exitos[mask_f], "eps": eps[mask_f],
            }
            print(f"  SVM Fase {f}: accuracy={sc_f.mean():.3f}")
        except Exception as e:
            print(f"  SVM Fase {f}: error — {e}")

    return ResultadoSVM(
        modelo=pipe, cv_scores=scores,
        cv_mean=float(np.nanmean(scores)), cv_std=float(np.nanstd(scores)),
        grid_ga0=g_r, grid_sa0=s_r, proba_grid=proba,
        ga0s=ga0s, sa0s=sa0s, exitos=exitos, eps=eps, fases=fases,
        reporte=reporte, kernel=kernel, C_param=C,
        fase_entrenamiento=fase_principal, por_fase=por_fase,
    )


# ── Figuras ────────────────────────────────────────────────────────────────────

def _panel_frontera(ax, grid_g, grid_s, proba, ga0s, sa0s,
                    exitos, eps, titulo, n_grid):
    """Panel reutilizable de frontera de decisión."""
    im = ax.contourf(grid_g, grid_s, proba,
                     levels=20, cmap="RdYlGn", alpha=0.6, vmin=0, vmax=1)
    ax.contour(grid_g, grid_s, proba, levels=[0.5],
               colors=["k"], linewidths=1.2, linestyles="--")
    plt.colorbar(im, ax=ax, label="$P$(éxito)", shrink=0.85)
    np.random.seed(42)
    idx = np.random.choice(len(ga0s), min(400, len(ga0s)), replace=False)
    ax.scatter(ga0s[idx][~exitos[idx]], sa0s[idx][~exitos[idx]],
               marker="x", color=C_FALLO, s=20, lw=0.7, alpha=0.85,
               label="Fracaso", zorder=3)
    ax.scatter(ga0s[idx][exitos[idx]], sa0s[idx][exitos[idx]],
               c=eps[idx][exitos[idx]], cmap="Blues_r",
               s=8, alpha=0.35, label="Éxito", zorder=2)
    ax.set_xlabel("$ga_0$ inicio", fontsize=10)
    ax.set_ylabel("$sa_0$ inicio", fontsize=10)
    ax.set_title(titulo, fontsize=10)
    return im


def fig_frontera_decision(res: ResultadoSVM, out: Path):
    """Figura principal: frontera fase 4 + CV scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    _panel_frontera(axes[0],
                    res.grid_ga0, res.grid_sa0, res.proba_grid,
                    res.ga0s[res.fases == res.fase_entrenamiento],
                    res.sa0s[res.fases == res.fase_entrenamiento],
                    res.exitos[res.fases == res.fase_entrenamiento],
                    res.eps[res.fases == res.fase_entrenamiento],
                    f"(a) Frontera SVM {res.kernel.upper()} — "
                    f"Fase {res.fase_entrenamiento} (política madura)",
                    len(res.grid_ga0))
    axes[0].legend(fontsize=9)

    # Panel CV scores — ylim dinámico
    ax2 = axes[1]
    folds = [f"Fold {i+1}" for i in range(len(res.cv_scores))]
    bars  = ax2.bar(folds, res.cv_scores * 100,
                    color=C_DIRECTO, alpha=0.8, width=0.5)
    ax2.axhline(res.cv_mean * 100, color=C_FALLO, lw=1.5, ls="--",
                label=f"Media = {res.cv_mean:.3f}")
    ymin = max(0, np.nanmin(res.cv_scores) * 100 - 5)
    ax2.set_ylim(ymin, 101)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("(b) Accuracy por fold — validación cruzada", fontsize=11)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars, res.cv_scores):
        mid = ymin + (val * 100 - ymin) * 0.5
        ax2.text(bar.get_x() + bar.get_width()/2, mid,
                 f"{val:.3f}", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_fronteras_por_fase(res: ResultadoSVM, out: Path):
    """
    Panel 2×2: frontera de decisión de cada fase.
    Muestra cómo la región de éxito se expande con el aprendizaje.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for i, f in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        if f not in res.por_fase:
            ax.text(0.5, 0.5, f"Fase {f}\nSin datos",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        pf = res.por_fase[f]
        acc = np.nanmean(pf["scores"])
        _panel_frontera(ax, pf["grid_g"], pf["grid_s"], pf["proba"],
                        pf["ga0s"], pf["sa0s"], pf["exitos"], pf["eps"],
                        f"Fase {f} — {LABELS_FASE[f].replace(chr(10),' ')}\n"
                        f"Accuracy = {acc:.3f}",
                        len(pf["grid_g"]))

    fig.suptitle("Evolución de la frontera de decisión SVM por fase del curriculum\n"
                 "Verde = zona de éxito  ·  Línea punteada = frontera P=0.5",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_accuracy_por_fase(res: ResultadoSVM, out: Path):
    """Barra de accuracy SVM por fase — muestra la mejora del agente."""
    fases_disp = sorted(res.por_fase.keys())
    accs  = [np.nanmean(res.por_fase[f]["scores"]) for f in fases_disp]
    errs  = [np.nanstd(res.por_fase[f]["scores"])  for f in fases_disp]
    cols  = [COLORES_FASE[f] for f in fases_disp]
    labs  = [LABELS_FASE[f].replace("\n", " ") for f in fases_disp]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labs, [a*100 for a in accs],
                  yerr=[e*100 for e in errs],
                  color=cols, alpha=0.8, capsize=5, width=0.5)
    ymin = max(0, min(accs)*100 - 5)
    ax.set_ylim(ymin, 103)
    for bar, acc in zip(bars, accs):
        mid = ymin + (acc*100 - ymin)*0.5
        ax.text(bar.get_x() + bar.get_width()/2, mid,
                f"{acc:.3f}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_ylabel("Accuracy SVM (%)", fontsize=11)
    ax.set_title("Accuracy del clasificador SVM por fase\n"
                 "Refleja qué tan predecible es el éxito desde el estado inicial",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_efecto_pesos_w(df: pd.DataFrame, out: Path, solo_fase: int = 4):
    """Distribución de pasos según peso w dominante — solo episodios exitosos."""
    df = df[(df["fase"] == solo_fase) & df["exito"] & (df["pasos"] > 1)].copy()
    w_cols = ["w_bif", "w_dist", "w_rob", "w_vir"]
    if not all(c in df.columns for c in w_cols):
        print("  AVISO: fig_efecto_pesos_w necesita columnas w_. Saltando.")
        return
    df["w_dom"] = df[w_cols].idxmax(axis=1)

    etiq   = {"w_bif": "$w_{bif}$\n(bifurcación)",
               "w_dist": "$w_{dist}$\n(distancia)",
               "w_rob": "$w_{rob}$\n(robustez)",
               "w_vir": "$w_{vir}$\n(virulencia)"}
    colores = {"w_bif": "#185FA5", "w_dist": "#1D9E75",
               "w_rob": "#BA7517", "w_vir": "#D85A30"}

    grupos = [df[df["w_dom"] == w]["pasos"].values for w in w_cols]
    medias = [g.mean() if len(g) > 0 else 0 for g in grupos]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot([g if len(g)>0 else [0] for g in grupos],
                    patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", lw=2))
    for patch, w in zip(bp["boxes"], w_cols):
        patch.set_facecolor(colores[w]); patch.set_alpha(0.75)
    for i, (m, w) in enumerate(zip(medias, w_cols)):
        if m > 0:
            ax.scatter(i+1, m, color=colores[w], zorder=5, s=70, marker="D")
            ax.text(i+1, m+0.5, f"{m:.1f}",
                    ha="center", va="bottom", fontsize=9, color=colores[w])
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([etiq[w] for w in w_cols], fontsize=10)
    ax.set_ylabel("Pasos hasta éxito", fontsize=11)
    ax.set_title(f"Efecto del peso dominante $w$ — Fase {solo_fase}\n"
                 "Diamante = media  |  Solo episodios exitosos no triviales",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")
