"""
src/analisis_semillas.py
=========================
Tema 3 — Selección de modelos    (LOO-CV sobre tabla de 20 semillas)
Tema 7 — Ensambles               (Random Forest sobre tabla de semillas)

La tabla de semillas incluye métricas por fase, lo que permite
comparar el costo del curriculum entre semillas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

from data_loader import COLORES_FASE, LABELS_FASE

C = {"rf": "#185FA5", "gb": "#1D9E75", "ridge": "#BA7517",
     "lasso": "#D85A30", "neutro": "#888780"}

# Usar métricas de fase 4 como variable respuesta (política madura)
TARGET   = "tasa_exito_f4"
FEATURES = ["exito_f1", "pasos_f1", "exito_f4", "pasos_f4_media",
            "r_rob_final50", "r_dist_final50", "pct_triviales",
            "pct_isola", "neck_final_media", "costo_curriculum"]


# TEMA 3 — Selección de modelos

def seleccion_modelos(tabla: pd.DataFrame) -> pd.DataFrame:
    feats = [f for f in FEATURES if f in tabla.columns and f != TARGET]
    if TARGET not in tabla.columns:
        # fallback
        target = "tasa_exito_final50" if "tasa_exito_final50" in tabla.columns \
                 else "tasa_exito_global"
    else:
        target = TARGET

    X = tabla[feats].fillna(0).values
    y = tabla[target].values

    modelos = {
        "Ridge":          Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
        "Lasso":          Pipeline([("sc", StandardScaler()),
                                     ("m", Lasso(max_iter=5000))]),
        "Random Forest":  RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boost": GradientBoostingRegressor(n_estimators=100,
                                                     random_state=42),
    }

    loo  = LeaveOneOut()
    rows = []
    for nombre, pipe in modelos.items():
        r2   = cross_val_score(pipe, X, y, cv=loo, scoring="r2")
        rmse = np.sqrt(-cross_val_score(pipe, X, y, cv=loo,
                        scoring="neg_mean_squared_error"))
        rows.append({"Modelo": nombre,
                     "R² LOO (media)": r2.mean(),
                     "R² LOO (std)":   r2.std(),
                     "RMSE LOO":        rmse.mean()})
        print(f"  {nombre}: R²={r2.mean():.3f} ± {r2.std():.3f}")

    return pd.DataFrame(rows).sort_values("R² LOO (media)", ascending=False)


def fig_seleccion_modelos(comparacion: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mods = comparacion["Modelo"].tolist()
    r2s  = comparacion["R² LOO (media)"].tolist()
    stds = comparacion["R² LOO (std)"].tolist()
    cols = [C["rf"], C["gb"], C["ridge"], C["lasso"]][:len(mods)]

    bars = ax.bar(mods, r2s, yerr=stds, color=cols, alpha=0.8,
                  capsize=5, width=0.5, error_kw={"lw": 1.5})
    ax.axhline(0, color=C["neutro"], lw=0.7, ls="--")
    for bar, val in zip(bars, r2s):
        ypos = val + 0.02 if val >= 0 else val - 0.06
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("$R^2$ LOO-CV", fontsize=11)
    ax.set_title("Selección de modelos — Leave-One-Out sobre 20 semillas\n"
                 f"Variable respuesta: {TARGET}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


# TEMA 7 — Ensambles

def ejecutar_random_forest(tabla: pd.DataFrame) -> dict:
    feats  = [f for f in FEATURES if f in tabla.columns and f != TARGET]
    target = TARGET if TARGET in tabla.columns else "tasa_exito_final50"
    X = tabla[feats].fillna(0).values
    y = tabla[target].values

    rf = RandomForestRegressor(n_estimators=200, random_state=42,
                                max_features="sqrt", oob_score=True)
    rf.fit(X, y)
    print(f"  RF OOB R²: {rf.oob_score_:.3f}")
    imp = dict(zip(feats, rf.feature_importances_))
    for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"    {k}: {v:.3f}")
    return {"modelo": rf, "importancias": imp, "oob_r2": rf.oob_score_,
            "feats": feats, "X": X, "y": y, "target": target}


def fig_feature_importance(rf_res: dict, out: Path):
    imp   = rf_res["importancias"]
    feats = rf_res["feats"]
    vals  = [imp[f] for f in feats]
    orden = np.argsort(vals)[::-1]

    etiq = {
        "exito_f1":         "Éxito Fase 1",
        "pasos_f1":         "Pasos Fase 1",
        "exito_f4":         "Éxito Fase 4",
        "pasos_f4_media":   "Pasos Fase 4",
        "r_rob_final50":    "$r_{rob}$ final 50",
        "r_dist_final50":   "$r_{dist}$ final 50",
        "pct_triviales":    "% triviales",
        "pct_isola":        "% Isola",
        "neck_final_media": "neck final",
        "costo_curriculum": "Costo curriculum",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Importancias
    ax = axes[0]
    cols = [C["rf"] if i < 3 else C["neutro"] for i in range(len(orden))]
    ax.barh([etiq.get(feats[i], feats[i]) for i in orden],
            [vals[i] for i in orden],
            color=[cols[j] for j, _ in enumerate(orden)], alpha=0.85)
    ax.set_xlabel("Importancia (Mean Decrease Impurity)", fontsize=11)
    ax.set_title(f"(a) Random Forest — importancia de features\n"
                 f"OOB $R^2$={rf_res['oob_r2']:.3f}  "
                 f"Target: {rf_res['target']}", fontsize=10)

    # Predicción vs real
    ax2 = axes[1]
    y     = rf_res["y"]
    y_pred = rf_res["modelo"].predict(rf_res["X"])
    ax2.scatter(y, y_pred, s=70, color=C["rf"], alpha=0.85, zorder=3)
    lims = [min(y.min(), y_pred.min())-0.02, max(y.max(), y_pred.max())+0.02]
    ax2.plot(lims, lims, "k--", lw=1, alpha=0.5, label="$y=\\hat{y}$")
    for i, (yi, yp) in enumerate(zip(y, y_pred)):
        ax2.annotate(f"s{i+1}", (yi, yp), fontsize=7,
                     xytext=(3,3), textcoords="offset points", alpha=0.7)
    ax2.set_xlabel("Valor real", fontsize=11)
    ax2.set_ylabel("Predicción RF", fontsize=11)
    ax2.set_title("(b) Predicción por semilla", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_dispersion_semillas(tabla: pd.DataFrame, out: Path):
    """Variabilidad entre semillas — métricas por fase."""
    metricas = ["exito_f1", "exito_f4", "tasa_exito_global",
                "pasos_f1", "pasos_f4_media", "costo_curriculum"]
    etiq = {"exito_f1": "Éxito\nFase 1", "exito_f4": "Éxito\nFase 4",
             "tasa_exito_global": "Éxito\nglobal",
             "pasos_f1": "Pasos\nFase 1", "pasos_f4_media": "Pasos\nFase 4",
             "costo_curriculum": "Costo\ncurriculum"}
    cols_fila = [COLORES_FASE[1], COLORES_FASE[4], C["neutro"],
                 COLORES_FASE[1], COLORES_FASE[4], "#9B59B6"]

    disp = [m for m in metricas if m in tabla.columns]
    fig, axes = plt.subplots(1, len(disp), figsize=(3*len(disp), 4.5))
    if len(disp) == 1: axes = [axes]

    np.random.seed(0)
    for ax, m, col in zip(axes, disp, cols_fila):
        vals = tabla[m].dropna().values
        ax.boxplot(vals, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.5),
                   medianprops=dict(color="white", lw=2))
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.ones(len(vals))+jitter, vals,
                   color=col, s=35, alpha=0.85, zorder=5)
        ax.set_xticks([1])
        ax.set_xticklabels([etiq.get(m, m)], fontsize=9)
        ax.set_title(f"$\\mu$={vals.mean():.3f}\n$\\sigma$={vals.std():.3f}",
                     fontsize=8)

    fig.suptitle("Variabilidad entre 20 semillas — métricas por fase del curriculum",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_curva_aprendizaje_semillas(dfs: list, out: Path, ventana: int = 30):
    """
    Curva de aprendizaje media ± std entre semillas.
    La figura más poderosa del análisis multi-semilla.
    """
    max_ep = max(df["ep"].max() for df in dfs)
    bins   = np.arange(1, max_ep + ventana, ventana)

    tasas_por_bin = []
    for df in dfs:
        tasa_bin = []
        for b in bins[:-1]:
            sub = df[(df["ep"] >= b) & (df["ep"] < b + ventana)]
            tasa_bin.append(sub["exito"].mean() if len(sub) > 0 else np.nan)
        tasas_por_bin.append(tasa_bin)

    mat   = np.array(tasas_por_bin)    # (n_semillas, n_bins)
    media = np.nanmean(mat, axis=0)
    std   = np.nanstd(mat, axis=0)
    x_mid = bins[:-1] + ventana // 2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_mid, media * 100, color=C["rf"], lw=2, label="Media (20 semillas)")
    ax.fill_between(x_mid,
                    (media - std) * 100,
                    (media + std) * 100,
                    alpha=0.2, color=C["rf"], label="± 1 std")

    # Líneas de fase
    from data_loader import FASES
    for f, (ini, fin) in FASES.items():
        ax.axvline(ini, color=COLORES_FASE[f], lw=1.5, ls="--", alpha=0.8)
        ax.text(ini + 5, 5, f"F{f}", color=COLORES_FASE[f],
                fontsize=9, fontweight="bold")

    ax.set_xlabel("Episodio", fontsize=11)
    ax.set_ylabel("Tasa de éxito (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.set_title("Curva de aprendizaje — media y desviación estándar entre 20 semillas\n"
                 "Las líneas verticales marcan el inicio de cada fase del curriculum",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")
