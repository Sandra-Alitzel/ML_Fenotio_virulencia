"""
src/analisis_bayesiano.py
==========================
Tema 1 — Clasificador Bayesiano Ingenuo
Tema 4 — Modelos Gráficos Probabilísticos
Análisis por fase del curriculum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap, BoundaryNorm
from dataclasses import dataclass, field
from pathlib import Path

from data_loader import FASES, COLORES_FASE, LABELS_FASE

C = {"isola": "#1D9E75", "hongo": "#D85A30", "multi": "#BA7517",
     "eq": "#888780"}


# TEMA 1 — Naive Bayes

def ejecutar_naive_bayes(df: pd.DataFrame,
                          fase: int = None) -> dict:
    """
    Clasifica tipo_final desde (ga0_inicio, sa0_inicio, w_dom_idx).
    fase: si se especifica, solo usa episodios de esa fase.
    """
    df2 = df.copy() if fase is None else df[df["fase"]==fase].copy()
    w_cols = ["w_bif","w_dist","w_rob","w_vir"]
    df2["w_dom_idx"] = df2[w_cols].values.argmax(axis=1).astype(float)

    X  = df2[["ga0_inicio","sa0_inicio","w_dom_idx"]].values
    le = LabelEncoder()
    y  = le.fit_transform(df2["tipo_final"].values)

    nb = GaussianNB()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if len(np.unique(y)) < 2:
        scores = np.array([np.nan]*5)
    else:
        scores = cross_val_score(nb, X, y, cv=cv, scoring="accuracy")
    nb.fit(X, y)

    print(f"  NB Fase {fase}: accuracy={np.nanmean(scores):.3f}")
    return {"modelo": nb, "encoder": le, "cv_scores": scores,
            "cv_mean": float(np.nanmean(scores)),
            "cv_std":  float(np.nanstd(scores)),
            "X": X, "y": y, "clases": le.classes_, "fase": fase}


def fig_naive_bayes_mapa(df: pd.DataFrame, nb_res: dict, out: Path):
    """Mapa de regiones de decisión NB en espacio (ga0, sa0)."""
    nb  = nb_res["modelo"]
    le  = nb_res["encoder"]
    n   = 80
    g_r = np.linspace(df["ga0_inicio"].min(), df["ga0_inicio"].max(), n)
    s_r = np.linspace(df["sa0_inicio"].min(), df["sa0_inicio"].max(), n)
    gg, ss = np.meshgrid(g_r, s_r)
    pred = nb.predict(
        np.column_stack([gg.ravel(), ss.ravel(), np.zeros(n*n)])
    ).reshape(n, n)

    col_map = {"Isola": C["isola"], "Hongo": C["hongo"],
               "Multiestabilidad": C["multi"], "Equilibrio estable": C["eq"]}
    clases  = le.classes_
    cmap    = ListedColormap([col_map.get(c, "#888780") for c in clases])
    bounds  = np.arange(-0.5, len(clases))
    norm    = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.pcolormesh(gg, ss, pred, cmap=cmap, norm=norm, alpha=0.5, shading="auto")

    df_plot = df if nb_res["fase"] is None else df[df["fase"]==nb_res["fase"]]
    np.random.seed(42)
    idx = np.random.choice(len(df_plot), min(600, len(df_plot)), replace=False)
    for clase, color in col_map.items():
        mask = df_plot["tipo_final"].values[idx] == clase
        ax.scatter(df_plot["ga0_inicio"].values[idx][mask],
                   df_plot["sa0_inicio"].values[idx][mask],
                   color=color, s=12, alpha=0.7, label=clase)

    fase_label = f"Fase {nb_res['fase']}" if nb_res["fase"] else "Todas las fases"
    ax.set_xlabel("$ga_0$ inicio", fontsize=11)
    ax.set_ylabel("$sa_0$ inicio", fontsize=11)
    ax.set_title(f"Regiones de decisión Naive Bayes — {fase_label}\n"
                 f"CV accuracy = {nb_res['cv_mean']:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


def fig_nb_accuracy_por_fase(df: pd.DataFrame, out: Path):
    """
    Accuracy del Naive Bayes por fase.
    Muestra si la distribución de tipos en el espacio se vuelve más
    predecible conforme el curriculum avanza.
    """
    accs, stds, labs, cols = [], [], [], []
    for f in [1, 2, 3, 4]:
        res = ejecutar_naive_bayes(df, fase=f)
        accs.append(res["cv_mean"])
        stds.append(res["cv_std"])
        labs.append(LABELS_FASE[f].replace("\n", " "))
        cols.append(COLORES_FASE[f])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labs, [a*100 for a in accs],
                  yerr=[s*100 for s in stds],
                  color=cols, alpha=0.8, capsize=5, width=0.5)
    ymin = max(0, min(accs)*100 - 5)
    ax.set_ylim(ymin, 105)
    for bar, acc in zip(bars, accs):
        mid = ymin + (acc*100 - ymin)*0.5
        ax.text(bar.get_x()+bar.get_width()/2, mid,
                f"{acc:.3f}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_ylabel("CV Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy del Naive Bayes por fase del curriculum\n"
                 "Predice el tipo morfológico final desde el estado inicial",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")


# TEMA 4 — Red Bayesiana

def _discretizar(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame()
    d["w_dom"]    = df[["w_bif","w_dist","w_rob","w_vir"]].idxmax(axis=1)
    d["zona_ga0"] = pd.cut(df["ga0_inicio"],
                            bins=[0,3,6,10],
                            labels=["ga0_bajo","ga0_medio","ga0_alto"])
    d["neck_cat"] = pd.cut(df["neck_final"],
                            bins=[-1,0,30,80,300],
                            labels=["cero","bajo","medio","alto"])
    d["tipo"]  = df["tipo_final"].replace({"Equilibrio estable": "Equilibrio"})
    d["exito"] = df["exito"].map({True: "Éxito", False: "Fracaso"})
    d["fase"]  = df["fase"]
    return d


def ejecutar_red_bayesiana(df: pd.DataFrame, fase: int = None) -> dict:
    df2 = df.copy() if fase is None else df[df["fase"]==fase].copy()
    d   = _discretizar(df2)

    def cp(df_d, target, given):
        t = df_d.groupby([given, target], observed=True).size().unstack(fill_value=0)
        return t.div(t.sum(axis=1), axis=0)

    return {
        "w_exito":    cp(d, "exito",      "w_dom"),
        "zona_tipo":  cp(d, "tipo",        "zona_ga0"),
        "tipo_neck":  cp(d, "neck_cat",    "tipo"),
        "neck_exito": cp(d, "exito",       "neck_cat"),
        "d": d, "fase": fase,
    }


def fig_red_bayesiana(rb_res: dict, out: Path):
    """4 heatmaps de probabilidades condicionales."""
    tablas = [
        (rb_res["w_exito"],   "P(éxito | $w$ dominante)",   "$w$ dominante"),
        (rb_res["zona_tipo"], "P(tipo | zona $ga_0$)",       "Zona $ga_0$"),
        (rb_res["tipo_neck"], "P(neck | tipo final)",        "Tipo final"),
        (rb_res["neck_exito"],"P(éxito | cat. neck)",        "Categoría neck"),
    ]
    fase_label = f"Fase {rb_res['fase']}" if rb_res["fase"] else "Todas"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    cmap = plt.cm.YlGn

    for ax, (tabla, titulo, xlabel) in zip(axes, tablas):
        try:
            im = ax.imshow(tabla.values.T, aspect="auto",
                           cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(tabla.index)))
            ax.set_xticklabels([str(x) for x in tabla.index],
                                rotation=25, ha="right", fontsize=9)
            ax.set_yticks(range(len(tabla.columns)))
            ax.set_yticklabels([str(c) for c in tabla.columns], fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for i in range(len(tabla.index)):
                for j in range(len(tabla.columns)):
                    val = tabla.values[i, j]
                    ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if val > 0.6 else "black")
            ax.set_title(titulo, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Sin datos\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

    fig.suptitle(f"Red Bayesiana — probabilidades condicionales ({fase_label})\n"
                 "$w \\to$ éxito  ·  zona $\\to$ tipo  ·  tipo $\\to$ neck  ·  neck $\\to$ éxito",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out}")
