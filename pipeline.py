"""
src/pipeline.py
================
Pipeline principal — genera todas las figuras organizadas por fase.

Uso:
    # Una sola semilla (análisis completo por fase)
    python pipeline.py --data "data/*.jsonl" --solo_demo

    # Todas las semillas
    python pipeline.py --data "data/*.jsonl"

    # Solo fase 4 en todos los análisis
    python pipeline.py --data "data/*.jsonl" --solo_demo --fase 4
"""

import argparse, sys, glob
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import (cargar_semilla, cargar_pasos,
                          cargar_todas_semillas, semilla_id, FASES)
from analisis_pca import (ejecutar_pca,
                           fig_varianza_y_scatter,
                           fig_trayectorias_prototipo,
                           fig_evolucion_temporal,
                           fig_fases_comparacion,
                           fig_trayectorias_por_fase,
                           fig_trayectoria_espacio_control)
from analisis_svm import (ejecutar_svm,
                           fig_frontera_decision,
                           fig_fronteras_por_fase,
                           fig_accuracy_por_fase,
                           fig_efecto_pesos_w)
from analisis_lasso import (ejecutar_lasso,
                             fig_lasso_coeficientes,
                             fig_r2_por_fase,
                             fig_acciones_vs_recompensas,
                             fig_distribucion_acciones)
from analisis_bayesiano import (ejecutar_naive_bayes,
                                 fig_naive_bayes_mapa,
                                 fig_nb_accuracy_por_fase,
                                 ejecutar_red_bayesiana,
                                 fig_red_bayesiana)
from analisis_semillas import (seleccion_modelos, fig_seleccion_modelos,
                                ejecutar_random_forest, fig_feature_importance,
                                fig_dispersion_semillas,
                                fig_curva_aprendizaje_semillas)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run_demo_semilla(path_demo: str, fase_principal: int = 4):
    banner(f"Cargando semilla demo: {path_demo}")
    df      = cargar_semilla(path_demo)
    df_pasos = cargar_pasos(path_demo, min_pasos=2)
    sid      = semilla_id(path_demo)

    # Estadísticas por fase
    print(f"\n  {'Fase':<6} {'n':>5} {'éxito':>7} {'pasos_med':>10}")
    for f in [1, 2, 3, 4]:
        sub = df[df["fase"] == f]
        print(f"  {f:<6} {len(sub):>5} {sub['exito'].mean():>7.3f} "
              f"{sub['pasos'].mean():>10.1f}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    banner("PCA — variables latentes")
    pca_res = ejecutar_pca(df)

    fig_varianza_y_scatter(
        pca_res, FIGURES_DIR / f"fig_pca_varianza_scatter_{sid}.png")
    fig_trayectorias_prototipo(
        pca_res, df, FIGURES_DIR / f"fig_pca_trayectorias_{sid}.png")
    fig_evolucion_temporal(
        pca_res, FIGURES_DIR / f"fig_pca_evolucion_temporal_{sid}.png")
    fig_fases_comparacion(
        pca_res, FIGURES_DIR / f"fig_pca_fases_comparacion_{sid}.png")
    fig_trayectorias_por_fase(
        pca_res, FIGURES_DIR / f"fig_pca_trayectorias_por_fase_{sid}.png")
    if len(df_pasos) > 0:
        fig_trayectoria_espacio_control(
            df_pasos, FIGURES_DIR / f"fig_espacio_control_f4_{sid}.png",
            solo_fase=4)
        # También todas las fases juntas para comparar
        fig_trayectoria_espacio_control(
            df_pasos, FIGURES_DIR / f"fig_espacio_control_todas_{sid}.png",
            solo_fase=None)

    # ── SVM ──────────────────────────────────────────────────────────────────
    banner("SVM — fronteras de decisión por fase")
    svm_res = ejecutar_svm(df, fase_principal=fase_principal)

    fig_frontera_decision(
        svm_res, FIGURES_DIR / f"fig_svm_frontera_{sid}.png")
    fig_fronteras_por_fase(
        svm_res, FIGURES_DIR / f"fig_svm_fronteras_fases_{sid}.png")
    fig_accuracy_por_fase(
        svm_res, FIGURES_DIR / f"fig_svm_accuracy_fases_{sid}.png")
    fig_efecto_pesos_w(
        df, FIGURES_DIR / f"fig_svm_pesos_w_{sid}.png",
        solo_fase=fase_principal)

    # ── Lasso ─────────────────────────────────────────────────────────────────
    banner("Lasso — regresión por fase")
    lasso_res = ejecutar_lasso(df, pca_res.Z, fase_principal=fase_principal)

    fig_lasso_coeficientes(
        lasso_res, FIGURES_DIR / f"fig_lasso_coef_{sid}.png")
    fig_r2_por_fase(
        lasso_res, FIGURES_DIR / f"fig_lasso_r2_fases_{sid}.png")
    if len(df_pasos) > 0:
        fig_acciones_vs_recompensas(
            df_pasos, FIGURES_DIR / f"fig_acciones_recompensas_{sid}.png",
            solo_fase=fase_principal)
        fig_distribucion_acciones(
            df_pasos, FIGURES_DIR / f"fig_distribucion_acciones_{sid}.png",
            solo_fase=fase_principal)

    # ── Bayesiano ─────────────────────────────────────────────────────────────
    banner("Naive Bayes + Red Bayesiana")
    # Fase 4 (política madura)
    nb_f4 = ejecutar_naive_bayes(df, fase=fase_principal)
    fig_naive_bayes_mapa(
        df, nb_f4, FIGURES_DIR / f"fig_naive_bayes_f4_{sid}.png")
    # Comparación de accuracy por fase
    fig_nb_accuracy_por_fase(
        df, FIGURES_DIR / f"fig_nb_accuracy_fases_{sid}.png")
    # Red bayesiana fase 4
    rb_f4 = ejecutar_red_bayesiana(df, fase=fase_principal)
    fig_red_bayesiana(
        rb_f4, FIGURES_DIR / f"fig_red_bayesiana_f4_{sid}.png")
    # Red bayesiana todas las fases
    rb_all = ejecutar_red_bayesiana(df, fase=None)
    fig_red_bayesiana(
        rb_all, FIGURES_DIR / f"fig_red_bayesiana_todas_{sid}.png")

    print(f"\n✓ Semilla demo completada — figuras en {FIGURES_DIR}/")
    return {"pca": pca_res, "svm": svm_res, "lasso": lasso_res,
            "df": df, "df_pasos": df_pasos, "sid": sid}


def run_todas_semillas(patron: str, path_demo: str = None,
                        fase_principal: int = 4):
    banner("Cargando todas las semillas")
    dfs, tabla = cargar_todas_semillas(patron)

    print(f"\n  Total semillas: {len(tabla)}")
    cols_show = [c for c in ["semilla","tasa_exito_global","tasa_exito_f4",
                              "exito_f1","pasos_f4_media","costo_curriculum"]
                 if c in tabla.columns]
    print(tabla[cols_show].to_string(index=False))

    # Curva de aprendizaje con banda
    banner("Curva de aprendizaje entre semillas")
    fig_curva_aprendizaje_semillas(
        dfs, FIGURES_DIR / "fig_curva_aprendizaje_semillas.png")

    # Dispersión entre semillas
    fig_dispersion_semillas(tabla, FIGURES_DIR / "fig_semillas_dispersion.png")

    if len(tabla) >= 5:
        banner("Selección de modelos — LOO-CV")
        comp = seleccion_modelos(tabla)
        print(comp.to_string(index=False))
        fig_seleccion_modelos(comp, FIGURES_DIR / "fig_seleccion_modelos.png")

        banner("Random Forest sobre tabla de semillas")
        rf_res = ejecutar_random_forest(tabla)
        fig_feature_importance(rf_res, FIGURES_DIR / "fig_random_forest.png")
    else:
        print("  (Menos de 5 semillas — saltando selección y RF)")

    tabla.to_csv("data/tabla_semillas.csv", index=False)
    print(f"\n  Tabla guardada en data/tabla_semillas.csv")
    print(f"\n✓ Análisis multi-semilla completado — figuras en {FIGURES_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/*.jsonl")
    parser.add_argument("--semilla_demo",default=None)
    parser.add_argument("--solo_demo",   action="store_true")
    parser.add_argument("--fase",        type=int, default=4,
                        help="Fase principal para los análisis (default=4)")
    args = parser.parse_args()

    # Determinar semilla demo
    if args.semilla_demo:
        path_demo = args.semilla_demo
    else:
        paths = sorted(glob.glob(args.data))
        if not paths:
            print(f"ERROR: No hay archivos con patrón: {args.data}")
            sys.exit(1)
        path_demo = paths[0]
        print(f"  Semilla demo: {path_demo}")

    # Análisis demo
    run_demo_semilla(path_demo, fase_principal=args.fase)

    # Análisis multi-semilla
    if not args.solo_demo:
        paths = sorted(glob.glob(args.data))
        if len(paths) > 1:
            run_todas_semillas(args.data, path_demo, args.fase)
        else:
            print("\n  Solo hay 1 semilla — saltando análisis multi-semilla")

    banner("PIPELINE COMPLETO")
    figs = sorted(FIGURES_DIR.glob("*.png"))
    print(f"  {len(figs)} figuras generadas en {FIGURES_DIR.resolve()}")
    for f in figs:
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
