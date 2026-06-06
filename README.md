# Análisis ML de trayectorias SAC multi-objetivo
## PCIC UNAM · 

---

## Estructura del proyecto

```
proyecto/
├── src/
│   ├── data_loader.py        # Carga de datos (semilla individual y múltiple)
│   ├── analisis_pca.py       # Tema 5: PCA + trayectorias en espacio control
│   ├── analisis_svm.py       # Tema 6: SVM + efecto de pesos w
│   ├── analisis_lasso.py     # Tema 2: Lasso + análisis de acciones
│   ├── analisis_bayesiano.py # Temas 1 y 4: Naive Bayes + Red Bayesiana
│   ├── analisis_semillas.py  # Temas 3 y 7: Selección de modelos + RF
│   └── pipeline.py           # Script principal que ejecuta todo
├── data/
│   └── *.jsonl               # Coloca aquí tus 20 archivos historial_*.jsonl
├── figures/                  # Figuras PDF generadas (auto-creado)
└── report/
    └── reporte.tex           # Reporte LaTeX completo
```

---

## Instalación

```bash
pip install numpy pandas scikit-learn matplotlib scipy seaborn
```

---

## Uso

### 1. Coloca tus datos

```bash
cp /ruta/a/tus/historiales/historial_*.jsonl data/
```

### 2. Ejecutar el pipeline completo (una semilla demo + 20 semillas)

```bash
cd proyecto
python src/pipeline.py --data "data/*.jsonl"
```

### 3. Solo análisis de una semilla

```bash
python src/pipeline.py --semilla_demo data/historial_20260603_122451.jsonl --solo_demo
```

### 4. El pipeline genera automáticamente todas las figuras en `figures/`

---


---

## Descripción de los análisis

| Módulo | Temas | Pregunta |
|--------|-------|----------|
| `analisis_pca.py`       | 5 (Variables latentes)     | ¿Qué patrones de trayectoria existen? |
| `analisis_svm.py`       | 6 (SVM y kernels)          | ¿Desde dónde puede ganar el agente? |
| `analisis_lasso.py`     | 2 (Regresión lineal)       | ¿Qué predice la robustez? |
| `analisis_bayesiano.py` | 1 (Naive Bayes) + 4 (MGP)  | ¿Cómo se propagan las dependencias? |
| `analisis_semillas.py`  | 3 (Selección) + 7 (Ensemb) | ¿Qué generaliza entre semillas? |

---

## Figuras generadas

| Figura | Descripción |
|--------|-------------|
| `fig_pca_varianza_scatter_*.pdf`   | Varianza explicada + scatter PC1-PC2 |
| `fig_pca_trayectorias_*.pdf`       | Trayectorias prototipo (directo/explorador/fallido) |
| `fig_espacio_control_*.pdf`        | Trayectorias en espacio (ga0, sa0) coloreadas por neck |
| `fig_svm_frontera_*.pdf`           | Frontera de decisión SVM + CV scores |
| `fig_svm_pesos_w_*.pdf`            | Efecto del peso dominante w en pasos |
| `fig_lasso_coef_*.pdf`             | Coeficientes Lasso + predicción vs real |
| `fig_acciones_recompensas_*.pdf`   | Correlación acciones vs recompensas |
| `fig_distribucion_acciones_*.pdf`  | Distribución de (Δga0, Δsa0) por tipo |
| `fig_naive_bayes_*.pdf`            | Regiones de decisión Naive Bayes |
| `fig_red_bayesiana_*.pdf`          | Heatmaps de probabilidades condicionales |
| `fig_semillas_dispersion.pdf`      | Variabilidad entre 20 semillas |
| `fig_seleccion_modelos.pdf`        | Comparación Ridge/Lasso/RF/GB LOO-CV |
| `fig_random_forest.pdf`            | Feature importance RF + predicción |

---

