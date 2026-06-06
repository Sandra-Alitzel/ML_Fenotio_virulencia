# Aprendizaje Automático aplicado a Fenotipo de Virulencia

Análisis de Machine Learning sobre trayectorias de un agente de aprendizaje por refuerzo que controla el fenotipo morfológico de un sistema biológico de virulencia. El agente aprende mediante **curriculum learning** en cuatro fases progresivas, y este repositorio aplica siete técnicas de ML para caracterizar su comportamiento.

---

## Contexto del problema

El agente actúa en un espacio de control `(ga0, sa0)` y busca llevar el sistema al fenotipo deseado. Cada episodio termina con un **tipo morfológico final**:

| Tipo | Descripción |
|------|-------------|
| `Isola` | Fenotipo aislado |
| `Hongo` | Fenotipo de hongo |
| `Multiestabilidad` | Múltiples estados estables |
| `Equilibrio estable` | Punto de equilibrio único |

El vector de pesos `w = (w_bif, w_dist, w_rob, w_vir)` modula los objetivos del agente (bifurcación, distancia, robustez, virulencia).

---

## Fases del curriculum

| Fase | Episodios | Descripción |
|------|-----------|-------------|
| 1 | 1 – 270 | Radio fijo ★ — `ga0=3`, `sa0=200` |
| 2 | 271 – 540 | Vecindad `±(1.0, 50)` alrededor del punto fijo |
| 3 | 541 – 810 | Zona media `±(3.0, 150)` |
| 4 | 811 – 1800 | Todo el mapa — política madura |

---

## Técnicas de ML implementadas

| Tema | Módulo | Técnica |
|------|--------|---------|
| 1 | `analisis_bayesiano.py` | Clasificador Naive Bayes |
| 2 | `analisis_lasso.py` | Regresión Lasso y clasificación lineal |
| 3 | `analisis_semillas.py` | Selección de modelos (LOO-CV) |
| 4 | `analisis_bayesiano.py` | Modelos gráficos probabilísticos (Red Bayesiana) |
| 5 | `analisis_pca.py` | Modelos de variables latentes (PCA) |
| 6 | `analisis_svm.py` | SVM con kernels — fronteras de decisión |
| 7 | `analisis_semillas.py` | Ensambles (Random Forest + Gradient Boosting) |

---

## Estructura del proyecto

```
.
├── data/                    # Historiales de episodios en formato JSONL
├── figures/                 # Figuras generadas por el pipeline
├── data_loader.py           # Carga, preprocesamiento y asignación de fases
├── analisis_pca.py          # PCA + trayectorias en espacio de control
├── analisis_svm.py          # SVM — fronteras de decisión por fase
├── analisis_lasso.py        # Lasso — coeficientes y R² por fase
├── analisis_bayesiano.py    # Naive Bayes + Red Bayesiana
├── analisis_semillas.py     # Selección de modelos + Random Forest multi-semilla
└── pipeline.py              # Pipeline principal — genera todas las figuras
```

---

## Uso

### Análisis de una semilla (demo)

```bash
python pipeline.py --data "data/*.jsonl" --solo_demo
```

### Todas las semillas

```bash
python pipeline.py --data "data/*.jsonl"
```

### Análisis enfocado en una fase específica

```bash
python pipeline.py --data "data/*.jsonl" --fase 4
```

### Semilla específica

```bash
python pipeline.py --data "data/*.jsonl" --semilla_demo data/historial_*.jsonl --solo_demo
```

---

## Figuras generadas

El pipeline produce figuras por semilla identificadas con el ID de la semilla (`_<sid>.png`):

| Figura | Descripción |
|--------|-------------|
| `fig_pca_varianza_scatter_*` | Varianza explicada + scatter PC1–PC2 |
| `fig_pca_trayectorias_*` | Trayectorias prototipo por fase |
| `fig_pca_evolucion_temporal_*` | PC1 medio a lo largo del entrenamiento |
| `fig_pca_fases_comparacion_*` | Scatter PC1–PC2 separado por fase (2×2) |
| `fig_espacio_control_*` | Trayectorias en `(ga0, sa0)` coloreadas por neck |
| `fig_svm_frontera_*` | Frontera de decisión SVM — fase 4 |
| `fig_svm_fronteras_fases_*` | Comparación de fronteras por fase (2×2) |
| `fig_svm_accuracy_fases_*` | Accuracy SVM por fase |
| `fig_svm_pesos_w_*` | Efecto del peso dominante `w` |
| `fig_lasso_coef_*` | Coeficientes Lasso + predicción vs real |
| `fig_lasso_r2_fases_*` | R² del Lasso por fase |
| `fig_acciones_recompensas_*` | Correlación Δ(ga0, sa0) vs recompensas |
| `fig_distribucion_acciones_*` | Histograma de acciones por tipo |
| `fig_naive_bayes_f4_*` | Regiones de decisión Naive Bayes |
| `fig_nb_accuracy_fases_*` | Accuracy Naive Bayes por fase |
| `fig_red_bayesiana_f4_*` | Probabilidades condicionales — Red Bayesiana (fase 4) |
| `fig_red_bayesiana_todas_*` | Probabilidades condicionales — Red Bayesiana (global) |

---

## Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Python 3.11+

---

## Formato de datos

Cada archivo `data/historial_*.jsonl` contiene un episodio por línea con los campos:

- `ep`: número de episodio
- `pasos`: duración del episodio
- `exito`: resultado booleano
- `tipo_final`: fenotipo morfológico al final del episodio
- `ga0_inicio`, `sa0_inicio`: condiciones iniciales
- `neck_final`: valor de cuello de botella final
- `w`: vector de pesos `[w_bif, w_dist, w_rob, w_vir]`
- `trayectoria`: lista de pasos con estado y acción
- `neck_por_paso`: evolución del neck durante el episodio
