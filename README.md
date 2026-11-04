# Proyecto RecSys — Recomendación de notas a evaluadores en Community Notes (X)

IIC3633 Sistemas Recomendadores, 2026-2 · Grupo 8 (Sebastián Barría, Antonia Parra, Jeremías Tapia).

Problema: la mayoría de las notas de Community Notes nunca llega a una decisión
(*Needs More Ratings*), porque X exige acuerdo entre evaluadores que normalmente discrepan
(*bridging*). Planteamos qué nota pendiente mostrarle a cada evaluador como un problema de
recomendación activa multiobjetivo, que combina dos términos:

- **Afinidad:** la probabilidad de que el evaluador evalúe la nota.
- **Valor para el consenso:** cuánto reduce su voto la incertidumbre sobre la utilidad de la nota.

## 1. Datos

Son los datos públicos de Community Notes:
<https://communitynotes.x.com/guide/en/under-the-hood/download-data> (en X:
`x.com/i/communitynotes/download-data`, con la sesión iniciada).

Descarguen **todos** los zip de notas (*notes*), evaluaciones (*ratings*) e historial de
estados (*note status history*) y déjenlos, sin descomprimir, en `data/raw/`. La carpeta
`data/` no se sube al repositorio (`.gitignore`).

Usamos las notas creadas entre enero y junio de 2025 (UTC) con todas sus evaluaciones. Se
dividen por mes de creación de la nota: train de enero a abril y test de mayo a junio.

## 2. Instalación

```
pip install -U polars numpy scipy scikit-learn matplotlib
pip install sentence-transformers      # solo para embeddings.py (opcional)
```

## 3. Orden de ejecución

| # | Script | Qué hace | Salidas principales |
|---|---|---|---|
| 1 | `preparar_datos.py` | Filtra la ventana ene–jun 2025 desde los zip y la guarda en Parquet | `data/subset/` |
| 2 | `analisis_descriptivo.py` | Análisis descriptivo (H1) | `tablas/resumen.csv`, `figuras/fig1`–`fig5` |
| 3 | `split_data.py` | Split temporal por nota: train ene–abr, test may–jun | `data/split/{train,test}/` |
| 4 | `baselines.py` | Baselines de H1: Random, Most Popular hasta t, Content-based TF-IDF | `tablas/baselines.csv` |
| 5 | `postura.py` | Postura f_u de cada evaluador con el modelo de X, usando solo datos anteriores a test | `data/midterm/postura_*`, `tablas/postura_validacion.csv`, `figuras/fig6_postura.png` |
| 6 | `embeddings.py` | *(opcional, ~1 h en CPU)* Embeddings multilingües del texto de las notas | `data/midterm/embeddings.npy` |
| 7 | `midterm.py` | Baseline entrenado (LTR), término de consenso y recomendador multiobjetivo | `tablas/midterm_*.csv`, `tablas/ltr_importancia.csv`, `figuras/fig7_tradeoff.png` |

Para una prueba rápida de `midterm.py`: `python midterm.py --max-events 20000 --max-val-events 10000`.

## 4. Protocolo de evaluación

Es el mismo en `baselines.py` y `midterm.py`:

- **Evento:** una evaluación (u, n, t) de una nota de test. La pregunta es: dado el evaluador
  u en el instante t, ¿qué nota pendiente le mostramos?
- **Relevante:** la nota n que el evaluador efectivamente evaluó. Cuenta **cualquier
  valoración** (útil, algo útil o no útil), porque cualquier voto ayuda a decidir la nota.
- **Candidatas en t:** notas de test creadas en las últimas 24 h, todavía sin decisión
  (Útil o No útil) en t y no evaluadas antes por u.
- **Evaluadores:** los que tienen al menos 10 evaluaciones en train, con hasta 3 eventos por
  evaluador y 100 mil eventos en total.
- **Eventos perdidos por nota ya decidida:** se reportan en `tablas/midterm_protocolo.csv`.

**Métricas:**
- Recall@K (tasa de acierto, porque hay un único relevante) y NDCG@K.
- Cobertura@K: fracción de las notas candidatas distintas que aparecen en algún top-K.
- AporteMinoria@K: % del top-K en que el evaluador está del lado (según el signo de f_u) con
  menos evaluaciones previas en esa nota.
- ReduccionVar@K: reducción de la varianza del intercepto de la nota que produciría su voto.

Todas las métricas llevan IC 95% por bootstrap sobre evaluadores.

## 5. Modelos

- **Random**, **Most Popular** (evaluaciones de la nota antes de t) y **Content-based**
  (coseno TF-IDF entre la nota y el perfil del evaluador). Son los baselines de H1.
- **LTR (entrenado):** HistGradientBoosting sobre atributos de la nota y del evaluador:
  popularidad hasta t, popularidad en la última hora, edad de la nota, similitud TF-IDF y de
  embeddings con el historial del evaluador, y coincidencia de escritura. Se entrena con
  eventos del último mes de train. Puede puntuar notas nuevas.
- **Consenso (ΔVar):** reducción de la varianza del intercepto b_n del modelo de X si el
  evaluador vota, calculada en forma cerrada con Sherman–Morrison. Inspirado en el active
  learning de Rubens & Sugiyama (RecSys 2007).
- **Propuesto:** `λ · percentil(afinidad LTR) + (1 − λ) · percentil(ΔVar)`. Con λ = 1 es el
  LTR y con λ = 0 es el término de consenso puro.

## 6. Estructura

```
data/            (no versionado) raw/, subset/, split/, baselines/, midterm/
figuras/         figuras de los informes
tablas/          tablas de resultados (versionadas)
*.py             scripts, en el orden de la sección 3
```
