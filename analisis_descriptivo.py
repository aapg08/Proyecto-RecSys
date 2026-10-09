"""

Uso (después de preparar_datos.py, en la misma carpeta):
  pip install -U polars matplotlib numpy
  python analisis_descriptivo.py

Salidas:
  data/subset/interacciones.parquet   u, i, t, y  (ids enteros, ~10x más liviano)
  data/subset/map_raters.parquet      raterParticipantId <-> u
  data/subset/map_notes.parquet       noteId <-> i (ordenado por fecha de creación)
  tablas/resumen.csv                  números para el informe
  figuras/*.png y *.pdf               figuras para el informe
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: F401
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIN = datetime(2025, 7, 1, tzinfo=timezone.utc)  # mismo FIN que preparar_datos.py
FIN_MS = int(FIN.timestamp() * 1000)
H = 3_600_000  # ms en una hora

BASE = Path(__file__).resolve().parent
SUB = BASE / "data" / "subset"
FIG = BASE / "figuras"
TAB = BASE / "tablas"

# ---------------------------------------------------------------------------
# Estilo de figuras
# ---------------------------------------------------------------------------
AZUL, NARANJA = "#2a78d6", "#eb6834"
GRIS, GRIS_CLARO = "#898781", "#c3c2b7"
TINTA, TINTA2 = "#0b0b0b", "#52514e"
COLOR_ESTADO = {
    "Útil": AZUL,
    "No útil": NARANJA,
    "Necesita más evaluaciones": GRIS,
    "Sin estado": GRIS_CLARO,
}
ETIQUETA_ESTADO = {
    "CURRENTLY_RATED_HELPFUL": "Útil",
    "CURRENTLY_RATED_NOT_HELPFUL": "No útil",
    "NEEDS_MORE_RATINGS": "Necesita más evaluaciones",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 9.5,
    "legend.frameon": False, "lines.linewidth": 2,
})


def guardar(fig, nombre):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{nombre}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  figuras/{nombre}.png")


def fmt(n):
    return f"{n:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Figuras (reciben arrays de numpy; no dependen de polars)
# ---------------------------------------------------------------------------
def fig_estados(estado):
    orden = ["Necesita más evaluaciones", "Útil", "No útil", "Sin estado"]
    n = np.array([(estado == e).sum() for e in orden])
    pct = 100 * n / n.sum()
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    y = np.arange(len(orden))[::-1]
    ax.barh(y, pct, color=[COLOR_ESTADO[e] for e in orden], height=0.62)
    for yy, p, k in zip(y, pct, n):
        ax.text(p + 1, yy, f"{p:.1f}%  ({fmt(k)})", va="center", color=TINTA2, fontsize=9)
    ax.set_yticks(y, orden)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de las notas creadas en la ventana")
    ax.grid(axis="y", visible=False)
    ax.set_title("Estado actual de las notas")
    guardar(fig, "fig1_estados")


def fig_cola_larga(cnt_note, cnt_user):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, cnt, quien, titulo in [
        (axes[0], cnt_note, "notas", "Evaluaciones por nota"),
        (axes[1], cnt_user, "evaluadores", "Evaluaciones por evaluador"),
    ]:
        c = np.sort(cnt[cnt > 0])[::-1]
        rank = np.arange(1, len(c) + 1)
        ax.loglog(rank, c, color=AZUL)
        top = int(np.ceil(0.2 * len(c)))
        share = 100 * c[:top].sum() / c.sum()
        ax.axvline(top, color=GRIS, lw=1, ls="--")
        ax.text(0.04, 0.05,
                f"el 20% de {quien} con más\nevaluaciones concentra\nel {share:.0f}% del total",
                color=TINTA2, fontsize=8.5, va="bottom", transform=ax.transAxes)
        ax.set_xlabel(f"Ranking de {quien} (log)")
        ax.set_ylabel("N.º de evaluaciones (log)")
        ax.set_title(titulo)
    fig.tight_layout()
    guardar(fig, "fig2_cola_larga")


def ecdf_log(x, bins):
    h, _ = np.histogram(np.clip(x, bins[0], bins[-1]), bins=bins)
    return bins[1:], np.cumsum(h) / max(len(x), 1)


def fig_evals_por_estado(cnt_note, estado):
    bins = np.logspace(0, np.log10(max(cnt_note.max(), 10)) + 0.01, 300)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for e in ["Útil", "No útil", "Necesita más evaluaciones"]:
        x = cnt_note[(estado == e) & (cnt_note > 0)]
        if len(x) == 0:
            continue
        xs, ys = ecdf_log(x, bins)
        ax.plot(xs, ys, color=COLOR_ESTADO[e], label=f"{e} (mediana {np.median(x):.0f})")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("N.º de evaluaciones de la nota (log)")
    ax.set_ylabel("Fracción acumulada de notas")
    ax.set_title("¿Cuántas evaluaciones reciben las notas según su estado?")
    ax.legend(loc="lower right")
    guardar(fig, "fig3_evaluaciones_por_estado")


def fig_tiempos(dt_eval_h, dt_dec_h):
    bins = np.logspace(-2, 4.5, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for x, color, nombre in [
        (dt_eval_h, AZUL, "Evaluaciones"),
        (dt_dec_h, NARANJA, "Primera decisión (Útil / No útil)"),
    ]:
        xs, ys = ecdf_log(x, bins)
        ax.plot(xs, ys, color=color, label=f"{nombre}: mediana {np.median(x):.1f} h")
    for h, txt in [(1, "1 h"), (24, "24 h"), (24 * 7, "7 días"), (24 * 30, "30 días")]:
        ax.axvline(h, color=GRIS, lw=0.8, ls="--")
        ax.text(h * 1.08, 0.03, txt, color=TINTA2, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Horas desde la creación de la nota (log)")
    ax.set_ylabel("Fracción acumulada")
    ax.set_title("¿Cuándo llegan las evaluaciones y las decisiones?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    guardar(fig, "fig4_tiempos")


def fig_semanal(sem_notas, n_notas, sem_evals, n_evals, fin):
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    axes[0].plot(sem_notas, n_notas, color=AZUL)
    axes[0].set_title("Notas creadas por semana")
    axes[1].plot(sem_evals, n_evals, color=AZUL)
    axes[1].set_title("Evaluaciones por semana (de las notas de la ventana)")
    for ax in axes:
        ax.axvline(fin, color=GRIS, lw=1, ls="--")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
    axes[1].text(fin, axes[1].get_ylim()[1] * 0.9, "  fin de la ventana", color=TINTA2, fontsize=8.5)
    fig.tight_layout()
    guardar(fig, "fig5_volumen_semanal")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def compactar():
    """Pasa los ids de texto a enteros y guarda interacciones.parquet."""
    inter = SUB / "interacciones.parquet"
    if inter.exists():
        print("  (ya existe, se reutiliza)")
        return
    notes = pl.read_parquet(SUB / "notes.parquet", columns=["noteId", "createdAtMillis"])
    map_notes = (notes.with_columns(pl.col("createdAtMillis").cast(pl.Int64))
                 .sort("createdAtMillis").with_row_index("i"))
    ratings = pl.scan_parquet(str(SUB / "ratings" / "*.parquet"))
    map_raters = (ratings.select("raterParticipantId").unique().collect()
                  .sort("raterParticipantId").with_row_index("u"))
    map_notes.write_parquet(SUB / "map_notes.parquet")
    map_raters.write_parquet(SUB / "map_raters.parquet")

    nivel = {"HELPFUL": 2, "SOMEWHAT_HELPFUL": 1, "NOT_HELPFUL": 0}
    (ratings.select("noteId", "raterParticipantId", "createdAtMillis", "helpfulnessLevel")
     .join(map_raters.lazy(), on="raterParticipantId")
     .join(map_notes.lazy().select("noteId", "i"), on="noteId")
     .select(
         "u", "i",
         pl.col("createdAtMillis").cast(pl.Int64).alias("t"),
         pl.col("helpfulnessLevel")
           .replace_strict(nivel, default=None, return_dtype=pl.Int8).alias("y"),
     )
     .sort("t")
     .sink_parquet(inter))


def main():
    TAB.mkdir(exist_ok=True)

    print("[1/3] Compactando interacciones a ids enteros...")
    compactar()

    print("[2/3] Cargando y calculando estadísticas...")
    X = pl.read_parquet(SUB / "interacciones.parquet")
    map_notes = pl.read_parquet(SUB / "map_notes.parquet")
    n_i = map_notes.height
    n_u = pl.read_parquet(SUB / "map_raters.parquet").height

    u = X["u"].to_numpy()
    i = X["i"].to_numpy()
    t = X["t"].to_numpy()
    y = X["y"].to_numpy(allow_copy=True)

    cnt_note = np.bincount(i, minlength=n_i)
    cnt_user = np.bincount(u, minlength=n_u)

    st = pl.read_parquet(SUB / "noteStatusHistory.parquet")
    notas = map_notes.join(st.select("noteId", "currentStatus", "timestampMillisOfFirstNonNMRStatus"),
                           on="noteId", how="left").sort("i")
    estado = np.array([ETIQUETA_ESTADO.get(s, "Sin estado")
                       for s in notas["currentStatus"].fill_null("").to_list()])
    creada = notas["createdAtMillis"].to_numpy()
    primera = notas["timestampMillisOfFirstNonNMRStatus"].cast(pl.Int64, strict=False).to_numpy()

    dt_eval_h = (t - creada[i]) / H
    dec = ~np.isnan(primera.astype(float)) & (primera > 0)
    dt_dec_h = (primera[dec] - creada[dec]) / H

    # --- tabla resumen ---
    n_r = len(X)
    activos_n = (cnt_note > 0).sum()
    activos_u = (cnt_user > 0).sum()
    c_sorted = np.sort(cnt_note[cnt_note > 0])[::-1]
    share_top20 = c_sorted[: int(np.ceil(0.2 * len(c_sorted)))].sum() / c_sorted.sum()
    yv = y[~np.isnan(y.astype(float))] if y.dtype.kind == "f" else y
    med_util = np.median(cnt_note[estado == "Útil"]) if (estado == "Útil").any() else np.nan
    nmr = estado == "Necesita más evaluaciones"

    filas = [
        ("Notas creadas en la ventana", fmt(n_i)),
        ("Notas con ≥1 evaluación", f"{fmt(activos_n)} ({100 * activos_n / n_i:.1f}%)"),
        ("Evaluadores", fmt(activos_u)),
        ("Evaluaciones", fmt(n_r)),
        ("Densidad de la matriz", f"{100 * n_r / (activos_n * activos_u):.4f}%"),
        ("Evaluaciones por nota (mediana / media / p90 / máx)",
         f"{np.median(cnt_note[cnt_note > 0]):.0f} / {cnt_note[cnt_note > 0].mean():.1f} / "
         f"{np.percentile(cnt_note[cnt_note > 0], 90):.0f} / {cnt_note.max()}"),
        ("Evaluaciones por evaluador (mediana / media / p90 / máx)",
         f"{np.median(cnt_user[cnt_user > 0]):.0f} / {cnt_user[cnt_user > 0].mean():.1f} / "
         f"{np.percentile(cnt_user[cnt_user > 0], 90):.0f} / {cnt_user.max()}"),
        ("Evaluadores con <5 evaluaciones", f"{100 * (cnt_user[cnt_user > 0] < 5).mean():.1f}%"),
        ("Evaluaciones en el 20% de notas más evaluadas", f"{100 * share_top20:.1f}%"),
    ]
    for e in ["Útil", "No útil", "Necesita más evaluaciones", "Sin estado"]:
        filas.append((f"Notas con estado '{e}'", f"{100 * (estado == e).mean():.1f}%"))
    filas += [
        (f"Notas 'Necesita más evaluaciones' con ≥{med_util:.0f} evaluaciones (mediana de las Útiles)",
         f"{100 * (cnt_note[nmr] >= med_util).mean():.1f}%"),
        ("Evaluaciones HELPFUL / SOMEWHAT / NOT_HELPFUL",
         " / ".join(f"{100 * (yv == k).mean():.1f}%" for k in (2, 1, 0))),
        ("Evaluaciones dentro de 24 h desde la creación", f"{100 * (dt_eval_h <= 24).mean():.1f}%"),
        ("Evaluaciones dentro de 7 días", f"{100 * (dt_eval_h <= 24 * 7).mean():.1f}%"),
        ("Evaluaciones hechas después del fin de la ventana", f"{100 * (t >= FIN_MS).mean():.1f}%"),
        ("Horas hasta la primera decisión (mediana, notas decididas)", f"{np.median(dt_dec_h):.1f}"),
    ]
    resumen = pl.DataFrame({"metrica": [f for f, _ in filas], "valor": [v for _, v in filas]})
    resumen.write_csv(TAB / "resumen.csv")
    print()
    for f, v in filas:
        print(f"  {f:<75s} {v}")

    # --- semanal ---
    sem_n = (map_notes.select(pl.from_epoch("createdAtMillis", time_unit="ms").dt.truncate("1w").alias("s"))
             .group_by("s").len().sort("s"))
    sem_r = (X.lazy().select(pl.from_epoch("t", time_unit="ms").dt.truncate("1w").alias("s"))
             .group_by("s").len().sort("s").collect())

    print("\n[3/3] Figuras")
    fig_estados(estado)
    fig_cola_larga(cnt_note, cnt_user)
    fig_evals_por_estado(cnt_note, estado)
    fig_tiempos(dt_eval_h, dt_dec_h)
    fig_semanal(sem_n["s"].to_numpy(), sem_n["len"].to_numpy(),
                sem_r["s"].to_numpy(), sem_r["len"].to_numpy(),
                np.datetime64(FIN.replace(tzinfo=None)))
    print("\nListo.")


if __name__ == "__main__":
    main()
