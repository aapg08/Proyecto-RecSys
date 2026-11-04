"""
Postura de los evaluadores (f_u): versión simplificada del modelo de X (Community Notes).

    rating(u, n) ≈ mu + b_u + b_n + f_u * g_n
    rating: útil = 1, algo útil = 0.5, no útil = 0
    b_n = intercepto de la nota (utilidad transversal), g_n = factor de la nota,
    f_u = postura del evaluador (1 dimensión).

Se ajusta por mínimos cuadrados alternados (ALS) con regularización L2. Como en X, los
interceptos se regularizan 5 veces más que los factores (l2_intercept_multiplier = 5,
l2_lambda = 0.03), lo que obliga a que solo las notas útiles para ambos lados del factor
tengan intercepto alto.

SIN FUGA DE INFORMACIÓN: se usan solo evaluaciones de train anteriores al inicio del periodo
de test (T_SPLIT = primera nota de test), igual que los perfiles de baselines.py.

Uso (después de split_data.py):
  python postura.py
  python postura.py --solo-validar    # rehace solo la validación y la figura con el ajuste guardado
Entrada : data/split/{train,test}/notes.parquet, data/split/train/{ratings,noteStatusHistory}.parquet
Salidas : data/midterm/postura_raters.parquet  (raterParticipantId, f_u, b_u, n_u)
          data/midterm/postura_notes.parquet   (noteId, b_n, g_n, n_n)
          data/midterm/postura_meta.json       (mu, lambdas, RMSE; los usa midterm.py)
          tablas/postura_validacion.csv        (estado predicho con las reglas de X vs estado real)
          figuras/fig6_postura.png / .pdf
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent
SPLIT = BASE / "data" / "split"
MID = BASE / "data" / "midterm"
TAB = BASE / "tablas"
FIG = BASE / "figuras"

LAMBDA = 0.03            # X: l2_lambda (factores)
MULT_INTERCEPTO = 5.0    # X: l2_intercept_multiplier
MIN_EVALS_EVALUADOR = 10
MIN_EVALS_NOTA = 5
ITERACIONES = 40
SEED = 0

# Reglas de estado (simplificadas; ver scoring/constants.py en github.com/twitter/communitynotes)
UMBRAL_UTIL = 0.40
MAX_FACTOR_UTIL = 0.50
UMBRAL_NO_UTIL = -0.05
PENDIENTE_NO_UTIL = 0.8

ESTADOS = ["CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL", "NEEDS_MORE_RATINGS"]
NOMBRE = {"CURRENTLY_RATED_HELPFUL": "Útil", "CURRENTLY_RATED_NOT_HELPFUL": "No útil",
          "NEEDS_MORE_RATINGS": "Necesita más ev."}
COLOR = {"CURRENTLY_RATED_HELPFUL": "#2a78d6", "CURRENTLY_RATED_NOT_HELPFUL": "#eb6834",
         "NEEDS_MORE_RATINGS": "#c3c2b7"}
GRIS, TINTA, TINTA2 = "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRIS,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": TINTA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "font.size": 9.5, "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Núcleo numérico (solo numpy)
# ---------------------------------------------------------------------------
def filtrar(u, i, min_u, min_n, rondas=3):
    """u, i: índices enteros compactos. Deja evaluadores con >= min_u y notas con >= min_n."""
    keep = np.ones(len(u), bool)
    nu, nn = int(u.max()) + 1, int(i.max()) + 1
    for _ in range(rondas):
        cu = np.bincount(u[keep], minlength=nu)
        cn = np.bincount(i[keep], minlength=nn)
        keep &= (cu[u] >= min_u) & (cn[i] >= min_n)
    return keep


def ajustar(u, i, r, iteraciones=ITERACIONES, seed=SEED, verbose=True):
    """ALS para r ≈ mu + b_u + b_n + f_u g_n. Cada paso es una ridge de 2 parámetros por
    evaluador (o por nota) con solución cerrada. Devuelve parámetros y lambdas usados."""
    uu, ui = np.unique(u, return_inverse=True)
    nn, ni = np.unique(i, return_inverse=True)
    U, N, R = len(uu), len(nn), len(r)
    # X minimiza MSE + lambda * promedio(param^2). En forma de suma equivale a una ridge
    # por parámetro de lambda * R / (n.º de parámetros de ese tipo).
    lam_u, lam_n = LAMBDA * R / U, LAMBDA * R / N
    cnt_u = np.bincount(ui, minlength=U).astype(float)
    cnt_n = np.bincount(ni, minlength=N).astype(float)

    rng = np.random.default_rng(seed)
    g = rng.normal(0, 0.1, N)
    bn, bu, f = np.zeros(N), np.zeros(U), np.zeros(U)
    mu = r.mean()

    def paso(idx, n_ent, cnt, lam, z, x):
        a11 = cnt + MULT_INTERCEPTO * lam
        a12 = np.bincount(idx, x, n_ent)
        a22 = np.bincount(idx, x * x, n_ent) + lam
        c1 = np.bincount(idx, z, n_ent)
        c2 = np.bincount(idx, z * x, n_ent)
        det = a11 * a22 - a12 ** 2
        return (a22 * c1 - a12 * c2) / det, (a11 * c2 - a12 * c1) / det

    rmse = np.nan
    for it in range(iteraciones):
        bu, f = paso(ui, U, cnt_u, lam_u, r - mu - bn[ni], g[ni])
        bn, g = paso(ni, N, cnt_n, lam_n, r - mu - bu[ui], f[ui])
        base = bu[ui] + bn[ni] + f[ui] * g[ni]
        mu = (r - base).mean()
        rmse = float(np.sqrt(np.mean((r - mu - base) ** 2)))
        if verbose and (it == 0 or it % 10 == 9):
            print(f"    iteración {it + 1:>2}: RMSE {rmse:.4f}")

    if np.mean(f > 0) < 0.5:   # el signo es arbitrario: positivo = lado con más evaluadores
        f, g = -f, -g
    return dict(u=uu, b_u=bu, f_u=f, n_u=cnt_u, i=nn, b_n=bn, g_n=g, n_n=cnt_n, mu=float(mu),
                rmse=rmse, lam_u=float(lam_u), lam_n=float(lam_n))


def estado_predicho(b, g, n, umbral_util=UMBRAL_UTIL):
    est = np.full(len(b), "NEEDS_MORE_RATINGS", dtype=object)
    ok = n >= MIN_EVALS_NOTA
    est[ok & (b >= umbral_util) & (np.abs(g) < MAX_FACTOR_UTIL)] = "CURRENTLY_RATED_HELPFUL"
    est[ok & (b <= UMBRAL_NO_UTIL - PENDIENTE_NO_UTIL * np.abs(g))] = "CURRENTLY_RATED_NOT_HELPFUL"
    return est


def diagnostico(b, g, n, real):
    """Percentiles del intercepto según el estado real (para ver la escala de b_n)."""
    print("\n  Intercepto b_n según estado real (notas con >= 5 evaluaciones):")
    print(f"  {'':18s}{'p10':>8s}{'p50':>8s}{'p90':>8s}{'p99':>8s}   |g_n| p50")
    for e in ESTADOS:
        m = (real == e) & (n >= MIN_EVALS_NOTA)
        if m.any():
            q = np.percentile(b[m], [10, 50, 90, 99])
            print(f"  {NOMBRE[e]:18s}" + "".join(f"{v:>8.3f}" for v in q) + f"   {np.median(np.abs(g[m])):.3f}")


def calibrar(b, g, n, real):
    """Umbral de intercepto para 'Útil' que maximiza el F1 de esa clase (manteniendo |g_n| < 0.5).
    Nuestro modelo simplificado no reproduce la escala exacta de los interceptos de X
    (X agrega filtros de evaluadores por utilidad, varios modelos y pesos en la pérdida)."""
    ok = (n >= MIN_EVALS_NOTA) & np.isin(real, ESTADOS)
    util = (real == "CURRENTLY_RATED_HELPFUL") & ok
    cand = ok & (np.abs(g) < MAX_FACTOR_UTIL)
    mejor = (UMBRAL_UTIL, -1.0)
    for tau in np.quantile(b[cand], np.linspace(0.5, 0.9995, 400)):
        pred = cand & (b >= tau)
        tp = int((pred & util).sum())
        f1 = 2 * tp / max(int(pred.sum()) + int(util.sum()), 1)
        if f1 > mejor[1]:
            mejor = (float(tau), f1)
    print(f"\n  Umbral 'Útil' calibrado: b_n >= {mejor[0]:.3f} (F1 de la clase Útil = {mejor[1]:.3f}); "
          f"el de X es {UMBRAL_UTIL}")
    return mejor[0]


def validar(b, g, n, real, umbral_util=UMBRAL_UTIL, archivo="postura_validacion.csv", titulo="reglas de X"):
    """real: estado actual (X) de cada nota ajustada. Escribe tablas/<archivo>."""
    pred = estado_predicho(b, g, n, umbral_util)
    ok = np.isin(real, ESTADOS)
    pred, real = pred[ok], real[ok]
    filas = ["estado_real,pred_util,pred_no_util,pred_nmr,total,acierto"]
    print(f"\n  Estado real (filas) vs predicho con {titulo} (columnas):")
    print(f"  {'':18s}{'Útil':>10s}{'No útil':>10s}{'NMR':>10s}{'acierto':>10s}")
    for e in ESTADOS:
        m = real == e
        c = [int((pred[m] == p).sum()) for p in ESTADOS]
        rec = c[ESTADOS.index(e)] / max(int(m.sum()), 1)
        print(f"  {NOMBRE[e]:18s}{c[0]:>10,d}{c[1]:>10,d}{c[2]:>10,d}{rec:>10.1%}")
        filas.append(f"{NOMBRE[e]},{c[0]},{c[1]},{c[2]},{int(m.sum())},{rec:.4f}")
    acc = float((pred == real).mean())
    bal = float(np.mean([(pred[real == e] == e).mean() for e in ESTADOS if (real == e).any()]))
    print(f"  Acuerdo total: {acc:.1%}  |  acuerdo balanceado (promedio por clase): {bal:.1%}")
    filas += [f"total,,,,{len(real)},{acc:.4f}", f"balanceado,,,,,{bal:.4f}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / archivo).write_text("\n".join(filas), encoding="utf-8")
    return acc, bal


def figura(f_u, b_n, g_n, real, umbral_cal=None, seed=SEED):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    lim = np.percentile(np.abs(f_u), 99.5)
    ax.hist(f_u, bins=80, range=(-lim, lim), color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=GRIS, lw=1, ls="--")
    ax.set_title("Postura inferida de los evaluadores (f_u)")
    ax.set_xlabel("f_u")
    ax.set_ylabel("N.º de evaluadores")
    ax.text(0.02, 0.97, f"f_u < 0: {np.mean(f_u < 0):.0%}\nf_u > 0: {np.mean(f_u > 0):.0%}",
            transform=ax.transAxes, va="top", color=TINTA2, fontsize=9)

    ax = axes[1]
    rng = np.random.default_rng(seed)
    for e in ["NEEDS_MORE_RATINGS", "CURRENTLY_RATED_NOT_HELPFUL", "CURRENTLY_RATED_HELPFUL"]:
        idx = np.where(real == e)[0]
        if len(idx):
            idx = rng.choice(idx, min(len(idx), 8000), replace=False)
        ax.scatter(g_n[idx], b_n[idx], s=3, alpha=0.5, color=COLOR[e], label=NOMBRE[e],
                   rasterized=True, linewidths=0)
    xs = np.linspace(-1.5, 1.5, 200)
    ax.plot([-MAX_FACTOR_UTIL, MAX_FACTOR_UTIL], [UMBRAL_UTIL] * 2, color=TINTA, lw=1.2)
    ax.plot(xs, UMBRAL_NO_UTIL - PENDIENTE_NO_UTIL * np.abs(xs), color=TINTA, lw=1.2, ls="--")
    if umbral_cal is not None:
        ax.plot([-MAX_FACTOR_UTIL, MAX_FACTOR_UTIL], [umbral_cal] * 2, color=TINTA, lw=1.2, ls=":")
        ax.text(MAX_FACTOR_UTIL + 0.05, umbral_cal, "umbral calibrado", va="center", color=TINTA, fontsize=8)
    ax.set_xlim(-1.5, 1.5)
    gl = np.percentile(b_n, [0.5, 99.5])
    ax.set_ylim(min(gl[0], -0.6), max(gl[1], 0.7))
    ax.set_title("Notas: intercepto vs factor (color = estado real en X)")
    ax.set_xlabel("Factor de la nota g_n (polarización)")
    ax.set_ylabel("Intercepto b_n (utilidad transversal)")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.text(0, UMBRAL_UTIL + 0.03, "umbral Útil", ha="center", color=TINTA, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_postura.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig6_postura.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  figuras/fig6_postura.png")


# ---------------------------------------------------------------------------
# Carga (polars) y main
# ---------------------------------------------------------------------------
def validacion(post_n, f_u):
    """Paso 4: compara con el estado real de X (umbral de X y umbral calibrado) y hace la figura."""
    import polars as pl
    print("[4/4] Validación contra el estado real de X y figura")
    st = pl.read_parquet(SPLIT / "train" / "noteStatusHistory.parquet", columns=["noteId", "currentStatus"])
    # todo en un mismo DataFrame: un join no garantiza conservar el orden de las filas
    val = post_n.join(st.unique("noteId"), on="noteId", how="left")
    real = val["currentStatus"].fill_null("SIN_ESTADO").to_numpy().astype(str)
    b_n, g_n, n_n = val["b_n"].to_numpy(), val["g_n"].to_numpy(), val["n_n"].to_numpy()
    diagnostico(b_n, g_n, n_n, real)
    validar(b_n, g_n, n_n, real)
    tau = calibrar(b_n, g_n, n_n, real)
    validar(b_n, g_n, n_n, real, tau, "postura_validacion_calibrada.csv", "el umbral Útil calibrado")
    meta_p = MID / "postura_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["umbral_util_calibrado"] = tau
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    figura(f_u, b_n, g_n, real, tau)


def main():
    import sys
    import polars as pl

    if "--solo-validar" in sys.argv:      # reutiliza el ajuste guardado (no vuelve a correr ALS)
        validacion(pl.read_parquet(MID / "postura_notes.parquet"),
                   pl.read_parquet(MID / "postura_raters.parquet")["f_u"].to_numpy())
        print("\nListo.")
        return

    print("[1/4] Cargando evaluaciones de train anteriores al inicio de test...")
    t_split = int(pl.read_parquet(SPLIT / "test" / "notes.parquet", columns=["createdAtMillis"])
                  ["createdAtMillis"].cast(pl.Int64).min())
    nivel = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.5, "NOT_HELPFUL": 0.0}
    r_scan = pl.scan_parquet(SPLIT / "train" / "ratings.parquet")
    # ids de 64 / 19 caracteres -> hash UInt64 (mismo truco que baselines.py, ~8x menos memoria)
    r = (r_scan.select(pl.col("raterParticipantId").hash().alias("r"),
                       pl.col("noteId").hash().alias("n"),
                       pl.col("createdAtMillis").cast(pl.Int64).alias("t"),
                       pl.col("helpfulnessLevel").replace_strict(nivel, default=None,
                                                                 return_dtype=pl.Float64).alias("y"))
         .filter(pl.col("t") < t_split).drop_nulls("y").select("r", "n", "y").collect())
    print(f"    {r.height:,} evaluaciones (corte: {np.datetime64(t_split, 'ms')})")
    r_h, n_h, y = r["r"].to_numpy(), r["n"].to_numpy(), r["y"].to_numpy()
    del r
    ur, u = np.unique(r_h, return_inverse=True)
    un, i = np.unique(n_h, return_inverse=True)
    del r_h, n_h

    print("[2/4] Ajuste del modelo (ALS)")
    k = filtrar(u, i, MIN_EVALS_EVALUADOR, MIN_EVALS_NOTA)
    print(f"    tras filtros: {k.sum():,} evaluaciones, {len(np.unique(u[k])):,} evaluadores, "
          f"{len(np.unique(i[k])):,} notas")
    m = ajustar(u[k], i[k], y[k])
    rater_hash, note_hash = ur[m["u"]], un[m["i"]]

    print("[3/4] Guardando")
    MID.mkdir(parents=True, exist_ok=True)
    # recuperar los ids originales de los evaluadores y notas ajustados
    ids_r = (r_scan.select("raterParticipantId", pl.col("raterParticipantId").hash().alias("r"))
             .unique("r").collect())
    post_r = (pl.DataFrame({"r": rater_hash, "f_u": m["f_u"], "b_u": m["b_u"], "n_u": m["n_u"]})
              .join(ids_r, on="r", how="left").drop("r"))
    post_r.write_parquet(MID / "postura_raters.parquet")
    notes_tr = pl.read_parquet(SPLIT / "train" / "notes.parquet", columns=["noteId"]).with_columns(
        pl.col("noteId").hash().alias("n"))
    post_n = (pl.DataFrame({"n": note_hash, "b_n": m["b_n"], "g_n": m["g_n"], "n_n": m["n_n"]})
              .join(notes_tr, on="n", how="left").drop("n"))
    post_n.write_parquet(MID / "postura_notes.parquet")
    meta = {k_: m[k_] for k_ in ("mu", "rmse", "lam_u", "lam_n")}
    meta.update(mult_intercepto=MULT_INTERCEPTO, l2_lambda=LAMBDA, t_split=t_split,
                n_evaluadores=int(len(rater_hash)), n_notas=int(len(note_hash)),
                n_evaluaciones=int(k.sum()))
    (MID / "postura_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    {len(rater_hash):,} evaluadores con postura | mu = {m['mu']:.3f} | RMSE = {m['rmse']:.4f}")
    print(f"    regularización por nota: intercepto {MULT_INTERCEPTO * m['lam_n']:.2f}, factor {m['lam_n']:.2f}")

    validacion(post_n, m["f_u"])
    print("\nListo.")


if __name__ == "__main__":
    main()
