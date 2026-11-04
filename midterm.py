"""
Midterm (H2): baseline entrenado, término de consenso y recomendador multiobjetivo.

Usa EXACTAMENTE el protocolo de baselines.py (H1) y agrega:

  ltr        Baseline entrenado (learning-to-rank puntual, HistGradientBoosting). Puede puntuar
             notas nuevas porque solo usa atributos de la nota y del evaluador, no un id:
               log_pop      log(1 + evaluaciones de la nota antes de t)
               log_pop_1h   log(1 + evaluaciones en la última hora antes de t)
               edad_h       horas desde la creación de la nota
               sim_tfidf    coseno TF-IDF entre la nota y el historial del evaluador
               sim_emb      coseno de embeddings multilingües (si existe embeddings.npy)
               escritura    fracción del historial del evaluador en la escritura de la nota
                            (latino, japonés, árabe, cirílico, ...)
               log_hist     log(1 + tamaño del historial del evaluador)
             Se entrena con eventos del ÚLTIMO MES DE TRAIN (abril), con el mismo protocolo
             (ventana de 24 h, notas pendientes), 1 positivo y --neg negativos por evento.
             Los perfiles de esos eventos usan solo evaluaciones anteriores al inicio de abril.

  consenso   Valor del voto para el consenso (active learning, Rubens & Sugiyama 2007):
             cuánto reduce el voto de u la incertidumbre del intercepto b_n de la nota, dado
             quiénes la evaluaron antes de t y su postura f (de postura.py). Para la nota n,
             con evaluadores previos de postura f_r, el modelo de X es una ridge sobre (b_n, g_n):
                 M = sum_r [1, f_r]^T [1, f_r] + diag(lam_b, lam_g)
                 Var(b_n) ∝ (M^-1)_00
             Agregar el voto de u (x = [1, f_u]) reduce esa varianza en (Sherman-Morrison):
                 dVar(u, n) = (M^-1 x)_0^2 / (1 + x^T M^-1 x)
             Es mayor cuando u está del lado que le falta a la nota: eso es el bridging.

  propuesto  score = lam * pct(afinidad ltr) + (1 - lam) * pct(dVar), con pct = percentil
             dentro de las candidatas del evento. lam = 1 es ltr; lam = 0 es consenso.

Métricas (K en --ks), con IC 95% por bootstrap sobre evaluadores:
  Recall@K, NDCG@K           relevante = la nota que el evaluador efectivamente evaluó
                             (cualquier valoración: útil, algo útil o no útil).
  Cobertura@K                fracción de las notas candidatas distintas que aparecen en algún top-K.
  AporteMinoria@K            % de las notas del top-K en que u está del lado (signo de f) con
                             MENOS evaluaciones previas en esa nota (empate = 0.5).
  ReduccionVar@K             dVar promedio del top-K (en unidades de la varianza del ruido).
  La fila "observado" mide lo mismo sobre la nota que el evaluador realmente evaluó.

Uso (después de split_data.py y postura.py; embeddings.py es opcional):
  pip install -U polars scikit-learn scipy numpy matplotlib
  python midterm.py
  python midterm.py --max-events 20000          # prueba rápida
Salidas:
  tablas/midterm_resultados.csv   todas las métricas por modelo y K
  tablas/midterm_lambda.csv       barrido de lambda (K = 10)
  tablas/midterm_protocolo.csv    conteos del protocolo (incluye eventos perdidos por nota decidida)
  tablas/ltr_importancia.csv      importancia por permutación de las features del baseline entrenado
  figuras/fig7_tradeoff.png/.pdf  afinidad vs consenso
  data/midterm/eventos_test.npz   rangos por modelo y evento
"""

import argparse
import html
import json
import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BASE = Path(__file__).resolve().parent
SPLIT = BASE / "data" / "split"
MID = BASE / "data" / "midterm"
TAB = BASE / "tablas"
FIG = BASE / "figuras"
H_MS = 3_600_000
DIA_MS = 24 * H_MS
URL = re.compile(r"https?://\S+")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--window-h", type=float, default=24.0)
    ap.add_argument("--min-train-ratings", type=int, default=10)
    ap.add_argument("--events-per-user", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=100_000, help="eventos de test (0 = sin tope)")
    ap.add_argument("--val-days", type=int, default=30, help="días finales de train para entrenar el LTR")
    ap.add_argument("--max-val-events", type=int, default=40_000)
    ap.add_argument("--neg", type=int, default=20, help="negativos por evento de entrenamiento")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.9, 0.75, 0.5, 0.25])
    ap.add_argument("--lambda-principal", type=float, default=0.5)
    ap.add_argument("--include-decided", action="store_true")
    ap.add_argument("--sin-embeddings", action="store_true")
    ap.add_argument("--chunk", type=int, default=400, help="eventos por lote de predicción")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def limpia(s):
    return URL.sub(" ", html.unescape(s or ""))


ESCRITURAS = [
    ("latina", r"[A-Za-zÀ-ɏ]"), ("japonesa", r"[぀-ヿ]"),
    ("han", r"[一-鿿]"), ("coreana", r"[가-힯]"), ("arabe", r"[؀-ۿ]"),
    ("cirilica", r"[Ѐ-ӿ]"), ("hebrea", r"[֐-׿]"), ("griega", r"[Ͱ-Ͽ]"),
    ("thai", r"[฀-๿]"), ("devanagari", r"[ऀ-ॿ]"),
]
_RX = [re.compile(p) for _, p in ESCRITURAS]
N_ESC = len(ESCRITURAS) + 1   # + "otra"


def escritura(texto):
    c = [len(rx.findall(texto)) for rx in _RX]
    if c[1] > 0:                 # kana presente: el texto es japonés aunque tenga kanji
        c[1] += c[2]
        c[2] = 0
    k = int(np.argmax(c))
    return k if c[k] > 0 else N_ESC - 1


# ---------------------------------------------------------------------------
# Carga de datos (única parte que usa polars)
# ---------------------------------------------------------------------------
class Datos:
    """Lee data/split/ y data/midterm/ y entrega arrays de numpy."""

    def __init__(self, args):
        import polars as pl
        self.pl, self.args = pl, args

    def _ratings(self, lado):
        pl = self.pl
        return pl.scan_parquet(SPLIT / lado / "ratings.parquet").select(
            "noteId", pl.col("raterParticipantId").hash().alias("r"),
            pl.col("createdAtMillis").cast(pl.Int64).alias("t"))

    def base(self):
        pl, a = self.pl, self.args
        cols = ["noteId", "createdAtMillis", "summary"]
        notes = (pl.concat([pl.read_parquet(SPLIT / lado / "notes.parquet", columns=cols)
                            .with_columns(pl.lit(lado == "test").alias("es_test"))
                            for lado in ("train", "test")])
                 .with_columns(pl.col("createdAtMillis").cast(pl.Int64))
                 .sort(["createdAtMillis", "noteId"]).with_row_index("g"))
        st = pl.concat([pl.read_parquet(SPLIT / lado / "noteStatusHistory.parquet",
                                        columns=["noteId", "timestampMillisOfFirstNonNMRStatus"])
                        for lado in ("train", "test")]).unique("noteId")
        fd = (notes.select("noteId", "g")
              .join(st.select("noteId", pl.col("timestampMillisOfFirstNonNMRStatus")
                              .cast(pl.Float64, strict=False).alias("f")), on="noteId", how="left")
              .sort("g")["f"].fill_null(np.inf).to_numpy())
        fd = np.where(fd > 0, fd, np.inf)          # <= 0 = nunca decidida
        t_split = int(notes.filter(pl.col("es_test"))["createdAtMillis"].min())
        t_val = t_split - a.val_days * DIA_MS
        self.gmap = notes.select("noteId", "g", pl.col("createdAtMillis").alias("c")).lazy()
        self.t_split = t_split

        r_all = pl.concat([self._ratings("train"), self._ratings("test")])
        log("leyendo evaluaciones de las notas de abril y de test...")
        stream = (r_all.join(self.gmap, on="noteId").filter(pl.col("c") >= t_val)
                  .select("t", "g", "r").collect().sort(["t", "g", "r"]))

        def warm(tmax):
            return np.sort(r_all.filter(pl.col("t") < tmax).group_by("r").len()
                           .filter(pl.col("len") >= a.min_train_ratings).collect()["r"].to_numpy())

        post = pl.read_parquet(MID / "postura_raters.parquet").select(
            pl.col("raterParticipantId").hash().alias("r"), "f_u").sort("r")
        meta = json.loads((MID / "postura_meta.json").read_text(encoding="utf-8"))

        E = None
        if not a.sin_embeddings and (MID / "embeddings.npy").exists():
            ids = pl.read_parquet(MID / "embeddings_ids.parquet").with_row_index("k")
            pos = (notes.select("noteId", "g").join(ids, on="noteId", how="left").sort("g")["k"]
                   .fill_null(-1).to_numpy())
            raw = np.load(MID / "embeddings.npy")
            E = np.zeros((notes.height, raw.shape[1]), np.float32)
            ok = pos >= 0
            E[ok] = raw[pos[ok]].astype(np.float32)
            log(f"embeddings: {ok.mean():.1%} de las notas tienen vector")
        elif not a.sin_embeddings:
            log("AVISO: no está data/midterm/embeddings.npy -> el LTR corre sin sim_emb")

        return dict(
            created=notes["createdAtMillis"].to_numpy(), first_dec=fd,
            es_test=notes["es_test"].to_numpy(), summary=notes["summary"].fill_null("").to_list(),
            T_SPLIT=t_split, T_VAL=t_val,
            s_t=stream["t"].to_numpy(), s_g=stream["g"].to_numpy().astype(np.int64),
            s_r=stream["r"].to_numpy(),
            warm_test=warm(t_split), warm_val=warm(t_val),
            post_r=post["r"].to_numpy(), post_f=post["f_u"].to_numpy(),
            lam_b=meta["mult_intercepto"] * meta["lam_n"], lam_g=meta["lam_n"], E=E,
        )

    def historial(self, raters):
        """Evaluaciones (r, g, t) anteriores al inicio de test de los evaluadores dados."""
        pl = self.pl
        sel = pl.DataFrame({"r": raters}).lazy()
        h = (self._ratings("train").filter(pl.col("t") < self.t_split).join(sel, on="r", how="semi")
             .join(self.gmap, on="noteId").select("r", "g", "t").collect())
        return h["r"].to_numpy(), h["g"].to_numpy().astype(np.int64), h["t"].to_numpy()


# ---------------------------------------------------------------------------
# Utilidades numpy
# ---------------------------------------------------------------------------
def buscar(claves, valores, q, defecto=0.0):
    """valores[claves == q] con claves ordenadas; defecto si no está."""
    if len(claves) == 0:
        return np.full(len(q), defecto)
    p = np.searchsorted(claves, q)
    p = np.minimum(p, len(claves) - 1)
    return np.where(claves[p] == q, valores[p], defecto)


def pct(s):
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(len(s))
    return r / max(len(s) - 1, 1)


def boot_ci(vals, groups, B, rng):
    """IC 95% de la media remuestreando evaluadores (como baselines.py), en lotes de memoria acotada."""
    if len(vals) == 0:
        return np.nan, np.nan
    _, g = np.unique(groups, return_inverse=True)
    g_sum = np.bincount(g, weights=vals)
    g_n = np.bincount(g).astype(float)
    G = len(g_sum)
    out = []
    lote = max(1, int(2e7 // G))
    for b0 in range(0, B, lote):
        idx = rng.integers(0, G, (min(lote, B - b0), G), dtype=np.int32)
        out.append(g_sum[idx].sum(1) / g_n[idx].sum(1))
    lo, hi = np.percentile(np.concatenate(out), [2.5, 97.5])
    return float(lo), float(hi)


def normalizar_filas(M):
    from sklearn.preprocessing import normalize
    return normalize(M, norm="l2", axis=1)


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def experimento(D, historial, args):
    rng_ev = np.random.default_rng(args.seed)        # muestreo de eventos
    rng_tr = np.random.default_rng(args.seed + 1)    # negativos del LTR
    rng_s = np.random.default_rng(args.seed + 2)     # puntajes aleatorios y desempates
    rng_b = np.random.default_rng(args.seed + 3)     # bootstrap
    KS, KMAX = sorted(args.ks), max(args.ks)
    W = int(args.window_h * H_MS)

    created, first_dec = D["created"], D["first_dec"]
    N = len(created)
    s_t, s_g, s_r = D["s_t"], D["s_g"], D["s_r"]
    T_SPLIT, T_VAL = D["T_SPLIT"], D["T_VAL"]
    g_test0 = int(np.searchsorted(created, T_SPLIT, "left"))
    g_val0 = int(np.searchsorted(created, T_VAL, "left"))
    E = D["E"]
    LB, LG = D["lam_b"], D["lam_g"]
    f_stream = buscar(D["post_r"], D["post_f"], s_r)
    log(f"notas {N:,} | val desde {np.datetime64(T_VAL, 'ms')} | test desde {np.datetime64(T_SPLIT, 'ms')}")
    log(f"evaluaciones en el flujo (abril + test): {len(s_t):,} | con postura conocida: {(f_stream != 0).mean():.1%}")
    log(f"regularización del término de consenso: lam_b = {LB:.2f}, lam_g = {LG:.2f}")

    # ---------------- eventos (mismo protocolo que baselines.py) ----------------
    protocolo = {}

    def eventos(nombre, mask, warm, cap, maxev, rng):
        idx = np.flatnonzero(mask)
        n0 = len(idx)
        idx = idx[np.isin(s_r[idx], warm)]
        n1 = len(idx)
        dt = s_t[idx] - created[s_g[idx]]
        idx = idx[(dt >= 0) & (dt < W)]
        n2 = len(idx)
        if not args.include_decided:
            idx = idx[first_dec[s_g[idx]] > s_t[idx]]
        n3 = len(idx)
        idx = idx[rng.permutation(len(idx))]
        o = np.argsort(s_r[idx], kind="stable")
        idx_s, r_s = idx[o], s_r[idx[o]]
        ini = np.r_[True, r_s[1:] != r_s[:-1]] if len(r_s) else np.zeros(0, bool)
        pos = np.arange(len(r_s))
        cum = pos - np.maximum.accumulate(np.where(ini, pos, 0))
        idx = idx_s[cum < cap]
        if maxev and len(idx) > maxev:
            idx = rng.choice(idx, maxev, replace=False)
        idx = np.sort(idx)       # el flujo está ordenado por t
        protocolo[nombre] = dict(
            evaluaciones_periodo=n0, de_evaluadores_con_historial=n1, en_ventana=n2,
            en_ventana_y_pendientes=n3, perdidos_por_nota_decidida=n2 - n3,
            pct_perdidos_por_nota_decidida=(n2 - n3) / max(n2, 1), eventos=len(idx),
            evaluadores=len(np.unique(s_r[idx])))
        log(f"eventos {nombre}: {n0:,} evaluaciones -> con historial {n1:,} -> en ventana {n2:,} "
            f"-> pendientes {n3:,} (perdidos por nota ya decidida: {n2 - n3:,}, {(n2 - n3) / max(n2, 1):.1%}) "
            f"-> muestra {len(idx):,}")
        return idx

    ev_val = eventos("val (entrenamiento LTR)", (s_g >= g_val0) & (s_g < g_test0) & (s_t < T_SPLIT),
                     D["warm_val"], args.events_per_user, args.max_val_events, rng_ev)
    ev_te = eventos("test", s_g >= g_test0, D["warm_test"], args.events_per_user, args.max_events, rng_ev)
    if len(ev_te) == 0 or len(ev_val) == 0:
        raise SystemExit("No quedaron eventos: revisen --window-h / --min-train-ratings / --val-days.")

    users_val, users_te = np.unique(s_r[ev_val]), np.unique(s_r[ev_te])
    users_all = np.union1d(users_val, users_te)

    # ---------------- texto: TF-IDF, escritura ----------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    log("TF-IDF (fit en notas de train, como baselines.py) y escritura de cada nota...")
    textos = [limpia(s) for s in D["summary"]]
    tfidf = TfidfVectorizer(min_df=5, max_df=0.5, max_features=300_000, sublinear_tf=True,
                            token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
    tr_mask = ~D["es_test"]
    tfidf.fit([t for t, m in zip(textos, tr_mask) if m])
    X = tfidf.transform(textos).tocsr()
    esc = np.array([escritura(t) for t in textos], np.int64)
    ESC = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), esc)), shape=(N, N_ESC))
    log(f"vocabulario {len(tfidf.vocabulary_):,} | escrituras: "
        + ", ".join(f"{n} {np.mean(esc == k):.1%}" for k, (n, _) in enumerate(ESCRITURAS) if np.mean(esc == k) > 0.005))

    # ---------------- perfiles de evaluadores ----------------
    log(f"historial de {len(users_all):,} evaluadores...")
    h_r, h_g, h_t = historial(users_all)

    def perfiles(users, tmax):
        m = np.isin(h_r, users) & (h_t < tmax)
        A = sp.csr_matrix((np.ones(int(m.sum()), np.float32), (np.searchsorted(users, h_r[m]), h_g[m])),
                          shape=(len(users), N))
        A.sum_duplicates()
        A.data[:] = 1
        P = normalizar_filas(A @ X).tocsr()
        PE = normalizar_filas(np.asarray(A @ E)) if E is not None else None
        SPm = np.asarray((A @ ESC).todense())
        SPm = SPm / np.maximum(SPm.sum(1, keepdims=True), 1)
        nh = np.asarray(A.sum(1)).ravel()
        return dict(users=users, P=P, PE=PE, SP=SPm, nh=nh)

    prof_val, prof_te = perfiles(users_val, T_VAL), perfiles(users_te, T_SPLIT)

    # evaluaciones del flujo hechas por cada evaluador (para no recomendar lo ya evaluado)
    pm = np.isin(s_r, users_all)
    pr, pt, pg = s_r[pm], s_t[pm], s_g[pm]
    o = np.argsort(pr, kind="stable")
    pr, pt, pg = pr[o], pt[o], pg[o]
    cortes = np.flatnonzero(pr[1:] != pr[:-1]) + 1
    claves = pr[np.r_[0, cortes]].tolist() if len(pr) else []
    previas = dict(zip(claves, zip(np.split(pt, cortes), np.split(pg, cortes))))
    vacio = (np.empty(0, np.int64), np.empty(0, np.int64))

    # ---------------- estado acumulado de cada nota antes de t ----------------
    cnt = np.zeros(N, np.int64)
    cnt_lag = np.zeros(N, np.int64)
    S1, S2 = np.zeros(N), np.zeros(N)
    npos, nneg = np.zeros(N, np.int64), np.zeros(N, np.int64)
    ptr = [0, 0]

    def avanzar(t):
        nuevo = int(np.searchsorted(s_t, t, "left"))       # evaluaciones estrictamente antes de t
        if nuevo > ptr[0]:
            gg, ff = s_g[ptr[0]:nuevo], f_stream[ptr[0]:nuevo]
            np.add.at(cnt, gg, 1)
            np.add.at(S1, gg, ff)
            np.add.at(S2, gg, ff * ff)
            np.add.at(npos, gg, (ff > 0).astype(np.int64))
            np.add.at(nneg, gg, (ff < 0).astype(np.int64))
            ptr[0] = nuevo
        nuevo = int(np.searchsorted(s_t, t - H_MS, "left"))
        if nuevo > ptr[1]:
            np.add.at(cnt_lag, s_g[ptr[1]:nuevo], 1)
            ptr[1] = nuevo

    def candidatos(t, r, g0, tgt):
        lo = max(int(np.searchsorted(created, t - W, "right")), g0)
        hi = int(np.searchsorted(created, t, "right"))
        cand = np.arange(lo, hi)
        keep = np.ones(len(cand), bool)
        if not args.include_decided:
            keep &= first_dec[lo:hi] > t
        ptu, pgu = previas.get(int(r), vacio)
        hechas = pgu[ptu < t]
        if len(hechas):
            keep &= ~np.isin(cand, hechas)
        cand = cand[keep]
        pos = int(np.searchsorted(cand, tgt))
        assert pos < len(cand) and cand[pos] == tgt, "el target debería estar entre los candidatos"
        return cand, pos

    nombres_feat = ["log_pop", "log_pop_1h", "edad_h", "sim_tfidf"] + (["sim_emb"] if E is not None else []) \
        + ["escritura", "log_hist"]
    IDX_TFIDF = 3

    def features(cand, prof, u, t):
        pop, lag = cnt[cand], cnt_lag[cand]
        cols = [np.log1p(pop), np.log1p(pop - lag), (t - created[cand]) / H_MS,
                (X[cand] @ prof["P"][u].T).toarray().ravel()]
        if E is not None:
            cols.append(E[cand] @ prof["PE"][u])
        cols.append(prof["SP"][u][esc[cand]])
        cols.append(np.full(len(cand), np.log1p(prof["nh"][u])))
        return np.column_stack(cols).astype(np.float32)

    def consenso(cand, fu):
        S0 = cnt[cand].astype(float)
        m00, m01, m11 = S0 + LB, S1[cand], S2[cand] + LG
        det = m00 * m11 - m01 ** 2
        a0 = (m11 - m01 * fu) / det
        a1 = (m00 * fu - m01) / det
        dvar = a0 ** 2 / (1.0 + a0 + fu * a1)
        lado = np.sign(fu)
        mino = np.sign(nneg[cand] - npos[cand])          # +1: faltan del lado +, -1: faltan del lado -
        aporte = np.where(mino == 0, 0.5, (mino == lado).astype(float))
        return dvar, aporte

    # ---------------- 1) entrenamiento del baseline entrenado ----------------
    log(f"construyendo datos de entrenamiento del LTR ({len(ev_val):,} eventos de val)...")
    F_list, y_list, e_list = [], [], []
    for e, k in enumerate(ev_val):
        t, tgt, r = int(s_t[k]), int(s_g[k]), s_r[k]
        avanzar(t)
        cand, pos = candidatos(t, r, g_val0, tgt)
        negs = np.delete(cand, pos)
        if len(negs) == 0:
            continue
        if len(negs) > args.neg:
            negs = rng_tr.choice(negs, args.neg, replace=False)
        filas = np.r_[tgt, negs]
        u = int(np.searchsorted(prof_val["users"], r))
        F_list.append(features(filas, prof_val, u, t))
        y_list.append(np.r_[1, np.zeros(len(negs), np.int8)])
        e_list.append(np.full(len(filas), e))
        if (e + 1) % 10000 == 0:
            log(f"  {e + 1:,}/{len(ev_val):,}")
    Ftr, ytr, etr = np.vstack(F_list), np.concatenate(y_list), np.concatenate(e_list)
    del F_list, y_list, e_list

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    hold = (etr % 10) == 0                    # 10% de los eventos como validación
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=30, random_state=args.seed)
    clf.fit(Ftr[~hold], ytr[~hold])
    auc = roc_auc_score(ytr[hold], clf.predict_proba(Ftr[hold])[:, 1])
    log(f"LTR: {len(ytr):,} filas, {clf.n_iter_} iteraciones, AUC en validación = {auc:.4f}")
    sub = np.flatnonzero(hold)
    if len(sub) > 60_000:
        sub = rng_tr.choice(sub, 60_000, replace=False)
    pi = permutation_importance(clf, Ftr[sub], ytr[sub], scoring="roc_auc", n_repeats=3,
                                random_state=args.seed)
    importancia = sorted(zip(nombres_feat, pi.importances_mean, pi.importances_std), key=lambda z: -z[1])
    for n_, m_, s_ in importancia:
        log(f"   importancia {n_:<11s} {m_:.4f} ± {s_:.4f}")
    protocolo["ltr"] = dict(filas_entrenamiento=int(len(ytr)), auc_validacion=float(auc),
                            iteraciones=int(clf.n_iter_), features=" ".join(nombres_feat))

    # ---------------- 2) evaluación en test ----------------
    lams = [lam for lam in args.lambdas if 0 < lam < 1]
    modelos = (["random", "most_popular", "content_based", "ltr", "consenso"]
               + [f"propuesto_l{lam:g}" for lam in lams])
    n_ev = len(ev_te)
    rank = {m: np.zeros(n_ev, np.int32) for m in modelos}
    recs = {m: np.full((n_ev, KMAX), -1, np.int32) for m in modelos}
    ap = {m: {K: np.full(n_ev, np.nan) for K in KS} for m in modelos}
    dv = {m: {K: np.full(n_ev, np.nan) for K in KS} for m in modelos}
    obs_ap, obs_dv = np.full(n_ev, np.nan), np.full(n_ev, np.nan)
    n_cand = np.zeros(n_ev, np.int32)
    visto = np.zeros(N, bool)
    ev_u = np.searchsorted(users_te, s_r[ev_te])
    f_ev = buscar(D["post_r"], D["post_f"], s_r[ev_te])
    log(f"evaluando {n_ev:,} eventos de test ({len(users_te):,} evaluadores; "
        f"{(f_ev != 0).mean():.1%} con postura)...")

    for c0 in range(0, n_ev, args.chunk):
        lote = []
        for e in range(c0, min(n_ev, c0 + args.chunk)):
            k = ev_te[e]
            t, tgt, r = int(s_t[k]), int(s_g[k]), s_r[k]
            avanzar(t)
            cand, pos = candidatos(t, r, g_test0, tgt)
            F = features(cand, prof_te, int(ev_u[e]), t)
            dvar, aporte = consenso(cand, f_ev[e])
            lote.append((e, cand, pos, F, dvar, aporte, cnt[cand].astype(float)))
        proba = clf.predict_proba(np.vstack([z[3] for z in lote]))[:, 1]
        off = 0
        for e, cand, pos, F, dvar, aporte, pop in lote:
            n = len(cand)
            p = proba[off:off + n]
            off += n
            n_cand[e] = n
            visto[cand] = True
            jit = rng_s.random(n) * 1e-9
            sc = {"random": rng_s.random(n),
                  "most_popular": pop + rng_s.random(n) * 0.5,
                  "content_based": F[:, IDX_TFIDF] + jit,
                  "ltr": p + jit,
                  "consenso": dvar + jit}
            pa, pv = pct(sc["ltr"]), pct(sc["consenso"])
            for lam in lams:
                sc[f"propuesto_l{lam:g}"] = lam * pa + (1 - lam) * pv + jit
            con_lado = f_ev[e] != 0
            obs_dv[e] = dvar[pos]
            if con_lado:
                obs_ap[e] = aporte[pos]
            kk = min(KMAX, n)
            for m, s in sc.items():
                rank[m][e] = 1 + int((s > s[pos]).sum())
                top = np.argpartition(-s, kk - 1)[:kk]
                top = top[np.argsort(-s[top])]
                recs[m][e, :kk] = cand[top]
                for K in KS:
                    tk = top[:K]
                    dv[m][K][e] = dvar[tk].mean()
                    if con_lado:
                        ap[m][K][e] = aporte[tk].mean()
        if (c0 // args.chunk) % 10 == 9 or c0 + args.chunk >= n_ev:
            log(f"  {min(n_ev, c0 + args.chunk):,}/{n_ev:,} eventos")

    # ---------------- 3) métricas ----------------
    filas = []
    den_cov = max(int(visto.sum()), 1)
    B = args.bootstrap
    for m in modelos + ["observado"]:
        for K in KS:
            fila = dict(modelo=m, K=K)
            if m != "observado":
                rk = rank[m]
                hit = (rk <= K).astype(float)
                ndcg = np.where(rk <= K, 1.0 / np.log2(rk + 1.0), 0.0)
                rr = recs[m][:, :K]
                fila.update(recall=hit.mean(), ndcg=ndcg.mean(),
                            cobertura=len(np.unique(rr[rr >= 0])) / den_cov)
                fila["recall_lo"], fila["recall_hi"] = boot_ci(hit, ev_u, B, rng_b)
                fila["ndcg_lo"], fila["ndcg_hi"] = boot_ci(ndcg, ev_u, B, rng_b)
                va, vd = ap[m][K], dv[m][K]
            else:
                va, vd = obs_ap, obs_dv
            ok = ~np.isnan(va)
            fila["aporte_minoria"] = float(va[ok].mean()) if ok.any() else np.nan
            fila["aporte_lo"], fila["aporte_hi"] = boot_ci(va[ok], ev_u[ok], B, rng_b)
            fila["reduccion_var"] = float(np.nanmean(vd))
            fila["redvar_lo"], fila["redvar_hi"] = boot_ci(vd, ev_u, B, rng_b)
            filas.append(fila)
    protocolo["test_resumen"] = dict(candidatas_mediana=float(np.median(n_cand)),
                                     candidatas_media=float(n_cand.mean()),
                                     random_recall10_esperado=float(np.mean(np.minimum(10, n_cand) / n_cand)))
    return dict(filas=filas, protocolo=protocolo, importancia=importancia, modelos=modelos, lams=lams,
                rank=rank, recs=recs, n_cand=n_cand, ev_u=ev_u)


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
COLS = ["modelo", "K", "recall", "recall_lo", "recall_hi", "ndcg", "ndcg_lo", "ndcg_hi", "cobertura",
        "aporte_minoria", "aporte_lo", "aporte_hi", "reduccion_var", "redvar_lo", "redvar_hi"]
NOMBRES = {"random": "Random", "most_popular": "Most Popular", "content_based": "Content-based",
           "ltr": "LTR (entrenado)", "consenso": "Consenso (dVar)", "observado": "Observado"}


def nombre(m):
    return NOMBRES.get(m, m.replace("propuesto_l", "Propuesto λ="))


def escribir_csv(path, filas, cols):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for f in filas:
            fh.write(",".join("" if f.get(c) is None or (isinstance(f.get(c), float) and np.isnan(f.get(c)))
                              else (f"{f[c]:.6f}" if isinstance(f[c], float) else str(f[c])) for c in cols) + "\n")


def figura_tradeoff(R, K=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AZUL, NARANJA, GRIS, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e"
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRIS,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": TINTA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
        "ytick.color": TINTA2, "font.size": 9.5, "legend.frameon": False})
    fk = {f["modelo"]: f for f in R["filas"] if f["K"] == K}
    curva = ["ltr"] + [f"propuesto_l{lam:g}" for lam in sorted(R["lams"], reverse=True)] + ["consenso"]
    etiq = {"ltr": "λ=1 (LTR)", "consenso": "λ=0 (consenso)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, met, ylab in [(axes[0], "aporte_minoria", f"Aporte al lado minoritario@{K}"),
                          (axes[1], "reduccion_var", f"Reducción de incertidumbre@{K}")]:
        xs = [fk[m]["recall"] for m in curva]
        ys = [fk[m][met] for m in curva]
        ax.plot(xs, ys, "-o", color=AZUL, ms=6, lw=2, zorder=3)
        for m, x, y in zip(curva, xs, ys):
            ax.annotate(etiq.get(m, m.replace("propuesto_l", "λ=")), (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=TINTA2)
        for m, dy in [("random", -10), ("most_popular", 6), ("content_based", -10)]:
            ax.plot(fk[m]["recall"], fk[m][met], "s", color=GRIS, ms=7, zorder=3)
            ax.annotate(nombre(m), (fk[m]["recall"], fk[m][met]), textcoords="offset points",
                        xytext=(6, dy), fontsize=8, color=TINTA2)
        obs = fk["observado"][met]
        ax.axhline(obs, color=NARANJA, lw=1.2, ls="--")
        ax.text(ax.get_xlim()[1], obs, "observado ", ha="right", va="bottom", color=NARANJA, fontsize=8)
        ax.set_xlabel(f"Recall@{K} (afinidad)")
        ax.set_ylabel(ylab)
    axes[0].set_title("Afinidad vs. aporte al consenso")
    axes[1].set_title("Afinidad vs. reducción de incertidumbre")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig7_tradeoff.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig7_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    log("figuras/fig7_tradeoff.png")


def guardar(R, args):
    TAB.mkdir(parents=True, exist_ok=True)
    MID.mkdir(parents=True, exist_ok=True)
    escribir_csv(TAB / "midterm_resultados.csv", R["filas"], COLS)
    lam_filas = [f for f in R["filas"] if f["K"] == 10 and (f["modelo"].startswith("propuesto")
                                                           or f["modelo"] in ("ltr", "consenso"))]
    for f in lam_filas:
        f["lambda"] = {"ltr": 1.0, "consenso": 0.0}[f["modelo"]] if f["modelo"] in ("ltr", "consenso") \
            else float(f["modelo"][len("propuesto_l"):])
    lam_filas.sort(key=lambda f: -f["lambda"])
    escribir_csv(TAB / "midterm_lambda.csv", lam_filas, ["lambda"] + COLS)
    with open(TAB / "midterm_protocolo.csv", "w", encoding="utf-8") as fh:
        fh.write("bloque,dato,valor\n")
        for b, d in R["protocolo"].items():
            for k, v in d.items():
                fh.write(f"\"{b}\",{k},{v:.4f}\n" if isinstance(v, float) else f"\"{b}\",{k},{v}\n")
    with open(TAB / "ltr_importancia.csv", "w", encoding="utf-8") as fh:
        fh.write("feature,importancia_auc,desv\n")
        for n_, m_, s_ in R["importancia"]:
            fh.write(f"{n_},{m_:.6f},{s_:.6f}\n")
    np.savez_compressed(MID / "eventos_test.npz", n_cand=R["n_cand"], ev_u=R["ev_u"],
                        **{f"rank_{m}": R["rank"][m] for m in R["modelos"]})
    if 10 in args.ks:
        figura_tradeoff(R, 10)

    print("\n" + "=" * 96)
    for K in sorted(args.ks):
        print(f"\nK = {K}")
        print(f"{'modelo':<22}{'Recall':>9}{'NDCG':>9}{'Cobert.':>9}{'AporteMin':>11}{'RedVar':>10}")
        for f in R["filas"]:
            if f["K"] != K:
                continue
            rec = f"{f['recall']:.4f}" if "recall" in f else "-"
            nd = f"{f['ndcg']:.4f}" if "ndcg" in f else "-"
            co = f"{f['cobertura']:.3f}" if "cobertura" in f else "-"
            print(f"{nombre(f['modelo']):<22}{rec:>9}{nd:>9}{co:>9}{f['aporte_minoria']:>11.4f}"
                  f"{f['reduccion_var']:>10.4f}")
    log("Listo. tablas/midterm_*.csv")


def main(argv=None):
    args = parse_args(argv)
    if not (MID / "postura_raters.parquet").exists():
        raise SystemExit("Falta data/midterm/postura_raters.parquet: corran primero postura.py")
    datos = Datos(args)
    D = datos.base()
    R = experimento(D, datos.historial, args)
    guardar(R, args)


if __name__ == "__main__":
    main()
