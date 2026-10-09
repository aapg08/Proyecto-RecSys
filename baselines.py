"""
Baselines H1: Random, Most Popular (hasta t) y Content-based (TF-IDF) para Community Notes.

Uso (después de preparar_datos.py y split_data.py):
  pip install -U polars scikit-learn scipy numpy
  python baselines.py                       # parámetros por defecto
  python baselines.py --window-h 48 --events-per-user 5 --max-events 200000

Entrada : data/split/{train,test}/{notes,noteStatusHistory,ratings}.parquet
Salida  : tablas/baselines.csv          (métricas por modelo y K, con IC 95%)
          data/baselines/events.parquet (eventos evaluados + posición del target por modelo)
          data/baselines/recs_<modelo>.npy (top-Kmax notas recomendadas por evento, índices de
                                            notes_test ordenadas por fecha; sirve para M1/M2 después)

PROTOCOLO (cada evento de test es una pregunta "¿qué nota le muestro a este evaluador ahora?")
  Evento  : un rating (u, n, t) de test. El target es la nota n.
  Candidatos en t: notas de test creadas en (t - W, t], aún sin decisión (Útil/No útil) en t,
            y que u no haya evaluado antes de t. Son las notas "pendientes y frescas".
  Random  : puntaje aleatorio.
  MostPop : nº de evaluaciones que la nota ya tiene ANTES de t (no el conteo final).
  Content : coseno entre el perfil TF-IDF de u (promedio de las notas que evaluó en train,
            solo ratings anteriores al inicio de test) y el texto de la nota candidata.
  Evaluadores: solo los con >= --min-train-ratings evaluaciones en train (como el mínimo de X).
  Muestreo: hasta --events-per-user eventos por evaluador (aleatorios), para que los muy
            activos no dominen el promedio.
Métricas: Recall@K (= hit rate, hay un único target), NDCG@K, item coverage@K.
"""

import argparse
import html
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parent
SPLIT = BASE / "data" / "split"
OUT_DIR = BASE / "data" / "baselines"
TAB = BASE / "tablas"
H_MS = 3_600_000

ap = argparse.ArgumentParser()
ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
ap.add_argument("--window-h", type=float, default=24.0, help="ventana W de frescura de candidatos")
ap.add_argument("--min-train-ratings", type=int, default=10)
ap.add_argument("--events-per-user", type=int, default=3)
ap.add_argument("--max-events", type=int, default=100_000, help="0 = sin tope")
ap.add_argument("--include-decided", action="store_true",
                help="no excluir de los candidatos las notas ya decididas (Útil/No útil) en t")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
KS, KMAX = sorted(args.ks), max(args.ks)
W_MS = int(args.window_h * H_MS)
rng = np.random.default_rng(args.seed)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Notas
# ---------------------------------------------------------------------------
notes_tr = pl.read_parquet(SPLIT / "train" / "notes.parquet", columns=["noteId", "createdAtMillis", "summary"])
notes_te = (pl.read_parquet(SPLIT / "test" / "notes.parquet", columns=["noteId", "createdAtMillis", "summary"])
            .sort("createdAtMillis").with_row_index("j"))           # j = posición por fecha
notes_tr = notes_tr.with_row_index("i")
T_SPLIT = int(notes_te["createdAtMillis"].min())                    # inicio del periodo de test
N_TE = notes_te.height
created = notes_te["createdAtMillis"].to_numpy()

# Primer momento en que la nota dejó de estar en "Necesita más evaluaciones" (inf = nunca)
st = pl.read_parquet(SPLIT / "test" / "noteStatusHistory.parquet",
                     columns=["noteId", "timestampMillisOfFirstNonNMRStatus"])
first_dec = (notes_te.select("noteId", "j")
             .join(st.select("noteId", pl.col("timestampMillisOfFirstNonNMRStatus")
                             .cast(pl.Float64, strict=False).alias("f")), on="noteId", how="left")
             .sort("j")["f"].fill_null(np.inf).to_numpy())
log(f"notas train {notes_tr.height:,} | test {N_TE:,} | inicio test {np.datetime64(T_SPLIT, 'ms')}")

# ---------------------------------------------------------------------------
# 2. Ratings
# ---------------------------------------------------------------------------
# El id del evaluador (64 caracteres) se reemplaza por su hash UInt64: mismo id <-> mismo hash,
# colisiones despreciables, y la memoria baja ~8x con los ~14 M de ratings de test.
r_tr = pl.scan_parquet(SPLIT / "train" / "ratings.parquet").select(
    "noteId", pl.col("raterParticipantId").hash().alias("rater"),
    pl.col("createdAtMillis").cast(pl.Int64).alias("t"))
# solo train anterior al test (evita usar evaluaciones del "futuro" para armar perfiles)
r_tr = r_tr.filter(pl.col("t") < T_SPLIT)
r_te = (pl.scan_parquet(SPLIT / "test" / "ratings.parquet")
        .select("noteId", pl.col("raterParticipantId").hash().alias("rater"),
                pl.col("createdAtMillis").cast(pl.Int64).alias("t"))
        .join(notes_te.select("noteId", "j").lazy(), on="noteId")
        .collect().sort("t"))
log(f"ratings test: {r_te.height:,}")

warm = (r_tr.group_by("rater").len().filter(pl.col("len") >= args.min_train_ratings)
        .select("rater").collect())
log(f"evaluadores con >= {args.min_train_ratings} evaluaciones en train: {warm.height:,}")

# ---------------------------------------------------------------------------
# 3. Eventos de evaluación
# ---------------------------------------------------------------------------
ev = r_te.join(warm, on="rater", how="semi")
n_ev0 = ev.height
log(f"ratings de test hechos por evaluadores sin historial suficiente (excluidos): "
    f"{r_te.height - n_ev0:,} ({(r_te.height - n_ev0) / r_te.height:.1%})")
j_arr = ev["j"].to_numpy()
t_arr = ev["t"].to_numpy()
ok = (t_arr >= created[j_arr]) & (t_arr - created[j_arr] < W_MS)
n_win = int(ok.sum())
if not args.include_decided:
    ok &= first_dec[j_arr] > t_arr
ev = ev.filter(pl.Series(ok))
log(f"eventos de evaluadores warm: {n_ev0:,} -> en ventana {n_win:,} -> también pendientes {ev.height:,}")

ev = (ev.with_columns(pl.Series("rnd", rng.random(ev.height)))
        .sort("rnd").group_by("rater", maintain_order=True).head(args.events_per_user))
if args.max_events and ev.height > args.max_events:
    ev = ev.sample(args.max_events, seed=args.seed)
ev = ev.drop("rnd").sort("t")
users = ev.select("rater").unique().with_row_index("u")
ev = ev.join(users, on="rater")
n_users = users.height
log(f"eventos evaluados: {ev.height:,} de {n_users:,} evaluadores")
if ev.height == 0:
    raise SystemExit("No quedaron eventos: revisen --window-h / --min-train-ratings.")

# Evaluaciones previas de test de esos evaluadores (para no recomendar lo ya evaluado)
prior = (r_te.join(users, on="rater").select("u", "j", "t").sort("t"))
prior_by_u = {}
for (u,), g in prior.group_by("u"):
    prior_by_u[u] = (g["t"].to_numpy(), g["j"].to_numpy())

# ---------------------------------------------------------------------------
# 4. Content-based: TF-IDF + perfiles de usuario
# ---------------------------------------------------------------------------
URL = re.compile(r"https?://\S+")


def limpia(s):
    return URL.sub(" ", html.unescape(s or ""))


log("TF-IDF (fit en notas de train)...")
tfidf = TfidfVectorizer(min_df=5, max_df=0.5, max_features=300_000, sublinear_tf=True,
                        token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
X_tr = tfidf.fit_transform([limpia(s) for s in notes_tr["summary"].to_list()])
X_te = tfidf.transform([limpia(s) for s in notes_te["summary"].to_list()]).tocsr()   # fila j
log(f"vocabulario {len(tfidf.vocabulary_):,} términos")

hist = (r_tr.join(users.lazy(), on="rater").join(notes_tr.select("noteId", "i").lazy(), on="noteId")
        .select("u", "i").unique().collect())
A = sp.csr_matrix((np.ones(hist.height, np.float32), (hist["u"].to_numpy(), hist["i"].to_numpy())),
                  shape=(n_users, notes_tr.height))
P = A @ X_tr                                              # suma de los vectores de sus notas
norm = np.sqrt(np.asarray(P.multiply(P).sum(axis=1)).ravel())
P = sp.diags(1.0 / np.maximum(norm, 1e-12)) @ P            # perfil normalizado (coseno)
P = P.tocsr()
log("perfiles listos")

# ---------------------------------------------------------------------------
# 5. Evaluación
# ---------------------------------------------------------------------------
MODELS = ["random", "most_popular", "content_based"]
ev_u, ev_j, ev_t = ev["u"].to_numpy(), ev["j"].to_numpy(), ev["t"].to_numpy()
E = len(ev_t)
rank = {m: np.zeros(E, np.int32) for m in MODELS}
recs = {m: np.full((E, KMAX), -1, np.int32) for m in MODELS}
n_cand = np.zeros(E, np.int32)
seen_cand = np.zeros(N_TE, bool)

all_t, all_j = r_te["t"].to_numpy(), r_te["j"].to_numpy()
cnt = np.zeros(N_TE, np.int64)             # evaluaciones acumuladas por nota ANTES de t
ptr = 0
JIT = 1e-6                                 # desempate aleatorio (evita sesgo por orden de creación)

for e in range(E):
    t, u, tgt = ev_t[e], ev_u[e], ev_j[e]
    new_ptr = np.searchsorted(all_t, t, side="left")        # ratings estrictamente anteriores a t
    if new_ptr > ptr:
        cnt += np.bincount(all_j[ptr:new_ptr], minlength=N_TE)
        ptr = new_ptr

    lo = np.searchsorted(created, t - W_MS, side="right")
    hi = np.searchsorted(created, t, side="right")
    cand = np.arange(lo, hi)
    keep = np.ones(len(cand), bool)
    if not args.include_decided:
        keep &= first_dec[lo:hi] > t
    pt, pj = prior_by_u.get(u, (np.empty(0), np.empty(0, np.int64)))
    done = pj[pt < t]
    if len(done):
        keep &= ~np.isin(cand, done)
    cand = cand[keep]
    n_cand[e] = len(cand)
    seen_cand[cand] = True
    pos = np.searchsorted(cand, tgt)                        # cand está ordenado
    assert pos < len(cand) and cand[pos] == tgt, "el target debería estar entre los candidatos"

    s_cb = (X_te[cand] @ P[u].T).toarray().ravel()
    scores = {
        "random": rng.random(len(cand)),
        "most_popular": cnt[cand] + rng.random(len(cand)) * 0.5,
        "content_based": s_cb + rng.random(len(cand)) * JIT,
    }
    for m, s in scores.items():
        rank[m][e] = 1 + int((s > s[pos]).sum())
        k = min(KMAX, len(cand))
        top = np.argpartition(-s, k - 1)[:k]
        recs[m][e, :k] = cand[top[np.argsort(-s[top])]]
    if (e + 1) % 5000 == 0:
        log(f"  {e + 1:,}/{E:,} eventos")

# ---------------------------------------------------------------------------
# 6. Métricas (+ IC 95% por bootstrap sobre evaluadores)
# ---------------------------------------------------------------------------
def boot_ci(vals, groups, B=1000):
    """IC 95% de la media, remuestreando evaluadores (los eventos de un evaluador van juntos)."""
    g_sum = np.bincount(groups, weights=vals)
    g_n = np.bincount(groups).astype(float)
    idx = rng.integers(0, len(g_sum), (B, len(g_sum)))
    m = g_sum[idx].sum(1) / g_n[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


rows = []
denom_cov = max(int(seen_cand.sum()), 1)
for m in MODELS:
    r = rank[m]
    for K in KS:
        hit = (r <= K).astype(float)
        ndcg = np.where(r <= K, 1.0 / np.log2(r + 1.0), 0.0)
        cov = len(np.unique(recs[m][:, :K][recs[m][:, :K] >= 0])) / denom_cov
        lo_r, hi_r = boot_ci(hit, ev_u)
        lo_n, hi_n = boot_ci(ndcg, ev_u)
        rows.append(dict(modelo=m, K=K, recall=hit.mean(), recall_lo=lo_r, recall_hi=hi_r,
                         ndcg=ndcg.mean(), ndcg_lo=lo_n, ndcg_hi=hi_n, item_coverage=cov))
res = pl.DataFrame(rows)
TAB.mkdir(exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
res.write_csv(TAB / "baselines.csv")
ev.with_columns([pl.Series(f"rank_{m}", rank[m]) for m in MODELS]
                + [pl.Series("n_cand", n_cand)]).write_parquet(OUT_DIR / "events.parquet")
for m in MODELS:
    np.save(OUT_DIR / f"recs_{m}.npy", recs[m])
notes_te.select("noteId", "j").write_parquet(OUT_DIR / "map_notes_test.parquet")

print("\n" + "=" * 78)
print(f"Eventos: {E:,} | evaluadores: {n_users:,} | ventana W = {args.window_h:g} h | "
      f"candidatos por evento: mediana {np.median(n_cand):.0f}, media {n_cand.mean():.0f}")
print(f"Random esperado: Recall@10 = {np.mean(np.minimum(10, n_cand) / n_cand):.4f}")
print("=" * 78)
for K in KS:
    print(f"\nK = {K}")
    print(f"{'modelo':<15}{'Recall@K':>10}{'  IC95%':>18}{'NDCG@K':>10}{'  IC95%':>18}{'Coverage':>10}")
    for r in res.filter(pl.col("K") == K).iter_rows(named=True):
        print(f"{r['modelo']:<15}{r['recall']:>10.4f}  [{r['recall_lo']:.4f},{r['recall_hi']:.4f}]"
              f"{r['ndcg']:>10.4f}  [{r['ndcg_lo']:.4f},{r['ndcg_hi']:.4f}]{r['item_coverage']:>10.3f}")
log("Listo. tablas/baselines.csv")