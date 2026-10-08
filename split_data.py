"""
Split temporal train/test de Community Notes, por mes de creación de la NOTA.

Uso (solamente uno se debe ejecutar):
  python split_train_test.py          # lee data/subset/ (salida de preparar_datos.py)
  python split_train_test.py --raiz   # lee los parquet junto a este script

Entrada por defecto (generada por preparar_datos.py): data/subset/
  notes.parquet, noteStatusHistory.parquet, ratings/*.parquet
Entrada con --raiz (misma carpeta que este script):
  notes.parquet, noteStatusHistory.parquet, ratings*.parquet (p. ej. ratings-00000.parquet)
Salida: data/split/
  train/{notes,noteStatusHistory,ratings}.parquet
  test/{notes,noteStatusHistory,ratings}.parquet

Criterio: cada nota (y TODOS sus ratings y su historial de estado) va a un único
lado del split, según el mes (UTC) en que fue creada la nota. Así ninguna nota de
test aparece en train, y los meses de test son posteriores a los de train.
"""

import argparse
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MESES_TEST = ["2025-05", "2025-06"]

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "split"

# ---------------------------------------------------------------------------
# Ubicación de los archivos de entrada
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Split train/test de Community Notes por mes.")
parser.add_argument("--raiz", action="store_true",
                    help="buscar los parquet en la misma carpeta que este script "
                         "en vez de en data/subset/")
args = parser.parse_args()

if args.raiz:
    SUBSET = BASE
    ratings_files = sorted(BASE.glob("ratings*.parquet"))
    ratings_desc = f"{BASE}/ratings*.parquet"
else:
    SUBSET = BASE / "data" / "subset"
    ratings_files = sorted((SUBSET / "ratings").glob("*.parquet"))
    ratings_desc = f"{SUBSET}/ratings/*.parquet"

notes_path = SUBSET / "notes.parquet"
st_path = SUBSET / "noteStatusHistory.parquet"

faltantes = []
if not notes_path.exists():
    faltantes.append(str(notes_path))
if not ratings_files:
    faltantes.append(ratings_desc)
if faltantes:
    donde = "en la carpeta del script" if args.raiz else "en data/subset/"
    print(f"No se encontraron los archivos necesarios {donde}:")
    for f in faltantes:
        print(f"  - {f}")
    raise SystemExit(1)

print(f"Leyendo desde: {SUBSET}")
print(f"  ratings: {len(ratings_files)} archivo(s)")

for lado in ("train", "test"):
    (OUT / lado).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Asignar cada nota a train/test según su mes de creación
# ---------------------------------------------------------------------------
notes = pl.read_parquet(notes_path).with_columns(
    pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms")
    .dt.strftime("%Y-%m")
    .alias("mes")
)

meses = sorted(notes["mes"].unique().to_list())
print("Notas por mes de creación:")
print(notes.group_by("mes").len().sort("mes"))

faltan = set(MESES_TEST) - set(meses)
if faltan:
    raise SystemExit(f"MESES_TEST contiene meses sin notas: {sorted(faltan)}. Disponibles: {meses}")

meses_train = [m for m in meses if m not in MESES_TEST]
if not meses_train:
    raise SystemExit("No quedan meses para train.")
if max(meses_train) > min(MESES_TEST):
    print("\n[AVISO] Algún mes de train es posterior a uno de test: el split no es "
          "estrictamente temporal (train < test).")

notes_train = notes.filter(~pl.col("mes").is_in(MESES_TEST)).drop("mes")
notes_test = notes.filter(pl.col("mes").is_in(MESES_TEST)).drop("mes")
ids_train = notes_train["noteId"]
ids_test = notes_test["noteId"]

print(f"\nTrain: meses {meses_train} -> {notes_train.height:,} notas")
print(f"Test:  meses {sorted(MESES_TEST)} -> {notes_test.height:,} notas")

notes_train.write_parquet(OUT / "train" / "notes.parquet")
notes_test.write_parquet(OUT / "test" / "notes.parquet")

# ---------------------------------------------------------------------------
# 2. Historial de estados (mismo criterio: por nota)
# ---------------------------------------------------------------------------
status = {}
if st_path.exists():
    st = pl.read_parquet(st_path)
    for lado, ids in (("train", ids_train), ("test", ids_test)):
        status[lado] = st.filter(pl.col("noteId").is_in(ids.implode()))
        status[lado].write_parquet(OUT / lado / "noteStatusHistory.parquet")
        print(f"noteStatusHistory {lado}: {status[lado].height:,} filas")
else:
    print("noteStatusHistory.parquet no encontrado, se omite.")

# ---------------------------------------------------------------------------
# 3. Ratings (en streaming, no se carga todo en memoria)
# ---------------------------------------------------------------------------
print("\nRatings (puede tardar)...")
ratings = pl.scan_parquet(ratings_files)
for lado, ids in (("train", ids_train), ("test", ids_test)):
    destino = OUT / lado / "ratings.parquet"
    ratings.filter(pl.col("noteId").is_in(ids.implode())).sink_parquet(destino)
    n = pl.scan_parquet(destino).select(pl.len()).collect().item()
    print(f"  ratings {lado}: {n:,}")

# ---------------------------------------------------------------------------
# 4. Diagnóstico del split
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
r_train = pl.scan_parquet(OUT / "train" / "ratings.parquet")
r_test = pl.scan_parquet(OUT / "test" / "ratings.parquet")

n_tr = r_train.select(pl.len()).collect().item()
n_te = r_test.select(pl.len()).collect().item()
print(f"Ratings train: {n_tr:,} | test: {n_te:,}"
      + (f" ({n_te / (n_tr + n_te):.1%} test)" if n_tr + n_te else ""))

if n_te:
    # Cold-start: evaluadores de test que nunca votaron en train
    raters_tr = r_train.select("raterParticipantId").unique().collect()["raterParticipantId"]
    raters_te = r_test.select("raterParticipantId").unique().collect()["raterParticipantId"]
    nuevos = raters_te.filter(~raters_te.is_in(raters_tr.implode()))
    print(f"Evaluadores train: {len(raters_tr):,} | test: {len(raters_te):,}")
    print(f"Evaluadores de test NO vistos en train: {len(nuevos):,} "
          f"({len(nuevos) / len(raters_te):.1%})")

    ratings_nuevos = (r_test.filter(pl.col("raterParticipantId").is_in(nuevos.implode()))
                      .select(pl.len()).collect().item())
    print(f"Ratings de test hechos por evaluadores nuevos: {ratings_nuevos:,} "
          f"({ratings_nuevos / n_te:.1%})")

for lado, df in status.items():
    if "currentStatus" in df.columns:
        print(f"\nEstado actual de las notas ({lado}):")
        print(df["currentStatus"].value_counts(sort=True))

print("=" * 50)
print(f"Listo. Archivos en {OUT}")