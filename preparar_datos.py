"""
Prepara el subconjunto de Community Notes para H1.

Uso:
  1. Creen la carpeta data/raw/ (al lado de este script) y pongan ahí TODOS los .zip
     descargados (notas, ratings, historial de estados, enrollment). No los descompriman.
  2. pip install -U polars
  3. python preparar_datos.py

Procesa un archivo a la vez: descomprime, filtra y (opcionalmente) borra el .tsv,
así no necesitan RAM ni disco para el dataset completo.

Resultado en data/subset/:
  notes.parquet, ratings/*.parquet, noteStatusHistory.parquet, userEnrollment.parquet
"""

import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
INICIO = datetime(2025, 1, 1, tzinfo=timezone.utc)
FIN = datetime(2025, 7, 1, tzinfo=timezone.utc)  # excluyente -> enero a junio 2025

BORRAR_TSV = True  # borra cada .tsv después de procesarlo (los .zip quedan intactos)

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "subset"
(OUT / "ratings").mkdir(parents=True, exist_ok=True)

ini_ms = int(INICIO.timestamp() * 1000)
fin_ms = int(FIN.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def zips_de(tipo):
    """Zips cuyo archivo interno empieza con `tipo` (notes, ratings, ...)."""
    res = []
    for z in sorted(RAW.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            miembros = [m for m in zf.namelist() if m.endswith(".tsv")]
        if any(Path(m).name.startswith(tipo) for m in miembros):
            res.append(z)
    return res


def extraer(z):
    """Descomprime un zip en data/raw/ y devuelve las rutas de sus .tsv."""
    with zipfile.ZipFile(z) as zf:
        miembros = [m for m in zf.namelist() if m.endswith(".tsv")]
        for m in miembros:
            if not (RAW / m).exists():
                zf.extract(m, RAW)
    return [RAW / m for m in miembros]


def scan(path):
    # Todo como texto (infer_schema_length=0) para evitar errores de tipos.
    # quote_char=None porque el texto de las notas trae comillas sueltas.
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema_length=0)


def ms_a_int(lf):
    return lf.with_columns(pl.col("createdAtMillis").cast(pl.Int64, strict=False))


def limpiar(paths):
    if BORRAR_TSV:
        for p in paths:
            p.unlink(missing_ok=True)


def mostrar_columnas(nombre, path):
    print(f"  columnas de {nombre}: {scan(path).collect_schema().names()}")


tipos = {t: zips_de(t) for t in ["notes", "ratings", "noteStatusHistory", "userEnrollment"]}
print("Zips encontrados en data/raw/:")
for t, zs in tipos.items():
    print(f"  {t:18s} {len(zs)} archivo(s)")
if not tipos["notes"] or not tipos["ratings"]:
    raise SystemExit("Faltan zips de notas o ratings en data/raw/")

# ---------------------------------------------------------------------------
# 1. Notas creadas en la ventana
# ---------------------------------------------------------------------------
print("\n[1/4] Notas")
partes = []
for i, z in enumerate(tipos["notes"]):
    tsvs = extraer(z)
    if i == 0:
        mostrar_columnas("notes", tsvs[0])
    for p in tsvs:
        df = (ms_a_int(scan(p))
              .filter(pl.col("createdAtMillis").is_between(ini_ms, fin_ms, closed="left"))
              .collect())
        print(f"  {p.name}: {df.height:,} notas en la ventana")
        partes.append(df)
    limpiar(tsvs)

notes = pl.concat(partes, how="diagonal").unique(subset="noteId")
notes.write_parquet(OUT / "notes.parquet")
ids = notes["noteId"]
print(f"  TOTAL: {notes.height:,} notas")

# ---------------------------------------------------------------------------
# 2. Ratings de esas notas (un archivo a la vez, en streaming)
# ---------------------------------------------------------------------------
print("\n[2/4] Ratings (esto es lo lento: varios minutos por archivo)")
for i, z in enumerate(tipos["ratings"]):
    tsvs = extraer(z)
    if i == 0:
        mostrar_columnas("ratings", tsvs[0])
    for p in tsvs:
        destino = OUT / "ratings" / f"{p.stem}.parquet"
        (ms_a_int(scan(p))
         .filter(pl.col("noteId").is_in(ids))
         .sink_parquet(destino))
        n = pl.scan_parquet(destino).select(pl.len()).collect().item()
        print(f"  {p.name}: {n:,} ratings")
    limpiar(tsvs)

ratings = pl.scan_parquet(str(OUT / "ratings" / "*.parquet"))
raters = ratings.select("raterParticipantId").unique().collect()["raterParticipantId"]

# ---------------------------------------------------------------------------
# 3. Historial de estados y evaluadores
# ---------------------------------------------------------------------------
print("\n[3/4] Historial de estados y enrollment")
for tipo, col, valores in [("noteStatusHistory", "noteId", ids),
                           ("userEnrollment", "participantId", raters)]:
    partes = []
    for i, z in enumerate(tipos[tipo]):
        tsvs = extraer(z)
        if i == 0:
            mostrar_columnas(tipo, tsvs[0])
        for p in tsvs:
            partes.append(scan(p).filter(pl.col(col).is_in(valores)).collect())
        limpiar(tsvs)
    if partes:
        df = pl.concat(partes, how="diagonal")
        df.write_parquet(OUT / f"{tipo}.parquet")
        print(f"  {tipo}: {df.height:,} filas")
    else:
        print(f"  {tipo}: no se encontró zip (descárguenlo también)")

# ---------------------------------------------------------------------------
# 4. Resumen
# ---------------------------------------------------------------------------
print("\n[4/4] Resumen")
por_nota = ratings.group_by("noteId").len().collect()["len"]
por_rater = ratings.group_by("raterParticipantId").len().collect()["len"]
n_ratings = int(por_nota.sum())

print("=" * 50)
print(f"Ventana:               {INICIO.date()} a {FIN.date()} (excluyente)")
print(f"Notas creadas:         {notes.height:,}")
print(f"Notas con >=1 rating:  {len(por_nota):,}")
print(f"Evaluadores:           {len(por_rater):,}")
print(f"Ratings:               {n_ratings:,}")
print(f"Densidad:              {n_ratings / (len(por_rater) * len(por_nota)):.5%}")
print(f"Ratings por nota:      mediana {por_nota.median():.0f}, media {por_nota.mean():.1f}")
print(f"Ratings por evaluador: mediana {por_rater.median():.0f}, media {por_rater.mean():.1f}")

st_path = OUT / "noteStatusHistory.parquet"
if st_path.exists():
    st = pl.read_parquet(st_path)
    if "currentStatus" in st.columns:
        print("\nEstado actual de las notas de la ventana:")
        print(st["currentStatus"].value_counts(sort=True))
print("=" * 50)
print(f"Listo. Archivos en {OUT}")
