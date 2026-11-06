"""
Une las partes de embeddings (parte_XXXX_<tam>_<n>.npy) en data/midterm/embeddings.npy + embeddings_ids.parquet,
sin recalcular nada y sin sentence-transformers.  Replica el final de embeddings.py.

1) Dejar las partes en  data/midterm/emb_partes/   (con sus nombres originales)
2) Desde la raíz del repo:  python unir_embeddings.py
"""
import re
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent
SPLIT, MID = BASE / "data" / "split", BASE / "data" / "midterm"
PARTES = MID / "emb_partes"

notes = (pl.concat([pl.read_parquet(SPLIT / lado / "notes.parquet", columns=["noteId"]) for lado in ("train", "test")])
         .unique("noteId").sort("noteId"))
n = notes.height
archivos = sorted(PARTES.glob("parte_*.npy"))
if not archivos:
    raise SystemExit(f"No hay archivos parte_*.npy en {PARTES}")
pat = re.compile(r"parte_(\d+)_(\d+)_(\d+)\.npy$")
info = []
for a in archivos:
    m = pat.match(a.name)
    if not m:
        raise SystemExit(f"Nombre inesperado: {a.name}")
    info.append((int(m[1]), int(m[2]), int(m[3])))
tams, ns = {t for _, t, _ in info}, {k for _, _, k in info}
if len(ns) != 1 or len(tams) != 1:
    raise SystemExit(f"Las partes son de corridas distintas (tamaños {tams}, n {ns})")
tam, n_partes = tams.pop(), ns.pop()
if n_partes != n:
    raise SystemExit(f"Las partes son de {n_partes:,} notas pero tus datos tienen {n:,}: no son del mismo split; no se pueden usar.")
esperadas = (n + tam - 1) // tam
if [p for p, _, _ in info] != list(range(esperadas)):
    raise SystemExit(f"Faltan partes: hay {len(info)} y se esperaban {esperadas} (índices 0..{esperadas - 1})")
E = np.vstack([np.load(a) for a in archivos])
assert E.shape[0] == n, (E.shape, n)
nrm = np.linalg.norm(E[:1000].astype(np.float32), axis=1)
print(f"{E.shape} dtype={E.dtype} | norma media (debe ser ~1): {nrm.mean():.3f}")
MID.mkdir(parents=True, exist_ok=True)
np.save(MID / "embeddings.npy", E)
notes.select("noteId").write_parquet(MID / "embeddings_ids.parquet")
print("Listo: data/midterm/embeddings.npy y embeddings_ids.parquet")