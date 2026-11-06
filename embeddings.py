"""
Embeddings multilingües del texto de las notas (para el baseline entrenado de midterm.py).

Modelo por defecto: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensiones, ~50 idiomas,
incluye alfabetos no latinos). Así se corrige la limitación de TF-IDF que reportamos en H1.

Uso (después de split_data.py):
  pip install sentence-transformers
  python embeddings.py                 # todas las notas de train + test
  python embeddings.py --limite 2000   # prueba rápida

En CPU puede tardar del orden de una hora. Se guarda por partes en data/midterm/emb_partes/:
si se interrumpe, al volver a correrlo continúa donde quedó.

"""

import argparse
import html
import re
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SPLIT = BASE / "data" / "split"
MID = BASE / "data" / "midterm"
PARTES = MID / "emb_partes"
URL = re.compile(r"https?://\S+")


def limpia(s):
    return URL.sub(" ", html.unescape(s or "")).strip()


def main():
    import polars as pl
    from sentence_transformers import SentenceTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--parte", type=int, default=20_000, help="notas por archivo parcial")
    ap.add_argument("--limite", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    notes = pl.concat([pl.read_parquet(SPLIT / lado / "notes.parquet", columns=["noteId", "summary"])
                       for lado in ("train", "test")]).unique("noteId").sort("noteId")
    if args.limite:
        notes = notes.head(args.limite)
    textos = [limpia(s) for s in notes["summary"].to_list()]
    n = len(textos)
    print(f"{n:,} notas | modelo {args.modelo}")

    modelo = SentenceTransformer(args.modelo, device="cpu")
    PARTES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hechas = 0
    for p, ini in enumerate(range(0, n, args.parte)):
        destino = PARTES / f"parte_{p:04d}_{args.parte}_{n}.npy"
        fin = min(n, ini + args.parte)
        if destino.exists():
            hechas += fin - ini
            continue
        v = modelo.encode(textos[ini:fin], batch_size=args.batch, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
        np.save(destino, v.astype(np.float16))
        hechas += fin - ini
        dt = time.time() - t0
        print(f"  {hechas:,}/{n:,} notas  ({dt / 60:.1f} min)", flush=True)

    partes = [np.load(PARTES / f"parte_{p:04d}_{args.parte}_{n}.npy")
              for p in range((n + args.parte - 1) // args.parte)]
    E = np.vstack(partes)
    assert E.shape[0] == n
    np.save(MID / "embeddings.npy", E)
    notes.select("noteId").write_parquet(MID / "embeddings_ids.parquet")
    print(f"Listo: data/midterm/embeddings.npy  {E.shape}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
