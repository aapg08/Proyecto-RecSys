"""
Rehace figuras/fig8_pareto_activo.* y figuras/fig9_paneles_activo.* a partir de tablas/midterm_resultados.csv
y tablas/midterm_protocolo.csv, SIN volver a correr midterm.py.   Uso (desde la raíz del repo):  python rehacer_figuras_activo.py

Necesario por un error en la implementación original (no requiere volver a ejecutarse después, los archivos afectados fueron actualizados)
"""
import csv
from pathlib import Path

import activo as act

BASE = Path(__file__).resolve().parent
TAB, FIG = BASE / "tablas", BASE / "figuras"
K = 10

filas = []
for r in csv.DictReader(open(TAB / "midterm_resultados.csv", encoding="utf-8")):
    f = {"modelo": r["modelo"], "K": int(r["K"])}
    for k, v in r.items():
        if k not in f:
            f[k] = float(v) if v not in ("", None) else float("nan")
    filas.append(f)
proto = {}
for r in csv.DictReader(open(TAB / "midterm_protocolo.csv", encoding="utf-8")):
    if r["bloque"] == "activo":
        proto[r["dato"]] = float(r["valor"])

betas = sorted(float(f["modelo"][4:]) for f in filas if f["modelo"].startswith("ev_b") and f["K"] == K)
recs = sorted(int(f["modelo"][9:]) for f in filas if f["modelo"].startswith("recorte_M") and f["K"] == K)
lams = sorted((float(f["modelo"][11:]) for f in filas if f["modelo"].startswith("propuesto_l") and f["K"] == K), reverse=True)
curvas = {
    "percentiles (original)": ["ltr"] + [f"propuesto_l{l:g}" for l in lams] + ["consenso"],
    "utilidad esperada (β)": ["ltr"] + [f"ev_b{b:g}" for b in betas] + ["valor"],
    "recorte por afinidad (M)": ["ltr"] + [f"recorte_M{m}" for m in recs] + ["valor"],
}
etq = {"activo_ev": f"β*={proto.get('beta_elegido', float('nan')):g}",
       "activo_recorte": f"M*={proto.get('recorte_elegido', float('nan')):g}",
       "activo_pct": f"λ*={proto.get('lambda_elegido', float('nan')):g}"}
FIG.mkdir(exist_ok=True)
act.figura_pareto(filas, curvas, K, FIG / "fig8_pareto_activo.png", FIG / "fig8_pareto_activo.pdf", elegidos=etq)
act.figura_paneles(filas, curvas, K, FIG / "fig9_paneles_activo.png", FIG / "fig9_paneles_activo.pdf", elegidos=etq)
print("listo: figuras/fig8_pareto_activo.* y figuras/fig9_paneles_activo.*")