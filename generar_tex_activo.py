"""
Genera, desde las tablas de midterm.py, los .tex de la parte "recomendador activo" del informe H2:
    numeros_activo.tex        macros con las cifras del texto (\\recLTR, \\valEV, ...)
    tabla_activo_principal.tex  tabla compacta (K=10) para la sección de resultados
    tabla_activo_anexo.tex      tabla completa (K=10) + tabla de selección en validación (Anexo A)

Uso (desde la raíz del repo de código):
    python generar_tex_activo.py --tablas tablas --salida <carpeta_del_proyecto_latex>
Si se repite midterm.py (p. ej. con embeddings) basta volver a correr esto y recompilar.
"""
import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tablas", default="tablas")
ap.add_argument("--salida", default=".")
ap.add_argument("--K", type=int, default=10)
args = ap.parse_args()
TAB, OUT, K = Path(args.tablas), Path(args.salida), args.K


def leer(nombre):
    with open(TAB / nombre, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in ("", None) else float("nan")


R = {r["modelo"]: r for r in leer("midterm_resultados.csv") if int(r["K"]) == K}
SEL = [r for r in leer("activo_seleccion.csv")]
PROTO = {r["dato"]: num(r["valor"]) for r in leer("midterm_protocolo.csv") if r["bloque"] == "activo"}
TEST = {r["dato"]: num(r["valor"]) for r in leer("midterm_protocolo.csv") if r["bloque"] in ("test", "test_resumen")}
VAL = {r["dato"]: num(r["valor"]) for r in leer("midterm_protocolo.csv") if r["bloque"].startswith("val")}
g = lambda m, c: num(R[m][c])


def d(x, n):                       # decimal con coma (formato del informe)
    return f"{x:.{n}f}".replace(".", "{,}")


def pc(x, n=0):                    # porcentaje
    return d(100 * x, n) + r"\,\%"


def miles(x):
    return f"{int(round(x)):,}".replace(",", r"\,")


def gp(x):                         # etiqueta de parámetro: 0.25 -> 0{,}25 ; 20.0 -> 20
    return d(x, 2).rstrip("0").rstrip("{,}") if x != int(x) else str(int(x))


bS, MS, lS = PROTO["beta_elegido"], int(PROTO["recorte_elegido"]), PROTO["lambda_elegido"]
mev, mrec, mpct = "activo_ev", "activo_recorte", "activo_pct"
ev4 = "ev_b0.4" if "ev_b0.4" in R else None

# ---------------------------------------------------------------- macros
M = {
    "nEventos": miles(TEST["eventos"]), "nEvaluadores": miles(TEST["evaluadores"]),
    "nCandMed": miles(TEST["candidatas_mediana"]), "nSel": miles(PROTO["eventos_seleccion"]),
    "TCal": gp(PROTO["T_calibracion"]), "bStar": gp(bS), "MStar": str(MS), "lStar": gp(lS),
    "tolAf": gp(PROTO["tol_afinidad"]), "recLTRval": d(PROTO["recall_ltr_validacion"], 3),
    "pctPerdTest": pc(TEST["pct_perdidos_por_nota_decidida"], 1), "pctPerdVal": pc(VAL["pct_perdidos_por_nota_decidida"], 1),
    "recMP": d(g("most_popular", "recall"), 3), "recCB": d(g("content_based", "recall"), 3),
    "recRandom": d(g("random", "recall"), 3), "recLTR": d(g("ltr", "recall"), 3), "recCons": d(g("consenso", "recall"), 4),
    "recPctNueve": d(g("propuesto_l0.9", "recall"), 3), "recEV": d(g(mev, "recall"), 3), "recPct": d(g(mpct, "recall"), 3),
    "recRec": d(g(mrec, "recall"), 3),
    "caidaPctNueve": pc(1 - g("propuesto_l0.9", "recall") / g("ltr", "recall")),
    "valRandom": d(g("random", "valor_decision") * 1e3, 2), "valLTR": d(g("ltr", "valor_decision") * 1e3, 2),
    "valEV": d(g(mev, "valor_decision") * 1e3, 2), "valObs": d(g("observado", "valor_decision") * 1e3, 2),
    "aporteRandom": d(g("random", "aporte_minoria"), 3), "aporteObs": d(g("observado", "aporte_minoria"), 3),
    "aporteEV": d(g(mev, "aporte_minoria"), 3), "aportePct": d(g(mpct, "aporte_minoria"), 3),
    "covLTR": d(g("ltr", "cobertura"), 2), "covEV": d(g(mev, "cobertura"), 2),
    "evRecRel": pc(g(mev, "recall") / g("ltr", "recall")),
    "evValX": d(g(mev, "valor_decision") / g("ltr", "valor_decision"), 1),
    "evVarX": d(g(mev, "reduccion_var") / g("ltr", "reduccion_var"), 1),
    "evVsPctValX": d(g(mev, "valor_decision") / g(mpct, "valor_decision"), 1),
    "evVsPctVarX": d(g(mev, "reduccion_var") / g(mpct, "reduccion_var"), 1),
    "evVsPctRec": pc(1 - g(mev, "recall") / g(mpct, "recall"), 1),
    "evVsRecValX": d(g(mev, "valor_decision") / g(mrec, "valor_decision"), 1),
    "evVsRecRec": pc(g(mev, "recall") / g(mrec, "recall") - 1),
}
if ev4:
    M.update({"valEVcuatro": d(g(ev4, "valor_decision") * 1e3, 2), "evCuatroRecRel": pc(g(ev4, "recall") / g("ltr", "recall"))})

# ---- macros adicionales y frases que dependen de los resultados (se recalculan en cada corrida) ----
M["umbralCRH"] = d(PROTO["umbral_crh"], 3) if "umbral_crh" in PROTO else "0{,}40"
M["aportePctNueve"] = d(g("propuesto_l0.9", "aporte_minoria"), 3)
betas_ord = sorted(float(m[4:]) for m in R if m.startswith("ev_b"))


def primer_beta(cond):
    for b in betas_ord:
        if cond(R[f"ev_b{b:g}"]):
            return b
    return None



def dominada(prefijos):
    """cuántos puntos de otras familias tienen algún punto EV con más Recall Y más V."""
    ev = [(g(m, "recall"), g(m, "valor_decision")) for m in R if m.startswith("ev_b")]
    ot = [(g(m, "recall"), g(m, "valor_decision")) for m in R if m.startswith(prefijos)]
    return sum(any(r >= r0 and v >= v0 for r, v in ev) for r0, v0 in ot), len(ot)


_dom = dominada(("propuesto_l", "recorte_M"))
M["nDomA"], M["nDomB"] = str(_dom[0]), str(_dom[1])
M["pctTopDiez"] = d(100 * 10 / TEST["candidatas_mediana"], 1) + r"\,\%"
M["aporteCons"] = d(g("consenso", "aporte_minoria"), 3)

vR, aR, aO = g("random", "valor_decision"), g("random", "aporte_minoria"), g("observado", "aporte_minoria")
bV = primer_beta(lambda r: num(r["valor_decision"]) > vR)
bA = primer_beta(lambda r: num(r["aporte_minoria"]) > aR)
if g(mev, "valor_decision") > vR:
    M["fraseVRandom"] = r"El punto elegido supera incluso a Random en $V$ (\valEV{} frente a \valRandom{}, en unidades de $10^{-3}$)."
else:
    t = r"El punto elegido (tolerancia \tolAf{}) aún queda bajo Random en $V$ (\valEV{} frente a \valRandom{}, en unidades de $10^{-3}$)"
    if bV is not None:
        rr = R[f"ev_b{bV:g}"]
        t += (rf"; superarlo exige $\beta\geq{gp(bV)}$ ($V={d(num(rr['valor_decision']) * 1e3, 2)}\times10^{{-3}}$), "
              rf"con {pc(num(rr['recall']) / g('ltr', 'recall'))} del Recall del $LTR$")
    M["fraseVRandom"] = t + "."
if g(mev, "aporte_minoria") > aR:
    t = r"El punto elegido de EV supera a la asignación aleatoria en AporteMinoría (\aporteEV{} frente a \aporteRandom{})"
else:
    t = r"El punto elegido de EV queda bajo la asignación aleatoria en AporteMinoría (\aporteEV{} frente a \aporteRandom{})"
    if bA is not None:
        rr = R[f"ev_b{bA:g}"]
        t += (rf"; la supera desde $\beta={gp(bA)}$ ({d(num(rr['aporte_minoria']), 3)}), con {pc(num(rr['recall']) / g('ltr', 'recall'))} del Recall del $LTR$")
M["fraseAporteRandom"] = t + r", y " + ("supera" if g(mev, "aporte_minoria") > aO else "no supera") + r" al flujo observado (\aporteObs{})."
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "numeros_activo.tex").write_text(
    "% GENERADO por generar_tex_activo.py -- no editar a mano\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n",
    encoding="utf-8")

# ---------------------------------------------------------------- tabla principal
COLS = [("recall", 4, 1), ("ndcg", 4, 1), ("cobertura", 4, 1), ("aporte_minoria", 4, 1), ("reduccion_var", 3, 1e3), ("valor_decision", 3, 1e3)]


def tabla(filas, caption, label, size="small", observado=True):
    """filas: lista de (etiqueta_latex, modelo) o '---' para separador."""
    mods = [m for _, m in filas if m != "---" and m != "observado"]
    mejor = {c: max(g(m, c) for m in mods) for c, _, _ in COLS}
    L = [r"\begin{table}[ht]", r"\centering", f"\\{size}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l cccccc@{}}", r"\toprule",
         r"\textbf{Modelo} & \textbf{Recall@%d} & \textbf{NDCG@%d} & \textbf{Cobertura} & \textbf{Aporte} & "
         r"\shortstack{\textbf{$\Delta$Var}\\($\times10^{-3}$)} & \shortstack{\textbf{$V$}\\($\times10^{-3}$)} \\ \midrule" % (K, K)]
    for et, m in filas:
        if m == "---":
            L.append(r"\midrule")
            continue
        celdas = []
        for c, nd, esc in COLS:
            x = g(m, c)
            if x != x:
                celdas.append("---")
                continue
            s = d(x * esc, nd)
            celdas.append(rf"\textbf{{{s}}}" if (m != "observado" and abs(x - mejor[c]) < 1e-12) else s)
        L.append(f"{et} & " + " & ".join(celdas) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


filas_p = [("Random", "random"), ("LTR", "ltr"), ("---", "---"),
           (rf"Percentiles ($\lambda=0{{,}}9$, orig.)", "propuesto_l0.9"),
           (rf"Percentiles ($\lambda^*={gp(lS)}$)", mpct), (rf"Recorte ($M^*={MS}$)", mrec),
           (rf"\textbf{{EV}} ($\beta^*={gp(bS)}$)", mev)]
if ev4:
    filas_p.append((r"EV ($\beta=0{,}4$; no elegido)", ev4))
filas_p += [(r"Valor $V$ puro", "valor"), ("---", "---"), ("Flujo observado (X)", "observado")]
(OUT / "tabla_activo_principal.tex").write_text(tabla(
    filas_p, rf"Recomendador multiobjetivo en el conjunto de prueba ($K={K}$). $^*$: punto elegido en validación. "
             r"Aporte = AporteMinoría. En negrita el mejor valor por columna (sin contar el flujo observado).",
    "tab:activo_principal", "footnotesize"), encoding="utf-8")

# ---------------------------------------------------------------- tabla anexo completa + selección
def orden(prefijo, clave):
    return sorted((m for m in R if m.startswith(prefijo)), key=lambda m: clave(m))


filas_a = [("Random", "random"), ("Most Popular", "most_popular"), ("Content-Based", "content_based"), ("LTR", "ltr"), ("---", "---"),
           (r"Consenso ($\lambda=0$)", "consenso")]
filas_a += [(rf"Percentiles ($\lambda={gp(float(m[11:]))}$)", m) for m in orden("propuesto_l", lambda m: -float(m[11:]))]
filas_a += [("---", "---")] + [(rf"EV ($\beta={gp(float(m[4:]))}$)", m) for m in orden("ev_b", lambda m: float(m[4:]))]
filas_a += [("---", "---")] + [(rf"Recorte ($M={m[9:]}$)", m) for m in orden("recorte_M", lambda m: int(m[9:]))]
filas_a += [("---", "---"), (r"Valor $V$ puro", "valor"), ("---", "---"), ("Flujo observado (X)", "observado")]
txt = tabla(filas_a, rf"Todas las reglas de combinación evaluadas en el conjunto de prueba ($K={K}$).", "tab:anexo_activo_completa", "footnotesize")

S = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
     rf"\caption{{Barrido en validación ({M['nSel']} eventos no usados para entrenar el LTR). Se elige, por familia, el parámetro de mayor $V$ "
     rf"con Recall@{K} $\geq {M['tolAf']}\times$ el del LTR ({M['recLTRval']}).}}", r"\label{tab:anexo_activo_seleccion}",
     r"\begin{tabular}{@{}l r cc cc@{}}", r"\toprule",
     rf"\textbf{{Familia}} & \textbf{{Parámetro}} & \textbf{{Recall@{K}}} & \textbf{{$V$ ($\times10^{{-3}}$)}} & \textbf{{Cumple tol.}} & \textbf{{Elegido}} \\ \midrule"]
NOM = {"ltr": "LTR", "ev": "EV ($\\beta$)", "recorte": "Recorte ($M$)", "pct": "Percentiles ($\\lambda$)"}
prev = None
for r in SEL:
    if prev is not None and r["familia"] != prev:
        S.append(r"\midrule")
    prev = r["familia"]
    p = "---" if r["param"] == "" else gp(float(r["param"]))
    cumple = {"1": r"$\checkmark$", "0": "", "": "---"}[r["cumple_tol"]]
    eleg = r"$\star$" if r["elegido"] == "1" else ""
    S.append(rf"{NOM[r['familia']]} & {p} & {d(num(r['recall']), 4)} & {d(num(r['valor']) * 1e3, 3)} & {cumple} & {eleg} \\")
S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "tabla_activo_anexo.tex").write_text(txt + "\n" + "\n".join(S) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- verificación de afirmaciones del texto
tol = PROTO["tol_afinidad"]
dom = dominada(("propuesto_l", "recorte_M"))
CHK = [
    ("EV* conserva >= tol x Recall del LTR en test", g(mev, "recall") >= tol * g("ltr", "recall")),
    ("EV* tiene mas V que el LTR", g(mev, "valor_decision") > g("ltr", "valor_decision")),
    ("EV* tiene mas DeltaVar que el LTR", g(mev, "reduccion_var") > g("ltr", "reduccion_var")),
    ("EV* sube la cobertura respecto del LTR", g(mev, "cobertura") > g("ltr", "cobertura")),
    ("EV* da mas V que percentiles* y que recorte*", g(mev, "valor_decision") > max(g(mpct, "valor_decision"), g(mrec, "valor_decision"))),
    ("EV* da mas DeltaVar que percentiles*", g(mev, "reduccion_var") > g(mpct, "reduccion_var")),
    ("percentiles* queda sobre EV* en AporteMinoria (se afirma en 'Matices')", g(mpct, "aporte_minoria") > g(mev, "aporte_minoria")),
    (f"'Toda la curva EV queda sobre las otras en V' (puntos pct/recorte dominados: {dom[0]}/{dom[1]})", dom[0] == dom[1]),
    ("EV* supera a Random en AporteMinoria (criterio de H1)", g(mev, "aporte_minoria") > aR),
    ("EV* supera al flujo observado en AporteMinoria", g(mev, "aporte_minoria") > aO),
    ("EV* supera a Random en V", g(mev, "valor_decision") > vR),
    ("el LTR tiene menos V que el flujo observado (se afirma en 'Matices')", g("ltr", "valor_decision") < g("observado", "valor_decision")),
    ("el flujo observado tiene menos AporteMinoria que Random", aO < aR),
    ("consenso puro (DeltaVar) tiene Recall < Random", g("consenso", "recall") < g("random", "recall")),
]
print("\n=== VERIFICAR afirmaciones del texto (REVISAR = reescribir la frase correspondiente) ===")
for txt, ok in CHK:
    print(f"  [{'OK     ' if ok else 'REVISAR'}] {txt}")
print()
print("generados: numeros_activo.tex, tabla_activo_principal.tex, tabla_activo_anexo.tex  en", OUT)