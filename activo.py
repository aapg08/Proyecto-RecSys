"""
Diseño del recomendador activo multiobjetivo (H2). Solo numpy/scipy (+ matplotlib para la figura).

Estado de la nota en t  : posterior ridge (b_n, g_n) con los parámetros de los evaluadores congelados
                        (postura.py).  r_v = y_v - mu - b_v = b_n + g_n f_v + eps,  eps ~ N(0, sigma2).
Valor de pedirle el voto: V(u,n) = RIESGO(n) x G(u,n)
    G(u,n)  = reducción RELATIVA de Var(b_n) si vota u = [(M⁻¹x)_0² / (1 + xᵀM⁻¹x)] / (M⁻¹)_00,  x=(1,f_u)
                (es el ΔVar de midterm.py dividido por Var(b_n); depende de f_u vía la covarianza => bridging)
    RIESGO  = prob., bajo la posterior, de que el estado terminal verdadero (CRH/CRNH) difiera del que
                daría el punto estimado actual (reglas de X). Anula el valor de notas que no pueden decidirse.
Afinidad calibrada      : q(n|u,t) = softmax_n( T * logit p_LTR(u,n) )  sobre las candidatas del evento
                        (probabilidad de que u elija n entre las candidatas; T se ajusta en validación).
Tres reglas de combinación (todas con 1 parámetro que recorre de "solo afinidad" a "solo valor"):
    ev       score = log q + β·log(V + ε)        β=1 es la utilidad esperada  P(vota)·V ; β=0 es el LTR
    recorte  top-M por afinidad y dentro de ese corte se ordena por V   (M=K es el LTR; M=|cand| es V puro)
    pct      λ·pct(LTR) + (1-λ)·pct(ΔVar)        <- regla original, se mantiene como referencia
Elección del punto de operación SIN mirar test: sobre eventos de validación NO usados para entrenar el LTR,
se toma el parámetro que maximiza el valor medio del top-K sujeto a Recall@K >= tol · Recall@K(LTR).
"""

import numpy as np
from scipy.special import logsumexp, ndtr

CRH_THR = 0.40
CRNH_I = -0.05
CRNH_M = -0.8


class EstadoNotas:
    """Estadísticos suficientes de la ridge 2x2 de cada nota. A = M y b = Σ x r."""
 
    def __init__(self, n, lam_b, lam_g, sigma2, prior_scale=1.0):
        self.a00 = np.full(n, lam_b * prior_scale)
        self.a01 = np.zeros(n)
        self.a11 = np.full(n, lam_g * prior_scale)
        self.b0 = np.zeros(n)
        self.b1 = np.zeros(n)
        self.sigma2 = sigma2
 
    def add(self, g, f, r):
        """g: notas, f: postura del evaluador, r = y - mu - b_u. Solo evaluadores con postura conocida."""
        np.add.at(self.a00, g, 1.0)
        np.add.at(self.a01, g, f)
        np.add.at(self.a11, g, f * f)
        np.add.at(self.b0, g, r)
        np.add.at(self.b1, g, f * r)
 
    def posterior(self, idx):
        a, b, d = self.a00[idx], self.a01[idx], self.a11[idx]
        det = a * d - b * b
        i00, i01, i11 = d / det, -b / det, a / det
        m_b = i00 * self.b0[idx] + i01 * self.b1[idx]
        m_g = i01 * self.b0[idx] + i11 * self.b1[idx]
        s = self.sigma2
        return m_b, m_g, s * i00, s * i01, s * i11


def valor_decision(est, cand, f_u):
    """V = riesgo x ganancia relativa de información, para el evaluador con postura f_u sobre las notas `cand`."""
    m_b, m_g, c00, c01, c11 = est.posterior(cand)
    c00 = np.maximum(c00, 1e-18)
    s = np.sqrt(c00)
    thr_l = CRNH_I + CRNH_M * np.abs(m_g)
    p_hi = 1.0 - ndtr((CRH_THR - m_b) / s)
    p_lo = ndtr((thr_l - m_b) / s)
    riesgo = np.where(m_b >= CRH_THR, 1.0 - p_hi, np.where(m_b <= thr_l, 1.0 - p_lo, p_hi + p_lo))
    sx0 = c00 + c01 * f_u
    xsx = c00 + 2.0 * f_u * c01 + f_u * f_u * c11
    ganancia = (sx0 * sx0 / (est.sigma2 + xsx)) / c00
    return riesgo * ganancia, ganancia, riesgo

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def calibrar_T(eventos, grilla=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    """T que maximiza la log-verosimilitud de la nota efectivamente evaluada bajo softmax(T·logit)."""
    ll = []
    for T in grilla:
        ll.append(sum(T * e["logit"][e["pos"]] - logsumexp(T * e["logit"]) for e in eventos) / max(len(eventos), 1))
    k = int(np.argmax(ll))
    return float(grilla[k]), dict(zip(grilla, ll))


def pct(s):
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(len(s))
    return r / max(len(s) - 1, 1)


def puntaje(familia, param, lg, V, dvar, T=1.0, eps_rel=0.01, jit=None):
    """Puntaje de cada candidata. lg = logit del LTR; V = valor de decisión; dvar = ΔVar original."""
    jit = 0.0 if jit is None else jit
    if familia == "ltr":
        return lg + jit
    if familia == "ev":
        logq = T * lg - logsumexp(T * lg)
        eps = eps_rel * float(V.mean()) + 1e-30
        return logq + param * np.log(V + eps) + jit
    if familia == "recorte":
        M = int(min(param, len(lg)))
        corte = np.argpartition(-lg, M - 1)[:M] if M < len(lg) else np.arange(len(lg))
        s = pct(lg) * 1e-3 + jit
        vc = V[corte]
        s[corte] = 1.0 + vc / (vc.max() + 1e-300) + 1e-6 * pct(lg)[corte]
        return s
    if familia == "pct":
        return param * pct(lg) + (1 - param) * pct(dvar) + jit
    if familia == "valor":
        return V + jit
    raise ValueError(familia)


def _topk(s, K):
    k = min(K, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    return top[np.argsort(-s[top])]


def barrido(eventos, configs, K, T, eps_rel, rng):
    """Recall@K y valor medio del top-K de cada (familia, parámetro) sobre eventos de validación."""
    filas = []
    for fam, par in [("ltr", None)] + list(configs):
        hits, vals = [], []
        for e in eventos:
            s = puntaje(fam, par, e["logit"], e["V"], e["dvar"], T, eps_rel, rng.random(len(e["V"])) * 1e-9)
            rank = 1 + int((s > s[e["pos"]]).sum())
            hits.append(rank <= K)
            vals.append(e["V"][_topk(s, K)].mean())
        filas.append(dict(familia=fam, param=np.nan if par is None else float(par),
                          recall=float(np.mean(hits)), valor=float(np.mean(vals)), K=K))
    return filas


def elegir(filas, tol):
    """Por familia: máximo valor sujeto a recall >= tol * recall(LTR); o el de mayor recall."""
    r_ltr = [f for f in filas if f["familia"] == "ltr"][0]["recall"]
    elegido = {}
    for fam in sorted({f["familia"] for f in filas} - {"ltr"}):
        c = [f for f in filas if f["familia"] == fam]
        ok = [f for f in c if f["recall"] >= tol * r_ltr]
        best = max(ok, key=lambda f: f["valor"]) if ok else max(c, key=lambda f: f["recall"])
        elegido[fam] = best["param"]
        for f in c:
            f["elegido"] = int(f is best)
            f["cumple_tol"] = int(f["recall"] >= tol * r_ltr)
    return elegido, r_ltr


def seleccionar(eventos, betas, recortes, lambdas, K, tol, eps_rel, seed=0):
    """Calibra T, recorre las tres familias y elige el punto de operación de cada una."""
    rng = np.random.default_rng(seed)
    T, ll = calibrar_T(eventos)
    configs = ([("ev", b) for b in betas] + [("recorte", m) for m in recortes] + [("pct", l) for l in lambdas])
    filas = barrido(eventos, configs, K, T, eps_rel, rng)
    elegido, r_ltr = elegir(filas, tol)
    return dict(T=T, ll_T=ll, filas=filas, elegido=elegido, recall_ltr_val=r_ltr)


# Figuras
def figura_pareto(filas, curvas, K, ruta_png, ruta_pdf=None, elegidos=()):
    """filas: dicts de midterm_resultados (modelo, K, recall, valor_decision). curvas: {titulo: [modelos]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fk = {f["modelo"]: f for f in filas if f["K"] == K}
    col = ["#2a78d6", "#eb6834", "#4a9d5b", "#8a5fbf"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for (tit, ms), c in zip(curvas.items(), col):
        ms = [m for m in ms if m in fk and "recall" in fk[m]]
        if ms:
            ax.plot([fk[m]["recall"] for m in ms], [max(fk[m]["valor_decision"], 1e-8) for m in ms], "-o", ms=4.5, lw=1.6, color=c, label=tit)
    for m, lab in [("random", "Random"), ("most_popular", "Most Popular"), ("ltr", "LTR"), ("valor", "Valor puro")]:
        if m in fk and "recall" in fk[m]:
            ax.plot(fk[m]["recall"], max(fk[m]["valor_decision"], 1e-8), "s", color="#898781", ms=6)
            ax.annotate(lab, (fk[m]["recall"], max(fk[m]["valor_decision"], 1e-8)), xytext=(5, -10), textcoords="offset points", fontsize=8)
    for m in elegidos:
        if m in fk:
            ax.plot(fk[m]["recall"], max(fk[m]["valor_decision"], 1e-8), "*", color="black", ms=13, zorder=5)
    if "observado" in fk:
        ax.axhline(fk["observado"]["valor_decision"], color="#eb6834", ls="--", lw=1)
        ax.text(ax.get_xlim()[0], fk["observado"]["valor_decision"], " observado", ha="left", va="top", fontsize=8, color="#eb6834")
    ax.set_xlabel(f"Recall@{K} (afinidad)")
    ax.set_ylabel(f"Valor de decisión@{K}")
    ax.set_yscale("log")
    ax.set_title("Compromiso afinidad vs. valor para el consenso\n(★ = punto elegido en validación)", fontsize=10, loc="left")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=200)
    if ruta_pdf:
        fig.savefig(ruta_pdf)
    plt.close(fig)