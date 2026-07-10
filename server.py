"""Tiny local server for the poisoning dashboard.

It calls the ULTIMATE-notebook functions EXACTLY AS-IS (extracted verbatim into
webcore.py) and returns the figures they already produce. The dropdowns/knobs are
just their parameters:

  functional_budget_sweep(name, kind, regime, M, ...)   -> sweep panel
  feasible_all(names=[name], kind, M, ...)               -> feasible-set panel
  ceiling_approach_3d(name, kind, M, ...)                -> ceiling panel
  boundary_walk(name, kind, caps, M, ...)                -> redistribution panel
  residual_inward(name, kind, M, ...)                    -> residual panel
  m_eps_sweep(name, kind, ...)                           -> m-eps panel

No new datasets, no new behaviour: `name` goes straight to webcore.DATASETS[name]
(the notebook's own loaders) and `M` straight to the solvers.

Run:  python3 server.py     then open  http://localhost:8000
"""
from __future__ import annotations
import io, json, os, sys, threading, hashlib, traceback
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import webcore as W
import dashboard_extras as DX

SWEEP_OUT = {}                      # cache of functional_budget_sweep `out` dicts for the summary
FEAS_CACHE = {}                     # cache of feasible_data dicts
REGIME_CAPS = {"budget": (False, False), "inlier": (False, True), "both": (True, True), "mse": (True, False)}

PORT = int(os.environ.get("PORT", "8000"))
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "fig_cache")
os.makedirs(CACHE, exist_ok=True)
_LOCK = threading.Lock()            # matplotlib / refit are not thread-safe -> serialise

DATASETS = (["casp"] + [d for d in W.DATASETS if d != "casp"]) if "casp" in W.DATASETS else list(W.DATASETS)
KINDS = ["linear_topq", "feature_seg", "feature_level", "level", "mse"]
PANELS = ["regression", "attackvec", "sweep", "feasible", "ceiling", "boundary", "residual", "meps", "defender", "heatmap"]

def _epsgrid(eps_max, lo=0.0, n=12):
    import numpy as _np
    return list(_np.round(_np.linspace(lo, float(eps_max), n), 3))


DEFAULT_METHODS = {
    "boundary": ("TR1", "TR3-relin", "bilevel@S*"),
    "ceiling":  ("TR1", "TR3", "bilevel"),
    "residual": ("TR1", "TR3-relin", "bilevel"),
}
# full option lists (exactly the strings each notebook function accepts) + the defaults, for the UI
DEFAULT_METHODS_ALL = {
    "boundary": {"all": ["TR1", "TR2", "TR3", "TR3-relin", "bilevel@S*", "bilevel@Rand"],
                 "default": list(DEFAULT_METHODS["boundary"])},
    "ceiling":  {"all": ["TR1", "TR2", "TR3", "TR3-relin", "bilevel"],
                 "default": list(DEFAULT_METHODS["ceiling"])},
    "residual": {"all": ["TR1", "TR2", "TR3", "TR3-relin", "bilevel"],
                 "default": list(DEFAULT_METHODS["residual"])},
}

# use the notebook's OWN defaults (relin_outer=60, bil_iter=150, bil_restarts=3) so the sweep reproduces
# the thesis figures exactly (e.g. TR3-relin climbing past bilevel under the inlier cap on casp/loan).
SWEEP_SOLVER = dict(n=400)
def skey(name, kind, m, regime, eps_max, lam, p, q, q_hi):
    return f"{name}|{kind}|{m}|{regime}|{eps_max}|{lam}|{p}|{q}|{q_hi}"

def build_fig(panel, name, kind, m, regime, methods, eps, eps_max, feat="move", lam=0.1, p=0.99, q=0.8, q_hi=1.0, knob="lam"):
    """Call the notebook function AS-IS. Dropdowns/sliders are its arguments:
       M=m, lam (defender ridge), sel_p/cap_p=p (defender inlier quantile tau_x), q/q_hi (attacked segment band),
       eps (single budget Rm), eps_max (sweep range), methods (which methods to draw)."""
    tgt = dict(q=q, q_hi=q_hi)                                       # -> make_target (via **tk / explicit)
    if panel == "regression":
        rmethod = (methods[0] if methods else "bilevel")
        return DX.regression_overlay(name=name, kind=kind, M=m, regime=regime, eps=eps, method=rmethod,
                                     feat=feat, lam=lam, p=p, q=q, q_hi=q_hi)
    if panel == "attackvec":
        return DX.force_balance_fig(name=name, kind=kind, M=m, regime=regime, eps=eps, lam=lam, p=p, q=q, q_hi=q_hi)
    if panel == "heatmap":
        return W.eta_influence_heatmap(kind=kind, names=[name], lam=lam, ncols=1, q=q, q_hi=q_hi)
    if panel == "defender":
        # single clean curvature surface (true ΔF vs TR3 quadratic model) — raise λ to flatten it
        return DX.curvature_surface(name=name, kind=kind, M=m, lam=lam, p=p, q=q, q_hi=q_hi)
    if panel == "sweep":
        fig, out = W.functional_budget_sweep(name=name, kind=kind, regime=regime, M=m, lam=lam, sel_p=p,
                                             eps_min=0.0, eps_max=float(eps_max), show_weak=True, q=q, q_hi=q_hi,
                                             **SWEEP_SOLVER)
        SWEEP_OUT[skey(name, kind, m, regime, eps_max, lam, p, q, q_hi)] = out
        return fig
    if panel == "ceiling":
        return W.ceiling_approach_3d(name=name, kind=kind, M=m, Rm=float(eps), lam=lam, sel_p=p,
                                     methods=methods or DEFAULT_METHODS["ceiling"], **tgt)
    if panel == "boundary":
        caps = {"budget": "budget", "inlier": "inlier", "both": "both", "mse": "budget"}[regime]
        return W.boundary_walk(name=name, kind=kind, caps=caps, M=m, lam=lam, sel_p=p, eps_grid=_epsgrid(eps_max, 0.3),
                               methods=methods or DEFAULT_METHODS["boundary"], **tgt)
    if panel == "residual":
        return W.residual_inward(name=name, kind=kind, M=m, Rm=float(eps), lam=lam, sel_p=p,
                                 methods=methods or DEFAULT_METHODS["residual"], **tgt)
    if panel == "meps":
        mm = methods[0] if methods else "TR1"; mc, ic = REGIME_CAPS.get(regime, (False, False))
        return W.m_eps_sweep(name=name, kind=kind, eps_grid=_epsgrid(eps_max, 0.0), method=mm,
                             mse_cap=mc, inlier_cap=ic, lam=lam, sel_p=p, **tgt)
    raise ValueError(f"unknown panel {panel!r}")


def feasible_json(name, kind, m, lam=0.1, p=0.99, q=0.8, q_hi=1.0):
    key = f"{name}|{kind}|{m}|{lam}|{p}|{q}|{q_hi}"
    if key not in FEAS_CACHE:
        FEAS_CACHE[key] = DX.feasible_data(name=name, kind=kind, M=int(m), lam=lam, p=p, q=q, q_hi=q_hi)
    return FEAS_CACHE[key]


def sweep_summary(name, kind, m, regime, eps_max, lam=0.1, p=0.99, q=0.8, q_hi=1.0):
    """Empirical max-reachable / saturation values from functional_budget_sweep `out`."""
    import numpy as _np
    key = skey(name, kind, m, regime, eps_max, lam, p, q, q_hi)
    out = SWEEP_OUT.get(key)
    if out is None:
        fig, out = W.functional_budget_sweep(name=name, kind=kind, regime=regime, M=m, lam=lam, sel_p=p,
                                             eps_min=0.0, eps_max=float(eps_max), show_weak=True, q=q, q_hi=q_hi,
                                             **SWEEP_SOLVER)
        plt.close(fig); SWEEP_OUT[key] = out
    epsg = _np.round(_np.arange(0.0, float(eps_max), 0.5), 3)   # functional_budget_sweep grid
    rows = {}
    for mth, vals in out.items():
        v = _np.asarray(vals, float)
        mx = float(_np.nanmax(v)); imx = int(_np.nanargmax(v))
        # empirical saturation: first eps within 1% of the max
        thr = mx - 0.01*abs(mx)
        isat = int(_np.argmax(v >= thr)) if _np.any(v >= thr) else imx
        rows[mth] = {"max": round(mx, 4), "eps_at_max": float(epsg[min(imx, len(epsg)-1)]),
                     "eps_sat": float(epsg[min(isat, len(epsg)-1)]), "end": round(float(v[-1]), 4)}
    return {"name": name, "kind": kind, "m": int(m), "regime": regime, "methods": rows}


def render_png(panel, name, kind, m, regime, methods=(), eps=3.0, eps_max=7.0, feat="move",
               lam=0.1, p=0.99, q=0.8, q_hi=1.0, knob="lam"):
    mkey = ",".join(methods)
    key = f"v4|{panel}|{name}|{kind}|{m}|{regime}|{mkey}|{eps}|{eps_max}|{feat}|{lam}|{p}|{q}|{q_hi}|{knob}"  # bump salt to bust stale cache
    path = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ".png")
    if os.path.exists(path):
        return open(path, "rb").read()
    with _LOCK:
        if os.path.exists(path):
            return open(path, "rb").read()
        fig = build_fig(panel, name, kind, int(m), regime, tuple(methods), float(eps), float(eps_max), feat,
                        float(lam), float(p), float(q), float(q_hi), knob)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        data = buf.getvalue()
        open(path, "wb").write(data)
        return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/datasets":
            body = json.dumps({"datasets": DATASETS, "kinds": KINDS, "panels": PANELS,
                               "methods": DEFAULT_METHODS_ALL}).encode()
            return self._send(200, body, "application/json")
        if u.path == "/api/summary":
            q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
            name, kind, m, regime = g("name", DATASETS[0]), g("kind", "linear_topq"), g("m", "8"), g("regime", "budget")
            eps_max = g("eps_max", "7.0"); lam, p, ql, qh = g("lam", "0.1"), g("p", "0.99"), g("q", "0.8"), g("q_hi", "1.0")
            if name not in DATASETS:
                return self._send(400, b"bad request", "text/plain")
            try:
                with _LOCK:
                    body = json.dumps(sweep_summary(name, kind, int(m), regime, float(eps_max),
                                                    float(lam), float(p), float(ql), float(qh))).encode()
                return self._send(200, body, "application/json")
            except Exception as ex:
                traceback.print_exc(); return self._send(500, f"{type(ex).__name__}: {ex}".encode(), "text/plain")
        if u.path == "/api/feasible":
            q = parse_qs(u.query); g = lambda k, dd: q.get(k, [dd])[0]
            name, kind, m = g("name", DATASETS[0]), g("kind", "linear_topq"), g("m", "8")
            lam, p, ql, qh = g("lam", "0.1"), g("p", "0.99"), g("q", "0.8"), g("q_hi", "1.0")
            if name not in DATASETS:
                return self._send(400, b"bad request", "text/plain")
            try:
                with _LOCK:
                    body = json.dumps(feasible_json(name, kind, int(m), float(lam), float(p), float(ql), float(qh))).encode()
                return self._send(200, body, "application/json")
            except Exception as ex:
                traceback.print_exc(); return self._send(500, f"{type(ex).__name__}: {ex}".encode(), "text/plain")
        if u.path == "/api/fig":
            q = parse_qs(u.query)
            g = lambda k, d: q.get(k, [d])[0]
            panel, name = g("panel", "sweep"), g("name", DATASETS[0])
            kind, m, regime = g("kind", "linear_topq"), g("m", "8"), g("regime", "budget")
            eps, eps_max, feat = g("eps", "3.0"), g("eps_max", "7.0"), g("feat", "move")
            lam, p, ql, qh = g("lam", "0.1"), g("p", "0.99"), g("q", "0.8"), g("q_hi", "1.0")
            knob = g("knob", "lam")
            methods = [x for x in g("methods", "").split(",") if x]
            if panel not in PANELS or name not in DATASETS:
                return self._send(400, b"bad request", "text/plain")
            try:
                png = render_png(panel, name, kind, m, regime, methods, eps, eps_max, feat,
                                 float(lam), float(p), float(ql), float(qh), knob)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")  # identical params -> browser cache, no round-trip
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            except Exception as ex:
                traceback.print_exc()
                msg = f"{type(ex).__name__}: {ex}".encode()
                self._send(500, msg, "text/plain")
            return
        self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    print(f"\n  Poisoning & Defence Explorer")
    print(f"  datasets: {', '.join(DATASETS)}")
    print(f"  open  ->  http://localhost:{PORT}\n  (first render of each panel runs the real solver; it is cached afterwards)\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
