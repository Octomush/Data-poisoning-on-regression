"""Precompute REAL 2-D detectability geometry for casp and bake it into a standalone, adjustable
explorer (budget ball + inlier circle + MSE quadric wall + KKT force arrows), seeded per
objective x regime x epsilon.  All the *curvature/alignment* is real; tau stays an adjustable slider.

  python3 build_casp_det.py            # compute casp presets + write attack_vector_casp.html
"""
from __future__ import annotations
import json, numpy as np
import webcore as W
from dashboard_extras import _target_triple, _cp_regime

NAME = "casp"
OBJS = ["linear_topq", "feature_seg", "feature_level", "mse"]
REGS = ["budget", "inlier", "both", "mse"]
EPS  = [1.5, 3.0, 5.0]
LAM, P, Q, Q_HI, M, N_SUB, SEED = 0.1, 0.99, 0.8, 1.0, 8, 220, 0
LABELS = {"linear_topq": "top-segment shift (linear)", "feature_seg": "feature-segment shift (linear)",
          "feature_level": "feature-segment → level (curved)", "mse": "MSE availability"}

def combo(kind, reg, eps, X, y, Xte, yte, fit):
    th0 = fit.theta; d = X.shape[1]; md = M*d; n = len(X)
    F_fn, gF_fn, hF_fn = _target_triple(kind, Xte, yte, th0, Q, Q_HI); NL = hF_fn is not None
    gF = gF_fn(th0); hessM = hF_fn(th0) if NL else None
    S = W.select_optimal_points(X, y, fit, gF, M, p=P, mode=("budget" if reg == "budget" else "cap"))["S"]
    A_S = W.stack_A(fit, X, y, S); lF = A_S.T @ gF; lFmag = float(np.linalg.norm(lF)) + 1e-12; u1 = lF/lFmag
    Hmv = lambda v: W.hmap_hvp(X, y, LAM, S, gF, v)
    Bmv = (lambda v: A_S.T @ (hessM @ (A_S @ v))) if hessM is not None else (lambda v: np.zeros(md))
    Mmv = lambda v: Hmv(v) + Bmv(v); u2 = W.top_perp(Mmv, u1, md)
    HM2 = W.restrict2(Hmv, u1, u2).tolist(); BF2 = W.restrict2(Bmv, u1, u2).tolist()
    lR = A_S.T @ W.grad_mse_theta(th0, X, y); lR2 = [float(lR @ u1), float(lR @ u2)]
    Rmv = lambda v: A_S.T @ ((2.0/n) * (X.T @ (X @ (A_S @ v))))
    MR2 = W.restrict2(Rmv, u1, u2).tolist()
    R = float(np.sqrt(M)*eps)
    cosFR = float(lF @ lR/(lFmag*(np.linalg.norm(lR)+1e-12)))
    angFR = float(np.degrees(np.arccos(np.clip(cosFR, -1, 1))))
    inl = reg in ("inlier", "both")
    return dict(lFmag=round(lFmag, 4), BF2=[[round(x, 5) for x in r] for r in BF2],
                HM2=[[round(x, 5) for x in r] for r in HM2], lR2=[round(x, 4) for x in lR2],
                MR2=[[round(x, 5) for x in r] for r in MR2], R=round(R, 3),
                ang=round(angFR, 1), cos=round(cosFR, 3), tau=round((0.72 if inl else 1.25)*R, 3),
                inl=inl, mse=reg in ("both", "mse"))

def build():
    ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (400, 150)))
    X, y = ds.X_train, ds.y_train
    if len(X) > N_SUB:
        i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; fit = W.ridge_fit(X, y, LAM)
    data = {}
    for kind in OBJS:
        data[kind] = {}
        for reg in REGS:
            data[kind][reg] = {f"{e:.1f}": combo(kind, reg, e, X, y, Xte, yte, fit) for e in EPS}
        print(f"  {kind}: done", flush=True)
    meta = dict(kinds=OBJS, regimes=REGS, eps=[f"{e:.1f}" for e in EPS], labels=LABELS)
    tmpl = open("_casp_template.html").read()
    html = tmpl.replace("__DATA__", json.dumps({"_meta": meta, "casp": data}))
    open("attack_vector_casp.html", "w").write(html)
    print("WROTE attack_vector_casp.html")

if __name__ == "__main__":
    build()
