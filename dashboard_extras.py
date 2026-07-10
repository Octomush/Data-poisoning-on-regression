"""Dashboard-only figures that are NOT in the notebook, built strictly on the
notebook's own quantities (webcore). Currently: feasible_surfaces, the
budget-ball + inlier-ellipsoid + MSE-quadric feasible-set view (matches the
three-column figure), which the notebook's feasible_all does not draw.

All geometry comes from the same objects the thesis uses:
  frame  u1 = influence e (lF/||lF||),  u2 = MSE-normal (orth lR),  u3 = inlier-normal (orth Sigma^{-1}(x-mu))
  inlier cap  : (x_i+delta - mu)^T Sigma^{-1} (x_i+delta - mu) <= tau^2      (Mahalanobis, chi2 tau)
  MSE cap     : lR^T delta + 1/2 delta^T M_R delta <= 0                       (M_R = B_R + H_map,R, full curvature)
  budget ball : ||delta|| <= R
for a single representative attacked point, so the two caps are exact quadrics
and can be drawn as surfaces (as in the photo).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.stats import chi2
import webcore as W

LAM, P, Q, N_SUB, SEED, LEVEL_BUMP = 1e-1, 0.99, 0.8, 350, 0, 6.0

def _orth(v, B):
    for b in B: v = v - (v @ b) * b
    nv = np.linalg.norm(v); return v/nv if nv > 1e-9 else v*0.0

def _sphere(n_u=40, n_v=20):
    u = np.linspace(0, 2*np.pi, n_u); v = np.linspace(0, np.pi, n_v)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

def _quadric_ellipsoid(ax, Q, l, c0, thr, color, lw=0.6, alpha=0.45):
    """Draw {a: aᵀQa + lᵀa + c0 = thr} as a wireframe ellipsoid if it is bounded (Q≻0, rhs>0)."""
    w, V = np.linalg.eigh(0.5*(Q+Q.T))
    if np.any(w <= 1e-9):
        return False
    vc = -0.5*np.linalg.solve(Q, l)
    rhs = thr - c0 - 0.5*float(l @ vc)                      # = thr - f(vc)
    if rhs <= 0:
        return False
    axes = np.sqrt(rhs / w)
    sx, sy, sz = _sphere()
    P0 = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], 1)  # unit sphere (N,3)
    world = (P0*axes) @ V.T + vc                            # scale in eigenbasis, rotate, shift
    X = world[:, 0].reshape(sx.shape); Y = world[:, 1].reshape(sx.shape); Z = world[:, 2].reshape(sx.shape)
    ax.plot_wireframe(X, Y, Z, color=color, lw=lw, alpha=alpha, rcount=18, ccount=18)
    return True


def feasible_surfaces(name="realestate", kind="linear_topq", M=8, lam=LAM, p=P, q=Q,
                      n=N_SUB, seed=SEED, Rfac=1.35, K=30000):
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X, y = ds.X_train, ds.y_train
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test
    N, d = X.shape
    fit = W.ridge_fit(X, y, lam); th0 = fit.theta
    if kind == "level":
        yh = Xte @ th0; sidx = np.where(yh >= np.quantile(yh, q))[0]
        c = float(Xte[sidx].mean(0) @ th0) + LEVEL_BUMP
        _, gF_fn, _ = W.make_target("level", Xte, idx=sidx, c=c)
    elif kind == "mse":
        _, gF_fn, _ = W.make_target("mse", Xte, Xt=Xte, yt=yte)
    else:
        _, gF_fn, _ = W.make_target("linear_topq", Xte, theta0=th0, q=q)
    gF = gF_fn(th0)
    i0 = W.select_optimal_points(X, y, fit, gF, M, p=p, mode="cap")["S"][0]     # representative attacked point
    A1 = W.stack_A(fit, X, y, [i0]); lF1 = A1.T @ gF; e = lF1/np.linalg.norm(lF1)
    gR = W.grad_mse_theta(th0, X, y); lR1 = A1.T @ gR
    _, _, MR1 = W.full_curvature(fit, X, y, lam, [i0], gR, (2.0/N)*(X.T @ X))   # M_R = B_R + H_map,R
    mu = X.mean(0); Sig = np.cov(X.T) + 1e-3*np.eye(d); Sinv = np.linalg.inv(Sig)
    tau2 = float(chi2.ppf(p, d)); v0 = X[i0] - mu; n0 = Sinv @ v0; n0 /= np.linalg.norm(n0)
    u1 = e; u2 = _orth(lR1.copy(), [u1]); u3 = _orth(n0.copy(), [u1, u2]); U = np.stack([u1, u2, u3], 1)
    dstar = W.tr_oneshot(fit, X, y, lam, [i0], 8.0, gF, None, 1, mse_cap=False, inlier_cap=True, cap_p=p)["delta"]
    R = float(Rfac*np.linalg.norm(dstar) + 1e-6)
    # reduced 3-D quadratics in normalized coords a=(a1,a2,a3), delta = R (a1 u1 + a2 u2 + a3 u3)
    Qi = (R**2)*(U.T @ Sinv @ U); li = (2*R)*(U.T @ (Sinv @ v0)); ci = float(v0 @ (Sinv @ v0))
    Qm = 0.5*(R**2)*(U.T @ MR1 @ U); lm = R*(U.T @ lR1)
    rng = np.random.default_rng(1)
    dirs = rng.standard_normal((K, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    abc = (rng.random(K)**(1/3))[:, None]*dirs
    fi = np.einsum('ki,ij,kj->k', abc, Qi, abc) + abc @ li + ci
    fm = np.einsum('ki,ij,kj->k', abc, Qm, abc) + abc @ lm
    m_inl, m_mse = fi <= tau2, fm <= 0.0; m_both = m_inl & m_mse
    sub = lambda mask, cap=1400: (lambda idx: idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))(np.where(mask)[0])
    sx, sy, sz = _sphere()

    fig = plt.figure(figsize=(15.5, 5.2))
    cols = [("inlier cap only", m_inl, "#2e6fb0", [("#d9a24a", Qi, li, ci, tau2)]),
            ("MSE cap only",    m_mse, "#2e8b57", [("#3f9c68", Qm, lm, 0.0, 0.0)]),
            ("inlier + MSE (feasible set)", m_both, "#8452a8",
             [("#d9a24a", Qi, li, ci, tau2), ("#3f9c68", Qm, lm, 0.0, 0.0)])]
    for j, (title, mask, cc, quads) in enumerate(cols):
        ax = fig.add_subplot(1, 3, j+1, projection="3d")
        ax.plot_wireframe(sx, sy, sz, color="#9fb6d6", lw=0.4, alpha=0.25, rcount=12, ccount=12)  # budget ball
        for (qc, Qq, lq, c0, thr) in quads:
            _quadric_ellipsoid(ax, Qq, lq, c0, thr, qc)
        idx = sub(mask)
        ax.scatter(abc[idx, 0], abc[idx, 1], abc[idx, 2], s=4, c=cc, alpha=0.35, linewidths=0)
        ax.set_title(f"{title}   [{100*mask.mean():.1f}% of ball]", fontsize=10)
        ax.set_xlabel("influence e", fontsize=8); ax.set_ylabel("MSE-normal", fontsize=8); ax.set_zlabel("inlier-normal", fontsize=8)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.view_init(20, -60); ax.set_box_aspect((1, 1, 1)); ax.tick_params(labelsize=6)
    fig.suptitle(f"{name} ({kind}, d={d}) — feasible set: budget ball, inlier ellipsoid (orange), MSE quadric (green)",
                 y=1.0, fontsize=12)
    plt.tight_layout()
    return fig


def feasible_data(name="realestate", kind="linear_topq", M=8, lam=LAM, p=P, q=Q, q_hi=1.0,
                  n=N_SUB, seed=SEED, Rfac=1.35, K=40000):
    """Reduced-quadric coefficients + feasible point clouds for the INTERACTIVE feasible-set view
    (rendered client-side in Plotly). Same geometry as feasible_surfaces, returned as JSON."""
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X, y = ds.X_train, ds.y_train
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; N, d = X.shape
    fit = W.ridge_fit(X, y, lam); th0 = fit.theta
    _, gF_fn, _ = _target_triple(kind, Xte, yte, th0, q, q_hi); gF = gF_fn(th0)
    i0 = W.select_optimal_points(X, y, fit, gF, M, p=p, mode="cap")["S"][0]
    A1 = W.stack_A(fit, X, y, [i0]); lF1 = A1.T @ gF; e = lF1/np.linalg.norm(lF1)
    gR = W.grad_mse_theta(th0, X, y); lR1 = A1.T @ gR
    _, _, MR1 = W.full_curvature(fit, X, y, lam, [i0], gR, (2.0/N)*(X.T @ X))
    mu = X.mean(0); Sig = np.cov(X.T) + 1e-3*np.eye(d); Sinv = np.linalg.inv(Sig)
    tau2 = float(chi2.ppf(p, d)); v0 = X[i0] - mu; n0 = Sinv @ v0; n0 /= np.linalg.norm(n0)
    u1 = e; u2 = _orth(lR1.copy(), [u1]); u3 = _orth(n0.copy(), [u1, u2]); U = np.stack([u1, u2, u3], 1)
    dstar = W.tr_oneshot(fit, X, y, lam, [i0], 8.0, gF, None, 1, mse_cap=False, inlier_cap=True, cap_p=p)["delta"]
    R = float(Rfac*np.linalg.norm(dstar) + 1e-6)
    Qi = (R**2)*(U.T @ Sinv @ U); li = (2*R)*(U.T @ (Sinv @ v0)); ci = float(v0 @ (Sinv @ v0))
    Qm = 0.5*(R**2)*(U.T @ MR1 @ U); lm = R*(U.T @ lR1)
    rng = np.random.default_rng(1)
    dirs = rng.standard_normal((K, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    abc = (rng.random(K)**(1/3))[:, None]*dirs
    fi = np.einsum('ki,ij,kj->k', abc, Qi, abc) + abc @ li + ci
    fm = np.einsum('ki,ij,kj->k', abc, Qm, abc) + abc @ lm
    m_inl, m_mse = fi <= tau2, fm <= 0.0; m_both = m_inl & m_mse
    rnd = lambda a: np.round(a, 3).tolist()
    def take(mask, cap):
        idx = np.where(mask)[0]
        if len(idx) > cap: idx = rng.choice(idx, cap, replace=False)
        return rnd(abc[idx])
    return dict(Qi=rnd(Qi), li=rnd(li), ci=round(ci, 6), tau2=round(tau2, 6), Qm=rnd(Qm), lm=rnd(lm),
                cloud_ball=take(np.ones(K, bool), 700), cloud_inlier=take(m_inl, 900),
                cloud_mse=take(m_mse, 900), cloud_both=take(m_both, 700),
                kept_inlier=round(float(m_inl.mean()), 4), kept_mse=round(float(m_mse.mean()), 4),
                kept_both=round(float(m_both.mean()), 4), d=int(d), name=name, kind=kind, m=int(M))


def _seg_mask(yh, q, q_hi):
    lo = np.quantile(yh, q); hi = np.quantile(yh, q_hi)
    m = (yh >= lo) & (yh <= hi) if q_hi < 0.999 else (yh >= lo)
    return m if m.any() else (yh >= lo)

def curvature_surface(name="realestate", kind="linear_topq", M=8, lam=LAM, p=P, q=Q, q_hi=1.0,
                      Rm=3.0, ngrid=19, n=N_SUB, seed=SEED):
    """Single clean panel of the notebook's curvature view (tr3_overlay_atlas math): the true ΔF
       surface with the TR3 quadratic model ℓ_F·δ + ½δᵀ(B_F+H_map)δ as a black grid, over the influence
       (ℓ_F) and top perpendicular-curvature directions. Raising λ flattens the surface."""
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X, y = ds.X_train, ds.y_train
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test
    fit = W.ridge_fit(X, y, lam); d = X.shape[1]; R = float(np.sqrt(M)*Rm); md = M*d
    inl = W.filter_outlier(X, p=p)
    F_fn, gF_fn, hF_fn = _target_triple(kind, Xte, yte, fit.theta, q, q_hi); gF = gF_fn(fit.theta)
    si = W.influence_scores(fit, X, y, gF)
    S = [int(inl[k]) for k in np.argsort(-si[inl])[:M]]
    lF = W.ell_F(fit, X, y, S, gF); nlF = float(np.linalg.norm(lF)); u1 = lF/nlF
    hvpH = lambda v: W.hmap_hvp(X, y, lam, S, gF, v)
    u2 = W.top_perp(hvpH, u1, md)
    if hF_fn is not None:
        A = W.stack_A(fit, X, y, S); HF = hF_fn(fit.theta)
        Bmv = lambda v: A.T @ (HF @ (A @ v))
    else:
        Bmv = lambda v: np.zeros(md)
    BH2 = W.restrict2(Bmv, u1, u2) + W.restrict2(hvpH, u1, u2)
    gv = np.linspace(-R, R, ngrid); Ag, Bg = np.meshgrid(gv, gv); mask = Ag**2 + Bg**2 <= R**2
    quad = np.where(mask, 0.5*(BH2[0, 0]*Ag**2 + 2*BH2[0, 1]*Ag*Bg + BH2[1, 1]*Bg**2), np.nan)
    ZTR3 = np.where(mask, nlF*Ag, np.nan) + quad
    F0 = float(F_fn(fit.theta)); Zt = np.full_like(Ag, np.nan)
    for a in range(ngrid):
        for b in range(ngrid):
            if mask[a, b]:
                Zt[a, b] = F_fn(W._refit_at_delta(X, y, lam, S, Ag[a, b]*u1 + Bg[a, b]*u2)[0].theta) - F0
    gap = float(np.nanmax(np.abs(Zt - ZTR3)))
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Ag, Bg, Zt, cmap="viridis", alpha=0.78, linewidth=0, antialiased=True)
    ax.plot_wireframe(Ag, Bg, ZTR3, color="k", linewidth=0.5, rstride=1, cstride=1)
    ax.set_xlabel("influence  ℓ_F", fontsize=9); ax.set_ylabel("perpendicular  ⊥", fontsize=9)
    ax.set_zlabel("ΔF", fontsize=9); ax.tick_params(labelsize=7); ax.view_init(22, -58)
    ax.set_title(f"{name} ({kind}) — curvature (λ={lam:g})\ntrue ΔF vs TR3 model · cubic gap = {gap:.2f}",
                 fontsize=9.5, pad=6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04)
    return fig


def attack_vectors_fig(name="realestate", kind="linear_topq", M=8, regime="both", eps=3.0,
                       lam=LAM, p=P, q=Q, q_hi=1.0, n=N_SUB, seed=SEED):
    """REAL attack vectors for the chosen dataset, projected into the (ℓ_F, ⊥) plane: TR1/TR2/TR3/
       TR3-relin under the current regime, the unconstrained TR3 (dashed) so you see the caps truncate &
       rotate, the budget circle, and the MSE-increasing direction ℓ_R. Tilt φ off ℓ_F is annotated."""
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X, y = ds.X_train, ds.y_train
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; d = X.shape[1]; md = M*d
    fit = W.ridge_fit(X, y, lam); th0 = fit.theta
    F_fn, gF_fn, hF_fn = _target_triple(kind, Xte, yte, th0, q, q_hi); NL = hF_fn is not None
    gF = gF_fn(th0); hessM = hF_fn(th0) if NL else None
    cp = dict(_cp_regime(regime, p))
    S = W.select_optimal_points(X, y, fit, gF, M, p=p, mode=("budget" if regime == "budget" else "cap"))["S"]
    R = float(np.sqrt(M)*eps)
    A_S = W.stack_A(fit, X, y, S); lF = A_S.T @ gF; u1 = lF/np.linalg.norm(lF)
    Mmv = lambda v: W.hmap_hvp(X, y, lam, S, gF, v) + (A_S.T @ (hessM @ (A_S @ v)) if hessM is not None else 0.0)
    u2 = W.top_perp(Mmv, u1, md)
    lR = A_S.T @ W.grad_mse_theta(th0, X, y)
    def atk(method, **cc):
        if method == "TR1": return W.tr_oneshot(fit, X, y, lam, S, R, gF, None,  1, **cc)["delta"]
        if method == "TR2": return W.tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 2, **cc)["delta"]
        if method == "TR3": return W.tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, **cc)["delta"]
        if NL: return W.tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, F_fn, hF_fn, **W._filter_kw(W.tr3_relin_nonlinear, max_outer=20, **cc))["delta"]
        return W.tr3_relin(fit, X, y, lam, S, R, gF, None, **W._filter_kw(W.tr3_relin, max_outer=20, **cc))["delta"]
    proj = lambda dl: (float(dl @ u1)/R, float(dl @ u2)/R)      # normalise so budget circle = unit
    vecs = {m: proj(atk(m, **cp)) for m in ["TR1", "TR2", "TR3", "TR3-relin"]}
    v_unc = proj(atk("TR3", mse_cap=False, inlier_cap=False))   # unconstrained reference
    lRp = np.array([lR @ u1, lR @ u2]); lRp = lRp/(np.linalg.norm(lRp)+1e-12)
    col = {"TR1": "#3f7fc0", "TR2": "#d99b2e", "TR3": "#3f9c68", "TR3-relin": "#c0392b"}
    tilt = lambda v: np.degrees(np.arctan2(abs(v[1]), v[0]))

    fig, ax = plt.subplots(figsize=(6.6, 6.2)); th = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), color="#c9c2b2", ls="--", lw=1)                        # budget circle
    ax.axhline(0, color="#eee6d8", lw=1); ax.axvline(0, color="#eee6d8", lw=1)
    ax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#8a8272", lw=2))  # ℓ_F
    ax.text(1.02, 0.02, "ℓ_F", color="#8a8272", fontsize=10)
    ax.annotate("", xy=(0.5*lRp[0], 0.5*lRp[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#c98a86", lw=1.6, ls="--"))
    ax.text(0.52*lRp[0], 0.52*lRp[1], "ℓ_R (MSE↑)", color="#c98a86", fontsize=9)
    if regime != "budget":
        ax.annotate("", xy=(v_unc[0], v_unc[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#3f9c68", lw=1.8, ls=(0, (4, 3)), alpha=.6))
        ax.text(v_unc[0], v_unc[1], "  TR3 (uncapped)", color="#3f9c68", fontsize=8, alpha=.7)
    for m, v in vecs.items():
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col[m], lw=3))
        ax.text(v[0]*1.04, v[1]*1.04, f"{m} ({tilt(v):.0f}°)", color=col[m], fontsize=9, fontweight="bold")
    lim = 1.25; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=.2)
    # angular alignment of the target functional with the MSE (risk) direction, in the FULL space
    cosFR = float(lF @ lR / (np.linalg.norm(lF)*np.linalg.norm(lR) + 1e-12))
    angFR = float(np.degrees(np.arccos(np.clip(cosFR, -1, 1))))
    verdict = "MSE cap ~FREE (anti-aligned)" if angFR > 95 else ("MSE cap COSTLY (aligned)" if angFR < 85 else "MSE cap ~neutral (orthogonal)")
    ax.text(-lim*0.96, lim*0.9, f"∠(ℓ_F, ℓ_R) = {angFR:.0f}°   cos = {cosFR:+.2f}\n{verdict}",
            fontsize=8.6, color="#8a5a56", va="top",
            bbox=dict(fc="white", ec="#e0cfcb", alpha=.92))
    ax.set_xlabel("along influence ℓ_F  (÷R)"); ax.set_ylabel("perpendicular ⊥  (÷R)")
    capn = {"budget": "budget-only", "inlier": "inlier cap", "both": "both caps", "mse": "MSE cap"}[regime]
    ax.set_title(f"{name} ({kind}) — real attack vectors, {capn}, ε={eps:.1f}, m={M}\n"
                 f"curvature rotates the ray off ℓ_F; caps truncate & re-rotate it", fontsize=10.5)
    plt.tight_layout()
    return fig


def _cp_regime(regime, p):
    base = {"budget": dict(mse_cap=False, inlier_cap=False),
            "inlier": dict(mse_cap=False, inlier_cap=True),
            "both":   dict(mse_cap=True,  inlier_cap=True),
            "mse":    dict(mse_cap=True,  inlier_cap=False)}[regime]
    return dict(base, cap_p=p)

def _whiten(X, g):
    """Blend the design toward whitened (decorrelated, variance-equalised) by strength g in [0,1].
       g=0 -> original X; g=1 -> features decorrelated (removes the ill-conditioning the attack rides on)."""
    if g <= 1e-6:
        return X
    mu = X.mean(0); Xc = X - mu; C = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
    w, V = np.linalg.eigh(C); wm = float(np.mean(w))
    scale = (1 - g) + g*np.sqrt(wm/np.maximum(w, 1e-8))     # equalise eigen-variances, keep overall scale
    return Xc @ (V * scale) @ V.T + mu


def defender_effect(name="realestate", kind="linear_topq", M=8, regime="inlier", knob="lam",
                    lam=LAM, p=P, q=Q, q_hi=1.0, n=N_SUB, seed=SEED, ebs=(2.0, 4.0, 6.0)):
    """How a DEFENDER knob shrinks what the attacker can reach. Sweeps one knob and plots the
       empirical max-reachable ΔF (bilevel over a few budgets, current regime), plus a second signal:
         knob='lam'  -> ridge strength λ (Γ* ~ 1/(nλ));           2nd axis: none
         knob='tau'  -> inlier quantile τ_x;                       2nd axis: feasible inlier fraction
         knob='cond' -> feature de-correlation (whitening) γ;      2nd axis: curvature ‖M_F‖ (the data curvature the attack rides on)
    """
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X0, y = ds.X_train, ds.y_train
    if len(X0) > n:
        i = np.random.default_rng(seed).choice(len(X0), n, replace=False); X0, y = X0[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; d = X0.shape[1]

    def reach_and_extra(Xd, lam_, p_):
        fit = W.ridge_fit(Xd, y, lam_); th0 = fit.theta
        F_fn, gF_fn, hF_fn = _target_triple(kind, Xte, yte, th0, q, q_hi); NL = hF_fn is not None
        gF = gF_fn(th0); hessM = hF_fn(th0) if NL else None; cp = _cp_regime(regime, p_)
        S = W.select_optimal_points(Xd, y, fit, gF, M, p=p_, mode=("budget" if regime == "budget" else "cap"))["S"]
        F0 = F_fn(th0); best = 0.0
        for e in ebs:
            R = float(np.sqrt(M)*e)
            dl = W.bilevel_at_S(fit, Xd, y, lam_, S, (gF_fn if NL else gF), R,
                                **W._filter_kw(W.bilevel_at_S, n_iter=60, n_restarts=2,
                                               F_fn=(F_fn if NL else None), **cp))["delta"]
            best = max(best, F_fn(W._refit_at_delta(Xd, y, lam_, S, dl)[0].theta) - F0)
        # curvature proxy ‖M_F‖ at the top selected point
        _, _, MF1 = W.full_curvature(fit, Xd, y, lam_, [S[0]], gF, hessM)
        return best, float(np.linalg.norm(MF1, 2))

    if knob == "lam":
        vals = [0.01, 0.03, 0.1, 0.3, 1.0]; xs = vals; reach = []; extra = []
        for v in vals:
            r_, mfn = reach_and_extra(X0, v, p); reach.append(r_); extra.append(mfn)
        xlab, x2 = "defender ridge λ  (log)", None; logx = True; title = "raising λ shrinks the reachable attack (Γ* ∼ 1/(nλ))"
    elif knob == "tau":
        vals = [0.90, 0.95, 0.99, 0.995, 0.999]; xs = vals; reach = []; extra = []
        from scipy.stats import chi2
        mu = X0.mean(0); Sig = np.cov(X0.T) + 1e-3*np.eye(d); Sinv = np.linalg.inv(Sig)
        Xc = X0 - mu; dM = np.sqrt(np.einsum('ij,jk,ik->i', Xc, Sinv, Xc))
        for v in vals:
            r_, _ = reach_and_extra(X0, lam, v); reach.append(r_)
            tau = np.sqrt(chi2.ppf(v, d)); extra.append(float((dM <= tau).mean()))
        xlab, x2 = "inlier quantile τ_x", "inlier fraction kept"; logx = False; title = "tightening τ_x shrinks the ellipsoid → less reachable, more clean points rejected"
    else:  # cond
        vals = [0.0, 0.25, 0.5, 0.75, 1.0]; xs = vals; reach = []; extra = []
        for v in vals:
            r_, mfn = reach_and_extra(_whiten(X0, v), lam, p); reach.append(r_); extra.append(mfn)
        xlab, x2 = "feature de-correlation γ (whitening)", "data curvature ‖M_F‖"; logx = False
        title = "de-correlating the design removes the ill-conditioning → curvature and reach both fall"

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(xs, reach, "o-", color="#c0392b", lw=2.4, ms=6, label="max reachable ΔF")
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlab); ax.set_ylabel("max reachable ΔF", color="#c0392b"); ax.tick_params(axis='y', labelcolor="#c0392b")
    ax.grid(alpha=.25)
    if x2 is not None:
        a2 = ax.twinx(); col = "#2e6fb0"
        a2.plot(xs, extra, "s--", color=col, lw=1.8, ms=5, label=x2)
        a2.set_ylabel(x2, color=col); a2.tick_params(axis='y', labelcolor=col)
    ax.set_title(f"{name} ({kind}) — defender knob: {knob}\n{title}", fontsize=11)
    plt.tight_layout()
    return fig


def _seg_feature(Xte, th0):
    """Feature that defines a feature-segment target: the most influential input (largest |theta|)."""
    return int(np.argmax(np.abs(th0)))

def _target_triple(kind, Xte, yte, th0, q=Q, q_hi=1.0):
    if kind == "level":
        sidx = np.where(_seg_mask(Xte @ th0, q, q_hi))[0]
        c = float(Xte[sidx].mean(0) @ th0) + LEVEL_BUMP
        return W.make_target("level", Xte, idx=sidx, c=c)
    if kind == "feature_seg":
        return W.make_target("feature_seg", Xte, fidx=_seg_feature(Xte, th0), q=q, q_hi=q_hi)
    if kind == "feature_level":
        fj = _seg_feature(Xte, th0); vals = Xte[:, fj]
        lo = np.quantile(vals, q); hi = np.quantile(vals, q_hi)
        mask = (vals >= lo) & (vals <= hi) if q_hi < 0.999 else (vals >= lo)
        if not mask.any(): mask = vals >= lo
        c = float(Xte[mask].mean(0) @ th0) + LEVEL_BUMP
        return W.make_target("feature_level", Xte, fidx=fj, q=q, q_hi=q_hi, c=c)
    if kind == "mse":
        return W.make_target("mse", Xte, Xt=Xte, yt=yte)
    return W.make_target("linear_topq", Xte, theta0=th0, q=q, q_hi=q_hi)


def regression_overlay(name="realestate", kind="linear_topq", M=8, regime="both", eps=3.0,
                       method="bilevel", feat="move", lam=LAM, p=P, q=Q, q_hi=1.0, n=N_SUB, seed=SEED):
    """THE key diagram: the clean regression vs the poisoned (changed) regression, overlaid.
       Left: partial-dependence line along the most influential feature (other features held at their
       mean) for the clean fit and the poisoned refit. Right: predicted-vs-actual, clean vs poisoned,
       with the targeted segment highlighted and its mean shift reported. Uses the notebook solvers as-is."""
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X, y = ds.X_train, ds.y_train
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; ym = float(ds.y_mean); d = X.shape[1]
    fit = W.ridge_fit(X, y, lam); th0 = fit.theta
    F_fn, gF_fn, hF_fn = _target_triple(kind, Xte, yte, th0, q, q_hi); NL = hF_fn is not None
    gF = gF_fn(th0); hessM = hF_fn(th0) if NL else None
    cp = {"budget": dict(mse_cap=False, inlier_cap=False),
          "inlier": dict(mse_cap=False, inlier_cap=True),
          "both":   dict(mse_cap=True,  inlier_cap=True),
          "mse":    dict(mse_cap=True,  inlier_cap=False)}[regime]
    cp = dict(cp, cap_p=p)                                    # defender inlier quantile tau_x
    S = W.select_optimal_points(X, y, fit, gF, M, p=p, mode=("budget" if regime == "budget" else "cap"))["S"]
    R = float(np.sqrt(M)*eps)
    if method == "TR1":   dl = W.tr_oneshot(fit, X, y, lam, S, R, gF, None,  1, **cp)["delta"]
    elif method == "TR2": dl = W.tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 2, **cp)["delta"]
    elif method == "TR3": dl = W.tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, **cp)["delta"]
    elif method == "TR3-relin":
        dl = (W.tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, F_fn, hF_fn, **W._filter_kw(W.tr3_relin_nonlinear, max_outer=18, **cp))["delta"]
              if NL else W.tr3_relin(fit, X, y, lam, S, R, gF, None, **W._filter_kw(W.tr3_relin, max_outer=18, **cp))["delta"])
    else:
        dl = W.bilevel_at_S(fit, X, y, lam, S, (gF_fn if NL else gF), R,
                            **W._filter_kw(W.bilevel_at_S, n_iter=60, n_restarts=2, F_fn=(F_fn if NL else None), **cp))["delta"]
    th1 = W._refit_at_delta(X, y, lam, S, dl)[0].theta

    feat_target = kind in ("feature_seg", "feature_level")   # segment defined on a FEATURE, not prediction
    yhc = Xte @ th0; yhp = Xte @ th1
    if feat_target:
        fseg = _seg_feature(Xte, th0); vals = Xte[:, fseg]
        flo = float(np.quantile(vals, q)); fhi = float(np.quantile(vals, q_hi)) if q_hi < 0.999 else float(vals.max())
        segmask = (vals >= flo) & (vals <= fhi) if q_hi < 0.999 else (vals >= flo)
        if not segmask.any(): segmask = vals >= flo
        seg = np.where(segmask)[0]
    else:
        seg = np.where(_seg_mask(yhc, q, q_hi))[0]
    mc, mp = float(yhc[seg].mean()), float(yhp[seg].mean())
    pct = (mp-mc)/(abs(mc+ym)+1e-12)*100.0

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    # --- left: partial-dependence regression line along a chosen feature ---
    if feat_target:
        fj = fseg                                              # x-axis = the feature that defines the segment
    elif feat == "influence":
        fj = int(np.argmax(np.abs(th0)))                       # most influential (largest |theta|)
    elif str(feat) in ("move", "auto", ""):
        fj = int(np.argmax(np.abs(th1 - th0)))                 # most prominent MOVEMENT (largest |dtheta|)
    else:
        fj = int(feat) % len(th0)                              # explicit feature index
    feat = fj
    fname = ds.feature_names[feat] if feat < len(ds.feature_names) else f"x[{feat}]"
    # targeted segment = predicted-value band [quantile(q), quantile(q_hi)]
    blo = float(np.quantile(yhc, q)) + ym; bhi = (float(np.quantile(yhc, q_hi)) + ym) if q_hi < 0.999 else (float((yhc+ym).max()))
    xf = np.linspace(Xte[:, feat].min(), Xte[:, feat].max(), 80)
    base = Xte.mean(0); rows = np.tile(base, (80, 1)); rows[:, feat] = xf
    if feat_target:
        a0.axvspan(flo, fhi, color="#f2c14e", alpha=.18, label=f"targeted FEATURE band [{q:.2f},{q_hi:.2f}]")
    else:
        a0.axhspan(blo, bhi, color="#f2c14e", alpha=.16, label=f"targeted segment [{q:.2f},{q_hi:.2f}]")
    a0.scatter(Xte[:, feat], yte+ym, s=10, c="#cbc4b4", alpha=.7, label="test data")
    xseg = Xte[seg, feat]
    a0.scatter(xseg, (yte+ym)[seg], s=16, c="#c77", alpha=.8, label="segment points")
    a0.plot(xf, rows @ th0 + ym, "-", color="#6b7186", lw=2.4, label="clean regression")
    a0.plot(xf, rows @ th1 + ym, "-", color="#e23b2e", lw=2.6, label="poisoned regression")
    a0.set_xlabel(f"feature: {fname}"); a0.set_ylabel("prediction ŷ")
    a0.set_title("regression line: clean vs poisoned"); a0.legend(fontsize=7.5); a0.grid(alpha=.25)
    # --- right: predicted vs actual overlay ---
    lo = float(min((yte+ym).min(), (yhc+ym).min(), (yhp+ym).min())); hi = float(max((yte+ym).max(), (yhc+ym).max(), (yhp+ym).max()))
    ns = np.setdiff1d(np.arange(len(yte)), seg)
    if not feat_target:
        a1.axhspan(blo, bhi, color="#f2c14e", alpha=.16, label=f"targeted band [{q:.2f},{q_hi:.2f}]")
    a1.plot([lo, hi], [lo, hi], "--", color="#d8d0c1", lw=1.2)
    a1.scatter((yte+ym)[ns], (yhc+ym)[ns], s=9, c="#cbc4b4", alpha=.6, label="clean")
    a1.scatter((yte+ym)[seg], (yhc+ym)[seg], s=22, c="#8a8272", edgecolor="w", lw=.4, label="segment · clean")
    a1.scatter((yte+ym)[seg], (yhp+ym)[seg], s=26, c="#e23b2e", edgecolor="w", lw=.5, label="segment · POISONED")
    for k in seg:
        a1.plot([(yte+ym)[k], (yte+ym)[k]], [(yhc+ym)[k], (yhp+ym)[k]], "-", color="#e6a23c", lw=.6, alpha=.5)
    a1.axhline(mc+ym, color="#8a8272", ls=":", lw=1.2); a1.axhline(mp+ym, color="#e23b2e", ls="-", lw=1.4)
    a1.set_xlabel("actual y"); a1.set_ylabel("predicted ŷ")
    a1.set_title(f"segment mean {mc+ym:.2f} → {mp+ym:.2f}  ({pct:+.1f}%)"); a1.legend(fontsize=8); a1.grid(alpha=.25)
    fig.suptitle(f"{name} ({kind}) — {method}, {regime} caps, ε={eps:.1f}, m={M}: how the regression changes", y=1.0, fontsize=12.5)
    plt.tight_layout()
    return fig


def force_balance_fig(name="casp", kind="linear_topq", M=8, regime="both", eps=3.0,
                      lam=LAM, p=P, q=Q, q_hi=1.0, n=N_SUB, seed=SEED):
    """Converged 'mechanism' view in the (ℓ_F, ⊥) plane: roll the attack out under the drive
       ℓ_F+Mδ until the caps' reactions balance it, then draw the END STATE — feasible region,
       the black net-δ (the real attack), the drive (blue) and the wall reactions (green inlier /
       grey budget / rose MSE) that cancel it, plus the uncapped optimum (green dashed) for contrast."""
    from matplotlib.patches import Circle, FancyArrowPatch
    from numpy.linalg import norm, eigh
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (800, 300)))
    X, y = ds.X_train, ds.y_train
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; d = X.shape[1]; md = M*d; nn = len(X)
    fit = W.ridge_fit(X, y, lam); th0 = fit.theta
    F, gF_fn, hF_fn = _target_triple(kind, Xte, yte, th0, q, q_hi); gF = gF_fn(th0); hM = hF_fn(th0) if hF_fn is not None else None
    S = W.select_optimal_points(X, y, fit, gF, M, p=p, mode=("budget" if regime == "budget" else "cap"))["S"]
    A = W.stack_A(fit, X, y, S); lFv = A.T @ gF; lFm = float(norm(lFv))+1e-12; u1 = lFv/lFm
    Hmv = lambda v: W.hmap_hvp(X, y, lam, S, gF, v)
    Bmv = (lambda v: A.T @ (hM @ (A @ v))) if hM is not None else (lambda v: np.zeros(md))
    Mmv = lambda v: Hmv(v)+Bmv(v); u2 = W.top_perp(Mmv, u1, md); M2 = W.restrict2(Mmv, u1, u2)
    lR = A.T @ W.grad_mse_theta(th0, X, y); lR2 = np.array([lR @ u1, lR @ u2])
    Rmv = lambda v: A.T @ ((2.0/nn)*(X.T @ (X @ (A @ v)))); MR2 = W.restrict2(Rmv, u1, u2)
    R = float(np.sqrt(M)*eps); tau = 0.72*R; lF = np.array([lFm, 0.0])
    show_inl = regime in ("inlier", "both"); use_mse = regime in ("both", "mse")
    Reff = tau if show_inl else R
    h = lambda u: lR2 @ u + 0.5*u @ MR2 @ u
    dd = np.zeros(2); AL = 0.05; drive = lF.copy(); rad = np.zeros(2); mse_r = np.zeros(2)
    for t in range(3000):
        g = lF + M2 @ dd; u = dd + AL*g; rc = np.zeros(2); mc = np.zeros(2)
        if norm(u) > Reff: nu = u*Reff/norm(u); rc = nu-u; u = nu
        if use_mse:
            for _ in range(5):
                hv = h(u)
                if hv <= 1e-9: break
                gh = lR2 + MR2 @ u; c = -(hv/(gh @ gh))*gh; u = u+c; mc = mc+c
        dd = u; drive = g; rad = rc/AL; mse_r = mc/AL
    def tr(l, Mx, Rb):
        w, V = eigh(Mx); b = V.T @ l; f = lambda gm: np.hypot(b[0]/(gm-w[0]), b[1]/(gm-w[1])); base = max(0, w.max())
        lo = base+1e-4
        for _ in range(200):
            if f(lo) >= Rb or lo-base < 1e-12: break
            lo = base+(lo-base)*0.5+1e-9
        hi = lo+1
        for _ in range(200):
            if f(hi) <= Rb: break
            hi *= 2
        for _ in range(100):
            m = (lo+hi)/2; lo, hi = (m, hi) if f(m) > Rb else (lo, m)
        gm = (lo+hi)/2; dU = V @ (b/(gm-w)); nd = norm(dU)
        return dU*Rb/nd if nd > Rb else dU
    dU = tr(lF, M2, R)
    # ---- scale forces modestly (animation-style) & auto-fit the view to everything drawn ----
    SC = (0.34*R)/max(norm(drive), lFm, 1e-9)
    lFref = np.array([0.92*R, 0.0])                     # ℓ_F drawn as a fixed-length axis reference
    tips = [dd, dd+drive*SC, lFref]
    if norm(rad) > 1e-6: tips.append(dd+rad*SC)
    if use_mse and norm(mse_r) > 1e-6: tips.append(dd+mse_r*SC)
    if regime != "budget": tips.append(dU)
    AXL = 1.13*max([norm(t) for t in tips] + [R])
    fig, ax = plt.subplots(figsize=(6.8, 6.9))
    ax.axhline(0, color="#eee6d8", lw=1, zorder=-1); ax.axvline(0, color="#eee6d8", lw=1, zorder=-1)
    ax.add_patch(Circle((0, 0), R, fill=False, ec="#c9c2b2", ls=(0, (4, 4)), lw=1.4))
    gx, gy = np.meshgrid(np.linspace(-AXL, AXL, 240), np.linspace(-AXL, AXL, 240))
    feas = gx**2+gy**2 <= Reff**2
    if use_mse: feas = feas & (lR2[0]*gx+lR2[1]*gy+0.5*(gx*(MR2[0, 0]*gx+MR2[0, 1]*gy)+gy*(MR2[1, 0]*gx+MR2[1, 1]*gy)) <= 1e-9)
    ax.contourf(gx, gy, feas.astype(float), levels=[.5, 1.5], colors=["#e7f0e9"], zorder=0)
    if show_inl: ax.add_patch(Circle((0, 0), tau, fill=False, ec="#7fa591", lw=2))
    if use_mse:
        tt = np.linspace(-AXL, AXL, 200); nlR = lR2/norm(lR2); pp = np.array([-nlR[1], nlR[0]]); kc = 0.5*(pp @ MR2 @ pp)
        wall = np.array([(-(kc*t*t)/norm(lR2))*nlR+t*pp for t in tt]); ax.plot(wall[:, 0], wall[:, 1], color="#c98a86", lw=2)
    def arr(p0, v, c, w=2.6, ls='-', ms=14, z=5):
        if norm(v) < 1e-9: return
        ax.add_patch(FancyArrowPatch(p0, (p0[0]+v[0], p0[1]+v[1]), arrowstyle='-|>', mutation_scale=ms, lw=w, color=c, linestyle=ls, zorder=z))
    def lbl(p, s, c, fs=10, fw='normal', dx=0.0, dy=0.0):
        ax.text(p[0]+dx, p[1]+dy, s, color=c, fontsize=fs, fontweight=fw, ha='center', va='center')
    up = np.array([-dd[1], dd[0]]); up = up/(norm(up)+1e-9)      # unit perp to delta, for offset labels
    arr([0, 0], lFref, "#c9c2b2", 1.6, ls=(0, (4, 3))); lbl(lFref, "ℓ_F", "#a99f8c", 10, dx=0.05*AXL)
    if regime != "budget":
        arr([0, 0], dU, "#7fb08f", 1.8, ls=(0, (5, 4))); lbl(dU, "uncapped", "#3f9c68", 8.5, dy=0.05*AXL)
    arr([0, 0], dd, "#111111", 3.2, z=4); lbl(dd*0.55 + up*0.07*AXL, "δ", "#111", 13, 'bold')
    arr(dd, drive*SC, "#2e6fb0", 2.6); lbl(dd + drive*SC + up*0.05*AXL, "drive", "#2e6fb0", 8.5)
    if norm(rad) > 1e-6: arr(dd, rad*SC, "#3f9c68" if show_inl else "#9a8f7e", 2.4, ls=(0, (2, 2)))
    if use_mse and norm(mse_r) > 1e-6: arr(dd, mse_r*SC, "#c98a86", 2.4, ls=(0, (2, 2)))
    ax.plot(dd[0], dd[1], 'o', color="#c0392b", ms=10, zorder=6, mec='w', mew=1)
    ang = np.degrees(np.arctan2(dd[1], dd[0])); cost = max(0, dU[0]-dd[0])
    ax.set_xlim(-AXL, AXL); ax.set_ylim(-AXL, AXL); ax.set_aspect('equal'); ax.axis('off')
    capn = {"budget": "budget only", "inlier": "inlier cap", "both": "both caps", "mse": "MSE cap"}[regime]
    ax.set_title(f"{name} ({kind}) — force balance @ optimum · {capn} · ε={eps:.1f} · m={M}\n"
                 f"black δ = final attack:  |δ|={norm(dd):.2f}, ∠={ang:.0f}°   ·   caps cost {min(100,100*cost/(abs(dU[0])+1e-9)):.0f}% of ℓ_F-reach", fontsize=9.5)
    ax.text(0, -AXL*1.03, "blue = drive ℓ_F+Mδ · green/grey = inlier/budget push · rose = MSE push · black = net δ · green-dashed = uncapped",
            fontsize=7.3, color="#8a8272", ha='center')
    plt.tight_layout(); return fig

