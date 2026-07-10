"""Standalone animation: how the attack curvature flattens as the defender raises ridge λ (0.01 -> 1).

Uses the notebook curvature construction (tr3_overlay_atlas math via webcore helpers): the true ΔF
surface and the TR3 quadratic model over the influence (ℓ_F) and top perpendicular-curvature axes.
The projection basis (attacked set S, directions u1,u2, target g) is FIXED at the smallest λ so the
axes and z-scale stay put and you literally watch the bowl flatten.

    python3 curvature_animation.py [dataset] [kind] [M]
    e.g.  python3 curvature_animation.py concrete linear_topq 8

Writes curvature_lambda_<dataset>_<kind>.gif
"""
from __future__ import annotations
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import webcore as W
from dashboard_extras import _target_triple

NAME = sys.argv[1] if len(sys.argv) > 1 else "concrete"
KIND = sys.argv[2] if len(sys.argv) > 2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
LAM_MIN, LAM_MAX, N_FRAMES = 0.01, 1.0, 16
P, Q, Q_HI, N_SUB, SEED, Rm, NGRID = 0.99, 0.8, 1.0, 350, 0, 3.0, 13

lams = np.round(10 ** np.linspace(np.log10(LAM_MIN), np.log10(LAM_MAX), N_FRAMES), 4)

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (800, 300)))
X, y = ds.X_train, ds.y_train
if len(X) > N_SUB:
    i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
# the "best" lambda = the one that minimises clean CV error on trusted data (thesis prescription)
_, LAM_CV = W.cv_lambda(X, y, list(lams))
CV_IDX = int(np.argmin([abs(np.log(l) - np.log(LAM_CV)) for l in lams]))
print(f"CV-optimal λ ≈ {LAM_CV:g}", flush=True)
Xte, yte = ds.X_test, ds.y_test
d = X.shape[1]; md = M * d; R = float(np.sqrt(M) * Rm)

# --- FIXED geometry at the smallest lambda (so axes/scale don't jump between frames) ---
fit0 = W.ridge_fit(X, y, LAM_MIN)
F_fn, gF_fn, hF_fn = _target_triple(KIND, Xte, yte, fit0.theta, Q, Q_HI); gF = gF_fn(fit0.theta)
inl = W.filter_outlier(X, p=P); si = W.influence_scores(fit0, X, y, gF)
S = [int(inl[k]) for k in np.argsort(-si[inl])[:M]]
lF0 = W.ell_F(fit0, X, y, S, gF); u1 = lF0 / np.linalg.norm(lF0)
u2 = W.top_perp(lambda v: W.hmap_hvp(X, y, LAM_MIN, S, gF, v), u1, md)
gv = np.linspace(-R, R, NGRID); Ag, Bg = np.meshgrid(gv, gv); mask = Ag**2 + Bg**2 <= R**2

def frame(lam):
    fit = W.ridge_fit(X, y, lam); F0 = float(F_fn(fit.theta))
    lF = W.ell_F(fit, X, y, S, gF); c1 = float(lF @ u1); c2 = float(lF @ u2)
    hvpH = lambda v: W.hmap_hvp(X, y, lam, S, gF, v)
    if hF_fn is not None:
        A = W.stack_A(fit, X, y, S); HF = hF_fn(fit.theta); Bmv = lambda v: A.T @ (HF @ (A @ v))
    else:
        Bmv = lambda v: np.zeros(md)
    BH2 = W.restrict2(Bmv, u1, u2) + W.restrict2(hvpH, u1, u2)
    Zt = np.full_like(Ag, np.nan)
    for a in range(NGRID):
        for b in range(NGRID):
            if mask[a, b]:
                Zt[a, b] = F_fn(W._refit_at_delta(X, y, lam, S, Ag[a, b]*u1 + Bg[a, b]*u2)[0].theta) - F0
    return Zt, float(np.linalg.norm(BH2, 2))    # true ΔF surface + curvature magnitude ‖M‖

print(f"computing {N_FRAMES} frames for {NAME}/{KIND} (M={M}) ...", flush=True)
frames = [frame(l) for l in lams]
z0 = frames[0][0]; zlo, zhi = float(np.nanmin(z0)), float(np.nanmax(z0))
pad = 0.1 * (zhi - zlo + 1e-9); zlo -= pad; zhi += pad
curv0 = frames[0][1]

fig = plt.figure(figsize=(6.4, 5.4)); ax = fig.add_subplot(111, projection="3d")
def update(k):
    ax.clear()
    Zt, curv = frames[k]
    ax.plot_surface(Ag, Bg, Zt, cmap="viridis", alpha=0.9, linewidth=0.2, edgecolor="k", antialiased=True, rstride=1, cstride=1)
    ax.set_zlim(zlo, zhi); ax.view_init(24, -58)
    ax.set_xlabel("influence  ℓ_F", fontsize=9); ax.set_ylabel("perpendicular  ⊥", fontsize=9); ax.set_zlabel("ΔF", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{NAME} ({KIND}) — curvature flattens as defender raises λ\n"
                 f"λ = {lams[k]:.3f}   ·   curvature ‖M‖ = {curv:.2f}   ·   best (CV-optimal) λ = {LAM_CV:g}",
                 fontsize=10.5, pad=8)
    return []

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=280, blit=False)
out = f"curvature_lambda_{NAME}_{KIND}.gif"
try:
    from matplotlib.animation import PillowWriter
    anim.save(out, writer=PillowWriter(fps=5))
except Exception as e:
    print("PillowWriter failed (%s); saving frames as PNGs instead" % e, flush=True)
    for k in range(N_FRAMES):
        update(k); fig.savefig(f"curv_frame_{k:02d}.png", dpi=110, bbox_inches="tight")
    raise SystemExit
import os
print(f"WROTE {out}  ({os.path.getsize(out)/1024:.0f} KB)", flush=True)
