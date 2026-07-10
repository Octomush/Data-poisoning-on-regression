"""Standalone animation: how TR3-relin re-linearises — the LOCAL curvature it sees changes at every
refit. Runs TR3-relin (inlier cap) on a dataset, captures the accepted perturbation path δ_t, and for
each step shows the local incremental attack surface  f(δ_t + p) − f(δ_t)  over a fixed (ℓ_F, ⊥) plane
(f(δ)=F(Θ_S(δ))−F(θ_clean), the exact refit objective). As the model refits, the bowl rotates/reshapes
and shrinks toward the constrained optimum — the geometry relin recomputes each step.

    python3 relin_curvature_animation.py [dataset] [kind] [M] [eps]
    e.g.  python3 relin_curvature_animation.py casp linear_topq 8 4
Writes relin_curvature_<dataset>_<kind>.gif
"""
from __future__ import annotations
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import webcore as W
from dashboard_extras import _target_triple

NAME = sys.argv[1] if len(sys.argv) > 1 else "casp"
KIND = sys.argv[2] if len(sys.argv) > 2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
EPS  = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
LAM, P, Q, Q_HI, N_SUB, SEED, NGRID, MAX_FRAMES = 0.1, 0.99, 0.8, 1.0, 300, 0, 11, 20

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (800, 300)))
X, y = ds.X_train, ds.y_train
if len(X) > N_SUB:
    i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
Xte, yte = ds.X_test, ds.y_test
fit0 = W.ridge_fit(X, y, LAM); d = X.shape[1]; md = M*d
F_fn, gF_fn, hF_fn = _target_triple(KIND, Xte, yte, fit0.theta, Q, Q_HI); gF = gF_fn(fit0.theta)
S = W.select_optimal_points(X, y, fit0, gF, M, p=P, mode="cap")["S"]
R = float(np.sqrt(M)*EPS)
F0c = float(F_fn(fit0.theta))
f = lambda dl: float(F_fn(W._refit_at_delta(X, y, LAM, S, dl)[0].theta)) - F0c   # exact fixed-set objective

# run TR3-relin (inlier cap) and grab the relinearisation path
res = W.tr3_relin(fit0, X, y, LAM, S, R, gF, None, max_outer=40, mse_cap=False, inlier_cap=True, return_traj=True)
traj = res["traj"]
if len(traj) > MAX_FRAMES:                                   # evenly subsample the path
    idx = np.linspace(0, len(traj)-1, MAX_FRAMES).round().astype(int)
    traj = [traj[j] for j in sorted(set(idx))]
print(f"{NAME}: relin accepted {len(res['traj'])} steps -> {len(traj)} frames, final ΔF={res['val']:.3f}", flush=True)

# FIXED plane from the clean point: u1 = influence ray, u2 = top perpendicular H_map direction
lF0 = W.ell_F(fit0, X, y, S, gF); u1 = lF0/np.linalg.norm(lF0)
u2 = W.top_perp(lambda v: W.hmap_hvp(X, y, LAM, S, gF, v), u1, md)
rho = 0.5*R; gv = np.linspace(-rho, rho, NGRID); Ag, Bg = np.meshgrid(gv, gv); mask = Ag**2 + Bg**2 <= rho**2

def local_surface(dt):
    base = f(dt)
    Z = np.full_like(Ag, np.nan)
    for a in range(NGRID):
        for b in range(NGRID):
            if mask[a, b]:
                Z[a, b] = f(dt + Ag[a, b]*u1 + Bg[a, b]*u2) - base
    Mnorm = float(np.linalg.norm(W.full_curvature(fit0, X, y, LAM, S, gF, delta0=dt)[2], 2))
    return Z, base, Mnorm

print("computing frames ...", flush=True)
frames = [local_surface(dt) for dt in traj]
allZ = np.concatenate([fr[0][~np.isnan(fr[0])] for fr in frames])
zlo, zhi = float(allZ.min()), float(allZ.max()); pad = 0.1*(zhi-zlo+1e-9); zlo -= pad; zhi += pad

fig = plt.figure(figsize=(6.6, 5.4)); ax = fig.add_subplot(111, projection="3d")
def update(k):
    ax.clear()
    Z, base, Mn = frames[k]
    ax.plot_surface(Ag, Bg, Z, cmap="magma", alpha=0.9, linewidth=0.2, edgecolor="k", rstride=1, cstride=1, antialiased=True)
    ax.set_zlim(zlo, zhi); ax.view_init(26, -60)
    ax.set_xlabel("influence  ℓ_F", fontsize=9); ax.set_ylabel("perpendicular  ⊥", fontsize=9); ax.set_zlabel("local ΔF gain", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{NAME} ({KIND}) — TR3-relin: local curvature at each refit\n"
                 f"refit step {k+1}/{len(frames)}   ·   cumulative ΔF = {base:.2f}   ·   local ‖M‖ = {Mn:.2f}",
                 fontsize=10.5, pad=8)
    return []

anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=650, blit=False)
out = f"relin_curvature_{NAME}_{KIND}.gif"
from matplotlib.animation import PillowWriter
anim.save(out, writer=PillowWriter(fps=1.8))
import os
print(f"WROTE {out}  ({os.path.getsize(out)/1024:.0f} KB)", flush=True)
