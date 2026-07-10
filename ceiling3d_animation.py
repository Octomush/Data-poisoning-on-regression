"""Side-by-side animation of the boundary walk:
   LEFT  = 3-D ceiling. Vertical axis = Mahalanobis d_M with the ceiling plane d_M = τ ON TOP; the
           horizontal plane is an ISOMAP(2) embedding. Points rise to the ceiling, pin, then slide
           TANGENTIALLY (black trail) or turn INWARD.
   RIGHT = the redistribution plot (headroom τ − d_M vs ε), same points, growing with ε.
   Each point is state-tracked:  approaching (its colour) · pinned at ceiling (gold) · REDROPPING (black).
   TR3-relin under the inlier cap (notebook solver).

    python3 ceiling3d_animation.py [dataset] [kind] [M] [caps] [method]
      caps: inlier (default) | budget      method: relin (default) | bilevel   (PGA, for comparison)
Writes ceiling3d_<dataset>_<kind>_<method>.gif
"""
from __future__ import annotations
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.manifold import Isomap
from scipy.stats import chi2
import webcore as W
from dashboard_extras import _target_triple

NAME = sys.argv[1] if len(sys.argv) > 1 else "casp"
KIND = sys.argv[2] if len(sys.argv) > 2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CAPS = sys.argv[4] if len(sys.argv) > 4 else "inlier"
METHOD = sys.argv[5] if len(sys.argv) > 5 else "relin"           # relin (TR3-relin) | bilevel (PGA)
CP = dict(mse_cap=False, inlier_cap=(CAPS == "inlier"))
MLABEL = {"relin": "TR3-relin", "bilevel": "bilevel (PGA)"}[METHOD]
LAM, P, Q, Q_HI, N_SUB, SEED = 0.1, 0.99, 0.8, 1.0, 300, 0
EPS = np.round(np.linspace(0.3, 7.0, 22), 3)

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (400, 150)))
X, y = ds.X_train, ds.y_train
if len(X) > N_SUB:
    i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
Xte, yte = ds.X_test, ds.y_test; n, d = X.shape
fit0 = W.ridge_fit(X, y, LAM)
F_fn, gF_fn, hF_fn = _target_triple(KIND, Xte, yte, fit0.theta, Q, Q_HI); gF = gF_fn(fit0.theta)
S = W.select_optimal_points(X, y, fit0, gF, M, p=P, mode="cap")["S"]
mu = X.mean(0); Xc = X - mu; cov = (Xc.T @ Xc)/max(n-1, 1) + 1e-3*np.eye(d); Lc = np.linalg.cholesky(cov)
tau = float(np.sqrt(chi2.ppf(P, d)))
dM = lambda pts: np.sqrt((np.linalg.solve(Lc, (pts - mu).T)**2).sum(0))

def attack(R):
    if METHOD == "bilevel":
        return W.bilevel_at_S(fit0, X, y, LAM, S, gF, R, n_iter=120, n_restarts=2, **CP)["delta"]
    return W.tr3_relin(fit0, X, y, LAM, S, R, gF, None, max_outer=30, **CP)["delta"]
print(f"computing {len(EPS)} ε-frames ({NAME}, {MLABEL}, caps={CAPS}) ...", flush=True)
Ppos = np.zeros((len(EPS), M, d)); Z = np.zeros((len(EPS), M))
for j, eps in enumerate(EPS):
    dl = attack(float(np.sqrt(M)*eps))
    Ppos[j] = np.array([X[S[k]] + dl[k*d:(k+1)*d] for k in range(M)])
    Z[j] = dM(Ppos[j])
H = tau - Z                                                     # headroom

# per-point state:  0 approaching · 1 pinned · 2 redropping
state = np.zeros((len(EPS), M), int); reached = np.zeros(M, bool)
for j in range(len(EPS)):
    for k in range(M):
        if Z[j, k] >= 0.85*tau: reached[k] = True
        if reached[k] and j > 0 and Z[j, k] < Z[j-1, k]-0.02*tau: state[j, k] = 2
        elif abs(Z[j, k]-tau) <= 0.06*tau: state[j, k] = 1
        else: state[j, k] = 0

inl = W.filter_outlier(X, p=P)
bg = X[inl][np.random.default_rng(1).choice(len(inl), min(160, len(inl)), replace=False)]
emb = Isomap(n_neighbors=min(10, len(bg)+len(S)+len(EPS)*M-1), n_components=2).fit_transform(
    np.vstack([bg, X[S], Ppos.reshape(-1, d)]))
nb = len(bg); bg_e = emb[:nb]; P_e = emb[nb+M:].reshape(len(EPS), M, 2); bg_z = dM(bg)
xl = [emb[:, 0].min(), emb[:, 0].max()]; yl = [emb[:, 1].min(), emb[:, 1].max()]; zt = max(tau*1.15, Z.max()*1.05)
cmap = plt.cm.viridis
rank = np.argsort(np.argsort(-(tau - Z[0])))                    # colour by initial headroom
def scol(k, st): return "#111111" if st == 2 else ("#e6a23c" if st == 1 else cmap(rank[k]/max(M-1, 1)))

fig = plt.figure(figsize=(13.2, 6.0))
ax3 = fig.add_subplot(1, 2, 1, projection="3d"); ax2 = fig.add_subplot(1, 2, 2)
def update(j):
    ax3.clear(); ax2.clear()
    # ---- LEFT: 3D ceiling ----
    xx, yy = np.meshgrid(np.linspace(*xl, 2), np.linspace(*yl, 2))
    ax3.plot_surface(xx, yy, np.full_like(xx, tau), color="crimson", alpha=0.14)
    ax3.scatter(bg_e[:, 0], bg_e[:, 1], bg_z, s=6, c="#d6cfc0", alpha=.3)
    for k in range(M):
        for t in range(j):
            dxy = np.hypot(P_e[t+1, k, 0]-P_e[t, k, 0], P_e[t+1, k, 1]-P_e[t, k, 1])
            tang = (Z[t, k] > 0.82*tau) and (dxy > 1.5*abs(Z[t+1, k]-Z[t, k]) + 1e-9)
            ax3.plot([P_e[t, k, 0], P_e[t+1, k, 0]], [P_e[t, k, 1], P_e[t+1, k, 1]], [Z[t, k], Z[t+1, k]],
                     color=("k" if tang else cmap(rank[k]/max(M-1, 1))), lw=(2.4 if tang else 1.0), alpha=(.95 if tang else .45))
        ax3.scatter([P_e[j, k, 0]], [P_e[j, k, 1]], [Z[j, k]], s=60, c=[scol(k, state[j, k])], edgecolor="k", lw=.5)
    ax3.set_xlim(*xl); ax3.set_ylim(*yl); ax3.set_zlim(0, zt); ax3.view_init(24, -64)
    ax3.set_xlabel("ISOMAP-1", fontsize=8); ax3.set_ylabel("ISOMAP-2", fontsize=8); ax3.set_zlabel("d_M (ceiling τ on top)", fontsize=8); ax3.tick_params(labelsize=6)
    ax3.set_title("3-D approach to the ceiling", fontsize=11)
    # ---- RIGHT: redistribution (headroom vs ε) ----
    for k in range(M):
        ax2.plot(EPS[:j+1], H[:j+1, k], "-", lw=1.5, color=cmap(rank[k]/max(M-1, 1)), alpha=.8)
        ax2.plot(EPS[j], H[j, k], "o", ms=8, color=scol(k, state[j, k]), mec="k", mew=.4, zorder=6)
    ax2.axhline(0, color="red", lw=2); ax2.axvline(EPS[j], color="#7fa591", lw=1.3, ls=":")
    ax2.set_xlim(EPS[0], EPS[-1]); ax2.set_ylim(min(-0.18*tau, float(H.min())*1.1), float(H.max())*1.08)
    ax2.set_xlabel("ε (budget)"); ax2.set_ylabel("headroom  τ − d_M"); ax2.set_title("redistribution (headroom vs ε)")
    npin = int((state[j] == 1).sum()); nre = int((state[j] == 2).sum())
    ax2.text(0.97, 0.96, f"pinned {npin}/{M}\nredropping {nre}/{M}", transform=ax2.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(fc="white", ec="0.7", alpha=.85))
    fig.suptitle(f"{NAME} ({KIND}) — {MLABEL} boundary walk, ε = {EPS[j]:.1f}   "
                 f"(gold = pinned at ceiling, BLACK = redropping / tangential)", y=0.99, fontsize=12.5)
    return []

fig.subplots_adjust(top=0.88, wspace=0.18, left=0.02, right=0.97, bottom=0.1)
anim = animation.FuncAnimation(fig, update, frames=len(EPS), interval=560, blit=False)
out = f"ceiling3d_{NAME}_{KIND}_{METHOD}.gif"
from matplotlib.animation import PillowWriter
anim.save(out, writer=PillowWriter(fps=2.2))
import os
print(f"WROTE {out}  ({os.path.getsize(out)/1024:.0f} KB)", flush=True)
