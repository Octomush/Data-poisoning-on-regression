"""Standalone animation across the budget ε for TR3-relin: three synced panels —
   (1) residuals sliding INWARD in the (ŷ, r) plane (poison disguising itself),
   (2) the BOUNDARY WALK: each poisoned point's headroom τ − d_M falling to the ceiling and
       redistributing (the relin bumps),
   (3) the reachable ΔF CLIMBING the curve.
All computed with the notebook solvers (tr3_relin under the inlier cap).

    python3 boundary_epsilon_animation.py [dataset] [kind] [M]
Writes boundary_eps_<dataset>_<kind>.gif
"""
from __future__ import annotations
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import webcore as W
from dashboard_extras import _target_triple

NAME = sys.argv[1] if len(sys.argv) > 1 else "realestate"
KIND = sys.argv[2] if len(sys.argv) > 2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CAPS = sys.argv[4] if len(sys.argv) > 4 else "inlier"        # 'inlier' (ceiling holds) or 'budget' (uncapped -> points cross)
CP = dict(mse_cap=False, inlier_cap=(CAPS == "inlier"))
LAM, P, Q, Q_HI, N_SUB, SEED = 0.1, 0.99, 0.8, 1.0, 400, 0
EPS = np.round(np.linspace(0.3, 7.0, 18), 3)

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (400, 150)))
X, y = ds.X_train, ds.y_train
if len(X) > N_SUB:
    i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
Xte, yte = ds.X_test, ds.y_test; n, d = X.shape; ym = float(ds.y_mean)
fit0 = W.ridge_fit(X, y, LAM); th0 = fit0.theta
F_fn, gF_fn, hF_fn = _target_triple(KIND, Xte, yte, th0, Q, Q_HI); gF = gF_fn(th0); F0 = float(F_fn(th0))
S = W.select_optimal_points(X, y, fit0, gF, M, p=P, mode="cap")["S"]
ctx = W.cap_context(X, y, fit0, S, P); tau = ctx["tau"]; L = ctx["L"]; Z = ctx["Z"]
yh0 = X @ th0; r0 = y - yh0
h0init = tau - np.linalg.norm(np.array(Z), axis=1); rank = np.argsort(np.argsort(-h0init))
cmap = plt.cm.viridis

print(f"computing {len(EPS)} ε-frames for {NAME}/{KIND} (M={M}, TR3-relin, inlier cap) ...", flush=True)
H = np.zeros((M, len(EPS))); DF = np.zeros(len(EPS)); YHP = []; RP = []
for j, eps in enumerate(EPS):
    R = float(np.sqrt(M) * eps)
    dl = W.tr3_relin(fit0, X, y, LAM, S, R, gF, None, max_outer=30, **CP)["delta"]
    Xp = X.copy()
    for k, ii in enumerate(S):
        Xp[ii] = Xp[ii] + dl[k*d:(k+1)*d]
        H[k, j] = tau - np.linalg.norm(Z[k] + np.linalg.solve(L, dl[k*d:(k+1)*d]))
    thp = W.ridge_fit(Xp, y, LAM).theta
    YHP.append(Xp @ thp); RP.append(y - thp @ Xp.T if False else y - Xp @ thp)
    DF[j] = float(gF @ thp) - F0
YHP = np.array(YHP); RP = np.array(RP)

# fixed limits
rx = [float((yh0).min()), float((yh0).max())]; ry = [float(r0.min()), float(r0.max())]
mrg = 0.08*(rx[1]-rx[0]); rx=[rx[0]-mrg,rx[1]+mrg]
fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.6))

def update(j):
    for a in (a0, a1, a2): a.clear()
    # (1) residual inward
    mask = np.ones(n, bool); mask[S] = False
    a0.scatter(yh0[mask]+ym, r0[mask], s=6, c="#d6cfc0", alpha=.6)
    a0.axhline(0, color="#b8b0a0", lw=.8)
    for k, ii in enumerate(S):
        a0.annotate("", xy=(YHP[j, ii]+ym, RP[j, ii]), xytext=(yh0[ii]+ym, r0[ii]),
                    arrowprops=dict(arrowstyle="-|>", color="#7a5fb0", lw=1.4, alpha=.85))
    a0.scatter([yh0[ii]+ym for ii in S], [r0[ii] for ii in S], s=90, marker="*", c="#e6a23c", edgecolor="k", lw=.4, zorder=5, label="clean ★")
    a0.scatter([YHP[j, ii]+ym for ii in S], [RP[j, ii] for ii in S], s=55, c="#2e7d57", edgecolor="k", lw=.4, zorder=6, label="poisoned ●")
    a0.set_xlim(rx[0]+ym, rx[1]+ym); a0.set_ylim(ry[0]*1.05, ry[1]*1.05)
    a0.set_xlabel("prediction ŷ"); a0.set_ylabel("residual r = y − ŷ"); a0.set_title("residuals move inward"); a0.legend(fontsize=8, loc="upper right")
    # (2) boundary walk
    for k in range(M):
        col = cmap(rank[k]/max(M-1, 1))
        a1.plot(EPS[:j+1], H[k, :j+1], "-", lw=1.6, color=col)
        a1.plot(EPS[j], H[k, j], "o", ms=5, color=col, mec="k", mew=.3)
    a1.axhline(0, color="red", lw=2)
    a1.axvline(EPS[j], color="#7fa591", lw=1.5, ls=":")
    a1.set_xlim(EPS[0], EPS[-1]); a1.set_ylim(min(-0.18*tau, float(H.min())*1.1), float(H.max())*1.08)  # room below ceiling
    a1.set_xlabel("ε (budget)"); a1.set_ylabel("headroom  τ − d_M"); a1.set_title("boundary walk (points → ceiling, redistribute)")
    # (3) ΔF climb
    a2.plot(EPS, DF, "-", color="#e0d3cf", lw=1.6)
    a2.plot(EPS[:j+1], DF[:j+1], "-", color="#c0392b", lw=2.6)
    a2.plot(EPS[j], DF[j], "o", ms=8, color="#c0392b", mec="k", mew=.4)
    a2.set_xlim(EPS[0], EPS[-1]); a2.set_ylim(min(0, float(DF.min()))*1.1, float(DF.max())*1.12)
    a2.set_xlabel("ε (budget)"); a2.set_ylabel("reachable ΔF"); a2.set_title("TR3-relin climbs the objective")
    fig.suptitle(f"{NAME} ({KIND}) — TR3-relin across ε = {EPS[j]:.1f}   (inlier cap, M={M})", y=0.99, fontsize=13)
    return []

fig.subplots_adjust(top=0.80, wspace=0.32, left=0.06, right=0.985, bottom=0.13)
anim = animation.FuncAnimation(fig, update, frames=len(EPS), interval=550, blit=False)
out = f"boundary_eps_{NAME}_{KIND}.gif"
from matplotlib.animation import PillowWriter
anim.save(out, writer=PillowWriter(fps=2.2))
import os
print(f"WROTE {out}  ({os.path.getsize(out)/1024:.0f} KB)", flush=True)
