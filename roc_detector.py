"""Why the detector AUC flags the nudge — with a real ROC, using the notebook's EXACT detector.

The thesis 'cluster recoverability' detector (webcore.detectability_scan):
  feats = standardised (ŷ, r);  KMeans(2) → bigger (clean-bulk) cluster;
  anom_i = ‖feats_i − center_big‖;  AUC = roc_auc_score(poison?, anom).   (unsupervised score)

We plot that ROC, and for CONTRAST the ROC of the two filters the attack was built to defeat:
  • TRIM / large-residual      score = |r|        (residual-magnitude space)
  • feature outlier filter     score = d_M(x)     (feature space x; natural threshold = τ)
Left panel: the (ŷ,r) plane with the two KMeans clusters and the poison marked.

    python3 roc_detector.py [dataset] [kind] [M] [regime] [eps] [method]
"""
from __future__ import annotations
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score, silhouette_samples
from scipy.stats import chi2
import webcore as W
from dashboard_extras import _target_triple, _cp_regime

NAME = sys.argv[1] if len(sys.argv) > 1 else "california"
KIND = sys.argv[2] if len(sys.argv) > 2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REG  = sys.argv[4] if len(sys.argv) > 4 else "both"
EPS  = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
METH = sys.argv[6] if len(sys.argv) > 6 else "TR3"
LAM, P, Q, Q_HI, N_SUB, SEED = 0.1, 0.99, 0.8, 1.0, 500, 0

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (400, 150)))
X, y = ds.X_train, ds.y_train
if len(X) > N_SUB:
    i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
Xte, yte = ds.X_test, ds.y_test; n, d = X.shape
fit0 = W.ridge_fit(X, y, LAM)
F_fn, gF_fn, hF_fn = _target_triple(KIND, Xte, yte, fit0.theta, Q, Q_HI); gF = gF_fn(fit0.theta)
hessM = hF_fn(fit0.theta) if hF_fn is not None else None
cp = dict(_cp_regime(REG, P))
S = W.select_optimal_points(X, y, fit0, gF, M, p=P, mode=("budget" if REG == "budget" else "cap"))["S"]
R = float(np.sqrt(M)*EPS)
if METH == "TR1": dl = W.tr_oneshot(fit0, X, y, LAM, S, R, gF, None, 1, **cp)["delta"]
elif METH == "TR3": dl = W.tr_oneshot(fit0, X, y, LAM, S, R, gF, hessM, 3, **cp)["delta"]
elif METH == "bilevel": dl = W.bilevel_at_S(fit0, X, y, LAM, S, gF, R, n_iter=120, n_restarts=2, **cp)["delta"]
else: dl = W.tr3_relin(fit0, X, y, LAM, S, R, gF, None, max_outer=25, **cp)["delta"]

Xp = X.copy()
for k, ii in enumerate(S): Xp[ii] = Xp[ii] + dl[k*d:(k+1)*d]
thp = W.ridge_fit(Xp, y, LAM).theta
yh = Xp @ thp; r = y - yh
lab = np.zeros(n, int); lab[S] = 1

# ---- notebook's exact (ŷ,r) detector ----
feats = np.column_stack([(yh - yh.mean())/(yh.std()+1e-9), (r - r.mean())/(r.std()+1e-9)])
km = KMeans(2, n_init=10, random_state=SEED).fit(feats)
big = int(np.argmax(np.bincount(km.labels_)))
anom = np.linalg.norm(feats - km.cluster_centers_[big], axis=1)     # distance to clean-bulk cluster
sil = float(silhouette_samples(feats, lab)[S].mean())
# ---- contrasts ----
s_trim = np.abs(r)
mu = X.mean(0); Sig = np.cov(X.T) + 1e-3*np.eye(d); Lc = np.linalg.cholesky(Sig)
s_feat = np.sqrt((np.linalg.solve(Lc, (Xp - mu).T)**2).sum(0))
tau = float(np.sqrt(chi2.ppf(P, d)))
pass_feat = float((s_feat[S] <= tau).mean())                        # fraction of poison a τ-filter LETS THROUGH
scores = [("(ŷ,r) KMeans detector — ours", anom, "#c0392b"),
          ("feature outlier  d_M(x)",       s_feat, "#3f7fc0"),
          ("TRIM  |r|",                     s_trim, "#d99b2e")]

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.4))
lb = km.labels_
a0.scatter(feats[lb == big, 0], feats[lb == big, 1], s=10, c="#c9c2b2", alpha=.55, label="clean-bulk cluster")
a0.scatter(feats[(lb != big) & (lab == 0), 0], feats[(lb != big) & (lab == 0), 1], s=14, c="#9aa7b0", alpha=.7, label="other cluster (clean)")
a0.scatter(feats[S, 0], feats[S, 1], s=75, c="#c0392b", edgecolor="k", lw=.6, label="poison (S)", zorder=6)
a0.scatter(*km.cluster_centers_[big], marker="+", c="#4d7562", s=160, zorder=7)
a0.axhline(0, color="#b8b0a0", lw=.7)
a0.set_xlabel("standardised prediction ŷ"); a0.set_ylabel("standardised residual r")
a0.set_title(f"{NAME}: poison splits off the clean bulk in (ŷ,r)\nsilhouette(poison)={sil:.2f}"); a0.legend(fontsize=8.5)
a1.plot([0, 1], [0, 1], "--", color="#c9c2b2", lw=1)
for nm, sc, c in scores:
    auc = roc_auc_score(lab, sc); fpr, tpr, _ = roc_curve(lab, sc)
    a1.plot(fpr, tpr, "-", color=c, lw=2.5, label=f"{nm}   AUC={auc:.2f}")
a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate (poison caught)")
a1.set_title("ROC on the same poison"); a1.legend(fontsize=9, loc="lower right"); a1.grid(alpha=.25)
fig.suptitle(f"Detecting the nudge — {NAME} ({KIND}), {METH}, {REG} caps, ε={EPS:.1f}, m={M}", y=1.02, fontsize=13)
fig.text(0.5, -0.02, f"TRIM ranks near chance (residuals move INWARD, so poison has small |r|).   "
         f"Feature filter's high AUC is a ranking artefact: at its natural threshold τ it lets {pass_feat*100:.0f}% of poison through "
         f"(attack built for d_M ≤ τ).\nOnly the joint (ŷ,r) detector both ranks AND clusters the poison (silhouette {sil:.2f}) — a usable, threshold-free flag.",
         ha="center", fontsize=8.6, color="#555")
plt.tight_layout()
out = f"roc_detector_{NAME}_{KIND}.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"WROTE {out}")
for nm, sc, c in scores: print(f"  {nm:32s} AUC={roc_auc_score(lab, sc):.3f}")
print(f"  silhouette(poison)={sil:.3f}   poison passing τ-feature-filter={pass_feat*100:.0f}%")
