"""Is the detector actually BLIND, or is it cheating with labels?
Score (KMeans distance to bulk in standardised (ŷ,r)) uses NO labels. Labels only grade it (AUC).
Here we run the label-free pipeline and, crucially, compare against a CLEAN control (no poison) run
through the identical procedure. If clean data produced the same alarm, the detector would be circular.

    python3 blind_detect.py [dataset] [kind] [M] [eps] [method]
"""
from __future__ import annotations
import sys, numpy as np
import webcore as W
from dashboard_extras import _target_triple, _cp_regime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, roc_auc_score

NAME = sys.argv[1] if len(sys.argv) > 1 else "california"
KIND = sys.argv[2] if len(sys.argv) > 2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
EPS  = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
METH = sys.argv[5] if len(sys.argv) > 5 else "TR3"
LAM, P, Q, Q_HI, N_SUB, SEED = 0.1, 0.99, 0.8, 1.0, 500, 0

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME, (400,150)))
X, y = ds.X_train, ds.y_train
if len(X) > N_SUB:
    i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
Xte, yte = ds.X_test, ds.y_test; n, d = X.shape
fit0 = W.ridge_fit(X, y, LAM)
F_fn, gF_fn, hF_fn = _target_triple(KIND, Xte, yte, fit0.theta, Q, Q_HI); gF = gF_fn(fit0.theta)
hessM = hF_fn(fit0.theta) if hF_fn is not None else None
cp = dict(_cp_regime("both", P))
S = W.select_optimal_points(X, y, fit0, gF, M, p=P, mode="cap")["S"]
R = float(np.sqrt(M)*EPS)
if METH=="TR1": dl = W.tr_oneshot(fit0,X,y,LAM,S,R,gF,None,1,**cp)["delta"]
elif METH=="TR3": dl = W.tr_oneshot(fit0,X,y,LAM,S,R,gF,hessM,3,**cp)["delta"]
elif METH=="bilevel": dl = W.bilevel_at_S(fit0,X,y,LAM,S,gF,R,n_iter=120,n_restarts=2,**cp)["delta"]
else: dl = W.tr3_relin(fit0,X,y,LAM,S,R,gF,None,max_outer=25,**cp)["delta"]

def yr(Xd):
    th = W.ridge_fit(Xd, y, LAM).theta; yh = Xd@th; r = y-yh
    return np.column_stack([(yh-yh.mean())/(yh.std()+1e-9), (r-r.mean())/(r.std()+1e-9)])

def blind_report(feats, tag, poison_idx=None):
    km = KMeans(2, n_init=10, random_state=SEED).fit(feats)
    sil = silhouette_score(feats, km.labels_)                       # LABEL-FREE: quality of 2-clustering
    sizes = np.bincount(km.labels_); minc = int(np.argmin(sizes))
    bic1 = GaussianMixture(1, random_state=SEED).fit(feats).bic(feats)
    bic2 = GaussianMixture(2, random_state=SEED).fit(feats).bic(feats)
    big = int(np.argmax(sizes)); anom = np.linalg.norm(feats - km.cluster_centers_[big], axis=1)
    top = set(np.argsort(-anom)[:M])                                # LABEL-FREE flag: M most anomalous
    # compactness of the flagged tail (poison should be a tight clump; clean tail should be scattered)
    P_ = feats[list(top)]; pair = np.linalg.norm(P_[:,None,:]-P_[None,:,:],axis=2)
    clump = pair[np.triu_indices(M,1)].mean()
    line = f"[{tag}]  2-cluster silhouette={sil:.3f}   BIC(2)-BIC(1)={bic2-bic1:+.0f} (neg⇒2 modes preferred)   " \
           f"minority cluster size={sizes[minc]}/{len(feats)}   top-{M}-anom clump(mean pairwise)={clump:.2f}"
    if poison_idx is not None:
        hit = len(top & set(poison_idx))
        order = np.argsort(-anom); rankpos = np.array([list(order).index(i) for i in poison_idx])
        Kall = int(rankpos.max())+1                                 # flag top-K to catch ALL poison
        n_clean_fp = Kall - M
        line += f"\n        unsup-score AUC vs truth={roc_auc_score(np.isin(np.arange(len(feats)),poison_idx).astype(int), anom):.3f}" \
                f"\n        poison sit at anom-ranks {sorted(rankpos)} (0=most anomalous of {len(feats)})" \
                f"\n        to catch ALL {M} poison blindly you must flag top-{Kall}  ⇒  {n_clean_fp} clean false-positives  (precision={M/Kall:.2f})"
    print(line)

Xp = X.copy()
for k,ii in enumerate(S): Xp[ii] = Xp[ii] + dl[k*d:(k+1)*d]
print(f"=== {NAME} {KIND} {METH} ε={EPS} m={M} ===")
blind_report(yr(X),  "CLEAN control (no poison)")
blind_report(yr(Xp), "POISONED", poison_idx=list(S))
