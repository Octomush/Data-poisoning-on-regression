"""2x2: the (ŷ,r) inner/outer clustering exists in BOTH clean and poisoned data.
Row 1 CLEAN   : (ŷ,r) with inner+outer clusters circled  |  ROC (no poison → random-label AUC ≈ 0.5)
Row 2 POISONED: same clustering, 8 poison marked, clean points MORE anomalous than any poison marked |
                ROC (AUC ≈ 0.94) with the 'catch-all-8' operating point annotated.

    python3 cluster_roc_2x2.py [dataset] [kind] [M] [eps] [method]
"""
from __future__ import annotations
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score
import webcore as W
from dashboard_extras import _target_triple, _cp_regime

NAME = sys.argv[1] if len(sys.argv)>1 else "california"
KIND = sys.argv[2] if len(sys.argv)>2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv)>3 else 8
EPS  = float(sys.argv[4]) if len(sys.argv)>4 else 1.5
METH = sys.argv[5] if len(sys.argv)>5 else "TR3"
LAM,P,Q,Q_HI,N_SUB,SEED = 0.1,0.99,0.8,1.0,500,0
rng = np.random.default_rng(SEED)

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME,(400,150)))
X,y = ds.X_train, ds.y_train
if len(X)>N_SUB:
    i = rng.choice(len(X),N_SUB,replace=False); X,y = X[i],y[i]
Xte,yte = ds.X_test, ds.y_test; n,d = X.shape
fit0 = W.ridge_fit(X,y,LAM)
F_fn,gF_fn,hF_fn = _target_triple(KIND,Xte,yte,fit0.theta,Q,Q_HI); gF = gF_fn(fit0.theta)
hessM = hF_fn(fit0.theta) if hF_fn is not None else None
cp = dict(_cp_regime("both",P))
S = W.select_optimal_points(X,y,fit0,gF,M,p=P,mode="cap")["S"]
R = float(np.sqrt(M)*EPS)
if METH=="TR1": dl = W.tr_oneshot(fit0,X,y,LAM,S,R,gF,None,1,**cp)["delta"]
elif METH=="TR3": dl = W.tr_oneshot(fit0,X,y,LAM,S,R,gF,hessM,3,**cp)["delta"]
elif METH=="bilevel": dl = W.bilevel_at_S(fit0,X,y,LAM,S,gF,R,n_iter=120,n_restarts=2,**cp)["delta"]
else: dl = W.tr3_relin(fit0,X,y,LAM,S,R,gF,None,max_outer=25,**cp)["delta"]
Xp = X.copy()
for k,ii in enumerate(S): Xp[ii] = Xp[ii] + dl[k*d:(k+1)*d]

def feats_of(Xd):
    th = W.ridge_fit(Xd,y,LAM).theta; yh = Xd@th; r = y-yh
    return np.column_stack([(yh-yh.mean())/(yh.std()+1e-9),(r-r.mean())/(r.std()+1e-9)])

def cluster(feats):
    km = KMeans(2,n_init=10,random_state=SEED).fit(feats)
    sizes = np.bincount(km.labels_); big = int(np.argmax(sizes)); small = 1-big
    anom = np.linalg.norm(feats-km.cluster_centers_[big],axis=1)
    return km,big,small,anom

def draw_clusters(ax,feats,km,big,small):
    for c,name,col in [(big,"inner (bulk)","#8a94a6"),(small,"outer tail","#c98a3a")]:
        pts = feats[km.labels_==c]; ctr = km.cluster_centers_[c]
        rad = np.percentile(np.linalg.norm(pts-ctr,axis=1),88)
        ax.add_patch(Circle(ctr,rad,fill=False,ec=col,lw=2,ls="--",zorder=4))
        ax.scatter(pts[:,0],pts[:,1],s=9,c=col,alpha=.35)
        ax.scatter(*ctr,marker="+",c=col,s=130,zorder=6)

def dist_panel(ax,anom,mark_idx,mark_color,thr=None,note="",title=""):
    ax.hist(anom,bins=42,color="#cfc8ba",alpha=.85,edgecolor="white",lw=.3)
    ymax=ax.get_ylim()[1]
    for i in mark_idx:
        ax.plot([anom[i],anom[i]],[0,ymax*0.20],color=mark_color,lw=1.5,alpha=.9,zorder=5)
    if thr is not None:
        ax.axvline(thr,color="#111",ls=":",lw=1.6)
        ax.axvspan(thr,anom.max()*1.02,color="#c0392b",alpha=.07)
    ax.set_xlabel("anomaly score  (distance to bulk)"); ax.set_ylabel("count")
    if note: ax.text(0.97,0.95,note,transform=ax.transAxes,ha="right",va="top",fontsize=8.4,
                     bbox=dict(fc="white",ec="0.8",alpha=.9))
    ax.set_title(title,fontsize=10.5)

fig,axes = plt.subplots(2,3,figsize=(18,10.4),gridspec_kw=dict(width_ratios=[1.15,1,1]))
# ---------- ROW 1: CLEAN ----------
fc = feats_of(X); km,big,small,anom = cluster(fc)
a=axes[0,0]; draw_clusters(a,fc,km,big,small)
top8 = np.argsort(-anom)[:M]
a.scatter(fc[top8,0],fc[top8,1],s=70,facecolors="none",edgecolors="#c0392b",lw=1.6,label="top-8 a detector would flag\n(all legitimate)")
a.axhline(0,color="#c9c2b2",lw=.7); a.set_title(f"CLEAN — (ŷ,r) always splits into inner+outer\n(no poison present)",fontsize=11)
a.set_xlabel("standardised ŷ"); a.set_ylabel("standardised r"); a.legend(fontsize=8,loc="upper left")
# clean ROC: no poison -> random-8 labels ~ chance (mean over draws)
aucs=[];
for t in range(200):
    lab=np.zeros(n,int); lab[np.random.default_rng(t).choice(n,M,replace=False)]=1
    aucs.append(roc_auc_score(lab,anom))
lab=np.zeros(n,int); lab[np.random.default_rng(3).choice(n,M,replace=False)]=1
fpr,tpr,_=roc_curve(lab,anom)
b=axes[0,1]; b.plot([0,1],[0,1],"--",color="#c9c2b2"); b.plot(fpr,tpr,color="#8a94a6",lw=2.3)
b.set_title(f"CLEAN ROC — no real anomalies\nAUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f}  (chance)",fontsize=11)
b.set_xlabel("false positive rate"); b.set_ylabel("true positive rate"); b.grid(alpha=.25)
dist_panel(axes[0,2],anom,top8,"#c0392b",thr=None,
           note="smooth, heavy tail\n(natural high-leverage points)",
           title="CLEAN — anomaly-score distribution\n(red ticks = top-8; just the natural tail)")
# ---------- ROW 2: POISONED ----------
fp = feats_of(Xp); kmp,bigp,smallp,anomp = cluster(fp)
a=axes[1,0]; draw_clusters(a,fp,kmp,bigp,smallp)
more_anom = np.array([i for i in range(n) if i not in set(S) and anomp[i]>anomp[S].max()])  # clean beyond ALL poison
a.scatter(fp[more_anom,0],fp[more_anom,1],s=46,facecolors="none",edgecolors="#c0392b",lw=1.5,
          label=f"{len(more_anom)} CLEAN points more\nanomalous than every poison")
a.scatter(fp[S,0],fp[S,1],s=80,c="#111111",edgecolor="w",lw=.6,label="8 poison (S)",zorder=7)
a.axhline(0,color="#c9c2b2",lw=.7); a.set_title("POISONED — same clustering; poison lands INSIDE the clean outer tail",fontsize=11)
a.set_xlabel("standardised ŷ"); a.set_ylabel("standardised r"); a.legend(fontsize=8,loc="upper left")
labp=np.zeros(n,int); labp[list(S)]=1; auc=roc_auc_score(labp,anomp); fpr,tpr,_=roc_curve(labp,anomp)
order=np.argsort(-anomp); ranks=sorted(list(order).index(i) for i in S); Kall=ranks[-1]+1; fp_all=(Kall-M)
b=axes[1,1]; b.plot([0,1],[0,1],"--",color="#c9c2b2"); b.plot(fpr,tpr,color="#c0392b",lw=2.6)
op_fpr=fp_all/(n-M); b.plot([op_fpr],[1.0],"o",ms=10,c="#111111",zorder=6)
b.annotate(f"catch all {M} → flag top-{Kall}\n({fp_all} clean false alarms, precision {M/Kall:.2f})",
           (op_fpr,1.0),(op_fpr+0.06,0.62),fontsize=8.5,arrowprops=dict(arrowstyle="->",color="#111"))
b.set_title(f"POISONED ROC — AUC = {auc:.2f}\npoison ranks {ranks[0]}–{ranks[-1]} of {n} (0 = most anomalous)",fontsize=11)
b.set_xlabel("false positive rate"); b.set_ylabel("true positive rate"); b.grid(alpha=.25)
dist_panel(axes[1,2],anomp,list(S),"#111111",thr=float(anomp[S].min()),
           note=f"{len(more_anom)} clean points sit\nto the RIGHT of every poison",
           title=f"POISONED — anomaly-score distribution\n(black ticks = poison; dotted = catch-all threshold)")
fig.suptitle(f"AUC {auc:.2f} looks strong, but the clustering is intrinsic and the poison hides in the clean tail — {NAME} ({KIND}, {METH}, ε={EPS})",
             fontsize=12.5,y=1.0)
plt.tight_layout()
out=f"cluster_roc_2x2_{NAME}_{KIND}.png"; fig.savefig(out,dpi=125,bbox_inches="tight")
print(f"WROTE {out}")
print(f"  clean AUC (random labels) = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"  poisoned AUC = {auc:.3f}; poison ranks {ranks}; catch-all top-{Kall} ⇒ {fp_all} FP; #clean beyond all poison = {len(more_anom)}")
