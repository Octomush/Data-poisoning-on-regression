"""Poisoning across THREE spaces as the budget ε grows, side by side:
   LEFT   feature space (PCA-2D of x): poison drifts toward the inlier boundary
   MIDDLE prediction-residual (ŷ, r): poison slides ANTIDIAGONALLY (Δr = -Δŷ), inward toward r≈0,
          out into the tail — the space the KMeans/AUC detector uses
   RIGHT  residual distribution: |r| of the poison shrinks (residuals move inward)
Trails show each poison point's path as ε increases.

    python3 poison_spaces_animation.py [dataset] [kind] [M] [method]
      method: TR3 (default, one-shot) | relin
Writes poison_spaces_<dataset>_<kind>_<method>.gif
"""
from __future__ import annotations
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.animation import PillowWriter
from scipy.stats import chi2
from sklearn.decomposition import PCA
import webcore as W
from dashboard_extras import _target_triple, _cp_regime

NAME = sys.argv[1] if len(sys.argv)>1 else "california"
KIND = sys.argv[2] if len(sys.argv)>2 else "linear_topq"
M    = int(sys.argv[3]) if len(sys.argv)>3 else 8
METH = sys.argv[4] if len(sys.argv)>4 else "TR3"
LAM,P,Q,Q_HI,N_SUB,SEED = 0.1,0.99,0.8,1.0,400,0
EPS = np.round(np.linspace(0.2,6.0,22),3)

ds = W.DATASETS[NAME](*W.DEFAULT_SIZE.get(NAME,(400,150)))
X,y = ds.X_train, ds.y_train
if len(X)>N_SUB:
    i = np.random.default_rng(SEED).choice(len(X),N_SUB,replace=False); X,y = X[i],y[i]
Xte,yte = ds.X_test, ds.y_test; n,d = X.shape
fit0 = W.ridge_fit(X,y,LAM)
F_fn,gF_fn,hF_fn = _target_triple(KIND,Xte,yte,fit0.theta,Q,Q_HI); gF = gF_fn(fit0.theta)
hessM = hF_fn(fit0.theta) if hF_fn is not None else None
cp = dict(_cp_regime("both",P))
S = list(W.select_optimal_points(X,y,fit0,gF,M,p=P,mode="cap")["S"])

th0 = fit0.theta; yh0 = X@th0; r0 = y-yh0
muY,sY = yh0.mean(), yh0.std()+1e-9; muR,sR = r0.mean(), r0.std()+1e-9
Xm = X.mean(0); pca = PCA(2).fit(X-Xm); Fclean = pca.transform(X-Xm)
C = np.cov(Fclean.T); wv,Vv = np.linalg.eigh(C); tau2 = np.sqrt(chi2.ppf(P,2))
ang = np.degrees(np.arctan2(Vv[1,-1],Vv[0,-1])); ew,eh = 2*tau2*np.sqrt(wv[-1]),2*tau2*np.sqrt(wv[-2])

def attack(R):
    if METH=="relin": return W.tr3_relin(fit0,X,y,LAM,S,R,gF,None,max_outer=25,**cp)["delta"]
    return W.tr_oneshot(fit0,X,y,LAM,S,R,gF,hessM,3,**cp)["delta"]

print(f"computing {len(EPS)} ε-frames ({NAME}, {METH}) ...", flush=True)
FP = np.zeros((len(EPS),M,2)); YR = np.zeros((len(EPS),n,2)); RES = np.zeros((len(EPS),n))
for j,eps in enumerate(EPS):
    dl = attack(float(np.sqrt(M)*eps)); Xp = X.copy()
    for k,ii in enumerate(S): Xp[ii] = Xp[ii] + dl[k*d:(k+1)*d]
    thp = W.ridge_fit(Xp,y,LAM).theta; yh = Xp@thp; r = y-yh
    YR[j] = np.column_stack([(yh-muY)/sY,(r-muR)/sR]); RES[j] = (r-muR)/sR
    FP[j] = pca.transform(Xp[S]-Xm)

cl = [i for i in range(n) if i not in set(S)]
fx = [min(Fclean[:,0].min(),FP[:,:,0].min())-.5, max(Fclean[:,0].max(),FP[:,:,0].max())+.5]
fy = [min(Fclean[:,1].min(),FP[:,:,1].min())-.5, max(Fclean[:,1].max(),FP[:,:,1].max())+.5]
yx = [YR[:,:,0].min()-.3, YR[:,:,0].max()+.3]; yy = [YR[:,:,1].min()-.3, YR[:,:,1].max()+.3]
rx = [RES.min()-.3, RES.max()+.3]
cmap = plt.cm.viridis; pcol = [cmap(k/max(M-1,1)) for k in range(M)]

fig,(a0,a1,a2) = plt.subplots(1,3,figsize=(18,5.8))
def update(j):
    a0.clear(); a1.clear(); a2.clear()
    # ---- feature space ----
    a0.scatter(Fclean[:,0],Fclean[:,1],s=8,c="#c9c2b2",alpha=.5)
    a0.add_patch(Ellipse((Fclean[:,0].mean(),Fclean[:,1].mean()),ew,eh,angle=ang,fill=False,ec="#7fa591",lw=2,ls="--"))
    for k in range(M):
        a0.plot(FP[:j+1,k,0],FP[:j+1,k,1],"-",color=pcol[k],lw=1.1,alpha=.5)
        a0.scatter(FP[j,k,0],FP[j,k,1],s=70,color=pcol[k],edgecolor="k",lw=.5,zorder=6)
    a0.set_xlim(*fx); a0.set_ylim(*fy); a0.set_xlabel("PCA-1"); a0.set_ylabel("PCA-2")
    a0.set_title("feature space x  (dashed = inlier boundary τ)",fontsize=11)
    # ---- (ŷ,r) space ----
    a1.scatter(YR[j,cl,0],YR[j,cl,1],s=9,c="#c9c2b2",alpha=.5)
    a1.axhline(0,color="#c9c2b2",lw=.7)
    lo=min(yx[0],yy[0]); hi=max(yx[1],yy[1]); a1.plot([lo,hi],[hi,lo],":",color="#c98a3a",lw=1,alpha=.6)  # antidiagonal
    for k,ii in enumerate(S):
        a1.plot(YR[:j+1,ii,0],YR[:j+1,ii,1],"-",color=pcol[k],lw=1.1,alpha=.5)
        a1.scatter(YR[j,ii,0],YR[j,ii,1],s=70,color=pcol[k],edgecolor="k",lw=.5,zorder=6)
    a1.set_xlim(*yx); a1.set_ylim(*yy); a1.set_xlabel("standardised ŷ"); a1.set_ylabel("standardised r")
    a1.set_title("prediction–residual (ŷ, r)  — antidiagonal inward drift",fontsize=11)
    # ---- residual distribution ----
    a2.hist(RES[j,cl],bins=40,color="#cfc8ba",alpha=.85,edgecolor="white",lw=.3)
    ymax=a2.get_ylim()[1]; a2.axvline(0,color="#7fa591",lw=1.4,ls="--")
    for k,ii in enumerate(S):
        a2.plot([RES[j,ii]]*2,[0,ymax*0.22],color=pcol[k],lw=1.8,zorder=6)
    a2.set_xlim(*rx); a2.set_xlabel("standardised residual r"); a2.set_ylabel("count")
    a2.set_title("residual distribution — poison |r| shrinks (moves inward)",fontsize=11)
    fig.suptitle(f"{NAME} ({KIND}) — poisoning across three spaces, {METH},  ε = {EPS[j]:.1f}   (m={M})",
                 y=1.0,fontsize=13)
    return []
fig.subplots_adjust(top=0.88,wspace=0.24,left=0.05,right=0.98,bottom=0.12)
anim = animation.FuncAnimation(fig,update,frames=len(EPS),interval=520,blit=False)
out=f"poison_spaces_{NAME}_{KIND}_{METH}.gif"; anim.save(out,writer=PillowWriter(fps=2.3))
print(f"WROTE {out}  ({os.path.getsize(out)/1024:.0f} KB)", flush=True)
