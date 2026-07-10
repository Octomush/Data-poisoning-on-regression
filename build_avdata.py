"""Precompute REAL attack-vector geometry (projected into the (ℓ_F, ⊥) plane) for a grid of
dataset × objective × regime × ε, so a STANDALONE html can draw real-data vectors with no server.

  python3 build_avdata.py <dataset>     # compute+merge that dataset into avdata.json
  python3 build_avdata.py html          # emit attack_vectors_real.html with the JSON baked in
"""
from __future__ import annotations
import sys, os, json, numpy as np
import webcore as W
from dashboard_extras import _target_triple, _cp_regime

OBJS  = ["linear_topq", "feature_seg", "feature_level", "mse"]
REGS  = ["budget", "inlier", "both"]
EPS   = [1.5, 3.0, 5.0]
DSETS = ["casp", "california", "airfoil", "concrete"]
LAM, P, Q, Q_HI, M, N_SUB, SEED = 0.1, 0.99, 0.8, 1.0, 8, 220, 0
RELIN_OUTER = 4
JSON = "avdata.json"

def combo(name, kind, regime, eps, X, y, Xte, yte, fit):
    th0 = fit.theta; d = X.shape[1]; md = M*d
    F_fn, gF_fn, hF_fn = _target_triple(kind, Xte, yte, th0, Q, Q_HI); NL = hF_fn is not None
    gF = gF_fn(th0); hessM = hF_fn(th0) if NL else None
    cp = dict(_cp_regime(regime, P))
    S = W.select_optimal_points(X, y, fit, gF, M, p=P, mode=("budget" if regime == "budget" else "cap"))["S"]
    R = float(np.sqrt(M)*eps)
    A_S = W.stack_A(fit, X, y, S); lF = A_S.T @ gF; nlF = np.linalg.norm(lF)+1e-12; u1 = lF/nlF
    Mmv = lambda v: W.hmap_hvp(X, y, LAM, S, gF, v) + (A_S.T @ (hessM @ (A_S @ v)) if hessM is not None else 0.0)
    u2 = W.top_perp(Mmv, u1, md)
    lR = A_S.T @ W.grad_mse_theta(th0, X, y)
    proj = lambda dl: [float(dl @ u1)/R, float(dl @ u2)/R]
    def atk(method, **cc):
        if method == "TR1": return W.tr_oneshot(fit, X, y, LAM, S, R, gF, None,  1, **cc)["delta"]
        if method == "TR2": return W.tr_oneshot(fit, X, y, LAM, S, R, gF, hessM, 2, **cc)["delta"]
        if method == "TR3": return W.tr_oneshot(fit, X, y, LAM, S, R, gF, hessM, 3, **cc)["delta"]
        if NL: return W.tr3_relin_nonlinear(fit, X, y, LAM, S, R, gF_fn, F_fn, hF_fn, **W._filter_kw(W.tr3_relin_nonlinear, max_outer=RELIN_OUTER, **cc))["delta"]
        return W.tr3_relin(fit, X, y, LAM, S, R, gF, None, **W._filter_kw(W.tr3_relin, max_outer=RELIN_OUTER, **cc))["delta"]
    v = {m: proj(atk(m, **cp)) for m in ["TR1", "TR2", "TR3", "relin"]}
    unc = proj(atk("TR3", mse_cap=False, inlier_cap=False))
    lRp = np.array([lR @ u1, lR @ u2]); lRp = (lRp/(np.linalg.norm(lRp)+1e-12)).tolist()
    cosFR = float(lF @ lR/(nlF*(np.linalg.norm(lR)+1e-12)))
    angFR = float(np.degrees(np.arccos(np.clip(cosFR, -1, 1))))
    tilt = lambda a: float(np.degrees(np.arctan2(abs(a[1]), a[0])))
    cost = max(0.0, unc[0]-v["TR3"][0]); reach = v["relin"][0]
    return dict(tr1=v["TR1"], tr2=v["TR2"], tr3=v["TR3"], relin=v["relin"], unc=unc, lR=lRp,
                ang=round(angFR, 1), cos=round(cosFR, 3), cost=round(cost, 3), reach=round(reach, 3),
                tilts={k: round(tilt(v[k]), 1) for k in v})

def run_dataset(name):
    data = json.load(open(JSON)) if os.path.exists(JSON) else {"_meta": {}}
    ds = W.DATASETS[name](*W.DEFAULT_SIZE.get(name, (400, 150)))
    X, y = ds.X_train, ds.y_train
    if len(X) > N_SUB:
        i = np.random.default_rng(SEED).choice(len(X), N_SUB, replace=False); X, y = X[i], y[i]
    Xte, yte = ds.X_test, ds.y_test; fit = W.ridge_fit(X, y, LAM)
    data.setdefault(name, {})
    meta = dict(datasets=DSETS, kinds=OBJS, regimes=REGS, eps=[f"{e:.1f}" for e in EPS],
                labels={"linear_topq": "top-segment shift (linear)", "feature_seg": "feature-segment shift (linear)",
                        "feature_level": "feature-segment → level (curved)", "mse": "MSE availability"})
    for kind in OBJS:
        if kind in data[name] and all(f"{e:.1f}" in data[name][kind].get(REGS[-1], {}) for e in EPS):
            print(f"  {name} {kind}: cached", flush=True); continue          # resume-friendly
        data[name].setdefault(kind, {})
        for reg in REGS:
            data[name][kind].setdefault(reg, {})
            for eps in EPS:
                data[name][kind][reg][f"{eps:.1f}"] = combo(name, kind, reg, eps, X, y, Xte, yte, fit)
        data["_meta"] = meta; json.dump(data, open(JSON, "w"))               # checkpoint per objective
        print(f"  {name} {kind}: done+saved", flush=True)
    print(f"WROTE {JSON} (+{name})", flush=True)

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Attack-vector explorer (real datasets)</title>
<style>
:root{--bg:#faf7f1;--ink:#3f3a32;--mut:#8a8272;--line:#e7dfd1;--card:#fff;}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:22px}
h1{font-size:19px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin:0 0 16px}
.wrap{display:flex;gap:22px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.ctl{margin:8px 0}label{font-size:12px;color:var(--mut);display:block;margin-bottom:4px}
select,.seg button{font:13px inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);cursor:pointer}
.seg{display:inline-flex;gap:4px;flex-wrap:wrap}.seg button.on{background:#6b7186;color:#fff;border-color:#6b7186}
#read{font-size:13px;margin-top:12px;padding:10px 12px;background:#fbf7ef;border:1px solid var(--line);border-radius:8px;min-width:300px}
.big{font-size:15px;font-weight:700}
</style></head><body>
<h1>Attack-vector explorer — real datasets</h1>
<p class="sub">TR1/TR2/TR3/TR3-relin projected into the (ℓ_F, ⊥) plane, computed on the actual data. Dashed green = unconstrained TR3; the caps truncate &amp; rotate the real ray. Fully standalone (no server).</p>
<div class="wrap">
  <div class="card" style="min-width:300px">
    <div class="ctl"><label>Dataset</label><select id="ds"></select></div>
    <div class="ctl"><label>Objective</label><select id="ob"></select></div>
    <div class="ctl"><label>Detectability regime</label><div class="seg" id="rg"></div></div>
    <div class="ctl"><label>Budget ε</label><div class="seg" id="ep"></div></div>
    <div id="read"></div>
  </div>
  <div class="card"><svg id="svg" width="440" height="440" viewBox="0 0 440 440"></svg></div>
</div>
<script>const AVDATA=__DATA__;
const M=AVDATA._meta, S=document.getElementById("svg");
let st={ds:M.datasets[0],ob:"linear_topq",rg:"both",ep:M.eps[1]};
const COL={tr1:"#3f7fc0",tr2:"#d99b2e",tr3:"#3f9c68",relin:"#c0392b"};
const NAME={tr1:"TR1",tr2:"TR2",tr3:"TR3",relin:"TR3-relin"};
const CX=220,CY=220,U=150;  // unit = U px
const px=(x,y)=>[CX+x*U, CY-y*U];
function opt(sel,arr,cur,lab){sel.innerHTML="";arr.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=lab?lab(v):v;if(v===cur)o.selected=true;sel.appendChild(o);});}
function seg(el,arr,cur,key){el.innerHTML="";arr.forEach(v=>{const b=document.createElement("button");b.textContent=v;if(v===cur)b.className="on";b.onclick=()=>{st[key]=v;draw();};el.appendChild(b);});}
function line(x1,y1,x2,y2,c,w,dash){const l=document.createElementNS("http://www.w3.org/2000/svg","line");l.setAttribute("x1",x1);l.setAttribute("y1",y1);l.setAttribute("x2",x2);l.setAttribute("y2",y2);l.setAttribute("stroke",c);l.setAttribute("stroke-width",w);if(dash)l.setAttribute("stroke-dasharray",dash);S.appendChild(l);}
function arrow(v,c,w,dash,label){const[ex,ey]=px(v[0],v[1]);line(CX,CY,ex,ey,c,w,dash);
  const a=Math.atan2(ey-CY,ex-CX),h=9;
  [[-0.4],[0.4]].forEach(([o])=>line(ex,ey,ex-h*Math.cos(a-o),ey-h*Math.sin(a-o),c,w));
  if(label){const t=document.createElementNS("http://www.w3.org/2000/svg","text");t.setAttribute("x",ex+6*Math.sign(v[0]||1));t.setAttribute("y",ey-4);t.setAttribute("fill",c);t.setAttribute("font-size","12");t.setAttribute("font-weight","700");t.textContent=label;S.appendChild(t);}}
function txt(x,y,s,c,sz){const t=document.createElementNS("http://www.w3.org/2000/svg","text");t.setAttribute("x",x);t.setAttribute("y",y);t.setAttribute("fill",c||"#8a8272");t.setAttribute("font-size",sz||11);t.textContent=s;S.appendChild(t);}
function draw(){
  const d=AVDATA[st.ds][st.ob][st.rg][st.ep];
  S.innerHTML="";
  // budget circle + axes
  const circ=document.createElementNS("http://www.w3.org/2000/svg","circle");circ.setAttribute("cx",CX);circ.setAttribute("cy",CY);circ.setAttribute("r",U);circ.setAttribute("fill","none");circ.setAttribute("stroke","#d8d0c1");circ.setAttribute("stroke-dasharray","4 4");S.appendChild(circ);
  line(CX-U-14,CY,CX+U+14,CY,"#efe8db",1);line(CX,CY-U-14,CX,CY+U+14,"#efe8db",1);
  arrow([1,0],"#8a8272",2,null,"ℓ_F");
  if(st.rg!=="budget")arrow(d.unc,"#3f9c68",1.8,"5 4","TR3 uncapped");
  arrow([d.lR[0]*0.5,d.lR[1]*0.5],"#c98a86",1.6,"4 3","ℓ_R (MSE↑)");
  [["tr1",d.tr1],["tr2",d.tr2],["tr3",d.tr3],["relin",d.relin]].forEach(([k,v])=>arrow(v,COL[k],3,null,`${NAME[k]} (${d.tilts[k]}°)`));
  txt(CX-U-6,CY+U+30,"along influence ℓ_F  (÷R)","#8a8272",11);
  // readout
  const verdict=d.ang>95?'<span style="color:#3f9c68">anti-aligned → MSE cap ~FREE</span>':(d.ang<85?'<span style="color:#c0392b">aligned → MSE cap COSTLY</span>':'<span style="color:#8a8272">orthogonal → ~neutral</span>');
  document.getElementById("read").innerHTML=
    `<div class="big">${st.ds} · ${M.labels[st.ob]||st.ob}</div>`+
    `<div style="margin-top:6px">reachable ℓ_Fᵀδ = <b>${d.reach.toFixed(2)}</b>·R &nbsp; (uncapped TR3 ${d.unc[0].toFixed(2)}·R)</div>`+
    `<div>caps cost <b>${(100*d.cost/(Math.abs(d.unc[0])+1e-9)).toFixed(0)}%</b> · TR3 rotated <b>${d.tilts.tr3}°</b> off ℓ_F</div>`+
    `<div style="margin-top:8px;padding-top:8px;border-top:1px solid #eee">∠(ℓ_F, ℓ_R) = <b>${d.ang}°</b> (cos ${d.cos>=0?'+':''}${d.cos}) → ${verdict}</div>`;
}
opt(document.getElementById("ds"),M.datasets,st.ds);document.getElementById("ds").onchange=e=>{st.ds=e.target.value;draw();};
opt(document.getElementById("ob"),M.kinds,st.ob,v=>M.labels[v]||v);document.getElementById("ob").onchange=e=>{st.ob=e.target.value;draw();};
seg(document.getElementById("rg"),M.regimes,st.rg,"rg");seg(document.getElementById("ep"),M.eps,st.ep,"ep");
draw();
</script></body></html>"""

def _complete(data, ds):
    if ds not in data: return False
    return all(f"{e:.1f}" in data[ds].get(k, {}).get(r, {}) for k in OBJS for r in REGS for e in EPS)

def emit_html():
    data = json.load(open(JSON))
    done = [d for d in DSETS if _complete(data, d)]
    data["_meta"]["datasets"] = done                                 # only fully-computed datasets appear
    html = HTML.replace("__DATA__", json.dumps(data))
    open("attack_vectors_real.html", "w").write(html)
    print(f"WROTE attack_vectors_real.html  (datasets: {', '.join(done)})")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "casp"
    if arg == "html":
        emit_html()
    elif arg == "all":                                               # run locally (no time cap): compute all + emit
        for d in DSETS:
            try: run_dataset(d)
            except Exception as e: print(f"  !! {d} failed: {e}", flush=True)
        emit_html()
    else:
        run_dataset(arg)

