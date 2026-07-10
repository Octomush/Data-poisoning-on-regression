# Data Poisoning on Regression

Research code for a thesis project on **feature‑perturbation data‑poisoning attacks against strongly‑convex
regression** — the "nudge, not inject" construction, its curvature‑driven attack surface (TR1/TR2/TR3/
TR3‑relin), the detectability ladder (budget · inlier · MSE caps), and the accompanying detection analysis.

The repository has two parts:

- **`Full_experiments.ipynb`** — the full experiment notebook (all attacks, sweeps, proofs‑in‑code, and figures).
- **an interactive explorer** — a small local dashboard and standalone pages that call the same functions and
  render their figures live (see [Interactive explorer](#interactive-explorer)).

---

## Download the data

The datasets are **not** stored in this repository (too large for GitHub upload). Download them from
the Google Drive folder below and place **all** files in the **same directory** as `Full_experiments.ipynb`:

**https://drive.google.com/drive/folders/11sUVKhZ0Lvuie7VSViaR1HzkEwyf1THF?usp=sharing**

Keep the file names unchanged — the notebook expects exactly these names. The repository should end up looking
like this:

```text
Data-poisoning-on-regression/
├── Full_experiments.ipynb
├── airfoil_self_noise.csv
├── blogData_train.csv
├── california.csv
├── casp.csv
├── communities.data
├── concrete.csv
├── house-processed.csv
├── loan_sample.csv
├── loan.csv.gz
├── realestate.csv
├── warfarin.csv
└── warfarin.xls
```

> **File paths:** the notebook assumes every dataset sits in the **same folder** as the notebook — directly,
> not inside a subfolder. Keep the notebook and the CSVs side by side.

---

## Running the notebook

```bash
# 1. clone
git clone https://github.com/[YOUR_USERNAME]/Data-poisoning-on-regression.git
cd Data-poisoning-on-regression

# 2. download the data (link above) into this same directory

# 3. install dependencies
pip install numpy scipy scikit-learn matplotlib pandas

# 4. open the notebook
jupyter notebook Full_experiments.ipynb      # or:  jupyter lab Full_experiments.ipynb
```

Then run the cells in order.

---

## Interactive explorer

The `dashboard/` folder is a self‑contained tool that calls the project's own functions and renders their
figures for whatever dataset / objective / regime / budget you choose — no notebook required.

```bash
cd dashboard
pip install -r requirements.txt
python3 server.py            # → http://localhost:8000
```

The first render of each setting runs the real solver (seconds on small datasets); results are cached. Panels
include the regression overlay, the **force‑balance mechanism** view, the functional budget sweep, m–ε sweep,
the interactive 3‑D feasible attack space, ceiling approach, budget redistribution, residual‑inward, and the
influence × η_max attackability heatmap.

**Animations** live in .gif forms — the force‑balance rollout, the ceiling approach
(TR3‑relin vs bilevel), poisoning across the three spaces, the boundary walk, the relinearization step, and the
curvature‑vs‑λ flattening. Regeneration code is: 

# --- ceiling approach (TR3-relin vs bilevel/PGA) ---
python3 ceiling3d_animation.py casp       linear_topq 8 inlier relin     # ceiling3d_casp_linear_topq_relin.gif
python3 ceiling3d_animation.py casp       linear_topq 8 inlier bilevel   # ceiling3d_casp_linear_topq_bilevel.gif
python3 ceiling3d_animation.py realestate linear_topq 8 inlier relin     # ceiling3d_realestate_linear_topq_relin.gif

# --- poisoning across the three spaces (feature · (ŷ,r) · residual) ---
python3 poison_spaces_animation.py california linear_topq 8 TR3          # poison_spaces_california_linear_topq_TR3.gif

# --- per-refit relinearization curvature ---
python3 relin_curvature_animation.py casp linear_topq 8 4.0              # relin_curvature_casp_linear_topq.gif

# --- boundary walk + residual inward across ε ---
python3 boundary_epsilon_animation.py realestate linear_topq 8 inlier    # boundary_eps_realestate_linear_topq.gif

# --- attack-surface curvature flattening as ridge λ grows ---
python3 curvature_animation.py concrete linear_topq 8                    # curvature_lambda_concrete_linear_topq.gif

> The datasets used by the explorer download automatically via scikit‑learn / OpenML, except `house`,
> `warfarin`, and `loan`, which read local CSVs (optional).

---

## Requirements

Python 3.9+, with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, and `pandas`. The dashboard server uses only
the Python standard library (no web framework).
