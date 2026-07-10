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

# Interactive Explorer

A small, self‑contained toolkit for exploring **feature‑perturbation data‑poisoning attacks on strongly‑convex regression**. It ships three things:

1. a **local web dashboard** (`server.py` + `index.html`) that calls the research code's own functions and renders their figures live for whatever dataset / objective / regime / budget you pick;
3. **animation & analysis scripts** that reproduce the figures in `animations/`.

Everything is built on the same core: `webcore.py` (the poisoning math and dataset loaders) and `dashboard_extras.py` (a few dashboard‑only figures built strictly on `webcore` quantities).

## Quick start — the live dashboard

```bash
pip install -r requirements.txt
python3 server.py            # → http://localhost:8000
```

The first render of each setting runs the real solver (seconds on small datasets, up to ~a minute on large ones); the figure is then cached in `fig_cache/`. Default dataset is **casp**. Controls map directly to function arguments: dataset, objective (`linear_topq`, `feature_seg`, `feature_level`, `level`, `mse`), poisoned‑point count *m*, regime (budget / inlier / both / MSE), and budget ε.

## Animations

Pre‑rendered in `animations/`:

- `force_balance_animation.gif` — attack rolled out under the drive until the wall reactions balance it (budget‑only → GREEN, caps → RED), with the net‑δ vector
- `ceiling3d_casp_linear_topq.gif` — approach to the inlier ceiling, TR3‑relin
- `ceiling3d_casp_linear_topq_bilevel.gif` — same, bilevel / PGA (for comparison)
- `ceiling3d_realestate_linear_topq.gif` — ceiling approach on the balanced/harder realestate geometry
- `relin_curvature_casp_linear_topq.gif` — local curvature re‑estimated each refit (relinearization)
- `poison_spaces_california_linear_topq_TR3.gif` — poisoning across three spaces (feature · (ŷ,r) · residual) as ε grows
- `boundary_eps_realestate_linear_topq.gif` — boundary walk + residual‑inward across ε
- `curvature_lambda_concrete_linear_topq.gif` — attack‑surface curvature flattening as the ridge λ grows

### Regenerate the animations

Run from this folder (each writes its `.gif` here):

```bash
python3 force_balance_animation.py casp mse
python3 ceiling3d_animation.py casp linear_topq 8 inlier relin
python3 ceiling3d_animation.py casp linear_topq 8 inlier bilevel
python3 ceiling3d_animation.py realestate linear_topq 8 inlier relin
python3 relin_curvature_animation.py casp linear_topq 8 4.0
python3 poison_spaces_animation.py california linear_topq 8 TR3
python3 boundary_epsilon_animation.py realestate linear_topq 8 inlier
python3 curvature_animation.py concrete linear_topq 8
```

Argument pattern is `dataset kind m [caps|method|eps]`; swap in any dataset/objective for variants. 

## Datasets

Most datasets download automatically via scikit‑learn / OpenML (`realestate`, `california`, `communities`, `concrete`, `airfoil`, `casp`, `blog`). Three need a local CSV in `data/`: `house` → `house-processed.csv`, `warfarin` → `warfarin.csv`, `loan` → `loan_sample.csv` (optional).

## Requirements

Python 3.9+, with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`. The server uses only the Python standard library.
