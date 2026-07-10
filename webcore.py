from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import sys, types
_ip=types.ModuleType("IPython"); _ip.get_ipython=lambda *a,**k: None; _disp=types.ModuleType("IPython.display")
_disp.display=lambda *a,**k: None
_ip.display=_disp; _ip.version_info=(7,34,0); sys.modules["IPython"]=_ip; sys.modules["IPython.display"]=_disp
import time
import os, sys, csv, argparse, warnings, time
import numpy as np
from sklearn.neighbors import kneighbors_graph
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Sequence
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.manifold import Isomap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Arc, FancyArrowPatch
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_samples
try:
    from sklearn.datasets import fetch_openml
    _HAS_OPENML = True
except Exception:
    _HAS_OPENML = False
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import KFold
from scipy.stats import chi2
import math
from scipy.stats import gaussian_kde

def _find_data_dir() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        for cand in (base / 'data', base / 'influence_atlas_v2' / 'data'):
            if (cand / 'concrete.csv').exists() or (cand / 'house-processed.csv').exists():
                return cand.resolve()
    return (Path.cwd() / 'data').resolve()
_DATA_DIR = _find_data_dir()

@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    y_mean: float
    description: str = ''

def polynomial_expand(X: np.ndarray, names: List[str], degree: int=2, interaction_only: bool=False) -> tuple:
    """Expand X by polynomial features up to `degree`.

    Useful to push real datasets into a HIGHER-DIM regime without losing the
    real correlation structure: e.g., diabetes (d=10, degree=3) -> d=285,
    house numerical block (d=37, degree=2 interaction-only) -> d~ 740.
    Avoids sklearn's PolynomialFeatures dependency on dtype quirks.
    """
    from itertools import combinations_with_replacement, combinations
    (n, d) = X.shape
    cols = [X[:, [i]] for i in range(d)]
    new_names = list(names)
    pick = combinations if interaction_only else combinations_with_replacement
    for deg in range(2, degree + 1):
        for combo in pick(range(d), deg):
            col = np.prod(np.stack([X[:, j] for j in combo], axis=1), axis=1, keepdims=True)
            cols.append(col)
            new_names.append('*'.join((names[j] for j in combo)))
    X_new = np.hstack(cols)
    return (X_new, new_names)

def _split_standardise(X: np.ndarray, y: np.ndarray, feature_names: List[str], train_size: int, test_size: int, seed: int, name: str, desc: str) -> Dataset:
    n = X.shape[0]
    if train_size + test_size > n:
        data_size = train_size + test_size
        train_size = int(train_size / data_size * n)
        test_size = n - train_size
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:train_size + test_size]
    (tr, te) = (idx[:train_size], idx[train_size:train_size + test_size])
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])
    Xte = sc.transform(X[te])
    y_mean = float(np.mean(y[tr]))
    return Dataset(name=name, X_train=Xtr, y_train=y[tr] - y_mean, X_test=Xte, y_test=y[te] - y_mean, feature_names=feature_names, y_mean=y_mean, description=desc)

def _load_cached_or_openml(csv_name: str, openml_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    csv_path = _DATA_DIR / csv_name
    if csv_path.exists():
        df = pd.read_csv(csv_path).dropna()
    else:
        if not _HAS_OPENML:
            raise RuntimeError(f'{csv_name} not cached and OpenML unavailable')
        d = fetch_openml(name=openml_name, as_frame=True, parser='auto')
        df = d.frame.copy()
        tcol = d.target.name if d.target is not None else df.columns[-1]
        df = df.rename(columns={tcol: 'target'}).dropna()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    y = df['target'].to_numpy(float)
    Xdf = df.drop(columns=['target']).select_dtypes('number')
    return (Xdf.to_numpy(float), y, list(Xdf.columns))

def load_house(csv_path: str | Path, train_size: int=840, test_size: int=280, seed: int=0) -> Dataset:
    """House dataset (the original baseline).  target is last column."""
    df = pd.read_csv(csv_path).dropna()
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float)
    names = list(df.columns[:-1])
    return _split_standardise(X, y, names, train_size, test_size, seed, name='house', desc=f'house-processed.csv (d={X.shape[1]}, n={X.shape[0]})')

def load_realestate(train_size: int=320, test_size: int=94, seed: int=0) -> Dataset:
    (X, y, names) = _load_cached_or_openml('realestate.csv', '')
    return _split_standardise(X, y, names, train_size, test_size, seed, name='realestate', desc=f'UCI real estate valuation (d={X.shape[1]}, n={X.shape[0]})')

def load_diabetes_(train_size: int=320, test_size: int=100, seed: int=0) -> Dataset:
    db = load_diabetes()
    (X, y) = (np.asarray(db.data, float), np.asarray(db.target, float))
    return _split_standardise(X, y, list(db.feature_names), train_size, test_size, seed, name='diabetes', desc=f'sklearn diabetes (d={X.shape[1]}, n={X.shape[0]})')

def load_airfoil(train_size: int=1100, test_size: int=400, seed: int=0, degree: int=1) -> Dataset:
    """NASA airfoil self-noise (acoustic sensor) regression.

    degree=2 applies a polynomial (quadratic) basis expansion -- still linear in
    theta, hence a *convex* quadratic-regression testbed for Part 14.
    """
    (X, y, names) = _load_cached_or_openml('airfoil_self_noise.csv', 'airfoil_self_noise')
    if degree > 1:
        (X, names) = polynomial_expand(X, names, degree=degree)
    return _split_standardise(X, y, names, train_size, test_size, seed, name=f"airfoil{('_quad' if degree > 1 else '')}", desc=f'NASA airfoil self-noise (d={X.shape[1]}, n={X.shape[0]}, deg={degree})')

def load_concrete(train_size: int=750, test_size: int=250, seed: int=0, degree: int=1) -> Dataset:
    """Concrete compressive-strength (materials-sensor) regression."""
    (X, y, names) = _load_cached_or_openml('concrete.csv', 'Concrete_Data')
    if degree > 1:
        (X, names) = polynomial_expand(X, names, degree=degree)
    return _split_standardise(X, y, names, train_size, test_size, seed, name=f"concrete{('_quad' if degree > 1 else '')}", desc=f'concrete compressive strength (d={X.shape[1]}, n={X.shape[0]}, deg={degree})')

def load_casp(train_size: int=2800, test_size: int=1000, seed: int=0, degree: int=1) -> Dataset:
    """CASP protein tertiary-structure RMSD regression (9 physico-chemical features)."""
    (X, y, names) = _load_cached_or_openml('casp.csv', '')
    if degree > 1:
        (X, names) = polynomial_expand(X, names, degree=degree)
    return _split_standardise(X, y, names, train_size, test_size, seed, name=f"casp{('_quad' if degree > 1 else '')}", desc=f'CASP protein RMSD (d={X.shape[1]}, n={X.shape[0]}, deg={degree})')

def _download_uci_ct_slices(cache_dir: Path) -> Path:
    """Download and cache the UCI CT-slice dataset locally."""
    import io, zipfile, urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / 'slice_localization_data.csv'
    if csv_path.exists():
        return csv_path
    url = 'https://archive.ics.uci.edu/static/public/206/relative+location+of+ct+slices+on+axial+axis.zip'
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    csv_name = next((n for n in names if n.lower().endswith('.csv')), None)
    if csv_name is None:
        raise RuntimeError(f'No CSV in UCI CT-slices zip; got {names}')
    with z.open(csv_name) as fh:
        csv_path.write_bytes(fh.read())
    return csv_path

def load_ct_slices(train_size: int=1500, test_size: int=500, seed: int=0, cache_dir: Optional[str]=None) -> Dataset:
    """UCI 'Relative location of CT slices on axial axis'  (d~384, n=53,500).

    Task: predict slice position (0–180) on axial axis from features extracted
    from a CT scan slice.  Canonical medium-to-high-dim regression benchmark,
    cited in the robust-statistics line adjacent to regression poisoning.

    First call downloads from UCI and caches under  ~/.cache/influence_atlas_v2.
    """
    cdir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'influence_atlas_v2'
    csv_path = _download_uci_ct_slices(cdir)
    df = pd.read_csv(csv_path)
    label_candidates = [c for c in df.columns if c.lower() in ('reference', 'target', 'class', 'label')]
    label = label_candidates[0] if label_candidates else df.columns[-1]
    y = df[label].to_numpy(float)
    drop = [label] + [c for c in df.columns if c.lower() in ('patientid', 'id')]
    Xdf = df.drop(columns=drop)
    X = Xdf.to_numpy(float)
    return _split_standardise(X, y, list(Xdf.columns), train_size, test_size, seed, name='ct_slices', desc=f'UCI CT slices relative location (d={X.shape[1]}, n={X.shape[0]})')

def _download_uci_communities(cache_dir: Path) -> Path:
    """Download and cache the UCI Communities & Crime dataset."""
    import urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / 'communities.data'
    if p.exists():
        return p
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
    with urllib.request.urlopen(url, timeout=30) as resp:
        p.write_bytes(resp.read())
    return p

def load_communities(train_size: int=1500, test_size: int=500, seed: int=0, cache_dir: Optional[str]=None) -> Dataset:
    """UCI 'Communities and Crime' (d=122, n=1994).

    Predict 'ViolentCrimesPerPop' (last column) from the 122 socioeconomic /
    demographic features.  Used in the robust-regression and certified-defense
    line (Steinhardt/Liang's group uses a related variant for classification).
    The first 5 columns are non-predictive (state, county codes) so we drop them.
    """
    p = _DATA_DIR / 'communities.data'
    if not p.exists():
        cdir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'influence_atlas_v2'
        p = _download_uci_communities(cdir)
    df = pd.read_csv(p, header=None, na_values='?')
    df = df.drop(columns=[0, 1, 2, 3, 4])
    df = df.dropna(axis=1, thresh=int(0.95 * len(df)))
    df = df.dropna(axis=0)
    y = df.iloc[:, -1].to_numpy(float)
    Xdf = df.iloc[:, :-1]
    X = Xdf.to_numpy(float)
    return _split_standardise(X, y, [f'f{i}' for i in range(X.shape[1])], train_size, test_size, seed, name='communities', desc=f'UCI Communities & Crime (d={X.shape[1]}, n={X.shape[0]})')

def load_warfarin(path: str | Path, train_size: int=4000, test_size: int=1000, seed: int=0) -> Dataset:
    """IWPC Warfarin dose dataset (Jagielski 2018 canonical).

    Accepts either the original PharmGKB .xls (sheet "Subject Data") or a CSV.
    Target: 'Therapeutic Dose of Warfarin' (mg/week), log-transformed.

    Preprocessing follows Jagielski 2018:
      - Drop ID columns (PharmGKB Subject/Sample ID, etc.) and free-text comments
      - One-hot encode categorical columns
      - Median-impute remaining numerics
    """
    path = str(path)
    if path.endswith('.xls') or path.endswith('.xlsx'):
        df = pd.read_excel(path, sheet_name='Subject Data')
    else:
        df = pd.read_csv(path)
    target_candidates = ['Therapeutic Dose of Warfarin', 'Therapeutic_Dose_of_Warfarin', 'therapeutic_dose', 'dose']
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f'Could not find target column.  Looked for {target_candidates}. Available columns: {list(df.columns)[:30]}...')
    drop_cols = [c for c in df.columns if 'Subject ID' in c or 'Sample ID' in c or 'Comments' in c or (c == target_col) or ('QC' in c) or ('genotype:' in c.lower()) or (c in ['Project Site', 'Indication for Warfarin Treatment', 'Medications', 'Estimated Target INR Range Based on Indication', 'Cyp2C9 genotypes', 'Race (Reported)', 'Ethnicity (Reported)', 'Comorbidities', 'Age'])]
    if 'Age' in df.columns:

        def age_to_num(x):
            try:
                s = str(x).split('-')
                return (float(s[0]) + float(s[1])) / 2 if len(s) == 2 else float(s[0])
            except:
                return float('nan')
        df['Age_num'] = df['Age'].apply(age_to_num)
    df_out = df.dropna(subset=[target_col]).copy()
    y = np.log1p(df_out[target_col].astype(float).clip(lower=0).to_numpy())
    Xdf = df_out.drop(columns=drop_cols, errors='ignore')
    obj_cols = Xdf.select_dtypes(include=['object']).columns
    for c in obj_cols:
        if Xdf[c].nunique() > 10:
            top = Xdf[c].value_counts().head(9).index
            Xdf[c] = Xdf[c].where(Xdf[c].isin(top), other='Other')
    Xdf = pd.get_dummies(Xdf, drop_first=True, dummy_na=False)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0)
    X = Xdf.to_numpy(float)
    return _split_standardise(X, y, list(Xdf.columns), train_size, test_size, seed, name='warfarin', desc=f'IWPC Warfarin (d={X.shape[1]}, n={X.shape[0]})')

def load_loan(csv_path: str | Path, train_size: int=4000, test_size: int=1000, seed: int=0, target_col: str='int_rate', max_features: int=200) -> Dataset:
    """LendingClub Loan dataset (Jagielski 2018 canonical).

    Manual setup required:
      1. Download from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
      2. Save as influence_atlas_v2/data/loan.csv (or pass csv_path).

    Preprocessing:
      - target_col = 'int_rate' (interest rate, regression target) by default
      - One-hot encode categoricals
      - Restrict to first `max_features` columns to keep things tractable
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found.  Available: {list(df.columns)[:30]}...")
    y_raw = df[target_col].astype(str).str.rstrip('%').astype(float, errors='ignore')
    y = y_raw.to_numpy()
    df = df.drop(columns=[target_col])
    keep = []
    for c in df.columns:
        if df[c].dtype.kind in 'fiu':
            keep.append(c)
        elif df[c].nunique() < 20:
            keep.append(c)
    df = df[keep]
    df = pd.get_dummies(df, drop_first=True).fillna(0)
    if df.shape[1] > max_features:
        df = df.iloc[:, :max_features]
    mask = ~np.isnan(y)
    X = df.to_numpy(float)[mask]
    y = y[mask]
    return _split_standardise(X, y, list(df.columns), train_size, test_size, seed, name='loan', desc=f'LendingClub Loan (d={X.shape[1]}, n={X.shape[0]})')

def _download_uci_blogfeedback(cache_dir: Path) -> Path:
    """Download and cache the UCI Blog Feedback dataset (zipped)."""
    import io, zipfile, urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv = cache_dir / 'blogData_train.csv'
    if csv.exists():
        return csv
    url = 'https://archive.ics.uci.edu/static/public/304/blogfeedback.zip'
    with urllib.request.urlopen(url, timeout=45) as resp:
        data = resp.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    names = [n for n in z.namelist() if 'blogData_train' in n and n.endswith('.csv')]
    if not names:
        names = [n for n in z.namelist() if n.endswith('.csv')]
    if not names:
        raise RuntimeError(f'No CSV found in blog feedback zip: {z.namelist()[:5]}')
    with z.open(names[0]) as fh:
        csv.write_bytes(fh.read())
    return csv

def load_blogfeedback(train_size: int=1500, test_size: int=500, seed: int=0, cache_dir: Optional[str]=None) -> Dataset:
    """UCI 'BlogFeedback' (d=280, n=52,397 train + 7,624 test).

    Task: predict the number of comments a blog post will receive in the next
    24 hours, from 280 features describing the post and its history.  A common
    medium-high-d regression benchmark; in the same dimensional regime as the
    Jagielski-style house dataset, but with much larger n and clean numerical
    features (no categorical one-hot dummies).
    """
    p = _DATA_DIR / 'blogData_train.csv'
    if not p.exists():
        cdir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'influence_atlas_v2'
        p = _download_uci_blogfeedback(cdir)
    df = pd.read_csv(p, header=None)
    y = df.iloc[:, -1].to_numpy(float)
    y = np.log1p(np.clip(y, 0, None))
    X = df.iloc[:, :-1].to_numpy(float)
    return _split_standardise(X, y, [f'f{i}' for i in range(X.shape[1])], train_size, test_size, seed, name='blog_feedback', desc=f'UCI Blog Feedback (d={X.shape[1]}, n={X.shape[0]}); target log1p-transformed')

def load_california(train_size: int=4000, test_size: int=1000, seed: int=0) -> Dataset:
    csv = _DATA_DIR / 'california.csv'
    if csv.exists():
        df = pd.read_csv(csv)
        y = df['target'].to_numpy(float)
        Xdf = df.drop(columns=['target'])
        X = Xdf.to_numpy(float)
        names = list(Xdf.columns)
    else:
        cal = fetch_california_housing()
        (X, y, names) = (np.asarray(cal.data, float), np.asarray(cal.target, float), list(cal.feature_names))
    return _split_standardise(X, y, names, train_size, test_size, seed, name='california', desc=f'California housing (d={X.shape[1]}, n={X.shape[0]})')

def synthetic_toy_2d(n_train: int=120, n_test: int=120, sigma_x: float=0.05, sigma_y: float=0.08, theta_true: tuple=(1.2, -0.7), t_lo: float=-1.0, t_hi: float=1.0, seed: int=7) -> Dataset:
    """The features lie on the parabolic manifold  x(t) = (t, t^2),  perturbed by
    Gaussian noise.  This lets us visualise everything in 2D — the regression
    curve, the attacked points, the rotation of theta in parameter space, and
    the angle between l_F and l_R.

    Returns a Dataset with the extra attribute  feature_names = ['t','t^2']
    and a side-channel `description` listing  theta_true and the t-range.
    """
    rng = np.random.default_rng(seed)

    def gen(n):
        t = rng.uniform(t_lo, t_hi, size=n)
        X = np.stack([t, t ** 2], axis=1)
        X = X + sigma_x * rng.normal(size=X.shape)
        theta_t = np.array(theta_true, dtype=float)
        y = X @ theta_t + sigma_y * rng.normal(size=n)
        return (X.astype(float), y.astype(float), t.astype(float))
    (Xtr, ytr, ttr) = gen(n_train)
    (Xte, yte, tte) = gen(n_test)
    return Dataset(name='toy_2d', X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, feature_names=['t', 't^2'], y_mean=0.0, description=f'toy 2D parabolic manifold (n_tr={n_train}, n_te={n_test}, theta_true={tuple(theta_true)})')
DATA = str(_DATA_DIR)
DATASETS = {'realestate': lambda tr, te: load_realestate(tr, te, seed=0), 'california': lambda tr, te: load_california(tr, te, seed=0), 'communities': lambda tr, te: load_communities(tr, te, seed=0), 'concrete': lambda tr, te: load_concrete(tr, te, degree=2, seed=0), 'airfoil': lambda tr, te: load_airfoil(tr, te, degree=2, seed=0), 'casp': lambda tr, te: load_casp(tr, te, degree=2, seed=0), 'house': lambda tr, te: load_house(os.path.join(DATA, 'house-processed.csv'), tr, te, seed=0), 'warfarin': lambda tr, te: load_warfarin(os.path.join(DATA, 'warfarin.csv'), tr, te, seed=0), 'loan': lambda tr, te: load_loan(os.path.join(DATA, 'loan_sample.csv'), tr, te, seed=0), 'blog': lambda tr, te: load_blogfeedback(tr, te, seed=0)}
DEFAULT_SIZE = {'realestate': (320, 94), 'concrete': (780, 250), 'airfoil': (1100, 400), 'communities': (1500, 494), 'house': (840, 280), 'warfarin': (4000, 1000), 'loan': (2000, 800), 'california': (3000, 1000), 'casp': (4000, 1500), 'blog': (3000, 1000)}

@dataclass
class RidgeFit:
    theta: np.ndarray
    lam: float
    H: np.ndarray
    train_mse: float = float('nan')
    test_mse: float = float('nan')
    test_r2: float = float('nan')

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> RidgeFit:
    (n, d) = X.shape
    H = X.T @ X + n * lam * np.eye(d)
    theta = np.linalg.solve(H, X.T @ y)
    train_mse = float(np.mean((y - X @ theta) ** 2))
    return RidgeFit(theta=theta, lam=lam, H=H, train_mse=train_mse)

def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ theta

def mse(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    r = y - X @ theta
    return float(np.mean(r * r))

def eval_metrics(fit: RidgeFit, X_te: np.ndarray, y_te: np.ndarray) -> RidgeFit:
    yhat = X_te @ fit.theta
    fit.test_mse = float(np.mean((y_te - yhat) ** 2))
    ss_res = float(np.sum((y_te - yhat) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    fit.test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    return fit

def grad_mse_theta(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of test MSE w.r.t. theta:  (2/n) X^T (X theta - y)."""
    n = X.shape[0]
    return 2.0 / n * (X.T @ (X @ theta - y))

def cv_lambda(X: np.ndarray, y: np.ndarray, lambdas: Sequence[float], n_splits: int=5, seed: int=0) -> Tuple[Dict[float, float], float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out: Dict[float, float] = {}
    for lam in lambdas:
        mses: List[float] = []
        for (tr, va) in kf.split(X):
            fit = ridge_fit(X[tr], y[tr], lam)
            mses.append(mse(fit.theta, X[va], y[va]))
        out[float(lam)] = float(np.mean(mses))
    best = min(out.keys(), key=lambda L: out[L])
    return (out, float(best))

def A_i(fit: RidgeFit, X: np.ndarray, y: np.ndarray, i: int) -> np.ndarray:
    x_i = X[i]
    r_i = float(y[i] - x_i @ fit.theta)
    d = X.shape[1]
    B_i = r_i * np.eye(d) - np.outer(x_i, fit.theta)
    return np.linalg.solve(fit.H, B_i)

def stack_A(fit, X, y, S) -> np.ndarray:
    return np.concatenate([A_i(fit, X, y, idx) for idx in S], axis=1)

def influence_scores(fit, X, y, grad_F) -> np.ndarray:
    r = y - X @ fit.theta
    h = np.linalg.solve(fit.H, grad_F)
    xh = X @ h
    M = np.outer(r, h) - np.outer(xh, fit.theta)
    return np.linalg.norm(M, axis=1)

def ell_F(fit, X, y, S, grad_F) -> np.ndarray:
    return stack_A(fit, X, y, S).T @ grad_F

def B_matrix(fit, X, y, S, hess_F) -> np.ndarray:
    A_S = stack_A(fit, X, y, S)
    return A_S.T @ hess_F @ A_S

def H_map_matrix(X, y, lam, S, grad_F, eps=0.001, delta0=None):
    """Solution-map curvature  H_map = sum_k (grad F)_k d^2 theta_k/d delta^2   (md x md).

    Central-difference of the analytic covector  ell_F(delta) = A_S(delta)^T grad_F
    with grad_F FROZEN at the passed vector -- that freezing is what isolates H_map
    from the functional curvature B.  delta0 = point to differentiate around; pass the
    CURRENT iterate during relinearisation so the curvature tracks delta (otherwise it
    stays the delta=0 Hessian).  Cost 2*md refits; we difference the GRADIENT not F, so
    roundoff stays ~1e-11 (vs ~1e-8 for a 2nd difference of F itself).
    """
    d = X.shape[1]
    md = len(S) * d
    if delta0 is None:
        delta0 = np.zeros(md)

    def gbar(delta):
        (fit_d, Xp) = _refit_at_delta(X, y, lam, S, delta)
        return stack_A(fit_d, Xp, y, S).T @ grad_F
    Hm = np.zeros((md, md))
    for j in range(md):
        ej = np.zeros(md)
        ej[j] = eps
        Hm[:, j] = (gbar(delta0 + ej) - gbar(delta0 - ej)) / (2 * eps)
    return 0.5 * (Hm + Hm.T)

def full_curvature(fit, X, y, lam, S, grad_F, hess_F=None, eps=0.001, delta0=None):
    """Returns (B, H_map, B + H_map).  hess_F=None  =>  linear F  =>  B = 0."""
    md = len(S) * X.shape[1]
    Bm = B_matrix(fit, X, y, S, hess_F) if hess_F is not None else np.zeros((md, md))
    Hm = H_map_matrix(X, y, lam, S, grad_F, eps, delta0=delta0)
    return (Bm, Hm, Bm + Hm)

def _refit_at_delta(X, y, lam, S, delta) -> Tuple[RidgeFit, np.ndarray]:
    d = X.shape[1]
    Xp = X.copy()
    for (k, i) in enumerate(S):
        Xp[i] = Xp[i] + delta[k * d:(k + 1) * d]
    return (ridge_fit(Xp, y, lam), Xp)

def _deltaF_true(fit, X, y, lam, S, grad_F, delta) -> float:
    """Exact  Delta F = F(theta(delta)) - F(theta(0))  by refitting.  LINEAR F (grad_F a vector)."""
    F0 = float(grad_F @ fit.theta)
    (fit_d, _) = _refit_at_delta(X, y, lam, S, delta)
    return float(grad_F @ fit_d.theta) - F0

def make_target(kind, X_test, theta0=None, q=0.8, idx=None, c=0.0, Xt=None, yt=None, q_hi=1.0, fidx=None):
    """Return the triple (F_fn, grad_F_fn, hess_F_fn) describing a target functional F(theta):
        F_fn(theta) -> float ,  grad_F_fn(theta) -> (d,) = grad F ,  hess_F_fn(theta) -> (d,d) or None.
    For point SELECTION use the vector  grad_F_fn(theta0)  (constant for linear F).

    kind:
      'linear_topq': push the mean prediction of the top-q test points.
          F = gF^T theta ,  gF = mean of X_test rows with yhat >= quantile(q).
          LINEAR -> hess_F_fn = None  (B = 0).  Your current default (q=0.8, needs theta0).
      'feature_seg': push the mean prediction of test points whose FEATURE fidx lies in the band
          [quantile(q), quantile(q_hi)] of that feature (segment defined in INPUT space, not prediction).
          F = gF^T theta ,  gF = mean of the selected rows.  The membership set is FIXED (depends only on
          x, not theta) so F is EXACTLY LINEAR -> hess_F_fn = None (B_F = 0): a clean instrument in which
          the entire attack-surface curvature comes from the DATA structure B, never the functional.
      'feature_level': drive that same feature-segment's mean prediction to a level c.
          F = -1/2 (ubar^T theta - c)^2 ,  ubar = mean of feature-band rows.  NONLINEAR -> B_F != 0
          (this is the curved version of the feature segment).
      'level'      : drive a segment's mean prediction to a level c.
          F = -1/2 (ubar^T theta - c)^2 ,  ubar = mean(X_test[idx]).
          NONLINEAR (interior optimum at ubar^T theta = c) -> B != 0.
      'mse'        : MSE on a reference set (Xt, yt) treated as the target functional.
          F = (1/n) || Xt theta - yt ||^2 .  NONLINEAR (quadratic) -> B = (2/n) Xt^T Xt.
    """
    if kind == 'linear_topq':
        yhat = X_test @ theta0
        lo = np.quantile(yhat, q); hi = np.quantile(yhat, q_hi)
        mask = (yhat >= lo) & (yhat <= hi) if q_hi < 0.999 else (yhat >= lo)   # selectable quantile band [q, q_hi]
        if not mask.any():
            mask = yhat >= lo
        gF = X_test[mask].mean(0)
        return (lambda th: float(gF @ th), lambda th: gF, None)
    if kind in ('feature_seg', 'feature_level'):
        if fidx is None:                                                       # default: most influential feature
            fidx = int(np.argmax(np.abs(theta0)))
        vals = X_test[:, fidx]
        lo = np.quantile(vals, q); hi = np.quantile(vals, q_hi)
        mask = (vals >= lo) & (vals <= hi) if q_hi < 0.999 else (vals >= lo)   # band on FEATURE fidx
        if not mask.any():
            mask = vals >= lo
        ubar = X_test[mask].mean(0)
        if kind == 'feature_seg':
            return (lambda th: float(ubar @ th), lambda th: ubar, None)        # linear (B_F = 0)
        if c == 0.0:                                                           # default level = current + bump
            c = float(ubar @ theta0) + 6.0
        return (lambda th: -0.5 * (float(ubar @ th) - c) ** 2, lambda th: -(float(ubar @ th) - c) * ubar, lambda th: -np.outer(ubar, ubar))
    if kind == 'level':
        ubar = (X_test if idx is None else X_test[idx]).mean(0)
        return (lambda th: -0.5 * (float(ubar @ th) - c) ** 2, lambda th: -(float(ubar @ th) - c) * ubar, lambda th: -np.outer(ubar, ubar))
    if kind == 'mse':
        n = len(yt)
        return (lambda th: float(np.mean((Xt @ th - yt) ** 2)), lambda th: 2.0 / n * (Xt.T @ (Xt @ th - yt)), lambda th: 2.0 / n * (Xt.T @ Xt))
    raise ValueError(f'unknown target kind {kind!r}')

def filter_outlier(X, p=0.99, return_info=False):
    """Mahalanobis inlier filter. Returns indices of INLIERS (outliers removed).

    d_M(x)^2 = (x-μ)^T Σ^{-1} (x-μ) ~ χ²_d under Gaussianity  ⇒  keep d_M ≤ τ,
    τ = sqrt(χ²_d quantile p).  Restrict selection to these: max-influence points
    are often already outliers, so attacking inliers only is the stealth-correct set.
    """
    (n, d) = X.shape
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / max(n - 1, 1) + 0.001 * np.eye(d)
    L = np.linalg.cholesky(cov)
    Z = np.linalg.solve(L, Xc.T)
    d_M = np.sqrt((Z * Z).sum(0))
    tau = float(np.sqrt(chi2.ppf(p, df=d)))
    inlier_idx = np.where(d_M <= tau)[0]
    if return_info:
        return (inlier_idx, d_M, tau)
    return inlier_idx

def hmap_hvp(X, y, lam, S, gvec, v, eps=0.001):

    def gb(dl):
        (fd, Xp) = _refit_at_delta(X, y, lam, S, dl)
        return stack_A(fd, Xp, y, S).T @ gvec
    return (gb(eps * v) - gb(-eps * v)) / (2 * eps)

def true_hvp(X, y, lam, S, gF_fn, v, eps=0.001):

    def gt(dl):
        (fd, Xp) = _refit_at_delta(X, y, lam, S, dl)
        return stack_A(fd, Xp, y, S).T @ gF_fn(fd.theta)
    return (gt(eps * v) - gt(-eps * v)) / (2 * eps)

def specnorm_matvec(matvec, md, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(md)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        w = matvec(v)
        nrm = np.linalg.norm(w)
        if nrm < 1e-300:
            return 0.0
        s = nrm
        v = w / nrm
    return s

def proj_mse(delta, lR):
    """MSE-stealth cap ALONE: Euclidean projection onto the half-space  ℓ_Rᵀδ ≤ 0  (don't raise train MSE)."""
    s = lR @ delta
    return delta - s / (lR @ lR + 1e-18) * lR if s > 0 else delta

def _mse_curv_shrink(dl, lR, MR):
    b = float(lR @ dl)
    c = float(dl @ MR @ dl)
    if b + 0.5 * c <= 0:
        return dl
    if c > 1e-12 and b < 0:
        return min(1.0, -2 * b / c) * dl
    return 0.0 * dl

def cap_context(X, y, fit, S, p=0.99, mse_full=True):
    (n, d) = X.shape
    mu = X.mean(0)
    cov = (X - mu).T @ (X - mu) / max(n - 1, 1) + 0.001 * np.eye(d)
    L = np.linalg.cholesky(cov)
    tau = float(np.sqrt(chi2.ppf(p, df=d)))
    gMSE = grad_mse_theta(fit.theta, X, y)
    lR = stack_A(fit, X, y, S).T @ gMSE
    MR = full_curvature(fit, X, y, fit.lam, S, gMSE, 2.0 / n * (X.T @ X))[2] if mse_full else None
    Z = [np.linalg.solve(L, X[i] - mu) for i in S]
    return dict(S=list(S), d=d, L=L, tau=tau, lR=lR, MR=MR, mse_full=mse_full, Z=Z, mu=mu)

def apply_caps(delta, R, ctx, mse_cap=True, inlier_cap=True, iters=30):
    (S, d, L, tau, lR, Z) = (ctx['S'], ctx['d'], ctx['L'], ctx['tau'], ctx['lR'], ctx['Z'])
    MR = ctx.get('MR')
    mf = ctx.get('mse_full', False) and MR is not None
    out = delta.copy()

    def P_inl(v):
        v = v.copy()
        for k in range(len(S)):
            w = np.linalg.solve(L, v[k * d:(k + 1) * d])
            ww = w @ w
            if ww < 1e-18:
                continue
            zw = Z[k] @ w
            disc = zw * zw - ww * (Z[k] @ Z[k] - tau * tau)
            a = 1.0 if disc < 0 else min(1.0, max(0.0, (-zw + np.sqrt(disc)) / ww))
            v[k * d:(k + 1) * d] *= a
        return v
    P_ball = lambda v: v * (R / np.linalg.norm(v)) if np.linalg.norm(v) > R else v
    mse_val = lambda v: lR @ v + (0.5 * v @ MR @ v if mf else 0.0)

    def feasible(v):
        if mse_cap and mse_val(v) > 1e-07:
            return False
        if np.linalg.norm(v) > R + 1e-06:
            return False
        if inlier_cap:
            for k in range(len(S)):
                if np.linalg.norm(Z[k] + np.linalg.solve(L, v[k * d:(k + 1) * d])) > tau + 1e-06:
                    return False
        return True
    for _ in range(iters):
        if mse_cap:
            out = _mse_curv_shrink(out, lR, MR) if mf else proj_mse(out, lR)
        if inlier_cap:
            out = P_inl(out)
        out = P_ball(out)
        if feasible(out):
            break
    s = 1.0
    while not feasible(out * s) and s > 1e-09:
        s *= 0.9
    return out * s

def _eta_max_all(fit, X, y, S, lF, dirs=None, delta=None, p=0.99):
    """Per-point Mahalanobis travel cap eta_max,i along a unit ray u_i.
       default u_i = lF_i/||lF_i||   (TR1 influence ray; this is what eps_leave/eps_sat use).
       Pass dirs = stacked unit directions, OR delta = a stacked perturbation
       (then u_i = delta_i/||delta_i||), to get the cap along a SPECIFIC attack's tilted direction."""
    ctx = cap_context(X, y, fit, S, p, mse_full=False)
    (L, tau, Z, d) = (ctx['L'], ctx['tau'], ctx['Z'], ctx['d'])
    m = len(S)
    si = np.array([np.linalg.norm(lF[k * d:(k + 1) * d]) for k in range(m)])
    etas = []
    for k in range(m):
        uk = dirs[k * d:(k + 1) * d] if dirs is not None else delta[k * d:(k + 1) * d] if delta is not None else lF[k * d:(k + 1) * d]
        nu = np.linalg.norm(uk)
        if nu < 1e-12:
            etas.append(0.0)
            continue
        u = uk / nu
        wu = np.linalg.solve(L, u)
        uu = wu @ wu
        zw = Z[k] @ wu
        zz = Z[k] @ Z[k]
        disc = zw * zw - uu * (zz - tau * tau)
        etas.append(max((-zw + np.sqrt(disc)) / uu if disc >= 0 and uu > 1e-18 else 0.0, 0.0))
    return (np.array(etas), si)

def _headroom_all(fit, X, y, S, delta, p=0.99):
    """Realized Mahalanobis headroom h_i = tau_x - d_M(x_i + delta_i) for the ACTUAL attack delta."""
    ctx = cap_context(X, y, fit, S, p, mse_full=False)
    (L, tau, Z, d) = (ctx['L'], ctx['tau'], ctx['Z'], ctx['d'])
    return np.array([tau - np.linalg.norm(Z[k] + np.linalg.solve(L, delta[k * d:(k + 1) * d])) for k in range(len(S))])

def trs_max(l, H, R):
    """Global max of  q(δ)=lᵀδ + ½δᵀHδ  s.t. ||δ||≤R.
    Boundary KKT: (λI−H)δ = l with λ > λ_max(H); ||δ(λ)|| is strictly decreasing
    in λ, so one scalar bisection on the secular equation is exact. Interior max
    (δ = −H⁻¹l) only when H≺0. One eigh = O(md³), then cheap scalar root."""
    (w, U) = np.linalg.eigh(0.5 * (H + H.T))
    b = U.T @ l
    if w.max() < -1e-12:
        z = -b / w
        if np.linalg.norm(z) <= R:
            d = U @ z
            return (float(l @ d + 0.5 * d @ (H @ d)), d)
    nrm = lambda lam: np.sqrt(np.sum((b / (lam - w)) ** 2))
    lo = w.max() + 1e-09
    while nrm(lo) < R:
        lo = w.max() + (lo - w.max()) * 0.5 + 1e-12
    hi = w.max() + 1.0
    while nrm(hi) > R:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if nrm(mid) > R else (lo, mid)
    lam = 0.5 * (lo + hi)
    d = U @ (b / (lam - w))
    return (float(l @ d + 0.5 * d @ (H @ d)), d)

def tr_oneshot(fit, X, y, lam, S, R, grad_F, hess_F=None, order=3, mse_cap=True, inlier_cap=True, cap_p=0.99, mse_full=True):
    """One-shot TRk.
       TR1: analytic influence ray  delta = R * lF/||lF||   (no curvature, no trs_max).
       TR2: H = B_F = A_S^T (hess_F) A_S  only              (H_map NEVER computed).
       TR3: H = B_F + H_map.
       Caps applied to the analytic step."""
    lF = ell_F(fit, X, y, S, grad_F)
    md = lF.size
    nl = np.linalg.norm(lF)
    if order == 1:
        d = R * lF / (nl + 1e-18)
        H = None
    else:
        if order == 2:
            A_S = stack_A(fit, X, y, S)
            H = A_S.T @ hess_F @ A_S if hess_F is not None else np.zeros((md, md))
        else:
            (_, _, H) = full_curvature(fit, X, y, lam, S, grad_F, hess_F)
        (_, d) = trs_max(lF, H, R)
    if mse_cap or inlier_cap:
        ctx = cap_context(X, y, fit, S, cap_p, mse_full=mse_full)
        d = apply_caps(d, R, ctx, mse_cap, inlier_cap)
    q = float(lF @ d + (0.0 if H is None else 0.5 * d @ (H @ d)))
    return dict(delta=d, q_model=q, lF=lF, H=H, order=order)

def _estimate_L3(fit, X, y, lam, S, gF, R, hess_F=None, radii=(0.25, 0.5, 1.0, 2.0, 4.0), n_rand=2, seed=0):
    md = len(S) * X.shape[1]
    rng = np.random.default_rng(seed)

    def hess(delta):
        (fit_d, Xp) = _refit_at_delta(X, y, lam, S, delta)
        H = H_map_matrix(X, y, lam, S, gF, 0.001, delta0=delta)
        if hess_F is not None:
            A = stack_A(fit_d, Xp, y, S)
            H = H + A.T @ hess_F @ A
        return 0.5 * (H + H.T)
    H0 = hess(np.zeros(md))
    (_, _, BH) = full_curvature(fit, X, y, lam, S, gF, hess_F)
    d1 = tr_oneshot(fit, X, y, lam, S, R, gF, None, 1)['delta']
    d3 = tr_oneshot(fit, X, y, lam, S, R, gF, hess_F, 3)['delta']
    dirs = [d1, d3, np.linalg.eigh(BH)[1][:, -1]] + [rng.standard_normal(md) for _ in range(n_rand)]
    L3 = 0.0
    for u in dirs:
        u = u / (np.linalg.norm(u) + 1e-15)
        for r in radii:
            r = min(r, R)
            L3 = max(L3, np.linalg.norm(hess(r * u) - H0, 2) / r)
    return max(L3, 1e-09)

def tr3_relin(fit, X, y, lam, S, R, grad_F, hess_F=None, delta0=None, eta=0.1, max_outer=80, hmap_move=0.25, eps=0.001, tol=1e-09, k=24, mse_cap=True, inlier_cap=True, cap_p=0.99, mse_full=True, return_traj=False):
    """TR3-relin for a constant (linear) target gradient. Matrix-free Lanczos TR; hmap_move kept
       only for signature compatibility (unused -- curvature is now refreshed every step cheaply).
       return_traj=True also returns 'traj', the list of accepted δ_t (the relinearisation path)."""
    md = len(S) * X.shape[1]
    ctx = cap_context(X, y, fit, S, cap_p, mse_full=mse_full) if mse_cap or inlier_cap else None
    proj = (lambda d: apply_caps(d, R, ctx, mse_cap, inlier_cap)) if ctx is not None else lambda d: d * (R / np.linalg.norm(d)) if np.linalg.norm(d) > R else d
    delta = np.zeros(md) if delta0 is None else proj(np.asarray(delta0, float).copy())
    cur = _deltaF_true(fit, X, y, lam, S, grad_F, delta)
    (best_v, best_d) = (cur, delta.copy())
    traj = [delta.copy()]
    Delta = 0.2 * R
    for _ in range(max_outer):
        (fd, Xp) = _refit_at_delta(X, y, lam, S, delta)
        A_S = stack_A(fd, Xp, y, S)
        lF = A_S.T @ grad_F

        def gbar(d):
            (fdd, Xpp) = _refit_at_delta(X, y, lam, S, d)
            return stack_A(fdd, Xpp, y, S).T @ grad_F

        def Hv(v):
            hv = (gbar(delta + eps * v) - gbar(delta - eps * v)) / (2 * eps)
            if hess_F is not None:
                hv = hv + A_S.T @ (hess_F @ (A_S @ v))
            return hv
        step = _lanczos_tr(Hv, lF, Delta, k)
        p_eff = proj(delta + step) - delta
        pred = float(lF @ p_eff + 0.5 * p_eff @ Hv(p_eff))
        cand = delta + p_eff
        v = _deltaF_true(fit, X, y, lam, S, grad_F, cand)
        rho = (v - cur) / (pred + 1e-18)
        if v > best_v:
            (best_v, best_d) = (v, cand.copy())
        if rho < 0.25:
            Delta *= 0.5
        elif rho > 0.75 and np.linalg.norm(p_eff) > 0.9 * Delta:
            Delta = min(R, 2 * Delta)
        if rho > eta:
            if np.linalg.norm(cand - delta) < tol:
                break
            delta = cand
            cur = v
            traj.append(delta.copy())
        if Delta < tol * R:
            break
    out = dict(delta=best_d, val=best_v)
    if return_traj:
        out["traj"] = traj
    return out

def tr3_relin_nonlinear(fit, X, y, lam, S, R, grad_F_fn, F_fn, hess_F_fn=None, delta0=None, eta=0.1, max_outer=80, hmap_move=0.0, eps=0.001, tol=1e-09, k=24, mse_cap=True, inlier_cap=True, cap_p=0.99, mse_full=True):
    """TR3-relin for a theta-dependent (nonlinear) target. Matrix-free: the curvature
       M = H_map(delta) + B_F(theta) is refreshed EVERY step (equivalent to hmap_move=0) but is
       used only through Hessian-vector products in a Lanczos TR solve -- no md x md build."""
    md = len(S) * X.shape[1]
    ctx = cap_context(X, y, fit, S, cap_p, mse_full=mse_full) if mse_cap or inlier_cap else None
    proj = (lambda d: apply_caps(d, R, ctx, mse_cap, inlier_cap)) if ctx is not None else lambda d: d * (R / np.linalg.norm(d)) if np.linalg.norm(d) > R else d
    F0 = F_fn(fit.theta)
    delta = np.zeros(md) if delta0 is None else proj(np.asarray(delta0, float).copy())
    val = lambda d: F_fn(_refit_at_delta(X, y, lam, S, d)[0].theta) - F0
    cur = val(delta)
    (best_v, best_d) = (cur, delta.copy())
    Delta = 0.2 * R
    for _ in range(max_outer):
        (fd, Xp) = _refit_at_delta(X, y, lam, S, delta)
        th = fd.theta
        gF = grad_F_fn(th)
        A_S = stack_A(fd, Xp, y, S)
        lF = A_S.T @ gF
        HF = hess_F_fn(th) if hess_F_fn is not None else None

        def gbar(d):
            (fdd, Xpp) = _refit_at_delta(X, y, lam, S, d)
            return stack_A(fdd, Xpp, y, S).T @ gF

        def Hv(v):
            hv = (gbar(delta + eps * v) - gbar(delta - eps * v)) / (2 * eps)
            if HF is not None:
                hv = hv + A_S.T @ (HF @ (A_S @ v))
            return hv
        step = _lanczos_tr(Hv, lF, Delta, k)
        p_eff = proj(delta + step) - delta
        pred = float(lF @ p_eff + 0.5 * p_eff @ Hv(p_eff))
        cand = delta + p_eff
        v = val(cand)
        rho = (v - cur) / (pred + 1e-18)
        if v > best_v:
            (best_v, best_d) = (v, cand.copy())
        if rho < 0.25:
            Delta *= 0.5
        elif rho > 0.75 and np.linalg.norm(p_eff) > 0.9 * Delta:
            Delta = min(R, 2 * Delta)
        if rho > eta:
            if np.linalg.norm(cand - delta) < tol:
                break
            delta = cand
            cur = v
        if Delta < tol * R:
            break
    return dict(delta=best_d, val=best_v)

def bilevel_at_S(fit, X, y, lam, S, grad_F, R, n_iter=300, n_restarts=1, seed=0, F_fn=None, mse_cap=True, inlier_cap=True, cap_p=0.99, warm_seeds=None, step='rm', a=2.0, b=10.0, track_best=True, mse_full=True):
    """PGA on the capped feasible set. step='rm' uses a proper Robbins-Monro vanishing schedule
       eta_k = a*R/(b+k)  (sum eta = inf, sum eta^2 < inf), so on a concave objective PGA converges;
       step='legacy' is the old normalized 0.25R/(1+0.01k). track_best returns the best feasible
       iterate along the trajectory (the fair 'best-so-far' PGA)."""
    rng = np.random.default_rng(seed)
    md = len(S) * X.shape[1]
    ctx = cap_context(X, y, fit, S, cap_p, mse_full=mse_full) if mse_cap or inlier_cap else None
    proj = (lambda dl: apply_caps(dl, R, ctx, mse_cap, inlier_cap)) if ctx is not None else lambda dl: dl * (R / np.linalg.norm(dl)) if np.linalg.norm(dl) > R else dl
    gradf = grad_F if callable(grad_F) else lambda th: grad_F
    if F_fn is not None:
        F0 = F_fn(fit.theta)
        value = lambda d: F_fn(_refit_at_delta(X, y, lam, S, d)[0].theta) - F0
    else:
        value = lambda d: _deltaF_true(fit, X, y, lam, S, grad_F, d)
    starts = [np.zeros(md)]
    if warm_seeds:
        starts += [proj(np.asarray(w, float).copy()) for w in warm_seeds]
    starts += [proj(rng.standard_normal(md) * 0.001) for _ in range(max(0, n_restarts - len(starts)))]
    (best_v, best_d) = (0.0, np.zeros(md))
    for delta in (s.copy() for s in starts):
        (cur_v, cur_d) = (value(delta), delta.copy())
        for k in range(n_iter):
            (fit_d, Xp) = _refit_at_delta(X, y, lam, S, delta)
            g = stack_A(fit_d, Xp, y, S).T @ gradf(fit_d.theta)
            eta = a * R / (b + k) if step == 'rm' else 0.25 * R / (1 + 0.01 * k)
            delta = proj(delta + eta * g / (np.linalg.norm(g) + 1e-12))
            if track_best:
                v = value(delta)
                if v > cur_v:
                    (cur_v, cur_d) = (v, delta.copy())
        v_end = cur_v if track_best else value(delta)
        d_end = cur_d if track_best else delta
        if v_end > best_v:
            (best_v, best_d) = (v_end, d_end)
    return dict(delta=best_d, val=best_v)

def _lanczos_tr(Hv, lF, Delta, k=24, tol=1e-08):
    """Matrix-free trust-region subproblem:  max_{||s||<=Delta}  lF·s + 0.5 s·M·s,
       via k-step Lanczos on M (M supplied only through the HVP Hv(v)=M@v). Reproduces
       trs_max on the full matrix, at ~k HVPs instead of forming M (2*md refits)."""
    md = lF.size
    beta0 = np.linalg.norm(lF)
    if beta0 < 1e-18:
        return np.zeros(md)
    Q = np.zeros((md, k))
    alphas = []
    betas = []
    q = lF / beta0
    Q[:, 0] = q
    qprev = np.zeros(md)
    b = 0.0
    for j in range(k):
        w = Hv(q)
        a = float(q @ w)
        alphas.append(a)
        w = w - a * q - b * qprev
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        b = np.linalg.norm(w)
        if b < tol or j == k - 1:
            break
        qprev = q
        q = w / b
        Q[:, j + 1] = q
        betas.append(b)
    kk = len(alphas)
    T = np.diag(np.array(alphas, float))
    for i in range(kk - 1):
        T[i, i + 1] = T[i + 1, i] = betas[i]
    g = np.zeros(kk)
    g[0] = beta0
    (_, h) = trs_max(g, T, Delta)
    return Q[:, :kk] @ h

def _estimate_L3(fit, X, y, lam, S, gF, R, hess_F=None, radii=(0.25, 0.5, 1.0, 2.0, 4.0), n_pi=12, n_rand=2, seed=0, eps=0.001):
    """Hessian-Lipschitz constant  L3 = max_r ||M(r u) - M(0)||_2 / r,  estimated MATRIX-FREE.
       M = B_F + H_map is used only through Hessian-vector products (2 refits each); the operator
       norm of the difference M(r u) - M(0) is obtained by power iteration -- no md x md build.
       ~10x faster than forming the full Hessian at every point, same L3."""
    md = len(S) * X.shape[1]
    rng = np.random.default_rng(seed)
    z = np.zeros(md)

    def gbar(delta):
        (fd, Xp) = _refit_at_delta(X, y, lam, S, delta)
        return stack_A(fd, Xp, y, S).T @ gF

    def Hv(delta, v):
        hv = (gbar(delta + eps * v) - gbar(delta - eps * v)) / (2 * eps)
        if hess_F is not None:
            (fd, Xp) = _refit_at_delta(X, y, lam, S, delta)
            A = stack_A(fd, Xp, y, S)
            hv = hv + A.T @ (hess_F @ (A @ v))
        return hv
    v1 = rng.standard_normal(md)
    v1 /= np.linalg.norm(v1)
    for _ in range(n_pi):
        w = Hv(z, v1)
        nw = np.linalg.norm(w)
        if nw < 1e-14:
            break
        v1 = w / nw
    d1 = tr_oneshot(fit, X, y, lam, S, R, gF, None, 1)['delta']
    d1 = d1 / (np.linalg.norm(d1) + 1e-15)
    dirs = [v1, d1] + [rng.standard_normal(md) for _ in range(n_rand)]
    L3 = 1e-09
    for u in dirs:
        u = u / (np.linalg.norm(u) + 1e-15)
        for r in radii:
            r = min(r, R)
            v = rng.standard_normal(md)
            v /= np.linalg.norm(v)
            nw = 0.0
            for _ in range(n_pi):
                w = Hv(r * u, v) - Hv(z, v)
                nw = np.linalg.norm(w)
                if nw < 1e-14:
                    break
                v = w / nw
            L3 = max(L3, nw / r)
    return max(L3, 1e-09)

def top_perp(hvp, u1, md, iters=15, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(md)
    v -= v @ u1 * u1
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = hvp(v)
        w -= w @ u1 * u1
        nw = np.linalg.norm(w)
        if nw < 1e-300:
            break
        v = w / nw
    return v

def restrict2(mv, u1, u2):
    (Mu1, Mu2) = (mv(u1), mv(u2))
    return np.array([[u1 @ Mu1, u1 @ Mu2], [u2 @ Mu1, u2 @ Mu2]])

def curvature_3d_atlas(kind='linear_topq', M=6, Rm=3.0, lam=0.1, ngrid=21, names=None, **tkw):
    names = names or list(DATASETS)
    ok = []
    for name in names:
        (tr, te) = DEFAULT_SIZE.get(name, (400, 150))
        try:
            ok.append((name, DATASETS[name](tr, te)))
            print(f'  loaded {name}')
        except Exception as e:
            print(f'  {name} SKIPPED ({type(e).__name__}: {str(e)[:40]})')
    fig = plt.figure(figsize=(18, 3.4 * len(ok)))
    for (r, (name, ds)) in enumerate(ok):
        (X, y, Xte, yte) = (ds.X_train, ds.y_train, ds.X_test, ds.y_test)
        fit = ridge_fit(X, y, lam)
        d = X.shape[1]
        R = np.sqrt(M) * Rm
        md = M * d
        inl = filter_outlier(X)
        (F_fn, gF_fn, hF_fn) = make_target(kind, Xte, theta0=fit.theta, Xt=Xte, yt=yte, **tkw)
        gF = gF_fn(fit.theta)
        si = influence_scores(fit, X, y, gF)
        S = [int(inl[k]) for k in np.argsort(-si[inl])[:M]]
        lF = ell_F(fit, X, y, S, gF)
        nlF = np.linalg.norm(lF)
        u1 = lF / nlF
        hvpH = lambda v: hmap_hvp(X, y, lam, S, gF, v)
        u2 = top_perp(hvpH, u1, md)
        Bmv = (lambda v, A=stack_A(fit, X, y, S), HF=hF_fn(fit.theta): A.T @ (HF @ (A @ v))) if hF_fn is not None else lambda v: np.zeros(md)
        B2 = restrict2(Bmv, u1, u2)
        H2 = restrict2(hvpH, u1, u2)
        BH2 = B2 + H2
        gv = np.linspace(-R, R, ngrid)
        (Ag, Bg) = np.meshgrid(gv, gv)
        mask = Ag ** 2 + Bg ** 2 <= R ** 2
        quad = lambda Q: np.where(mask, 0.5 * (Q[0, 0] * Ag ** 2 + 2 * Q[0, 1] * Ag * Bg + Q[1, 1] * Bg ** 2), np.nan)
        Zlin = np.where(mask, nlF * Ag, np.nan)
        F0 = float(F_fn(fit.theta))
        Zt = np.full_like(Ag, np.nan)
        for i in range(ngrid):
            for j in range(ngrid):
                if mask[i, j]:
                    v = Ag[i, j] * u1 + Bg[i, j] * u2
                    Zt[i, j] = F_fn(_refit_at_delta(X, y, lam, S, v)[0].theta) - F0
        panels = [('ℓ_F·δ (linear)', Zlin), ('½δᵀBδ', quad(B2)), ('½δᵀH_map δ', quad(H2)), ('½δᵀ(B+H_map)δ', quad(BH2)), ('true ΔF', Zt)]
        for (c, (lab, Z)) in enumerate(panels):
            ax = fig.add_subplot(len(ok), 5, 5 * r + c + 1, projection='3d')
            ax.plot_surface(Ag, Bg, Z, cmap='viridis', linewidth=0, antialiased=True)
            ax.set_title(f'{name}\n{lab}', fontsize=8)
            ax.set_xlabel('ℓ_F', fontsize=7)
            ax.set_ylabel('⊥', fontsize=7)
            ax.tick_params(labelsize=5)
    plt.tight_layout()
    return fig

def tr3_overlay_atlas(kind='linear_topq', M=6, Rm=3.0, lam=0.1, ngrid=25, ncols=3, names=None, **tkw):
    """One panel per dataset: true ΔF (filled surface) with the TR3 additive model
    ℓ_F·δ + ½δᵀ(B+H_map)δ overlaid as a black wireframe grid. Where grid hugs the
    surface the 2nd-order model is exact; lift-off = cubic remainder."""
    names = names or list(DATASETS)
    ok = []
    for name in names:
        (tr, te) = DEFAULT_SIZE.get(name, (400, 150))
        try:
            ok.append((name, DATASETS[name](tr, te)))
            print(f'  loaded {name}')
        except Exception as e:
            print(f'  {name} SKIPPED ({type(e).__name__}: {str(e)[:40]})')
    ncols = min(ncols, len(ok))
    nrows = math.ceil(len(ok) / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows))
    for (p, (name, ds)) in enumerate(ok):
        (X, y, Xte, yte) = (ds.X_train, ds.y_train, ds.X_test, ds.y_test)
        fit = ridge_fit(X, y, lam)
        d = X.shape[1]
        R = np.sqrt(M) * Rm
        md = M * d
        inl = filter_outlier(X)
        (F_fn, gF_fn, hF_fn) = make_target(kind, Xte, theta0=fit.theta, Xt=Xte, yt=yte, **tkw)
        gF = gF_fn(fit.theta)
        si = influence_scores(fit, X, y, gF)
        S = [int(inl[k]) for k in np.argsort(-si[inl])[:M]]
        lF = ell_F(fit, X, y, S, gF)
        nlF = np.linalg.norm(lF)
        u1 = lF / nlF
        hvpH = lambda v: hmap_hvp(X, y, lam, S, gF, v)
        u2 = top_perp(hvpH, u1, md)
        Bmv = (lambda v, A=stack_A(fit, X, y, S), HF=hF_fn(fit.theta): A.T @ (HF @ (A @ v))) if hF_fn is not None else lambda v: np.zeros(md)
        B2 = restrict2(Bmv, u1, u2)
        H2 = restrict2(hvpH, u1, u2)
        BH2 = B2 + H2
        gv = np.linspace(-R, R, ngrid)
        (Ag, Bg) = np.meshgrid(gv, gv)
        mask = Ag ** 2 + Bg ** 2 <= R ** 2
        quad = lambda Q: np.where(mask, 0.5 * (Q[0, 0] * Ag ** 2 + 2 * Q[0, 1] * Ag * Bg + Q[1, 1] * Bg ** 2), np.nan)
        ZTR3 = np.where(mask, nlF * Ag, np.nan) + quad(BH2)
        F0 = float(F_fn(fit.theta))
        Zt = np.full_like(Ag, np.nan)
        for i in range(ngrid):
            for j in range(ngrid):
                if mask[i, j]:
                    v = Ag[i, j] * u1 + Bg[i, j] * u2
                    Zt[i, j] = F_fn(_refit_at_delta(X, y, lam, S, v)[0].theta) - F0
        gap = np.nanmax(np.abs(Zt - ZTR3))
        ax = fig.add_subplot(nrows, ncols, p + 1, projection='3d')
        ax.plot_surface(Ag, Bg, Zt, cmap='viridis', alpha=0.72, linewidth=0, antialiased=True)
        ax.plot_wireframe(Ag, Bg, ZTR3, color='k', linewidth=0.55, rstride=1, cstride=1)
        ax.set_title(f'{name}: true ΔF vs TR3 model\nmax cubic gap = {gap:.2f}', fontsize=9)
        ax.set_xlabel('ℓ_F', fontsize=7)
        ax.set_ylabel('⊥', fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f'TR3 reconstruction (grid) over true ΔF (surface) — {kind}', fontsize=12, y=1.0)
    plt.tight_layout()
    return fig

def _waterfill_nu(s, eta, R):
    """Smallest nu>0 with sum_i min(eta_i, s_i/nu)^2 = R^2 (monotone -> bisection).
       Returns inf when full saturation already fits the budget (sum eta^2 <= R^2)."""
    s = np.asarray(s, float)
    eta = np.asarray(eta, float)
    if np.sum(eta ** 2) <= R * R:
        return np.inf
    f = lambda nu: np.sum(np.minimum(eta, s / nu) ** 2) - R * R
    (lo, hi) = (1e-12, 1.0)
    while f(hi) > 0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def select_optimal_points(X, y, fit, grad_F, m, p=0.99, mode='cap', filter_outliers=None, eps_cov=0.001, R=None):
    """Select the m attacked points S*.

    mode = 'cap'    -> inlier + MSE aware: score = g~*eta_max, candidates = inliers.
    mode = 'inlier' -> inlier cap, along the INFLUENCE RAY u_i = l_i/||l_i|| (Theorem 7.8):
                       ceiling eta_i along u_i, and (if R given) the GLOBAL BUDGET is kept via
                       water-filling  t_i = min(eta_i, s_i/nu),  sum_i t_i^2 = R^2.
                       score = s_i * t_i  (= s_i*eta_i when R=None, i.e. budget slack / saturated).
    mode = 'budget' -> budget-only: score = ||l_i||_2, candidates = ALL n points.

    R : global L2 budget (sqrt(M)*eps). None -> inlier-saturated (budget slack).
    filter_outliers : None -> follow mode (cap/inlier => inliers, budget => all n).
    """
    (n, d) = X.shape
    theta = fit.theta
    H = fit.H
    gMSE = grad_mse_theta(theta, X, y)
    h = np.linalg.solve(H, grad_F)
    hR = np.linalg.solve(H, gMSE)
    r = y - X @ theta
    Xh = X @ h
    XhR = X @ hR
    L = np.outer(r, h) - np.outer(Xh, theta)
    LR = np.outer(r, hR) - np.outer(XhR, theta)
    mu = X.mean(0)
    Xc = X - mu
    Sig = Xc.T @ Xc / max(n - 1, 1) + eps_cov * np.eye(d)
    Sinv = np.linalg.inv(Sig)
    tau2 = float(chi2.ppf(p, df=d))
    tau = np.sqrt(tau2)
    LS = L @ Sig
    c = np.sum(LS * L, 1)
    a = np.sum(LS * LR, 1)
    b = np.sum(LR @ Sig * LR, 1) + 1e-18
    ap = np.maximum(a, 0.0)
    gtil = np.sqrt(np.maximum(c - ap * ap / b, 0.0))
    Dir = (L - (ap / b)[:, None] * LR) @ Sig
    U = Dir / np.maximum(np.linalg.norm(Dir, axis=1, keepdims=True), 1e-18)
    XS = Xc @ Sinv
    dM = np.sqrt(np.sum(XS * Xc, 1))
    SU = U @ Sinv
    aeta = np.sum(SU * U, 1)
    beta = np.sum(XS * U, 1)
    ceta = dM ** 2 - tau2
    disc = beta ** 2 - aeta * ceta
    eta = np.maximum(np.where(disc >= 0, (-beta + np.sqrt(np.maximum(disc, 0.0))) / aeta, 0.0), 0.0)
    ell_eucl = np.linalg.norm(L, axis=1)
    s = ell_eucl
    Uinf = L / np.maximum(s[:, None], 1e-18)
    aei = np.sum(Uinf @ Sinv * Uinf, 1)
    bei = np.sum(XS * Uinf, 1)
    dii = bei ** 2 - aei * (dM ** 2 - tau2)
    eta_ray = np.maximum(np.where(dii >= 0, (-bei + np.sqrt(np.maximum(dii, 0.0))) / aei, 0.0), 0.0)
    ci_inl = np.where(dM <= tau)[0]
    if mode == 'inlier' and R is not None:
        sel = ci_inl[np.argsort(-(s * eta_ray)[ci_inl])[:m]]
        nu = np.inf
        for _ in range(3):
            nu = _waterfill_nu(s[sel], eta_ray[sel], R)
            t = np.minimum(eta_ray, s / nu)
            sel = ci_inl[np.argsort(-(s * t)[ci_inl])[:m]]
        score_inlier = s * np.minimum(eta_ray, s / nu)
    else:
        score_inlier = s * eta_ray
    if mode == 'cap':
        score = gtil * eta
    elif mode == 'inlier':
        score = score_inlier
    elif mode == 'budget':
        score = ell_eucl
    else:
        raise ValueError(f"mode must be 'cap','inlier','budget', got {mode!r}")
    use_filter = mode in ('cap', 'inlier') if filter_outliers is None else bool(filter_outliers)
    cand = ci_inl if use_filter else np.arange(n)
    S = [int(i) for i in cand[np.argsort(-score[cand])[:m]]]
    return dict(S=S, gtilde=gtil, eta_max=eta, eta_ray=eta_ray, score=score, cand=cand, ell=L, ellR=LR, U=U, ell_eucl=ell_eucl, score_inlier=score_inlier, mode=mode)

def feasible_all(names=None, kind='linear_topq', M=3, lam=0.1, p=0.99, n=140, K=22000, ncols=5, seed=0, Rfac=1.35, q=0.8, nonlin_names=(), savedir=None, **tk):
    """Data-anchored feasible sets (real inlier ellipsoid + FULL MSE curvature quadric), one 3-D
       slice per dataset. Frame: u1=influence e (lF/||lF||), u2=MSE-normal (orth lR),
       u3=inlier-normal (orth Sinv(x-mu)); R=Rfac*||delta*_inlier-saturated||. Blue cloud = feasible
       poison displacements: all M blocks Mahalanobis-inlier AND lR.d + 0.5 d^T M_R d <= 0
       (M_R = B_R + H_map,R, the FULL MSE curvature), normalized by R. kept = feasible vol fraction."""
    import numpy as np, matplotlib.pyplot as plt, math, os
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import chi2

    def orth(v, B):
        for b in B:
            v = v - v @ b * b
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-09 else v * 0

    def one(name):
        ds = DATASETS[name](*DEFAULT_SIZE.get(name, (800, 300)))
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = ridge_fit(X, y, lam)
        d = X.shape[1]
        (F, gF_fn, hF) = make_target(kind, ds.X_test, theta0=fit.theta, q=q, Xt=ds.X_test, yt=ds.y_test, **tk)
        gF = gF_fn(fit.theta)
        S = select_optimal_points(X, y, fit, gF, M, p=p, mode='cap')['S']
        A = stack_A(fit, X, y, S)
        lF = A.T @ gF
        e = lF / np.linalg.norm(lF)
        gR = grad_mse_theta(fit.theta, X, y)
        lR = A.T @ gR
        (_, _, MR) = full_curvature(fit, X, y, lam, S, gR, 2 / len(X) * (X.T @ X))
        mu = X.mean(0)
        Sig = np.cov(X.T) + 0.001 * np.eye(d)
        Sinv = np.linalg.inv(Sig)
        tau2 = chi2.ppf(p, d)
        n0 = np.concatenate([Sinv @ (X[i] - mu) for i in S])
        n0 /= np.linalg.norm(n0)
        u1 = e
        u2 = orth(lR.copy(), [u1])
        u3 = orth(n0.copy(), [u1, u2])
        dstar = tr_oneshot(fit, X, y, lam, S, np.sqrt(M) * 8.0, gF, None, 1, mse_cap=False, inlier_cap=True)['delta']
        R = Rfac * np.linalg.norm(dstar) + 1e-06
        rng = np.random.default_rng(1)
        dirs = rng.standard_normal((K, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        abc = (R * rng.random(K) ** (1 / 3))[:, None] * dirs
        Xs = np.array([X[i] for i in S])
        feas = np.zeros(K, bool)
        for s in range(0, K, 4000):
            ab = abc[s:s + 4000]
            dl = ab[:, 0:1] * u1 + ab[:, 1:2] * u2 + ab[:, 2:3] * u3
            blk = dl.reshape(len(ab), M, d)
            zc = Xs[None] + blk - mu[None, None]
            m2 = np.einsum('kmd,de,kme->km', zc, Sinv, zc)
            mse_q = dl @ lR + 0.5 * np.einsum('ki,ij,kj->k', dl, MR, dl)
            feas[s:s + len(ab)] = (m2 <= tau2).all(1) & (mse_q <= 0)
        return (abc[feas] / R, d, feas.mean())
    names = names or list(DATASETS)
    nrows = math.ceil(len(names) / ncols)
    fig = plt.figure(figsize=(3.6 * ncols, 5.1 * nrows))
    for (k, nm) in enumerate(names):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection='3d')
        try:
            (P, d, kept) = one(nm)
            ss = 7 if len(P) < 900 else 3
            al = 0.3 if len(P) < 900 else 0.12
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=ss, c='#780606', alpha=al, linewidths=1)
            lab = 'nonlinear' if nm in nonlin_names else 'linear'
            ax.set_title(f'{nm} ({lab}, d={d})  kept={kept:.2f}', fontsize=9)
            print(f'  {nm:12s} d={d} feas={len(P)} kept={kept:.2f}', flush=True)
        except Exception as ex:
            ax.set_title(f'{nm}\nSKIP', fontsize=9, color='gray')
            print(f'  {nm:12s} SKIP ({type(ex).__name__}: {ex})', flush=True)
        ax.set_xlabel('influence e', fontsize=7)
        ax.set_ylabel('MSE-norm', fontsize=7)
        ax.set_zlabel('inlier-norm', fontsize=7)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(20, -60)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    if savedir:
        pth = os.path.join(savedir, 'feasible_all.png')
        fig.savefig(pth, dpi=115, bbox_inches='tight')
        print('saved', pth, flush=True)
    return fig

def _ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))

def _tilt_exact(lF, M, R):
    """Exact tilt (deg) of the TRS maximiser of l·d + ½ dᵀM d over ||d||<=R, via the secular equation."""
    (w, U) = np.linalg.eigh(0.5 * (M + M.T))
    b = U.T @ lF
    nl = np.linalg.norm(lF)
    wk = (b / nl) ** 2
    m2 = lambda lam: np.sum(wk / (lam - w) ** 2)
    lo = w.max() + 1e-09
    while nl * np.sqrt(m2(lo)) < R:
        lo = w.max() + (lo - w.max()) * 0.5 + 1e-12
    hi = w.max() + 1.0
    while nl * np.sqrt(m2(hi)) > R:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if nl * np.sqrt(m2(mid)) > R else (lo, mid)
    lam = 0.5 * (lo + hi)
    m1 = np.sum(wk / (lam - w))
    return float(np.degrees(np.arccos(np.clip(nl * m1 / R, -1, 1))))

def rho_table(kind='linear_topq', names=None, M=8, Rm=3.0, lam=0.1, n=350, seed=0, hess_F=None, sel_mode='cap', sel_p=0.99, **tk):
    """rho = R||M||/||l_F|| per dataset (bar chart + rho=1 line), with the tilt in the TABLE:
       phi_star = analytic exact tilt, tilt_deg = measured (trs_max). Prints progress per dataset."""
    names = names or list(DATASETS)
    rows = []
    for nm in names:
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            gF = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)[1](fit.theta)
            S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
            R = np.sqrt(M) * Rm
            lF = ell_F(fit, X, y, S, gF)
            nlF = np.linalg.norm(lF)
            (_, _, Mc) = full_curvature(fit, X, y, lam, S, gF, hess_F)
            Mc = 0.5 * (Mc + Mc.T)
            nM = float(np.linalg.norm(Mc, 2))
            rho = R * nM / nlF
            tilt = _ang(trs_max(lF, Mc, R)[1], lF)
            phis = _tilt_exact(lF, Mc, R)
            rows.append(dict(dataset=nm, rho=round(rho, 3), phi_star=round(phis, 1), tilt_deg=round(tilt, 1), normlF=round(nlF, 3), normM=round(nM, 3), R=round(R, 2), tilts=bool(rho >= 1)))
            print(f'  rho_table: {nm:12s} rho={rho:5.2f}  tilt={tilt:4.0f}°  (phi*={phis:.0f}°)', flush=True)
        except Exception as e:
            print(f'  rho_table: {nm:12s} SKIP ({type(e).__name__})', flush=True)
    df = pd.DataFrame(rows).sort_values('rho', ascending=False).reset_index(drop=True)
    (fig, ax) = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(df))
    ax.bar(x, df['rho'], color=['steelblue' if t else 'crimson' for t in df['tilts']])
    ax.axhline(1, ls='--', color='k', lw=1.4, label='$\\rho=1$ (tilt onset)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.set_ylabel('$\\rho = R\\|M\\|/\\|\\ell_F\\|$')
    ax.set_title('Curvature ratio $\\rho$   (blue: $\\rho\\geq1$ tilts,  red: $\\rho<1$)')
    ax.legend()
    plt.tight_layout()
    return (df, fig)
import numpy as np, math, pandas as pd, matplotlib.pyplot as plt

def _ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))

def method_tilt(kind='linear_topq', names=None, M=8, Rm=3.0, lam=0.1, n=400, seed=0, q=0.8, level_bump=6.0, sel_mode='cap', sel_p=0.99, cols=5):
    """Per dataset: attack DIRECTION for TR1, TR2, TR3 and how much each tilts off l_F.
       TR1 = along l_F (0 tilt);  TR2 adds the TARGET curvature B_F = A_S^T (grad^2 F) A_S;
       TR3 adds the DATA curvature H_map.  TR2 tilts only when grad^2 F has curvature off l_F
       (true for the 'mse' target; the 'level' target's B_F is rank-1 ALONG l_F, so TR2=0)."""
    names = names or list(DATASETS)
    nr = math.ceil(len(names) / cols)
    (fig, axes) = plt.subplots(nr, cols, figsize=(3.2 * cols, 3.2 * nr), squeeze=False)
    rows = []
    COL = {'TR1': '#1f6fff', 'TR2': '#e1a730', 'TR3': '#16a01a'}
    for (k, nm) in enumerate(names):
        ax = axes[k // cols][k % cols]
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            if kind == 'linear_topq':
                (F_fn, gF_fn, hF_fn) = make_target('linear_topq', ds.X_test, theta0=fit.theta, q=q)
            elif kind == 'level':
                yh = ds.X_test @ fit.theta
                idx = np.where(yh >= np.quantile(yh, q))[0]
                c = float(ds.X_test[idx].mean(0) @ fit.theta) + level_bump
                (F_fn, gF_fn, hF_fn) = make_target('level', ds.X_test, idx=idx, c=c)
            elif kind == 'mse':
                (F_fn, gF_fn, hF_fn) = make_target('mse', ds.X_test, Xt=ds.X_test, yt=ds.y_test)
            elif kind in ('feature_seg', 'feature_level'):
                (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, q=q)
            else:
                raise ValueError(f'unknown kind {kind!r}')
            gF = gF_fn(fit.theta)
            hessF = hF_fn(fit.theta) if hF_fn is not None else None
            S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
            R = np.sqrt(M) * Rm
            lF = ell_F(fit, X, y, S, gF)
            u1 = lF / np.linalg.norm(lF)
            (B, Hmap, Mc) = full_curvature(fit, X, y, lam, S, gF, hessF)
            d1 = trs_max(lF, np.zeros_like(Mc), R)[1]
            d2 = trs_max(lF, 0.5 * (B + B.T), R)[1]
            d3 = trs_max(lF, 0.5 * (Mc + Mc.T), R)[1]
            w = d3 - d3 @ u1 * u1
            u2 = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-09 else np.zeros_like(u1)
            ang2 = lambda d: np.arctan2(float(d @ u2), float(d @ u1))
            (t2, t3) = (_ang(d2, lF), _ang(d3, lF))
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=':', color='gray', lw=0.7))
            for (lbl, dd, rad) in [('TR1', d1, 1.0), ('TR2', d2, 0.94), ('TR3', d3, 1.0)]:
                a = ang2(dd)
                ax.annotate('', xy=(rad * np.cos(a), rad * np.sin(a)), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=COL[lbl], lw=2.4, alpha=0.9))
                ax.annotate(lbl, xy=((rad + 0.06) * np.cos(a), (rad + 0.06) * np.sin(a) + 0.03), fontsize=7.5, color=COL[lbl])
            ax.set_xlim(-0.2, 1.3)
            ax.set_ylim(-0.2, max(0.35, float(np.sin([ang2(d2), ang2(d3)]).max()) + 0.2))
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{nm}\nTR2 {t2:.0f}°, TR3 {t3:.0f}° off $\\ell_F$', fontsize=8.5)
            rows.append(dict(dataset=nm, tilt_TR1=0.0, tilt_TR2=round(t2, 1), tilt_TR3=round(t3, 1)))
            print(f'  method_tilt: {nm:12s} {kind:11s} TR2={t2:5.0f}°  TR3={t3:5.0f}°', flush=True)
        except Exception as e:
            ax.axis('off')
            ax.set_title(f'{nm}\nSKIP', fontsize=8, color='gray')
            print(f'  method_tilt: {nm:12s} SKIP ({type(e).__name__})', flush=True)
    for q_ in range(len(names), nr * cols):
        axes[q_ // cols][q_ % cols].axis('off')
    fig.suptitle(f'Attack-vector tilt by method ({kind}):  TR1 (=ℓ_F) → TR2 (+target curv) → TR3 (+data curv)', y=1.0, fontsize=10.5)
    plt.tight_layout()
    return (fig, pd.DataFrame(rows))

def eigen_alignment(kind='linear_topq', names=None, M=6, Rm=3.0, lam=0.1, n=350, seed=0, sel_mode='cap', sel_p=0.99, **tk):
    """Two rows per dataset. TOP: the curvature tilt as an ANGLE — ℓ_F (first-order/TR1 direction)
    vs the trust-region TR3 attack δ, with the φ* arc between them. BOTTOM: WHY — a stem of ℓ_F's
    energy w_k=⟨ℓ̂_F,u_k⟩² at each curvature μ_k, over the gain curve 1/(λ−μ). The trust region
    amplifies high-gain (high-μ) modes, so δ's energy centroid (red) shifts toward higher gain
    relative to ℓ_F's (blue) — that shift IS the φ* tilt. φ* is the measured angle ∠(ℓ_F, δ)."""
    from matplotlib.patches import FancyArrowPatch, Arc
    names = names or list(DATASETS)
    ok = []
    for nm in names:
        try:
            ok.append((nm, DATASETS[nm](*DEFAULT_SIZE.get(nm, (400, 150)))))
        except Exception:
            pass
    C = len(ok)
    (fig, axes) = plt.subplots(2, C, figsize=(4.5 * C, 7.4), squeeze=False, gridspec_kw=dict(height_ratios=[1.05, 1]))
    for (q, (nm, ds)) in enumerate(ok):
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            ii = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[ii], y[ii])
        fit = ridge_fit(X, y, lam)
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
        gF = gF_fn(fit.theta)
        S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
        R = np.sqrt(M) * Rm
        lF = ell_F(fit, X, y, S, gF)
        nlF = np.linalg.norm(lF) + 1e-18
        (B, Hmap, Mc) = full_curvature(fit, X, y, lam, S, gF, hF_fn(fit.theta) if hF_fn is not None else None)
        Mc = 0.5 * (Mc + Mc.T)
        (mu, U) = np.linalg.eigh(Mc)
        o = np.argsort(mu)[::-1]
        (mu, U) = (mu[o], U[:, o])
        a = U.T @ lF
        wk = (a / nlF) ** 2
        (_, dT) = trs_max(lF, Mc, R)
        ndT = np.linalg.norm(dT) + 1e-18
        wd = (U.T @ dT / ndT) ** 2
        ang = math.degrees(math.acos(np.clip(lF @ dT / (nlF * ndT), -1, 1)))
        try:
            ang_ex = float(tilt_exact(lF, Mc, R))
            lbl = f'φ*={ang:.0f}°  (exact {ang_ex:.0f}°)'
        except Exception:
            lbl = f'φ*={ang:.0f}°'
        lam_kkt = float(dT @ (lF + Mc @ dT) / (dT @ dT + 1e-18))
        lam_gain = max(lam_kkt, float(mu.max()) + 0.05 * float(mu.max() - mu.min() + 1e-09))
        (mu_lF, mu_dT) = (float(mu @ wk), float(mu @ wd))
        axt = axes[0][q]
        axt.set_aspect('equal')
        axt.axis('off')
        th = math.radians(ang)
        axt.add_patch(FancyArrowPatch((0, 0), (1, 0), arrowstyle='-|>', mutation_scale=18, lw=2.4, color='0.45'))
        axt.add_patch(FancyArrowPatch((0, 0), (math.cos(th), math.sin(th)), arrowstyle='-|>', mutation_scale=18, lw=2.4, color='#c0392b'))
        axt.add_patch(Arc((0, 0), 0.7, 0.7, theta1=0, theta2=max(ang, 0.5), color='k', lw=1.4))
        axt.text(0.4 * math.cos(th / 2), 0.4 * math.sin(th / 2) + 0.04, lbl, fontsize=13, fontweight='bold')
        axt.text(1.03, 0.0, 'ℓ_F\n(first-order /\nTR1 direction)', fontsize=8.5, va='center', color='0.3')
        axt.text(math.cos(th) * 1.05, math.sin(th) * 1.05, 'TR3 attack\n(curvature-tilted)', fontsize=8.5, va='bottom', color='#c0392b')
        axt.set_xlim(-0.25, 1.75)
        axt.set_ylim(-0.25, 1.35)
        axt.set_title(f'{nm}: curvature rotates the attack {ang:.0f}°', fontsize=10.5, fontweight='bold')
        axb = axes[1][q]
        axb.stem(mu, wk, linefmt='C0-', markerfmt='C0o', basefmt=' ')
        axb.set_xlabel('curvature  μ_k   (←flattening      sharpening→)')
        axb.set_ylabel('attack-direction energy  w_k')
        axb.set_ylim(0, 1.05)
        ax2 = axb.twinx()
        xx = np.linspace(mu.min(), mu.max(), 200)
        gg = 1.0 / (lam_gain - xx)
        gg = (gg - gg.min()) / (gg.max() - gg.min() + 1e-12)
        ax2.plot(xx, gg, '--', color='0.6', lw=1.3)
        ax2.set_ylim(-0.02, 1.05)
        ax2.set_ylabel('trust-region gain 1/(λ−μ)  (norm.)', color='0.5', fontsize=8)
        ax2.tick_params(labelcolor='0.5')
        axb.axvline(mu_lF, color='#2e86ab', lw=1.5, ls=':')
        axb.axvline(mu_dT, color='#c0392b', lw=1.5, ls=':')
        axb.annotate('', xy=(mu_dT, 0.9), xytext=(mu_lF, 0.9), arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=2))
        axb.text((mu_lF + mu_dT) / 2, 0.95, 'δ leans toward\nhigher gain', ha='center', fontsize=8, color='#c0392b')
        mtop = mu[np.argmax(wk)]
        kindw = 'flattening' if mtop < 0 else 'sharpening'
        axb.set_title(f'~{wk.max() * 100:.0f}% of ℓ_F on one {kindw} mode (μ={mtop:+.3f})', fontsize=9)
    fig.suptitle("Curvature tilt: the trust region reweights ℓ_F's energy by the gain 1/(λ−μ) → attack rotates by φ*", fontsize=12, y=1.0)
    plt.tight_layout()
    return fig
import numpy as np, math, matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.manifold import Isomap

def mahalanobis_info(X, p=0.99):
    (n, d) = X.shape
    Xc = X - X.mean(0)
    cov = Xc.T @ Xc / max(n - 1, 1) + 0.001 * np.eye(d)
    L = np.linalg.cholesky(cov)
    Z = np.linalg.solve(L, Xc.T)
    d_M = np.sqrt((Z * Z).sum(0))
    tau = float(np.sqrt(chi2.ppf(p, df=d)))
    return (d_M, tau, d_M <= tau)

def top_influence_isomap(kind='linear_topq', names=None, m=40, m_sweep=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20), n=400, k=10, lam=0.1, seed=0, p=0.99, **tk):
    """ISOMAP-2D. Points ranked by the BUDGET-ONLY selection (select_optimal_points mode='budget'
       = Euclidean ‖ℓ_i‖). Four groups: inlier (grey) · outlier (crimson) · top-m budget (navy ring) ·
       top-m AND outlier (gold star). Inset: % of top-m that are outliers vs m, base dashed."""
    names = names or list(DATASETS)
    cols = min(3, len(names))
    rows = math.ceil(len(names) / cols)
    (fig, axes) = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.8 * rows), squeeze=False)
    summ = {}
    for (pi, nm) in enumerate(names):
        ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = ridge_fit(X, y, lam)
        gF = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)[1](fit.theta)
        sel = select_optimal_points(X, y, fit, gF, m, p=p, mode='budget')
        s = sel['score']
        (dM, tau, inl) = mahalanobis_info(X, p)
        order = np.argsort(-s)
        top = order[:m]
        out = ~inl
        top_out = np.array([j for j in top if out[j]])
        E = Isomap(n_neighbors=k, n_components=2).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-09))
        ax = axes[pi // cols][pi % cols]
        ax.scatter(E[inl, 0], E[inl, 1], c='lightgray', s=10, alpha=0.55, label=f'inlier ({inl.sum()})')
        ax.scatter(E[out, 0], E[out, 1], c='crimson', s=20, alpha=0.75, label=f'outlier ({out.sum()})')
        ax.scatter(E[top, 0], E[top, 1], facecolors='none', edgecolors='navy', s=95, linewidths=1.5, label=f'top-{m} budget S*')
        if len(top_out):
            ax.scatter(E[top_out, 0], E[top_out, 1], marker='*', c='gold', edgecolors='k', s=190, linewidths=0.7, zorder=5, label=f'top-S* & outlier ({len(top_out)})')
        ms = [mm for mm in m_sweep if mm <= len(X)]
        fr = [100 * np.mean(out[order[:mm]]) for mm in ms]
        base = 100 * np.mean(out)
        ins = ax.inset_axes([0.6, 0.6, 0.37, 0.36])
        ins.plot(ms, fr, 'o-', color='crimson', ms=3.5, lw=1.4)
        ins.axhline(base, color='gray', ls='--', lw=1, label='base')
        ins.set_title('% of top-m that are outliers', fontsize=6)
        ins.tick_params(labelsize=5)
        ins.set_xlabel('m', fontsize=5.5)
        ins.set_ylim(0, 105)
        ins.legend(fontsize=5, loc='upper right')
        enrich = np.mean(out[top]) / max(np.mean(out), 1e-09)
        ax.set_title(f'{nm}: top-{m} outlier {100 * np.mean(out[top]):.0f}% vs base {base:.1f}%  ({enrich:.1f}× enrich)', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9)
        summ[nm] = fr
    for q in range(len(names), rows * cols):
        axes[q // cols][q % cols].axis('off')
    plt.tight_layout()
    return (fig, summ)

def influence_decay(kind='linear_topq', names=None, m=80, n=600, lam=0.1, seed=0, inliers_only=True, logy=True, **tk):
    """Sorted |influence| decay across ranked points (top-m, normalized, log-y).
    Metrics: top1/top10 share, participation ratio PR=(Σs)²/Σs² (effective # influential points),
    half-mass rank (points holding 50% of total |influence|)."""
    names = names or list(DATASETS)
    (fig, ax) = plt.subplots(figsize=(8, 5))
    rec = []
    cmap = plt.get_cmap('tab10')
    for (ci, nm) in enumerate(names):
        ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = ridge_fit(X, y, lam)
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
        s = np.abs(influence_scores(fit, X, y, gF_fn(fit.theta)))
        if inliers_only:
            s = s[filter_outlier(X)]
        s = np.sort(s)[::-1]
        mm = min(m, len(s))
        top = s[:mm]
        col = cmap(ci % 10)
        ax.plot(np.arange(1, mm + 1), top / top[0], color=col, lw=1.8, label=nm)
        cum = np.cumsum(s) / s.sum()
        PR = s.sum() ** 2 / (np.sum(s ** 2) + 1e-18)
        half = int(np.searchsorted(cum, 0.5)) + 1
        rec.append(dict(dataset=nm, top1_share=round(float(s[0] / s.sum()), 4), top10_share=round(float(s[:10].sum() / s.sum()), 3), eff_pts_PR=round(float(PR), 1), half_mass_rank=half))
    ax.set_yscale('log' if logy else 'linear')
    ax.set_xlabel('rank')
    ax.set_ylabel('|influence| / max')
    ax.set_title('Influence decay (top-m, normalized)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return (fig, pd.DataFrame(rec))

def combinatorial_selection_landscape(kind='linear_topq', names=None, M=6, Rm=3.0, lam=0.1, n=400, seed=0, n_rand=25, ncols=3, sel_mode='cap', sel_p=0.99, bil_iter=60, mse_full=True, caps='both', **tk):
    """Each point = one m-selection scored by the SAME CAPPED TR3 attack's (ΔF, ΔMSE); only the
       SELECTION differs. gray = random m-subsets; red ★ = S* (select_optimal_points, cap-aware);
       green ■ = bilevel-pick = the m inliers a cap-aware PGA over the whole inlier pool moves most.
       Efficient corner = strong ΔF (right) + stealthy ΔMSE (down). Selection AND evaluation use the
       SAME caps (caps='inlier' -> inlier only; caps='both' -> inlier+MSE) so the comparison is fair."""
    names = names or list(DATASETS)
    ok = []
    for nm in names:
        try:
            ok.append((nm, DATASETS[nm](*DEFAULT_SIZE.get(nm, (400, 150)))))
        except Exception:
            print(f'  {nm} load SKIP', flush=True)
    nr = math.ceil(len(ok) / ncols)
    (fig, axes) = plt.subplots(nr, ncols, figsize=(5 * ncols, 4 * nr), squeeze=False)
    rng = np.random.default_rng(seed)
    cp = dict(inlier_cap=True, mse_cap=caps == 'both')
    for (q, (nm, ds)) in enumerate(ok):
        ax = axes[q // ncols][q % ncols]
        try:
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                ii = rng.choice(len(X), n, replace=False)
                (X, y) = (X[ii], y[ii])
            fit = ridge_fit(X, y, lam)
            (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
            gF = gF_fn(fit.theta)
            hessM = hF_fn(fit.theta) if hF_fn is not None else None
            nonlin = hF_fn is not None
            inl = filter_outlier(X)
            R = np.sqrt(M) * Rm
            d = X.shape[1]
            F0 = F_fn(fit.theta)
            mse0 = mse(fit.theta, X, y)

            def ev(S):
                dl = tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, mse_full=mse_full, **cp)['delta']
                (fd, Xp) = _refit_at_delta(X, y, lam, S, dl)
                return (F_fn(fd.theta) - F0, mse(fd.theta, Xp, y) - mse0)
            pts = np.array([ev(list(rng.choice(inl, M, replace=False))) for _ in range(n_rand)])
            (dFs, dMs) = ev(select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S'])
            db = bilevel_at_S(fit, X, y, lam, list(inl), gF_fn if nonlin else gF, R, n_iter=bil_iter, n_restarts=1, F_fn=F_fn if nonlin else None, mse_full=False, **cp)['delta']
            per = np.array([np.linalg.norm(db[k * d:(k + 1) * d]) for k in range(len(inl))])
            Sbp = [int(inl[k]) for k in np.argsort(-per)[:M]]
            (dFb, dMb) = ev(Sbp)
            ax.scatter(pts[:, 0], pts[:, 1], c='lightgray', s=22, edgecolor='gray', linewidth=0.3, label='random m-subsets')
            ax.scatter([dFb], [dMb], marker='s', s=120, c='green', edgecolor='k', zorder=5, label='bilevel-pick (whole pool)')
            ax.scatter([dFs], [dMs], marker='*', s=320, c='red', edgecolor='k', zorder=6, label='S* (select_optimal_points)')
            ax.set_xlabel('targeted shift ΔF (stronger →)')
            ax.set_ylabel('ΔMSE (stealthier ↓)')
            ax.set_title(nm, fontsize=9)
            ax.grid(alpha=0.25)
            if q == 0:
                ax.legend(fontsize=7)
            Ss = set(map(int, select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']))
            print(f'  {nm:11s} S*(ΔF={dFs:.3f},ΔMSE={dMs:.3f}) bp(ΔF={dFb:.3f},ΔMSE={dMb:.3f}) |S*∩bp|={len(Ss & set(Sbp))}/{M}', flush=True)
        except Exception as e:
            ax.axis('off')
            ax.set_title(f'{nm}\nSKIP', fontsize=9, color='gray')
            print(f'  {nm:11s} SKIP ({type(e).__name__}: {str(e)[:60]})', flush=True)
    for q in range(len(ok), nr * ncols):
        axes[q // ncols][q % ncols].axis('off')
    fig.suptitle(f'Fixed-m combinatorial selection landscape (same CAPPED TR3 attack, caps={caps}; only selection differs)', fontsize=12, y=1.0)
    plt.tight_layout()
    return fig

def _ci(fit, X, gF):
    return X @ np.linalg.solve(fit.H, gF)

def ci_density(kind='linear_topq', names=None, lam=0.1, n=600, seed=0, **tk):
    names = names or list(DATASETS)
    (fig, ax) = plt.subplots(figsize=(9, 5.5))
    xs = np.linspace(-4, 4, 400)
    for nm in names:
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
            c = _ci(fit, X, gF_fn(fit.theta))[filter_outlier(X)]
            z = (c - c.mean()) / (c.std() + 1e-12)
            ax.plot(xs, gaussian_kde(z)(xs), lw=2, label=nm)
        except Exception as e:
            print(f'  {nm} SKIP {type(e).__name__}')
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k:', label='N(0,1)')
    ax.axvline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel('standardised $c_i$')
    ax.set_ylabel('density')
    ax.set_title('$c_i$ density (inliers) — skew ⇒ translation bias')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    return fig

def translation_rotation_split(kind='linear_topq', names=None, M=8, lam=0.1, n=500, seed=0, sel_mode='cap', sel_p=0.99, **tk):
    names = names or list(DATASETS)
    rows = []
    for nm in names:
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
            gF = gF_fn(fit.theta)
            S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
            c = _ci(fit, X, gF)[S]
            tot = np.sum(c ** 2) + 1e-18
            trans = np.sum(c) ** 2 / len(c)
            rows.append((nm, 100 * trans / tot, 100 * (tot - trans) / tot, c.mean() / (np.abs(c).mean() + 1e-12)))
        except Exception as e:
            print(f'  {nm} SKIP {type(e).__name__}')
    nm = [r[0] for r in rows]
    tr = [r[1] for r in rows]
    ro = [r[2] for r in rows]
    bc = [r[3] for r in rows]
    (fig, ax) = plt.subplots(figsize=(9, 5))
    x = np.arange(len(nm))
    ax.bar(x, tr, color='#1f77b4', label='translation %')
    ax.bar(x, ro, bottom=tr, color='#ff7f0e', label='rotation %')
    for (i, b) in enumerate(bc):
        ax.annotate(f'c̄/|c̄|={b:+.2f}', (x[i], 101), ha='center', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(nm, rotation=30, ha='right')
    ax.set_ylabel('%')
    ax.set_ylim(0, 108)
    ax.set_title('Sign decomposition on S*: translation vs rotation')
    ax.legend()
    plt.tight_layout()
    return (fig, rows)

def top_point_bias(kind='linear_topq', names=None, M=8, lam=0.1, n=600, seed=0, sel_mode='cap', sel_p=0.99, **tk):
    names = names or list(DATASETS)
    rows = []
    for nm in names:
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
            gF = gF_fn(fit.theta)
            inl = filter_outlier(X)
            c = _ci(fit, X, gF)
            cs = (c - c[inl].mean()) / (c[inl].std() + 1e-12)
            S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
            rows.append((nm, float(np.abs(cs[inl]).mean()), float(np.abs(cs[S]).mean())))
        except Exception as e:
            print(f'  {nm} SKIP {type(e).__name__}')
    nm = [r[0] for r in rows]
    allc = [r[1] for r in rows]
    topc = [r[2] for r in rows]
    (fig, ax) = plt.subplots(figsize=(9, 5))
    x = np.arange(len(nm))
    w = 0.38
    ax.bar(x - w / 2, allc, w, label='all inliers  mean|c_std|', color='#888')
    ax.bar(x + w / 2, topc, w, label='top S*  mean|c_std|', color='#c44')
    ax.set_xticks(x)
    ax.set_xticklabels(nm, rotation=30, ha='right')
    ax.set_ylabel('mean |standardised c|')
    ax.set_title('Are top points more sign-biased? (S* vs population)')
    ax.legend()
    plt.tight_layout()
    return (fig, rows)

def tau_table(names=None, p=0.99, n=None, ridge=1e-06):
    """Per-dataset inlier ceiling τ_x = sqrt(χ²_d.ppf(p)) plus outlier counts."""
    from scipy.stats import chi2
    names = names or list(DATASETS)
    rec = []
    for nm in names:
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            X = ds.X_train
            if n and len(X) > n:
                X = X[np.random.default_rng(0).choice(len(X), n, replace=False)]
            d = X.shape[1]
            mu = X.mean(0)
            Sig = np.cov(X, rowvar=False) + ridge * np.eye(d)
            L = np.linalg.cholesky(Sig)
            dM = np.sqrt((np.linalg.solve(L, (X - mu).T) ** 2).sum(0))
            tau = float(np.sqrt(chi2.ppf(p, d)))
            inl = dM <= tau
            rec.append(dict(dataset=nm, n=len(X), d=d, tau_x=round(tau, 3), n_outlier=int((~inl).sum()), pct_outlier=round(100 * np.mean(~inl), 2), max_dM=round(float(dM.max()), 2)))
        except Exception as e:
            rec.append(dict(dataset=nm, note=f'{type(e).__name__}: {e}'))
    return pd.DataFrame(rec)
from scipy.stats import skew as _skew

def _influence_covectors(fit, X, y, gF):
    h = np.linalg.solve(fit.H, gF)
    return np.outer(y - X @ fit.theta, h) - np.outer(X @ h, fit.theta)

def eta_max_along(X, S, delta, p=0.99, eps_cov=0.001):
    """Per-point inlier ceiling along the ACTUAL attack direction u_i = delta_i/||delta_i||,
    in the Mahalanobis metric M = Sigma^{-1}.  Returns (eta_max[m], feasible[m], U[m,d]).
    Solves  a eta^2 + 2b eta + c <= 0 with a=u'Mu, b=x'Mu, c=x'Mx - tau^2 (no explicit inverse)."""
    (n, d) = X.shape
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / max(n - 1, 1) + eps_cov * np.eye(d)
    L = np.linalg.cholesky(cov)
    tau2 = float(chi2.ppf(p, df=d))
    Minner = lambda a, b: np.linalg.solve(L, a) @ np.linalg.solve(L, b)
    eta = np.full(len(S), np.nan)
    feas = np.zeros(len(S), bool)
    U = np.zeros((len(S), d))
    for (kk, i) in enumerate(S):
        xi = Xc[i]
        di = delta[kk * d:(kk + 1) * d]
        nu = np.linalg.norm(di)
        if nu < 1e-12:
            continue
        u = di / nu
        U[kk] = u
        a = Minner(u, u)
        b = Minner(xi, u)
        c = Minner(xi, xi) - tau2
        disc = b * b - a * c
        feas[kk] = disc >= 0
        eta[kk] = max((-b + np.sqrt(disc)) / a, 0.0) if disc >= 0 else 0.0
    return (eta, feas, U)

def eta_influence_heatmap(kind='linear_topq', names=None, lam=0.1, n=800, seed=0, ncols=5, topk=20, **tk):
    """2D density of (log10|s_i|, eta_i^max) over inliers, top-influence points starred.
       eta_i^max = inlier travel ceiling along the point's own influence ray l_i."""
    names = names or list(DATASETS)
    ok = []
    for nm in names:
        try:
            ok.append((nm, DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))))
        except Exception:
            print(f'  {nm} load SKIP', flush=True)
    nr = math.ceil(len(ok) / ncols)
    (fig, axes) = plt.subplots(nr, ncols, figsize=(4 * ncols, 3.4 * nr), squeeze=False)
    for (q, (nm, ds)) in enumerate(ok):
        ax = axes[q // ncols][q % ncols]
        try:
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            gF = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)[1](fit.theta)
            s = influence_scores(fit, X, y, gF)
            L = _influence_covectors(fit, X, y, gF)
            (eta, _, _) = eta_max_along(X, np.arange(len(X)), L.reshape(-1))
            inl = filter_outlier(X)
            (si, ei) = (s[inl], eta[inl])
            m = np.isfinite(ei) & (si > 0)
            ls = np.log10(si[m] + 1e-12)
            ax.hist2d(ls, ei[m], bins=28, cmap='viridis')
            top = inl[np.argsort(-s[inl])[:topk]]
            ft = np.isfinite(eta[top])
            ax.scatter(np.log10(s[top][ft] + 1e-12), eta[top][ft], marker='*', c='yellow', s=30, edgecolor='k', linewidth=0.2)
            ax.set_xlabel('log10 |s_i|')
            ax.set_ylabel('$\\eta_i^{\\max}$')
            ax.set_title(f'{nm} (d={X.shape[1]})', fontsize=8)
            print(f'  {nm:11s} done', flush=True)
        except Exception as e:
            ax.axis('off')
            print(f'  {nm:11s} SKIP ({type(e).__name__})', flush=True)
    for q in range(len(ok), nr * ncols):
        axes[q // ncols][q % ncols].axis('off')
    fig.suptitle('2D density of $(\\log_{10}|s_i|,\\ \\eta_i^{\\max})$ — top-influence stars overlaid', fontsize=12, y=1.0)
    plt.tight_layout()
    return fig

def eta_max_distribution(kind='linear_topq', names=None, lam=0.1, n=1500, seed=0, ncols=5, **tk):
    """Per-dataset distribution of eta_i^max over inliers. Right-skewed, shaped by the data
       covariance; NO point mass at 0 (inliers always have eta>0)."""
    names = names or list(DATASETS)
    ok = []
    for nm in names:
        try:
            ok.append((nm, DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))))
        except Exception:
            print(f'  {nm} load SKIP', flush=True)
    nr = math.ceil(len(ok) / ncols)
    (fig, axes) = plt.subplots(nr, ncols, figsize=(4 * ncols, 3.4 * nr), squeeze=False)
    for (q, (nm, ds)) in enumerate(ok):
        ax = axes[q // ncols][q % ncols]
        try:
            (X, y) = (ds.X_train, ds.y_train)
            if len(X) > n:
                i = np.random.default_rng(seed).choice(len(X), n, replace=False)
                (X, y) = (X[i], y[i])
            fit = ridge_fit(X, y, lam)
            gF = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)[1](fit.theta)
            L = _influence_covectors(fit, X, y, gF)
            (eta, _, _) = eta_max_along(X, np.arange(len(X)), L.reshape(-1))
            inl = filter_outlier(X)
            e = eta[inl]
            e = e[np.isfinite(e)]
            ax.hist(e, bins=30, color='steelblue', edgecolor='k', linewidth=0.3)
            ax.axvline(np.median(e), color='crimson', ls='--', lw=1.2)
            ax.set_xlabel('$\\eta_i^{\\max}$')
            ax.set_ylabel('count')
            ax.set_title(f'{nm} (d={X.shape[1]})\nmedian={np.median(e):.2f}, skew={_skew(e):.2f}', fontsize=8)
            print(f'  {nm:11s} median={np.median(e):.2f} skew={_skew(e):.2f}', flush=True)
        except Exception as e2:
            ax.axis('off')
            print(f'  {nm:11s} SKIP ({type(e2).__name__})', flush=True)
    for q in range(len(ok), nr * ncols):
        axes[q // ncols][q % ncols].axis('off')
    fig.suptitle('$\\eta_i^{\\max}$ distribution per dataset (inliers) — right-skewed, no point mass at 0', fontsize=12, y=1.0)
    plt.tight_layout()
    return fig
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.manifold import Isomap

def _covec(fit, X, y, gF):
    h = np.linalg.solve(fit.H, gF)
    return np.outer(y - X @ fit.theta, h) - np.outer(X @ h, fit.theta)

def ceiling_approach_3d(name='diabetes', kind='linear_topq', M=8, Rm=3.0, lam=0.1, n=500, seed=0, methods=('TR1', 'TR3', 'bilevel'), relin_outer=15, bil_iter=120, sel_mode='cap', sel_p=0.99, eps_cov=0.001, **tk):
    """z = inlier headroom  h(x) = tau - d_M(x)  (Mahalanobis; ceiling h=0 plane on top). For each
       method the attacked points drop their headroom: green star BEFORE (h at x_i) -> salmon dot
       AFTER (h at x_i+delta_i), purple stem. Direction-free & EXACT (robust to delta_i=0, so TR3's
       concentrated/tangential moves are shown correctly). Dispatches relin/bilevel by target
       linearity; MSE cap auto-off for kind='mse'."""
    ds = DATASETS[name](*DEFAULT_SIZE.get(name, (500, 150)))
    (X, y) = (ds.X_train, ds.y_train)
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False)
        (X, y) = (X[i], y[i])
    fit = ridge_fit(X, y, lam)
    if kind == 'mse':
        (F_fn, gF_fn, hF_fn) = make_target('mse', ds.X_test, Xt=ds.X_test, yt=ds.y_test)
    elif kind == 'level':
        yh = ds.X_test @ fit.theta
        idx = np.where(yh >= np.quantile(yh, tk.get('q', 0.8)))[0]
        c = float(ds.X_test[idx].mean(0) @ fit.theta) + tk.get('level_bump', 6.0)
        (F_fn, gF_fn, hF_fn) = make_target('level', ds.X_test, idx=idx, c=c)
    else:
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, **tk)
    gF = gF_fn(fit.theta)
    hessM = hF_fn(fit.theta) if hF_fn is not None else None
    nl = hF_fn is not None
    S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
    R = np.sqrt(M) * Rm
    d = X.shape[1]
    cp = dict(mse_cap=kind != 'mse', inlier_cap=True)

    def sol(mn):
        if mn == 'TR1':
            return tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, **cp)['delta']
        if mn == 'TR2':
            return tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 2, **cp)['delta']
        if mn == 'TR3':
            return tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, **cp)['delta']
        if mn == 'TR3-relin':
            return tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, F_fn, hF_fn, max_outer=relin_outer, **cp)['delta'] if nl else tr3_relin(fit, X, y, lam, S, R, gF, None, max_outer=relin_outer, **cp)['delta']
        if mn == 'bilevel':
            return bilevel_at_S(fit, X, y, lam, S, gF_fn if nl else gF, R, n_iter=bil_iter, n_restarts=2, F_fn=F_fn if nl else None, **cp)['delta']
        raise ValueError(f'unknown method {mn!r}')
    mu = X.mean(0)
    cov = (X - mu).T @ (X - mu) / max(len(X) - 1, 1) + eps_cov * np.eye(d)
    Lc = np.linalg.cholesky(cov)
    tau = float(np.sqrt(chi2.ppf(sel_p, df=d)))
    head = lambda pts: tau - np.sqrt((np.linalg.solve(Lc, (pts - mu).T) ** 2).sum(0))
    head_all = head(X)
    inl = np.where(head_all >= 0)[0]
    P = Isomap(n_neighbors=8, n_components=2).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-09))
    zbase = float(head_all[inl].max()) * 1.05 + 1e-09
    fig = plt.figure(figsize=(6.5 * len(methods), 5.5))
    for (mi, mn) in enumerate(methods):
        ax = fig.add_subplot(1, len(methods), mi + 1, projection='3d')
        try:
            dl = sol(mn)
            Xp_S = np.array([X[S[k]] + dl[k * d:(k + 1) * d] for k in range(M)])
            eb = head_all[S]
            ea = head(Xp_S)
            ax.scatter(P[inl, 0], P[inl, 1], head_all[inl], c=head_all[inl], cmap='viridis', s=8, alpha=0.35)
            (xx, yy) = np.meshgrid(np.linspace(P[:, 0].min(), P[:, 0].max(), 2), np.linspace(P[:, 1].min(), P[:, 1].max(), 2))
            ax.plot_surface(xx, yy, np.zeros_like(xx), color='crimson', alpha=0.12)
            for (k, i) in enumerate(S):
                ax.plot([P[i, 0], P[i, 0]], [P[i, 1], P[i, 1]], [eb[k], ea[k]], color='purple', lw=1.5, zorder=5)
                ax.scatter([P[i, 0]], [P[i, 1]], [eb[k]], marker='*', c='green', s=80, edgecolor='k', zorder=6)
                ax.scatter([P[i, 0]], [P[i, 1]], [ea[k]], marker='o', c='salmon', s=70, edgecolor='k', zorder=6)
            ax.set_zlim(zbase, min(0.0, float(ea.min())))
            ax.set_xlabel('ISOMAP-1')
            ax.set_ylabel('ISOMAP-2')
            ax.set_zlabel('headroom $\\tau - d_M$')
            ax.set_title(f'{mn}: approach to ceiling', fontsize=10)
            print(f'  {name}/{kind} {mn:10s}: mean headroom {eb.mean():.3f} -> {ea.mean():.3f}', flush=True)
        except Exception as e:
            ax.axis('off')
            ax.set_title(f'{mn}\nSKIP', fontsize=9, color='gray')
            print(f'  {mn} SKIP ({type(e).__name__})', flush=True)
    fig.suptitle(f'{name} ({kind}): poisoning = approach to the inlier ceiling h=0  (★ before → ● after)', fontsize=12, y=1.0)
    plt.tight_layout()
    return fig
import math

def mse_angular(names=None, kind='linear_topq', M=8, Rm=2.5, lam=0.1, n=350, seed=0, ncols=5, sel_mode='cap', sel_p=0.99, **tk):
    """ℓ_F vs ℓ_R drawn in their 2-plane with the MSE half-space ℓ_Rᵀδ≤0. The exact linear MSE-cap
    cost is R‖ℓ_F‖(1−sinφ) for φ=∠(ℓ_F,ℓ_R) when cosφ>0, and 0 (FREE) when cosφ≤0."""
    names = names or list(DATASETS)
    ok = [(nm, DATASETS[nm](*DEFAULT_SIZE.get(nm, (300, 120)))) for nm in names]
    nr = math.ceil(len(ok) / ncols)
    (fig, axes) = plt.subplots(nr, ncols, figsize=(4.6 * ncols, 4.4 * nr), squeeze=False)
    for (q, (nm, ds)) in enumerate(ok):
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = ridge_fit(X, y, lam)
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
        gF = gF_fn(fit.theta)
        S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
        R = np.sqrt(M) * Rm
        SA = stack_A(fit, X, y, S)
        lF = SA.T @ gF
        lR = SA.T @ grad_mse_theta(fit.theta, X, y)
        (cF, cR) = (lF / np.linalg.norm(lF), lR / np.linalg.norm(lR))
        cos = float(cF @ cR)
        phi = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        cost = R * np.linalg.norm(lF) * (1 - np.sqrt(max(1 - max(cos, 0) ** 2, 0)))
        ax = axes[q // ncols][q % ncols]
        ax.set_aspect('equal')
        ax.axhline(0, color='0.8', lw=0.5)
        ax.axvline(0, color='0.8', lw=0.5)
        from matplotlib.patches import FancyArrowPatch, Arc
        th = np.radians(phi)
        ax.add_patch(FancyArrowPatch((0, 0), (1, 0), arrowstyle='-|>', mutation_scale=16, lw=2.4, color='#2e86ab'))
        ax.add_patch(FancyArrowPatch((0, 0), (np.cos(th), np.sin(th)), arrowstyle='-|>', mutation_scale=16, lw=2.4, color='#c0392b'))
        ax.add_patch(Arc((0, 0), 0.5, 0.5, theta1=0, theta2=phi, color='k', lw=1.2))
        ax.text(1.02, 0, 'ℓ_F', color='#2e86ab', fontsize=10)
        ax.text(np.cos(th) * 1.03, np.sin(th) * 1.03, 'ℓ_R (MSE)', color='#c0392b', fontsize=10)
        ax.set_xlim(-1.2, 1.4)
        ax.set_ylim(-1.2, 1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{nm}: φ={phi:.0f}°, cos={cos:+.2f}', fontsize=9)
    for q in range(len(ok), nr * ncols):
        axes[q // ncols][q % ncols].axis('off')
    plt.tight_layout()
    return fig

def _mse_full_shrink(X, y, lam, S, dl, mse0, ref='clean'):
    """EXACT full-curvature MSE cap (all orders, via refit): shrink δ→αδ until true ΔRisk ≤ 0."""
    a = 1.0
    for _ in range(40):
        (fd, Xp) = _refit_at_delta(X, y, lam, S, a * dl)
        Xref = X if ref == 'clean' else Xp
        if mse(fd.theta, Xref, y) - mse0 <= 0:
            break
        a *= 0.85
    return a * dl

def _mse_curv_shrink(dl, lR, MR):
    """ANALYTIC 2nd-order curvature cap: scale δ→αδ so ℓ_Rᵀδ+½δᵀM_Rδ ≤ 0, M_R=B_R+H_map_R.
    ΔRisk_proxy(α)=bα+½cα², b=ℓ_Rᵀδ, c=δᵀM_Rδ. Largest feasible α∈[0,1]."""
    b = float(lR @ dl)
    c = float(dl @ MR @ dl)
    if b + 0.5 * c <= 0:
        return dl
    if c > 1e-12 and b < 0:
        return min(1.0, -2 * b / c) * dl
    return 0.0 * dl

def mse_fullcurv_cost(name='concrete', kind='linear_topq', M=8, eps_grid=None, lam=0.1, n=350, seed=0, ref='clean', sel_mode='cap', sel_p=0.99, **tk):
    """Three MSE-stealth caps compared: (1) LINEAR ℓ_Rᵀδ≤0, (2) ANALYTIC 2nd-order curvature
    ℓ_Rᵀδ+½δᵀM_Rδ≤0 (M_R=B_R+H_map_R), (3) EXACT full-curvature (true ΔRisk≤0 via refit, all orders).
    Left: ΔF cost. Right: true ΔRisk achieved — linear leaks up, analytic-curvature lands near 0
    (the residual is the 3rd-order gap), exact refit hits 0."""
    ds = DATASETS[name](*DEFAULT_SIZE.get(name, (350, 150)))
    (X, y) = (ds.X_train, ds.y_train)
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False)
        (X, y) = (X[i], y[i])
    fit = ridge_fit(X, y, lam)
    (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
    gF = gF_fn(fit.theta)
    S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
    d = X.shape[1]
    F0 = gF @ fit.theta
    mse0 = mse(fit.theta, X, y)
    eps_grid = np.array(eps_grid) if eps_grid is not None else np.linspace(0.5, 4, 8)
    lR = stack_A(fit, X, y, S).T @ grad_mse_theta(fit.theta, X, y)
    (_, _, MR) = full_curvature(fit, X, y, lam, S, grad_mse_theta(fit.theta, X, y), 2 / len(X) * (X.T @ X))
    risk = lambda th, Xp: mse(th, X if ref == 'clean' else Xp, y) - mse0
    (dF_lin, dM_lin, dF_curv, dM_curv, dF_full, dM_full) = ([], [], [], [], [], [])
    for eps in eps_grid:
        R = np.sqrt(M) * eps
        dl = tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, mse_cap=True, inlier_cap=True)['delta']
        (fd, Xp) = _refit_at_delta(X, y, lam, S, dl)
        dF_lin.append(gF @ fd.theta - F0)
        dM_lin.append(risk(fd.theta, Xp))
        d2 = tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, mse_cap=False, inlier_cap=True)['delta']
        dc = _mse_curv_shrink(d2, lR, MR)
        (fdc, Xpc) = _refit_at_delta(X, y, lam, S, dc)
        dF_curv.append(gF @ fdc.theta - F0)
        dM_curv.append(risk(fdc.theta, Xpc))
        df = _mse_full_shrink(X, y, lam, S, d2, mse0, ref)
        (fdf, Xpf) = _refit_at_delta(X, y, lam, S, df)
        dF_full.append(gF @ fdf.theta - F0)
        dM_full.append(risk(fdf.theta, Xpf))
    (fig, (a0, a1)) = plt.subplots(1, 2, figsize=(12, 4.6))
    a0.plot(eps_grid, dF_lin, 'o-', label='linear-truncated cap')
    a0.plot(eps_grid, dF_curv, '^--', color='green', label='analytic curvature (B_R+H_map_R)')
    a0.plot(eps_grid, dF_full, 's--', color='purple', label='exact full-curvature (refit, all orders)')
    a0.set_xlabel('ε')
    a0.set_ylabel('ΔF (prediction)')
    a0.legend(fontsize=8)
    a0.grid(alpha=0.3)
    a0.set_title(f'{name}: ΔF cost of the MSE cap')
    a1.plot(eps_grid, dM_lin, 'o-', label='linear')
    a1.plot(eps_grid, dM_curv, '^--', color='green', label='analytic curvature')
    a1.plot(eps_grid, dM_full, 's--', color='purple', label='exact refit')
    a1.axhline(0, color='k', lw=0.6)
    a1.set_xlabel('ε')
    a1.set_ylabel(f'true ΔRisk ({ref})')
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a1.set_title('MSE stealth achieved')
    plt.tight_layout()
    return fig

def residual_inward(name='diabetes', kind='linear_topq', M=8, Rm=2.5, lam=0.1, n=400, seed=0, methods=('TR1', 'TR2', 'TR3', 'TR3-relin', 'bilevel'), sel_mode='cap', sel_p=0.99, **tk):
    """All residuals shift faintly (the global θ̂ move from refitting); the M ATTACKED points are
    bold, sized by their budget allocation ‖δ_i‖, and move INWARD (|r|↓) along the slope −1
    antidiagonal (ŷ_i+r_i=y_i, label fixed). Uses the cap-placed solvers (caps on by default)."""
    ds = DATASETS[name](*DEFAULT_SIZE.get(name, (300, 120)))
    (X, y) = (ds.X_train, ds.y_train)
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False)
        (X, y) = (X[i], y[i])
    fit = ridge_fit(X, y, lam)
    (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
    gF = gF_fn(fit.theta)
    hF = hF_fn(fit.theta) if hF_fn is not None else None
    S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
    R = np.sqrt(M) * Rm
    (n_, d) = X.shape
    th0 = fit.theta
    yh0 = X @ th0
    r0 = y - yh0
    mask = np.ones(n_, bool)
    mask[S] = False
    cp = dict(inlier_cap=True, mse_cap=True)

    def dl_(m):
        if m == 'TR1':
            return tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, **cp)['delta']
        if m == 'TR2':
            return tr_oneshot(fit, X, y, lam, S, R, gF, hF, 2, **cp)['delta']
        if m == 'TR3':
            return tr_oneshot(fit, X, y, lam, S, R, gF, hF, 3, **cp)['delta']
        if m == 'TR3-relin':
            if hF_fn is not None:
                return tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, hF_fn, **cp)['delta']
            return tr3_relin(fit, X, y, lam, S, R, gF, hF, **cp)['delta']
        if hF_fn is not None:
            return bilevel_at_S(fit, X, y, lam, S, gF, R, gF_fn=gF_fn, F_fn=F_fn, **cp)['delta']
        return bilevel_at_S(fit, X, y, lam, S, gF, R, **cp)['delta']
    C = len(methods)
    (fig, axes) = plt.subplots(1, C, figsize=(3.4 * C, 4.0), sharey=True)
    if C == 1:
        axes = [axes]
    for (ax, m) in zip(axes, methods):
        delta = dl_(m)
        Xp = X.copy()
        for (k, i) in enumerate(S):
            Xp[i] = Xp[i] + delta[k * d:(k + 1) * d]
        thp = ridge_fit(Xp, y, lam).theta
        yhp = Xp @ thp
        rp = y - yhp
        ax.scatter(yh0[mask], r0[mask], s=5, c='0.8', zorder=1)
        for i in np.where(mask)[0]:
            ax.annotate('', xy=(yhp[i], rp[i]), xytext=(yh0[i], r0[i]), arrowprops=dict(arrowstyle='-', color='0.85', lw=0.3, alpha=0.5))
        bud = np.array([np.linalg.norm(delta[k * d:(k + 1) * d]) for k in range(M)])
        bn = bud / (bud.max() + 1e-09)
        for (k, i) in enumerate(S):
            ax.annotate('', xy=(yhp[i], rp[i]), xytext=(yh0[i], r0[i]), arrowprops=dict(arrowstyle='-|>', color='#7a5fb0', lw=1.0 + 2.5 * bn[k], alpha=0.9))
        ax.scatter([yh0[i] for i in S], [r0[i] for i in S], s=90, marker='*', c='orange', edgecolor='k', linewidth=0.4, zorder=5, label='attacked ★ before')
        ax.scatter([yhp[i] for i in S], [rp[i] for i in S], s=40 + 160 * bn, c='#2e7d57', edgecolor='k', linewidth=0.4, zorder=6, label='● after (size=‖δ_i‖)')
        ax.axhline(0, color='k', lw=0.7)
        ra = np.array([r0[i] for i in S])
        rpa = np.array([rp[i] for i in S])
        ax.set_xlabel('$\\hat y_i$')
        ax.set_title(f'{m}\nattacked Δ|r|̄={np.mean(np.abs(rpa)) - np.mean(np.abs(ra)):+.1f}', fontsize=8)
    axes[0].set_ylabel('residual $r_i=y_i-\\hat y_i$')
    axes[0].legend(fontsize=6.5, loc='upper right')
    fig.suptitle(f'{name}: all residuals shift faintly (global θ̂ move); attacked points (bold, size=budget ‖δ_i‖) move inward', y=1.02)
    plt.tight_layout()
    return fig

def m_eps_sweep(name='house', kind='linear_topq', m_list=(1, 2, 3, 5, 8, 13, 21, 30), eps_grid=None, lam=0.1, n=600, seed=0, sel_mode='cap', sel_p=0.99, method='TR1', mse_cap=False, inlier_cap=False, **tk):
    """ΔF vs ε for several m. Curves nest monotonically (no crossing): increasing m or ε only
    moves ΔF UP. Closed form ΔF(m,ε) ≈ ε·c(m) + ½mε²·(û'Mû),  c(m)=√m·‖ℓ_F(S_m)‖.
    method in {TR1,TR2,TR3,TR3-relin,bilevel}; caps default off = original TR1 budget-only behaviour."""
    ds = DATASETS[name](*DEFAULT_SIZE.get(name, (600, 200)))
    (X, y) = (ds.X_train, ds.y_train)
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False)
        (X, y) = (X[i], y[i])
    fit = ridge_fit(X, y, lam)
    (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
    gF = gF_fn(fit.theta); hF = hF_fn(fit.theta) if hF_fn is not None else None; nl = hF_fn is not None
    F0 = F_fn(fit.theta)
    cp = dict(mse_cap=mse_cap, inlier_cap=inlier_cap)
    def _atk(mm, Sm, R):
        if method == 'TR1': return tr_oneshot(fit, X, y, lam, Sm, R, gF, None, 1, **cp)['delta']
        if method == 'TR2': return tr_oneshot(fit, X, y, lam, Sm, R, gF, hF, 2, **cp)['delta']
        if method == 'TR3': return tr_oneshot(fit, X, y, lam, Sm, R, gF, hF, 3, **cp)['delta']
        if method == 'TR3-relin':
            if nl: return tr3_relin_nonlinear(fit, X, y, lam, Sm, R, gF_fn, F_fn, hF_fn, **_filter_kw(tr3_relin_nonlinear, max_outer=16, **cp))['delta']
            return tr3_relin(fit, X, y, lam, Sm, R, gF, None, **_filter_kw(tr3_relin, max_outer=16, **cp))['delta']
        return bilevel_at_S(fit, X, y, lam, Sm, gF_fn if nl else gF, R, **_filter_kw(bilevel_at_S, n_iter=45, n_restarts=2, F_fn=F_fn if nl else None, **cp))['delta']
    eps_grid = np.array(eps_grid) if eps_grid is not None else np.linspace(0, 0.8, 12)
    (fig, ax) = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap('viridis')
    for (j, m) in enumerate(m_list):
        Sm = select_optimal_points(X, y, fit, gF, m, p=sel_p, mode=sel_mode)['S']
        dF = [F_fn(_refit_at_delta(X, y, lam, Sm, _atk(m, Sm, np.sqrt(m) * e))[0].theta) - F0 for e in eps_grid]
        ax.plot(eps_grid, dF, 'o-', ms=4, color=cmap(j / max(len(m_list) - 1, 1)), label=f'm={m}')
    ax.set_xlabel('attack budget ε')
    ax.set_ylabel('ΔF')
    _capn = 'both' if (mse_cap and inlier_cap) else 'inlier' if inlier_cap else 'mse' if mse_cap else 'budget'
    ax.set_title(f'{name}: m–ε sweep ({method}, {_capn})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
import inspect

def _filter_kw(fn, **kw):
    p = inspect.signature(fn).parameters
    return {k: v for (k, v) in kw.items() if k in p}

def detectability_scan(names=None, kind='linear_topq', M=8, eps=1.5, methods=('TR1', 'TR2', 'TR3', 'TR3-relin', 'bilevel@S*', 'bilevel@Rand'), lam=0.1, n=None, seed=0, sel_mode='cap', sel_p=0.99, q=0.8, level_bump=6.0, relin_outer=60, bil_iter=150, bil_restarts=3, auc_thr=0.85, sil_thr=0.25):
    """Self-contained fixed-ε detectability, caps ON. FOUR signals kept separate:
       one bar chart per signal (2×2) AND one pivot table per signal (datasets × methods).
       'cluster' = imbalance-robust distance-to-clean detector AUC (0.5 = invisible, →1 separable);
       replaces the old ARI, which flatlines at ~0 when the poison fraction M/n is tiny.
       Datasets ordered by decreasing mean AUC. Returns (fig, df)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, roc_auc_score
    names = names or list(DATASETS)
    ok = []
    for nm in names:
        try:
            ok.append((nm, DATASETS[nm](*DEFAULT_SIZE.get(nm, (300, 120)))))
            print(f'  loaded {nm}', flush=True)
        except Exception as e:
            print(f'  {nm} SKIPPED ({type(e).__name__}: {str(e)[:40]})', flush=True)
    rec = []
    for (nm, ds) in ok:
        (X, y) = (ds.X_train, ds.y_train)
        if n and len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = ridge_fit(X, y, lam)
        th0 = fit.theta
        tn0 = np.linalg.norm(th0)
        Xc0 = X - X.mean(0)
        Sig0 = Xc0.T @ Xc0 / max(len(X) - 1, 1)
        nS0 = np.linalg.norm(Sig0)
        yh = ds.X_test @ th0
        if kind == 'linear_topq':
            (F_fn, gF_fn, hF_fn) = make_target('linear_topq', ds.X_test, theta0=th0, q=q)
        elif kind == 'level':
            idx = np.where(yh >= np.quantile(yh, q))[0]
            c = float(ds.X_test[idx].mean(0) @ th0) + level_bump
            (F_fn, gF_fn, hF_fn) = make_target('level', ds.X_test, idx=idx, c=c)
        elif kind == 'mse':
            (F_fn, gF_fn, hF_fn) = make_target('mse', ds.X_test, Xt=ds.X_test, yt=ds.y_test)
        elif kind in ('feature_seg', 'feature_level'):
            (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=th0, q=q)
        else:
            raise ValueError(f'unknown kind {kind!r}')
        gF = gF_fn(th0)
        hessM = hF_fn(th0) if hF_fn is not None else None
        S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
        inl = filter_outlier(X)
        Srand = list(np.random.default_rng(seed + 7).choice(inl, M, replace=False))
        R = np.sqrt(M) * eps
        cp = dict(mse_cap=kind != 'mse', inlier_cap=True)

        def atk(m):
            if m == 'TR1':
                return (tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, **cp)['delta'], S)
            if m == 'TR2':
                return (tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 2, **cp)['delta'], S)
            if m == 'TR3':
                return (tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, **cp)['delta'], S)
            if m == 'TR3-relin':
                if hF_fn is not None:
                    d = tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, F_fn, hF_fn, **_filter_kw(tr3_relin_nonlinear, max_outer=relin_outer, n_outer=relin_outer, **cp))['delta']
                else:
                    d = tr3_relin(fit, X, y, lam, S, R, gF, None, **_filter_kw(tr3_relin, max_outer=relin_outer, n_outer=relin_outer, **cp))['delta']
                return (d, S)
            Su = S if m == 'bilevel@S*' else Srand
            gA = gF_fn if hF_fn is not None else gF
            extra = dict(F_fn=F_fn) if hF_fn is not None else {}
            d = bilevel_at_S(fit, X, y, lam, Su, gA, R, **_filter_kw(bilevel_at_S, n_iter=bil_iter, n_restarts=bil_restarts, **cp, **extra))['delta']
            return (d, Su)

        def sigs(Su, dl):
            (fd, Xp) = _refit_at_delta(X, y, lam, Su, dl)
            thp = fd.theta
            lab = np.zeros(len(X), int)
            lab[list(Su)] = 1
            yhp = Xp @ thp
            rp = y - yhp
            feats = np.column_stack([(yhp - yhp.mean()) / (yhp.std() + 1e-09), (rp - rp.mean()) / (rp.std() + 1e-09)])
            km = KMeans(2, n_init=10, random_state=seed).fit(feats)
            big = np.argmax(np.bincount(km.labels_))
            anom = np.linalg.norm(feats - km.cluster_centers_[big], axis=1)
            Xcp = Xp - Xp.mean(0)
            Sigp = Xcp.T @ Xcp / max(len(Xp) - 1, 1)
            return dict(cluster=float(roc_auc_score(lab, anom)), silhouette=float(silhouette_samples(feats, lab)[list(Su)].mean()), param=float(np.linalg.norm(thp - th0) / (tn0 + 1e-12)), covdrift=float(np.linalg.norm(Sigp - Sig0) / (nS0 + 1e-12)))
        for m in methods:
            (dl, Su) = atk(m)
            sig = sigs(Su, dl)
            rec.append(dict(dataset=nm, method=m, **sig, flagged=bool(sig['cluster'] >= auc_thr or sig['silhouette'] >= sil_thr)))
            print(f"    {nm}: {m:12s} AUC={sig['cluster']:.2f}  sil={sig['silhouette']:.2f}", flush=True)
        print(f'  done {nm}', flush=True)
    df = pd.DataFrame(rec)
    dsl = list(df.groupby('dataset')['cluster'].mean().sort_values(ascending=False).index)
    mth = list(methods)
    pv = lambda col: df.pivot(index='dataset', columns='method', values=col).reindex(index=dsl, columns=mth).round(3)
    signals = [('cluster', 'cluster recoverability (distance-to-clean detector AUC)'), ('silhouette', 'silhouette of poisoned cluster on (ŷ, r)'), ('param', 'parameter movement  ‖Δθ‖ / ‖θ₀‖'), ('covdrift', 'covariance drift  ‖ΔΣ‖_F / ‖Σ₀‖_F')]
    print(f'\n=== Detectability at ε={eps}, caps ON, M={M}  (flagged = AUC≥{auc_thr} or silhouette≥{sil_thr}) ===')
    for (key, label) in signals:
        print(f'\n[ {label} ]')
        print(pv(key).to_string())
    print('\nfraction of datasets flagged per method:')
    print(df.groupby('method')['flagged'].mean().reindex(mth).round(2).to_string())
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
    w = 0.8 / len(mth)
    xpos = np.arange(len(dsl))
    ctr = (len(mth) - 1) * w / 2
    (fig, axes) = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.ravel()
    for (c, (key, label)) in enumerate(signals):
        ax = axes[c]
        for (j, m) in enumerate(mth):
            vals = [df[(df.dataset == nm) & (df.method == m)][key].values[0] for nm in dsl]
            ax.bar(xpos + j * w, vals, w, label=m, color=colors[j], edgecolor='k', linewidth=0.3)
        ax.set_xticks(xpos + ctr)
        ax.set_xticklabels(dsl, rotation=45, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        if key == 'cluster':
            ax.axhline(auc_thr, color='k', lw=0.8, ls=':')
            ax.axhline(0.5, color='gray', lw=0.8, ls='--')
        if key == 'silhouette':
            ax.axhline(sil_thr, color='k', lw=0.8, ls=':')
        if c == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'Detectability at ε={eps} (caps on) — four signals, datasets by decreasing detector AUC', y=1.0)
    plt.tight_layout()
    return (fig, df)

def detectability_method_summary(df, methods=None, auc_thr=0.85, sil_thr=0.25):
    """Per-method means across datasets for each of the four signals (one bar panel each).
       Consumes the df from detectability_scan — no external helpers."""
    methods = methods or list(dict.fromkeys(df['method']))
    if 'flagged' not in df:
        df = df.assign(flagged=(df.cluster >= auc_thr) | (df.silhouette >= sil_thr))
    g = df.groupby('method').agg(cluster=('cluster', 'mean'), silhouette=('silhouette', 'mean'), param=('param', 'mean'), covdrift=('covdrift', 'mean'), frac=('flagged', 'mean')).reindex(methods)
    print('\nper-method means (four signals + fraction flagged):')
    print(g.round(3).to_string())
    signals = [('cluster', 'mean detector AUC'), ('silhouette', 'mean silhouette'), ('param', 'mean parameter movement'), ('covdrift', 'mean covariance drift')]
    x = np.arange(len(methods))
    cols = plt.get_cmap('tab10')(np.linspace(0, 1, 10))[:len(methods)]
    (fig, axes) = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for (c, (key, label)) in enumerate(signals):
        ax = axes[c]
        ax.bar(x, g[key].values, color=cols, edgecolor='k')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        if key == 'cluster':
            ax.axhline(auc_thr, color='k', lw=0.8, ls=':')
            ax.axhline(0.5, color='gray', lw=0.8, ls='--')
        if key == 'silhouette':
            ax.axhline(sil_thr, color='k', lw=0.8, ls=':')
    fig.suptitle('Per-method detectability, averaged over datasets', y=1.0)
    plt.tight_layout()
    return fig

def _eta_max_all(fit, X, y, S, lF, dirs=None, delta=None, p=0.99):
    """Per-point Mahalanobis travel cap eta_max,i along a unit ray u_i.
       default u_i = lF_i/||lF_i||  (TR1 influence ray; this is what eps_leave/eps_sat use)."""
    ctx = cap_context(X, y, fit, S, p, mse_full=False)
    (L, tau, Z, d) = (ctx['L'], ctx['tau'], ctx['Z'], ctx['d'])
    m = len(S)
    si = np.array([np.linalg.norm(lF[k * d:(k + 1) * d]) for k in range(m)])
    etas = []
    for k in range(m):
        uk = dirs[k * d:(k + 1) * d] if dirs is not None else delta[k * d:(k + 1) * d] if delta is not None else lF[k * d:(k + 1) * d]
        nu = np.linalg.norm(uk)
        if nu < 1e-12:
            etas.append(0.0)
            continue
        u = uk / nu
        wu = np.linalg.solve(L, u)
        uu = wu @ wu
        zw = Z[k] @ wu
        zz = Z[k] @ Z[k]
        disc = zw * zw - uu * (zz - tau * tau)
        etas.append(max((-zw + np.sqrt(disc)) / uu if disc >= 0 and uu > 1e-18 else 0.0, 0.0))
    return (np.array(etas), si)

def eps_sat_empirical(eps_list, vals, tol_frac=0.01):
    """First eps where the realized objective stops increasing (dGamma/deps ~ 0)."""
    eps_list = np.asarray(eps_list, float)
    vals = np.asarray(vals, float)
    rng = vals.max() - vals.min() + 1e-12
    for i in range(1, len(vals)):
        if (vals[i] - vals[i - 1]) / rng < tol_frac:
            return float(eps_list[i])
    return float(eps_list[-1])

def add_saturation_overlay(ax, eps_list, out, sc, which='bilevel', tol_frac=0.01):
    """Overlay two saturation budgets on a ΔF-vs-ε axis:
       - analytic  ε_sat  = sc['eps_sat']  (inlier-cap travel limit from _theory_scales), orange dashed;
       - empirical ε_sat  = where the realized `which` curve plateaus (dΔF/dε ~ 0), purple dash-dot."""
    eps_list = np.asarray(eps_list, float)
    (lo, hi) = (float(eps_list.min()), float(eps_list.max()))
    ytop = ax.get_ylim()[1]
    if not np.isfinite(ytop):
        ytop = 1.0
    e_an = float(sc.get('eps_sat', np.inf)) if isinstance(sc, dict) else np.inf
    if np.isfinite(e_an) and lo <= e_an <= hi:
        ax.axvline(e_an, color='#cc6600', ls='--', lw=1.8)
        ax.text(e_an, ytop, 'TR1 $\\epsilon_{\\rm sat}$ (inlier)', color='#cc6600', rotation=90, va='top', ha='right', fontsize=7.5)
    vals = out.get(which) if isinstance(out, dict) else None
    if not vals:
        vals = next((v for v in (out.values() if isinstance(out, dict) else []) if v), None)
    e_emp = np.inf
    if vals is not None and len(vals) == len(eps_list):
        e_emp = eps_sat_empirical(eps_list, vals, tol_frac)
        if np.isfinite(e_emp) and lo <= e_emp <= hi:
            ax.axvline(e_emp, color='purple', ls='-.', lw=1.8)
            ax.text(e_emp, ytop, 'empirical $\\epsilon_{\\rm sat}$', color='purple', rotation=90, va='top', ha='left', fontsize=7.5)
    return dict(eps_sat_analytic=e_an, eps_sat_empirical=e_emp)

def _theory_scales(fit, X, y, lam, S, gF, hessM, mm, R_L3, eta=0.1, K=60):
    """Analytic budget-scale markers (in eps units) + curvature quantities for one (S, target)."""
    lF = ell_F(fit, X, y, S, gF)
    nl = np.linalg.norm(lF)
    e = lF / nl
    (_, _, BH) = full_curvature(fit, X, y, lam, S, gF, hessM)
    A = np.linalg.norm(BH, 2)
    b1 = float(e @ BH @ e)
    L3 = _estimate_L3(fit, X, y, lam, S, gF, R_L3, hess_F=hessM)
    DF = lambda d: _deltaF_true(fit, X, y, lam, S, gF, d)
    h = 0.5
    c3 = (DF(2 * h * e) - 2 * DF(h * e) + 2 * DF(-h * e) - DF(-2 * h * e)) / (2 * h ** 3)
    (aa, bb, cc) = (0.5 * c3, b1, nl)
    disc = bb * bb - 4 * aa * cc
    roots = [r for r in ([(-bb + np.sqrt(disc)) / (2 * aa), (-bb - np.sqrt(disc)) / (2 * aa)] if disc >= 0 and aa != 0 else []) if r > 0]
    Rbf = min(roots) if roots else nl / abs(b1) if b1 < 0 else np.inf
    (etas, si) = _eta_max_all(fit, X, y, S, lF)
    eps_leave = float(np.min(etas * nl / (np.sqrt(mm) * np.maximum(si, 1e-12))))
    eps_sat = float(np.sqrt(np.sum(etas ** 2) / mm))
    return dict(eps_full=nl / (np.sqrt(mm) * A), eps_cubic=3 * A / (np.sqrt(mm) * L3), eps_backfire=Rbf / np.sqrt(mm), eps_13=3 / (2 * L3) * (np.sqrt(A * A + 4 / 3 * eta * nl * L3) - A) / np.sqrt(mm), eps_3relin=np.sqrt(3 * eta * nl / (L3 * (1 - K ** (-2)))) / np.sqrt(mm), eps_leave=eps_leave, eps_sat=eps_sat, gamma_max_tr1=nl * Rbf + 0.5 * b1 * Rbf ** 2 + c3 * Rbf ** 3 / 6 if np.isfinite(Rbf) else np.inf, gamma_ceiling=float(np.sum(si * etas)), normlF=nl, Mop=A, b1=b1, c3=c3, L3=L3)
import inspect
import numpy as np, matplotlib.pyplot as plt, os

def _filter_kw(fn, **kw):
    """Keep only the kwargs fn actually declares (safe for max_outer/n_outer/tr_frac uncertainty)."""
    p = inspect.signature(fn).parameters
    return {k: v for (k, v) in kw.items() if k in p}

def boundary_walk(name, kind='linear_topq', caps='both', M=8, eps_grid=None, methods=('TR1', 'TR2', 'TR3', 'TR3-relin', 'bilevel@S*', 'bilevel@Rand'), lam=0.1, n=400, seed=0, sel_mode='cap', sel_p=0.99, relin_outer=22, relin_frac=0.12, bil_iter=70, sat_frac=0.05, savedir=None, **tk):
    """ONE dataset -> ONE 2x3 figure (3 methods top, 3 bottom), CAPPED attack.

    Tracks each selected point's Mahalanobis headroom  h_i(eps) = tau - d_M(x_i+delta_i)  vs budget
    eps, with the attack subject to the detectability cap(s):
        caps="inlier" -> inlier_cap=True, mse_cap=False
        caps="both"   -> inlier_cap=True, mse_cap=True   (default)
    The cap holds every poisoned point inside the ellipsoid, so h decreases to the ceiling. The eps
    at which a point first HITS h=0 (exact, interpolated) is its stealth-exhaustion budget; the order
    in which points reach it is the water-filling schedule. relin/bilevel are re-optimised at every eps.

    Annotations: red line = ceiling h=0; faint orange band [0,sat_frac*tau] = near-ceiling guide;
    dots ON the ceiling = per-point exact h=0 crossings; grey dotted line + band = median & IQR of the
    crossing eps; curve colour (viridis) = initial headroom h0 (dark = started near the ceiling).
    """
    ds = DATASETS[name](*DEFAULT_SIZE.get(name, (300, 120)))
    (X, y) = (ds.X_train, ds.y_train)
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False)
        (X, y) = (X[i], y[i])
    d = X.shape[1]
    fit = ridge_fit(X, y, lam)
    (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
    gF = gF_fn(fit.theta)
    hF = hF_fn(fit.theta) if hF_fn is not None else None
    nonlin = hF_fn is not None
    eps_grid = np.asarray(eps_grid) if eps_grid is not None else np.linspace(0.3, 10.0, 12)
    Sstar = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
    inl = filter_outlier(X)
    Srand = list(np.random.default_rng(seed + 7).choice(inl, M, replace=False))
    cp = dict(inlier_cap=(caps in ('inlier', 'both')), mse_cap=caps == 'both')   # caps='budget'/'none' -> uncapped

    def atk(m, S, R):
        if m == 'TR1':
            return tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, **cp)['delta']
        if m == 'TR2':
            return tr_oneshot(fit, X, y, lam, S, R, gF, hF, 2, **cp)['delta']
        if m == 'TR3':
            return tr_oneshot(fit, X, y, lam, S, R, gF, hF, 3, **cp)['delta']
        if m == 'TR3-relin':
            if nonlin:
                kw = _filter_kw(tr3_relin_nonlinear, max_outer=relin_outer, n_outer=relin_outer, tr_frac=relin_frac, **cp)
                return tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, F_fn, **kw)['delta']
            kw = _filter_kw(tr3_relin, max_outer=relin_outer, n_outer=relin_outer, tr_frac=relin_frac, **cp)
            return tr3_relin(fit, X, y, lam, S, R, gF, hF, **kw)['delta']
        grad_arg = gF_fn if nonlin else gF
        kw = _filter_kw(bilevel_at_S, n_iter=bil_iter, F_fn=F_fn if nonlin else None, **cp)
        return bilevel_at_S(fit, X, y, lam, S, grad_arg, R, **kw)['delta']

    def headrooms(m, S):
        ctx = cap_context(X, y, fit, S, sel_p)
        (tau, L, Z) = (ctx['tau'], ctx['L'], ctx['Z'])
        H = np.empty((M, len(eps_grid)))
        for (j, eps) in enumerate(eps_grid):
            dl = atk(m, S, np.sqrt(M) * eps)
            for k in range(M):
                H[k, j] = tau - np.linalg.norm(Z[k] + np.linalg.solve(L, dl[k * d:(k + 1) * d]))
        return (H, tau, Z)

    def cross_eps(eps, h):
        below = np.where(h <= 0.0)[0]
        if len(below) == 0:
            return None
        j = below[0]
        if j == 0:
            return float(eps[0])
        return float(eps[j - 1] + (0.0 - h[j - 1]) * (eps[j] - eps[j - 1]) / (h[j] - h[j - 1]))
    nsel = len(methods); ncols = min(nsel, 3); nrows = int(np.ceil(nsel / ncols))
    cell = 6.2 if nsel == 1 else (4.8 if nsel == 2 else 3.8)
    (fig, axes) = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * 0.9 * nrows), squeeze=False)
    cmap = plt.cm.viridis
    for (c, m) in enumerate(methods):
        ax = axes[c // ncols][c % ncols]
        S = Srand if m == 'bilevel@Rand' else Sstar
        try:
            (H, tau, Z) = headrooms(m, S)
            tol = sat_frac * tau
            h0 = tau - np.linalg.norm(Z, axis=1)
            rank = np.argsort(np.argsort(-h0))
            ax.axhspan(0.0, tol, color='orange', alpha=0.1)
            crosses = []
            for k in range(M):
                col = cmap(rank[k] / max(M - 1, 1))
                ax.plot(eps_grid, H[k], '-', lw=1.1, alpha=0.85, color=col)
                e = cross_eps(eps_grid, H[k])
                if e is not None:
                    crosses.append(e)
                    ax.plot(e, 0.0, 'o', ms=4, color=col, mec='k', mew=0.3, zorder=6)
            ax.axhline(0, color='red', lw=2.0)
            ax.set_ylim(bottom=min(-0.12 * tau, float(np.nanmin(H)) * 1.05))   # leave room below the ceiling
            if crosses:
                med = np.median(crosses)
                (q1, q3) = np.percentile(crosses, [25, 75])
                ax.axvspan(q1, q3, color='0.5', alpha=0.1)
                ax.axvline(med, color='0.4', ls=':', lw=1.0)
                ax.text(0.97, 0.92, f'ε_st≈{med:.1f}\n{len(crosses)}/{M} cross', transform=ax.transAxes, ha='right', va='top', fontsize=7.5, bbox=dict(fc='white', ec='0.7', alpha=0.7, pad=1.5))
            else:
                ax.text(0.97, 0.92, '0 reach h=0', transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
            print(f'  {name:11s} [{caps:5s}] {m:13s} cross={len(crosses)}/{M}', flush=True)
        except Exception as ex:
            ax.text(0.5, 0.5, f'{type(ex).__name__}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')
            print(f'  {name:11s} [{caps}] {m:13s} SKIP ({type(ex).__name__}: {ex})', flush=True)
        ax.set_title(m, fontsize=9)
        if c // ncols == nrows - 1:
            ax.set_xlabel('ε (budget)')
        if c % ncols == 0:
            ax.set_ylabel('headroom  τ − d_M')
    for cc in range(len(methods), nrows * ncols):
        axes[cc // ncols][cc % ncols].axis('off')
    fig.suptitle(f'{name} — boundary walk [caps={caps}, target={kind}]: headroom vs ε (red = ceiling h=0)', y=1.0, fontsize=12)
    plt.tight_layout()
    if savedir:
        p = os.path.join(savedir, f'boundary_walk_{name}_{kind}_{caps}.png')
        fig.savefig(p, dpi=130, bbox_inches='tight')
        print(f'  saved {p}', flush=True)
    return fig

def boundary_walk_all(names, kinds=('linear_topq',), caps_list=('both',), savedir=None, **kw):
    """Driver: one 2x3 figure per (dataset, kind, caps). Prints progress per panel."""
    figs = {}
    for kind in kinds:
        for caps in caps_list:
            for nm in names:
                print(f'[{kind} | {caps}] {nm} ...', flush=True)
                try:
                    figs[nm, kind, caps] = boundary_walk(nm, kind=kind, caps=caps, savedir=savedir, **kw)
                except Exception as e:
                    print(f'  {nm} FAILED: {type(e).__name__}: {e}', flush=True)
    return figs

def relin_projection(names=None, kind='linear_topq', M=6, lam=0.1, n=350, seed=0, ncols=5, sel_mode='cap', sel_p=0.99, include_mse=True, q=0.8, kill_thr=0.35, keep_thr=0.65, savedir=None, **tk):
    """Projected-curvature gap (SCALE-FREE): split the top curvature mode v1 of M_F against the single
       aggregate outward wall direction n0 (stacked inlier normals, + MSE normal if include_mse).
       kept = ||P_perp v1|| = sin<(v1,n0) = tangential (surviving) fraction;
       lam-ratio = lam_max(P_perp M_F P_perp)_+ / lam_max(M_F)_+ (the factor in the gap bound).
       Small kept / small lam-ratio => curvature killed => TR1 catches up. 5-column grid."""
    import numpy as np, matplotlib.pyplot as plt, math, os
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def one(name):
        ds = DATASETS[name](*DEFAULT_SIZE.get(name, (500, 150)))
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        d = X.shape[1]
        fit = ridge_fit(X, y, lam)
        (F, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, q=q, Xt=ds.X_test, yt=ds.y_test, **tk)
        gF = gF_fn(fit.theta)
        hF = hF_fn(fit.theta) if hF_fn is not None else None
        S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
        (B, Hmap, MF) = full_curvature(fit, X, y, lam, S, gF, hF)
        MF = 0.5 * (MF + MF.T)
        (w, V) = np.linalg.eigh(MF)
        v1 = V[:, -1]
        v1 /= np.linalg.norm(v1)
        lam_full = max(float(w[-1]), 0.0)
        md = len(S) * d
        mu = X.mean(0)
        Sig = np.cov(X.T) + 0.001 * np.eye(d)
        Sinv = np.linalg.inv(Sig)
        n_inl = np.concatenate([Sinv @ (X[idx] - mu) for idx in S])
        n0 = n_inl / (np.linalg.norm(n_inl) + 1e-18)
        if include_mse:
            lR = stack_A(fit, X, y, S).T @ grad_mse_theta(fit.theta, X, y)
            n0 = n0 + lR / (np.linalg.norm(lR) + 1e-18)
            n0 /= np.linalg.norm(n0) + 1e-18
        align = abs(float(v1 @ n0))
        kept = math.sqrt(max(1 - align ** 2, 0.0))
        MT = MF - np.outer(MF @ n0, n0) - np.outer(n0, n0 @ MF) + n0 @ MF @ n0 * np.outer(n0, n0)
        MT = 0.5 * (MT + MT.T)
        lam_T = max(float(np.linalg.eigvalsh(MT)[-1]), 0.0)
        lam_ratio = lam_T / (lam_full + 1e-18)
        return (kept, lam_ratio, d)
    names = names or list(DATASETS)
    nrows = math.ceil(len(names) / ncols)
    fig = plt.figure(figsize=(3.7 * ncols, 3.7 * nrows))
    for (k, nm) in enumerate(names):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection='3d')
        try:
            (kept, lr, d) = one(nm)
            nc = math.sqrt(max(1 - kept ** 2, 0.0))
            status = 'KILLED' if kept < kill_thr else 'KEPT' if kept >= keep_thr else 'PARTIAL'
            tail = {'KILLED': ' · TR1 catches up', 'KEPT': ' · gap persists', 'PARTIAL': ''}[status]
            ax.add_collection3d(Poly3DCollection([[(-0.05, -0.05, 0), (1, -0.05, 0), (1, 1, 0), (-0.05, 1, 0)]], color='#c0392b', alpha=0.1))
            ax.quiver(0, 0, 0, 0, 0, 0.9, color='#c0392b', lw=2.0, arrow_length_ratio=0.12)
            ax.text(0, 0, 0.97, 'n', color='#c0392b', fontsize=11)
            ax.quiver(0, 0, 0, kept, 0, nc, color='k', lw=2.2, arrow_length_ratio=0.1)
            ax.text(kept, 0, nc + 0.03, '$v_1$', color='k', fontsize=11)
            ax.quiver(0, 0, 0, kept, 0, 0, color='#2e7d32', lw=3.0, arrow_length_ratio=0.15)
            ax.plot([kept, kept], [0, 0], [0, nc], 'r--', lw=1.4)
            ax.text(max(kept * 0.5, 0.05), 0, -0.12, f'$\\|P_Tv_1\\|$={kept:.2f}', color='#2e7d32', fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.view_init(18, -60)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([])
            ax.set_zticks([0, 0.5, 1])
            ax.tick_params(labelsize=7)
            ax.set_title(f'{nm} (d={d})\nkept {kept:.2f} · λ-ratio {lr:.2f} · {status}{tail}', fontsize=9.5)
            print(f'  {nm:12s} d={d} kept={kept:.2f} align={(align if False else 1 - kept ** 2):.2f} λ-ratio={lr:.2f} {status}', flush=True)
        except Exception as ex:
            ax.set_title(f'{nm}\nSKIP', fontsize=9, color='gray')
            ax.axis('off')
            print(f'  {nm:12s} SKIP ({type(ex).__name__}: {ex})', flush=True)
    fig.suptitle('Projected-curvature gap: top curvature mode $v_1$ split against the active wall (small kept $\\Rightarrow$ curvature killed $\\Rightarrow$ TR1 catches up)', fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.45, wspace=0.15)
    if savedir:
        p = os.path.join(savedir, 'relin_projection.png')
        fig.savefig(p, dpi=120, bbox_inches='tight')
        print('saved', p, flush=True)
    return fig

def functional_budget_sweep(name='realestate', kind='linear_topq', regime=None, caps=False, M=8, eps_list=None, lam=0.1, n=400, seed=0, q=0.8, q_hi=1.0, level_bump=6.0, relin_outer=60, bil_iter=150, bil_restarts=3, sel_mode='cap', sel_p=0.99, budget_mode=True, show_scales=True, eps_min=0.0, eps_max=7.01, warm_start=True, continuation=True, show_weak=True):
    """Budget sweep for ANY functional. regime in {'budget','inlier','both'}:
         budget -> no caps ;  inlier -> inlier cap only ;  both -> inlier + MSE caps.
       (caps=True/False is a fallback: True->'both', False->'budget'.)
       When kind=='mse', the MSE cap is auto-disabled even under 'both' (it would fight itself);
       the inlier cap stays. continuation=True path-follows; show_weak adds poor bilevel baselines.

       budget_mode=True (default): in the 'budget' regime the POINT SELECTION also switches to
       mode='budget' (genuine budget-only S*, Euclidean ‖ℓ_i‖ over all points), so selection AND
       displacement are budget-only end-to-end. Set budget_mode=False to keep cap-selection even
       under the budget regime (to isolate the selection effect)."""
    if regime is None:
        regime = 'both' if caps else 'budget'
    assert regime in ('budget', 'inlier', 'both'), f'bad regime {regime!r}'
    eps_list = np.asarray(eps_list) if eps_list is not None else np.arange(eps_min, eps_max, 0.5)
    ds = DATASETS[name](*DEFAULT_SIZE.get(name, (400, 150)))
    (X, y) = (ds.X_train, ds.y_train)
    if len(X) > n:
        i = np.random.default_rng(seed).choice(len(X), n, replace=False)
        (X, y) = (X[i], y[i])
    fit = ridge_fit(X, y, lam)
    th0 = fit.theta
    yh = ds.X_test @ th0
    if kind == 'linear_topq':
        (F_fn, gF_fn, hF_fn) = make_target('linear_topq', ds.X_test, theta0=th0, q=q, q_hi=q_hi)
    elif kind == 'level':
        idx = np.where(yh >= np.quantile(yh, q))[0]
        c = float(ds.X_test[idx].mean(0) @ th0) + level_bump
        (F_fn, gF_fn, hF_fn) = make_target('level', ds.X_test, idx=idx, c=c)
    elif kind == 'mse':
        (F_fn, gF_fn, hF_fn) = make_target('mse', ds.X_test, Xt=ds.X_test, yt=ds.y_test)
    elif kind in ('feature_seg', 'feature_level'):
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=th0, q=q, q_hi=q_hi)
    else:
        raise ValueError(f'unknown kind {kind!r}')
    inlier_on = regime in ('inlier', 'both')
    mse_on = regime == 'both' and kind != 'mse'
    if regime == 'both' and kind == 'mse':
        print("  note: kind='mse' -> MSE cap auto-DISABLED (objective is MSE); inlier cap stays", flush=True)
    cp = dict(mse_cap=mse_on, inlier_cap=inlier_on)
    any_cap = inlier_on or mse_on
    gF = gF_fn(th0)
    hessM = hF_fn(th0) if hF_fn is not None else None
    sel_mode_eff = 'budget' if regime == 'budget' and budget_mode else sel_mode
    S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode_eff)['S']
    print(f'  {name}/{kind}: regime={regime}, selection mode={sel_mode_eff}', flush=True)
    inl = filter_outlier(X)
    Srand = list(np.random.default_rng(seed + 7).choice(inl, M, replace=False))
    F0 = F_fn(th0)
    DFtrue = lambda d: F_fn(_refit_at_delta(X, y, lam, S, d)[0].theta) - F0
    sc = _theory_scales(fit, X, y, lam, S, gF, hessM, M, R_L3=np.sqrt(M) * float(max(eps_list)))
    keys = ['TR1', 'TR2', 'TR3', 'TR3-relin', 'bilevel'] + (['bilevel(weak)', 'bilevel@rand'] if show_weak else [])
    out = {k: [] for k in keys}
    (prev_rel, prev_bil) = (None, None)
    for eps in eps_list:
        R = np.sqrt(M) * eps
        d1 = tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, **cp)['delta']
        d2 = tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 2, **cp)['delta']
        d3 = tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, **cp)['delta']
        rel_pool = ([d1] if warm_start else []) + ([prev_rel] if continuation and prev_rel is not None else [])
        bil_pool = ([d1] if warm_start else []) + ([prev_bil] if continuation and prev_bil is not None else [])
        rel_delta0 = max(rel_pool, key=DFtrue) if rel_pool else None
        rel = tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, F_fn, hF_fn, **_filter_kw(tr3_relin_nonlinear, delta0=rel_delta0, max_outer=relin_outer, n_outer=relin_outer, **cp))
        bil = bilevel_at_S(fit, X, y, lam, S, gF_fn, R, F_fn=F_fn, **_filter_kw(bilevel_at_S, warm_seeds=bil_pool or None, n_iter=bil_iter, n_restarts=bil_restarts, **cp))
        (prev_rel, prev_bil) = (rel['delta'], bil['delta'])
        out['TR1'].append(DFtrue(d1))
        out['TR2'].append(DFtrue(d2))
        out['TR3'].append(DFtrue(d3))
        out['TR3-relin'].append(rel['val'])
        out['bilevel'].append(max(bil['val'], DFtrue(d1)))
        if show_weak:
            bilw = bilevel_at_S(fit, X, y, lam, S, gF_fn, R, F_fn=F_fn, **_filter_kw(bilevel_at_S, n_iter=40, n_restarts=1, **cp))
            bilr = bilevel_at_S(fit, X, y, lam, Srand, gF_fn, R, F_fn=F_fn, **_filter_kw(bilevel_at_S, n_iter=bil_iter, n_restarts=1, **cp))
            out['bilevel(weak)'].append(bilw['val'])
            out['bilevel@rand'].append(bilr['val'])
        print(f'  {name}/{kind} regime={regime} eps={eps:.2f} done', flush=True)
    (fig, ax) = plt.subplots(figsize=(8, 5))
    style = {'bilevel': ('k', 'o'), 'TR3-relin': ('#e6194B', 'P'), 'TR3': ('#16a01a', '^'), 'TR2': ('#e1a730', 'D'), 'TR1': ('#1f6fff', 's'), 'bilevel(weak)': ('#8a8a8a', 'x'), 'bilevel@rand': ('#8B4513', 'v')}
    for k in keys:
        (c, mk) = style[k]
        ax.plot(eps_list, out[k], color=c, marker=mk, lw=1.8, label=k, ls='--' if k in ('bilevel(weak)', 'bilevel@rand') else '-')
    if show_scales:
        (lo, hi) = (float(min(eps_list)), float(max(eps_list)))
        ytop = ax.get_ylim()[1]
        for (key, lab, c) in [('eps_full', '$\\epsilon^\\star_{\\rm full}$', '#777'), ('eps_backfire', '$\\epsilon^{\\rm TR1}_{\\rm back}$', '#1f6fff'), ('eps_3relin', '$\\epsilon_{3\\approx{\\rm rel}}$', '#e6194B')]:
            xv = sc.get(key, np.nan)
            if np.isfinite(xv) and lo <= xv <= hi:
                ax.axvline(xv, color=c, ls=':', lw=1.1, alpha=0.6)
                ax.text(xv, ytop, lab, color=c, rotation=90, va='top', ha='right', fontsize=7)
        if any_cap:
            try:
                add_saturation_overlay(ax, eps_list, out, sc, which='bilevel')
            except Exception:
                pass
    cap_txt = {'budget': 'BUDGET-ONLY', 'inlier': 'INLIER CAP ONLY', 'both': 'INLIER CAP (MSE cap off)' if kind == 'mse' else 'INLIER + MSE CAPS'}[regime]
    sel_txt = f', {sel_mode_eff}-selected S*'
    ax.set_xlabel('ε')
    ax.set_ylabel('ΔF')
    ax.set_title(f'{name}: {kind} target, {cap_txt}{sel_txt}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return (fig, out)
import os
SAVE = '/Users/dhlee/Downloads/Thesis research'
import time as _time
import pandas as pd
_CAP = {'budget-only': dict(mse_cap=False, inlier_cap=False), 'all-caps': dict(mse_cap=True, inlier_cap=True)}

def comp_collect(names=None, kind='linear_topq', M=8, Rm=3.0, lam=0.1, n=400, seed=0, relin_outer=120, bil_iter=200, bil_restarts=3, eps_max=8.0, reps=3, sel_mode='cap', sel_p=0.99, regimes=('budget-only', 'all-caps'), **tk):
    """Per dataset x regime: wall-time per method (best of `reps` runs) + analytic ε-scales.
       Nonlinear targets (level/mse) dispatch through tr3_relin_nonlinear / gF_fn,F_fn."""
    names = names or list(DATASETS)
    base = ridge_fit
    cnt = {'n': 0}

    def counted(*a, **k):
        cnt['n'] += 1
        return base(*a, **k)
    (rows, scales) = ([], {})
    print(f'comp_collect: {len(names)} datasets, regimes={list(regimes)}, reps={reps}', flush=True)
    for (di, nm) in enumerate(names, 1):
        print(f'[{di}/{len(names)}] {nm}: loading...', flush=True)
        ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
        (X, y) = (ds.X_train, ds.y_train)
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = base(X, y, lam)
        R = np.sqrt(M) * Rm
        md = M * X.shape[1]
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
        gF = gF_fn(fit.theta)
        hessM = hF_fn(fit.theta) if hF_fn is not None else None
        nonlin = hF_fn is not None
        S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
        print(f'    d={X.shape[1]}, md={md}, |S|={len(S)}, nonlinear={nonlin}', flush=True)
        for rg in regimes:
            cp = _CAP[rg]
            print(f'    regime={rg:11s}: ', end='', flush=True)

            def _relin(cp=cp):
                if nonlin:
                    return tr3_relin_nonlinear(fit, X, y, lam, S, R, gF_fn, hF_fn, max_outer=relin_outer, **cp)['delta']
                return tr3_relin(fit, X, y, lam, S, R, gF, None, max_outer=relin_outer, **cp)['delta']

            def _bil(cp=cp):
                if nonlin:
                    return bilevel_at_S(fit, X, y, lam, S, gF, R, n_iter=bil_iter, n_restarts=bil_restarts, gF_fn=gF_fn, F_fn=F_fn, **cp)['delta']
                return bilevel_at_S(fit, X, y, lam, S, gF, R, n_iter=bil_iter, n_restarts=bil_restarts, **cp)['delta']
            meths = {'TR1': lambda cp=cp: tr_oneshot(fit, X, y, lam, S, R, gF, None, 1, **cp)['delta'], 'TR2': lambda cp=cp: tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 2, **cp)['delta'], 'TR3': lambda cp=cp: tr_oneshot(fit, X, y, lam, S, R, gF, hessM, 3, **cp)['delta'], 'TR3-relin': _relin, 'bilevel': _bil}
            for (mn, fn) in meths.items():
                globals()['ridge_fit'] = counted
                cnt['n'] = 0
                t0 = _time.perf_counter()
                d = fn()
                dt = _time.perf_counter() - t0
                refits = cnt['n']
                for _ in range(max(0, reps - 1)):
                    t0 = _time.perf_counter()
                    d = fn()
                    dt = min(dt, _time.perf_counter() - t0)
                globals()['ridge_fit'] = base
                print(f'{mn} {1000 * dt:6.0f}ms ', end='', flush=True)
                rows.append(dict(dataset=nm, regime=rg, method=mn, refits=refits, ms=1000 * dt, dF=float(_deltaF_true(fit, X, y, lam, S, gF, d))))
            print('', flush=True)
        sc = _theory_scales(fit, X, y, lam, S, gF, hessM, M, R_L3=np.sqrt(M) * eps_max)
        scales[nm] = sc
        print(f"    ε: leave={sc['eps_leave']:.2f} sat={sc['eps_sat']:.2f} full={sc['eps_full']:.2f} back={sc['eps_backfire']:.2f} cubic={sc['eps_cubic']:.2f} 3~rel={sc['eps_3relin']:.2f}", flush=True)
    print('comp_collect: done.', flush=True)
    return (pd.DataFrame(rows), scales)

def error_below_bound(names=None, kind='linear_topq', M=8, Rm=3.0, lam=0.1, n=400, seed=0, relin_K=40, n_probe=4000, L3_safety=1.15, sel_mode='cap', sel_p=0.99, **tk):
    """Empirical suboptimality gap G_j = f* - f(delta_TRj) vs the analytic certified bound, per method
       per dataset, on the UNCAPPED budget ball:
         G1 <= ||B_F+H_map|| R^2 + L3/3 R^3 ;  G2 <= ||H_map|| R^2 + L3/3 R^3 ;
         G3 <= L3/3 R^3 ;  G_relin <= L3/(3 K^2) R^3 .
       f* = best of {all methods, random multi-start probe on the sphere ||d||=R} (a real optimum
       estimate, not just best-of-methods). L3 is scaled by L3_safety(>=1) for a conservative bound."""
    names = names or list(DATASETS)
    print(f'error_below_bound: {len(names)} datasets, M={M}, Rm={Rm}, relin_K={relin_K}, probe={n_probe}', flush=True)

    def _relin_un(fit, X, y, lam, S, R, gF_fn, hF_fn, K, nonlin):
        md = len(S) * X.shape[1]
        Delta = 0.12 * R
        delta = np.zeros(md)
        best = 0.0
        for _ in range(K):
            (fd, Xp) = _refit_at_delta(X, y, lam, S, delta)
            g_now = gF_fn(fd.theta)
            SA = stack_A(fd, Xp, y, S)
            lF = SA.T @ g_now
            Hmap = H_map_matrix(X, y, lam, S, g_now, 0.001, delta0=delta)
            Mcur = SA.T @ hF_fn(fd.theta) @ SA + Hmap if nonlin else Hmap
            (_, st) = trs_max(lF, Mcur, Delta)
            delta = delta + st
            if np.linalg.norm(delta) > R:
                delta *= R / np.linalg.norm(delta)
            best = max(best, _deltaF_true(fit, X, y, lam, S, gF_fn(fit.theta), delta))
        return best
    rows = []
    for (di, nm) in enumerate(names, 1):
        print(f'[{di}/{len(names)}] {nm}: loading...', flush=True)
        try:
            ds = DATASETS[nm](*DEFAULT_SIZE.get(nm, (500, 150)))
            (X, y) = (ds.X_train, ds.y_train)
        except Exception as e:
            print(f'  {nm} SKIP {type(e).__name__}')
            continue
        if len(X) > n:
            i = np.random.default_rng(seed).choice(len(X), n, replace=False)
            (X, y) = (X[i], y[i])
        fit = ridge_fit(X, y, lam)
        R = np.sqrt(M) * Rm
        (F_fn, gF_fn, hF_fn) = make_target(kind, ds.X_test, theta0=fit.theta, Xt=ds.X_test, yt=ds.y_test, **tk)
        gF = gF_fn(fit.theta)
        hF = hF_fn(fit.theta) if hF_fn is not None else None
        nonlin = hF_fn is not None
        S = select_optimal_points(X, y, fit, gF, M, p=sel_p, mode=sel_mode)['S']
        md = M * X.shape[1]
        print(f'    d={X.shape[1]}, md={md}, |S|={len(S)}: curvature...', flush=True)
        lF = ell_F(fit, X, y, S, gF)
        (B, Hmap, Mc) = full_curvature(fit, X, y, lam, S, gF, hF)
        (nM, nH) = (np.linalg.norm(Mc, 2), np.linalg.norm(Hmap, 2))
        DF = lambda d: _deltaF_true(fit, X, y, lam, S, gF, d)
        print(f'    one-shot TR1/TR2/TR3...', flush=True)
        d1 = trs_max(lF, np.zeros((md, md)), R)[1]
        d2 = trs_max(lF, B, R)[1]
        d3 = trs_max(lF, Mc, R)[1]
        print(f'    relin ({relin_K} steps) + L3 estimate + global probe...', flush=True)
        frel = _relin_un(fit, X, y, lam, S, R, gF_fn, hF_fn, relin_K, nonlin)
        L3 = L3_safety * _estimate_L3(fit, X, y, lam, S, gF, R, hess_F=hF)
        rng = np.random.default_rng(seed + 99)
        P = rng.normal(size=(n_probe, md))
        P *= R / np.linalg.norm(P, axis=1, keepdims=True)
        fprobe = max((DF(P[j]) for j in range(n_probe)), default=0.0)
        fstar = max(frel, DF(d1), DF(d2), DF(d3), fprobe)
        emp = {'TR1': fstar - DF(d1), 'TR2': fstar - DF(d2), 'TR3': fstar - DF(d3), 'relin': fstar - frel}
        bnd = {'TR1': nM * R ** 2 + L3 / 3 * R ** 3, 'TR2': nH * R ** 2 + L3 / 3 * R ** 3, 'TR3': L3 / 3 * R ** 3, 'relin': L3 / (3 * relin_K ** 2) * R ** 3}
        for mth in emp:
            g = max(emp[mth], 0.0)
            rows.append(dict(dataset=nm, method=mth, gap=g, bound=bnd[mth], ratio=g / (bnd[mth] + 1e-18), holds=bool(g <= bnd[mth] + 1e-06)))
        oks = {mth: emp[mth] <= bnd[mth] + 1e-06 for mth in emp}
        print(f'    L3={L3:.3g} (x{L3_safety}) fprobe={fprobe:.3g}  holds: ' + ' '.join((f"{mth}={('OK' if oks[mth] else 'CHECK')}" for mth in ['TR1', 'TR2', 'TR3', 'relin'])), flush=True)
    print('error_below_bound: assembling table...', flush=True)
    df = pd.DataFrame(rows)
    methods = ['TR1', 'TR2', 'TR3', 'relin']
    piv = df.pivot(index='dataset', columns='method')[['gap', 'bound', 'ratio']]
    piv = piv.swaplevel(axis=1).sort_index(axis=1).reindex(columns=methods, level=0).round(4)
    holds = df.pivot(index='dataset', columns='method')['holds'].reindex(columns=methods)
    print(f'Empirical gap vs analytic bound (uncapped budget ball).  holds = gap<=bound;  ALL OK = {df.holds.all()}', flush=True)
    display(piv)
    display(holds)
    return df
COMP_NAMES = list(DATASETS)
_ORDER = ['TR1', 'TR2', 'TR3', 'TR3-relin', 'bilevel']
_COL = {'TR1': '#1f6fff', 'TR2': '#ff8c00', 'TR3': '#16a01a', 'TR3-relin': '#e6194B', 'bilevel': 'k'}
_REGIMES = ['budget-only', 'all-caps']
from IPython.display import display

def table_computation_time(df=None):
    """Wall-time (ms) per dataset x method, split by regime — as a table."""
    df = COMP_TIME if df is None else df
    t = df.pivot_table(index='dataset', columns=['regime', 'method'], values='ms')
    t = t.reindex(columns=pd.MultiIndex.from_product([_REGIMES, _ORDER])).round(1)
    print('Computation time (ms)')
    display(t)
    return t

def table_eps_values(scales=None):
    """Analytic epsilon budgets per dataset — as a table."""
    scales = COMP_SCALES if scales is None else scales
    keys = ['eps_leave', 'eps_sat', 'eps_full', 'eps_backfire', 'eps_cubic', 'eps_3relin']
    t = pd.DataFrame({d: {k: scales[d].get(k, np.nan) for k in keys} for d in scales}).T
    t = t[keys].round(3)
    t.index.name = 'dataset'
    print('Analytic ε budgets')
    display(t)
    return t