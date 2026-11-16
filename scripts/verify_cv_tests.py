#!/usr/bin/env python
"""Recompute the cross-validation significance tests of the paper from the per-fold results saved by the notebooks.

    python scripts/verify_cv_tests.py --results results/

Notebook 01: cv_runs.csv (one row per head x fold x seed)  ->  cv_tests.csv
Notebook 02: benchmark_cv_heads.csv + benchmark_cv_kernels.csv  ->  benchmark_cv_tests.csv

Test: Nadeau & Bengio (2003) corrected resampled t-test on the paired per-(fold, seed) macro-F1 differences,
t = mean(d) / sqrt((1/J + n_val/n_train) var(d)), J - 1 degrees of freedom, Holm correction across the pre-declared pairs.
Means and standard deviations per head are printed for comparison with the CV tables of the paper.
"""
import argparse, math, os, sys
import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results')
args = ap.parse_args()
R = args.results
ok, done = True, False


def corrected_ttest(d, n_train, n_test):
    d = np.asarray(d, float); J = len(d); v = d.var(ddof=1)
    if J < 2 or v == 0:
        return 0.0, 1.0
    t = d.mean() / math.sqrt((1 / J + n_test / n_train) * v)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


def compare(name, mine, saved, cols):
    global ok
    diff = max(float(np.nanmax(np.abs(mine[c].values.astype(float) - saved[c].values.astype(float)))) for c in cols)
    good = diff < 1e-6
    ok &= good
    print(f'  recomputed vs saved {name}: max |diff| = {diff:.1e}', 'PASS' if good else 'FAIL')


# ---------------------------------------------------------------- notebook 01
if os.path.exists(f'{R}/cv_runs.csv'):
    done = True
    CV = pd.read_csv(f'{R}/cv_runs.csv')
    print('Notebook 01: CV macro-F1 (%), mean +- sd over folds x seeds')
    print((CV.groupby('head').f1.agg(['mean', 'std']) * 100).round(2).to_string())
    ref = CV[CV['head'] == 'hybrid_vqc'].sort_values(['fold', 'seed']).set_index(['fold', 'seed'])
    n_tr, n_te = CV.n_train.mean(), CV.n_val.mean()
    rows = []
    for h in [x for x in CV['head'].unique() if x != 'hybrid_vqc']:
        o = CV[CV['head'] == h].sort_values(['fold', 'seed']).set_index(['fold', 'seed'])
        d = (ref.f1 - o.f1).values
        t, p = corrected_ttest(d, n_tr, n_te)
        rows.append(dict(comparison=f'hybrid_vqc - {h}', delta_f1_pp=100 * d.mean(), sd_pp=100 * d.std(ddof=1), t=t, p=p))
    T = pd.DataFrame(rows); T['p_holm'] = holm(T.p)
    print(T.round(4).to_string(index=False))
    if os.path.exists(f'{R}/cv_tests.csv'):
        S = pd.read_csv(f'{R}/cv_tests.csv').set_index('comparison').loc[T.comparison].reset_index()
        compare('cv_tests.csv', T, S, ['delta_f1_pp', 't', 'p', 'p_holm'])

# ---------------------------------------------------------------- notebook 02
if os.path.exists(f'{R}/benchmark_cv_heads.csv'):
    done = True
    CV = pd.read_csv(f'{R}/benchmark_cv_heads.csv')
    if os.path.exists(f'{R}/benchmark_cv_kernels.csv'):
        CV = pd.concat([CV, pd.read_csv(f'{R}/benchmark_cv_kernels.csv').assign(seed=0)], ignore_index=True)
    print('\nNotebook 02: CV macro-F1 (%), mean +- sd')
    print((CV.groupby('head').f1.agg(['mean', 'std']) * 100).round(2).to_string())
    saved = pd.read_csv(f'{R}/benchmark_cv_tests.csv')
    rows = []
    for q, c in zip(saved.quantum, saved.classical):           # the pre-declared pairs, in the saved order
        a = CV[CV['head'] == q].set_index(['fold', 'seed']).f1
        b = CV[CV['head'] == c].set_index(['fold', 'seed']).f1
        d = (a - b).dropna().values
        t, p = corrected_ttest(d, CV[CV['head'] == q].n_train.mean(), CV[CV['head'] == q].n_val.mean())
        rows.append(dict(quantum=q, classical=c, delta_f1_pp=100 * d.mean(), folds_quantum_better=f'{(d > 0).sum()}/{len(d)}', t=t, p=p))
    T = pd.DataFrame(rows); T['p_holm'] = holm(T.p)
    print(T.round(4).to_string(index=False))
    compare('benchmark_cv_tests.csv', T, saved, ['delta_f1_pp', 't', 'p', 'p_holm'])

if not done:
    sys.exit(f'no cv_runs.csv or benchmark_cv_heads.csv in {R}')
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
