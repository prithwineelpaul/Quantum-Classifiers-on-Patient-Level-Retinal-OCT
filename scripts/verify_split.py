#!/usr/bin/env python
"""Recompute the patient-level split of Kermany OCT2017 and the locked-test hash, without a GPU or any deep-learning package.

The procedure is a line-by-line copy of the split cell in both notebooks (Cell 3 of notebook 01, Cell 6 of notebook 02).

    python scripts/verify_split.py /path/to/OCT2017                      # folder holding train/ and test/
    python scripts/verify_split.py /path/to/OCT2017 --manifest results/split_manifest.json

Checks
  1. pool sizes (images / patients per class) are printed for comparison with Table 1 of the paper;
  2. all pools are patient- and image-disjoint, and no patient of the official test set is in any training pool;
  3. SHA-256 of the sorted locked-test file names must equal --expect (paper: b735ec67e3e4531b...);
  4. optionally, the pools equal the split_manifest.json written by the notebooks.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('root', help='Kermany OCT2017 folder that contains train/ and test/')
ap.add_argument('--manifest', default='', help='split_manifest.json from a notebook run (optional)')
ap.add_argument('--expect', default='b735ec67e3e4531ba2b911a0aa3221810d19b4c5aa1f91973c6927540a804961',
                help='locked-test SHA-256 reported in the paper (the full hash or its first 16 characters)')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--fast-dev-run', action='store_true', help='reproduce the tiny FAST_DEV_RUN subset (only for checking a smoke run)')
args = ap.parse_args()

CLASSES = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
CFG = dict(locked_test_frac=0.15, backbone_frac=0.50, bb_val_frac=0.10)


def patient_id(path):                                   # Kermany file names: CLASS-PATIENTID-N.jpeg
    return os.path.basename(path).split('-')[1]


def index_split(split):
    rows = []
    for lab, c in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(args.root, split, c, '*.jpeg'))):
            rows.append((f, lab, patient_id(f)))
    return pd.DataFrame(rows, columns=['path', 'label', 'patient'])


train_all, official = index_split('train'), index_split('test')
if train_all.empty or official.empty:
    sys.exit(f'no images found under {args.root}/train or {args.root}/test')
if args.fast_dev_run:
    def strat_sample(df, n_per_class, seed):
        parts = [d.sample(min(len(d), n_per_class), random_state=seed) for _, d in df.groupby('label')]
        return pd.concat(parts).sort_index().reset_index(drop=True)
    train_all, official = strat_sample(train_all, 120, args.seed), strat_sample(official, 20, args.seed)
shared = set(train_all.patient) & set(official.patient)
print(f'train folder: {len(train_all)} images / {train_all.patient.nunique()} patients; '
      f'official test: {len(official)} images / {official.patient.nunique()} patients')
print(f'patients in both: {len(shared)} -> {train_all.patient.isin(shared).sum()} train-folder images removed')
train_all = train_all[~train_all.patient.isin(shared)].reset_index(drop=True)

pat_label = train_all.groupby('patient')['label'].agg(lambda s: s.value_counts().idxmax())
rng = np.random.RandomState(args.seed)
assign = {}
for lab, grp in pat_label.groupby(pat_label):
    pats = np.array(sorted(grp.index)); rng.shuffle(pats)
    n = len(pats)
    a = int(round(n * CFG['locked_test_frac'])); b = a + int(round(n * CFG['backbone_frac']))
    for p in pats[:a]:  assign[p] = 'locked_test'
    for p in pats[a:b]: assign[p] = 'backbone'
    for p in pats[b:]:  assign[p] = 'head'
train_all['pool'] = train_all.patient.map(assign)
bb_pats = pat_label[pat_label.index.isin(train_all.loc[train_all.pool == 'backbone', 'patient'])]
for lab, grp in bb_pats.groupby(bb_pats):
    pats = np.array(sorted(grp.index)); rng.shuffle(pats)
    for p in pats[:max(1, int(round(len(pats) * CFG['bb_val_frac'])))]:
        assign[p] = 'backbone_val'
train_all['pool'] = train_all.patient.map(assign)

POOLS = {k: train_all[train_all.pool == k].reset_index(drop=True) for k in ['backbone', 'backbone_val', 'head', 'locked_test']}
POOLS['official_test'] = official.reset_index(drop=True)

ok = True
names = list(POOLS)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        if set(POOLS[a].patient) & set(POOLS[b].patient) or set(POOLS[a].path) & set(POOLS[b].path):
            print(f'FAIL: overlap between {a} and {b}'); ok = False
summary = pd.DataFrame({k: v.label.value_counts().reindex(range(len(CLASSES)), fill_value=0).values for k, v in POOLS.items()},
                       index=CLASSES).T
summary['images'] = summary.sum(axis=1); summary['patients'] = [v.patient.nunique() for v in POOLS.values()]
print(summary.to_string())
print('pools are patient- and image-disjoint' if ok else 'POOLS OVERLAP')

h = hashlib.sha256('\n'.join(sorted(POOLS['locked_test'].path.map(os.path.basename))).encode()).hexdigest()
print('locked-test SHA-256 (sorted file names):', h)
if args.expect:
    match = h.startswith(args.expect.lower())
    print(f"expected {args.expect}:", 'PASS' if match else 'FAIL')
    ok &= match

if args.manifest:
    man = json.load(open(args.manifest))
    for k, v in POOLS.items():
        same = sorted(v.path.map(os.path.basename)) == man.get(k)
        print(f'manifest {k:14s}:', 'PASS' if same else 'FAIL'); ok &= same

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
