#!/usr/bin/env python
"""Re-derive the IBM hardware results of the paper from the raw files saved by notebook 02, independently of the notebook.

    python scripts/verify_hardware.py --results results/

Needs only numpy, scipy, pandas, scikit-learn and torch (torch only to read the saved head weights).

Inputs (written by notebook 02, see results/README.md)
  hw_inputs.npz               ANG (N x 20 circuit angles), QW_NP (4 x 2 x 5 x 3 trained circuit weights), EV_EXACT (N x 20)
  hw_heads_fold0_seed0.pt     trained hybrid VQC head (fold 0, seed 0); only its output layer Linear(20 -> 4) is used here
  hw_images.csv               the 300 locked-test images (file, patient, label)
  hw_bits_main.npy            raw IBM measurement bits, N x shots x 20 (unmitigated Sampler job)
  hw_ev_mitigated.npy         mitigated <Y> per image and qubit (Estimator job; optional)

Checks
  1. an independent numpy state-vector simulation of the circuit reproduces EV_EXACT (the exact <Y> the head was trained on);
  2. macro-F1 of exact simulation and of hardware, patient-cluster bootstrap CIs, the hardware - exact difference,
     the patient-clustered McNemar test, attenuation slope and Pearson r, compared with hw_summary.json / the paper;
  3. if present, the mitigated run (clipped to [-1, 1] before classification, as in the notebook);
  4. if ibm_jobs/job-<id>-result.json files (downloaded from the IBM Quantum dashboard) are present and
     qiskit-ibm-runtime is installed: the saved arrays are identical to IBM's own records of the jobs.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results')
ap.add_argument('--n-boot', type=int, default=2000)
args = ap.parse_args()
R = args.results
CLASSES = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
PAPER = dict(f1_exact=90.39, f1_hw=89.42, f1_hw_lo=85.49, f1_hw_hi=92.85, delta_pp=-0.97, delta_lo=-2.94, delta_hi=0.63,
             p_clustered=0.2568, p_boot=0.241, slope=0.7467, pearson_r=0.9559, n_images=300, patients=164)
ok = True


def check(name, value, ref, tol):
    global ok
    good = ref is None or abs(value - ref) <= tol
    ok &= good
    print(f'  {name:42s} {value:10.4f}' + ('' if ref is None else f'   reference {ref:10.4f}   {"PASS" if good else "FAIL"}'))


# ---------------------------------------------------------------------------- 1. independent circuit simulation
def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def apply_1q(psi, U, w, n):
    psi = np.moveaxis(psi.reshape((2,) * n), w, 0)
    psi = np.tensordot(U, psi, axes=(1, 0))
    return np.moveaxis(psi, 0, w).reshape(-1)


def apply_cnot(psi, c, t, n):
    psi = psi.reshape((2,) * n).copy()
    idx = [slice(None)] * n; idx[c] = 1
    sub = psi[tuple(idx)]
    tt = t - (t > c)
    psi[tuple(idx)] = np.flip(sub, axis=tt)
    return psi.reshape(-1)


def expval_y(psi, w, n):
    Y = np.array([[0, -1j], [1j, 0]])
    return float(np.real(np.vdot(psi, apply_1q(psi, Y, w, n))))


def circuit_ev(angles, qw):
    """One n-qubit sub-circuit: RY(a) embedding -> StronglyEntanglingLayers (PennyLane convention) -> <Y> on each wire."""
    L, n, _ = qw.shape
    psi = np.zeros(2 ** n, dtype=complex); psi[0] = 1
    for i in range(n):
        psi = apply_1q(psi, ry(angles[i]), i, n)
    for l in range(L):
        for i in range(n):
            phi, theta, omega = qw[l, i]
            psi = apply_1q(psi, rz(omega) @ ry(theta) @ rz(phi), i, n)     # Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
        r = (l % (n - 1)) + 1
        for i in range(n):
            psi = apply_cnot(psi, i, (i + r) % n, n)
    return np.array([expval_y(psi, i, n) for i in range(n)])


z = np.load(f'{R}/hw_inputs.npz')
ANG, QW, EV_EXACT = z['ANG'], z['QW_NP'], z['EV_EXACT']
C, L, n, _ = QW.shape
ev_np = np.stack([np.concatenate([circuit_ev(ANG[k, c * n:(c + 1) * n], QW[c]) for c in range(C)]) for k in range(len(ANG))])
err = np.abs(ev_np - EV_EXACT).max()
print(f'1. independent numpy simulation vs saved exact <Y> ({len(ANG)} images x {C * n} qubits): max |diff| = {err:.2e}',
      'PASS' if err < 1e-4 else 'FAIL')
ok &= err < 1e-4

# ---------------------------------------------------------------------------- 2. hardware vs exact
import torch
ck = torch.load(f'{R}/hw_heads_fold0_seed0.pt', map_location='cpu', weights_only=False)
Wo, bo = ck['vqc']['out.weight'].double().numpy(), ck['vqc']['out.bias'].double().numpy()


def head(ev):
    lg = ev @ Wo.T + bo
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg)
    return p / p.sum(1, keepdims=True)


imgs = pd.read_csv(f'{R}/hw_images.csv', dtype={'patient': str})
y_all, g_all = imgs.label.map(CLASSES.index).values, imgs.patient.astype(str).values
BITS = np.load(f'{R}/hw_bits_main.npy')
n_hw = BITS.shape[0]
y, g, EVX = y_all[:n_hw], g_all[:n_hw], EV_EXACT[:n_hw]
EV_HW = 1 - 2 * BITS.mean(1)
pe, ph = head(EVX).argmax(1), head(EV_HW).argmax(1)

r_ = np.random.RandomState(99)                      # same bootstrap draws as the notebook
ug, inv = np.unique(g, return_inverse=True)
mem = [np.where(inv == k)[0] for k in range(len(ug))]
BOOTS = [np.concatenate([mem[k] for k in r_.randint(0, len(ug), len(ug))]) for _ in range(args.n_boot)]
f1 = lambda yy, pp: f1_score(yy, pp, average='macro')


def clustered_mcnemar(correct_a, correct_b, groups):
    d = pd.DataFrame({'g': groups, 'a': (correct_a & ~correct_b).astype(int), 'b': (~correct_a & correct_b).astype(int)}).groupby('g').sum()
    diff = (d.a - d.b).values.astype(float)
    return 1.0 if (diff ** 2).sum() == 0 else float(stats.chi2.sf(diff.sum() ** 2 / (diff ** 2).sum(), 1))


bh = [f1(y[i], ph[i]) for i in BOOTS]
d = np.array([f1(y[i], ph[i]) - f1(y[i], pe[i]) for i in BOOTS])
slope = float((EV_HW * EVX).sum() / (EVX ** 2).sum()); r = float(np.corrcoef(EV_HW.ravel(), EVX.ravel())[0, 1])
p_boot = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
is_paper = n_hw == PAPER['n_images'] and len(ug) == PAPER['patients']
P = PAPER if is_paper else {k: None for k in PAPER}
print(f'\n2. hardware vs exact simulation: {n_hw} images, {len(ug)} patients, {BITS.shape[1]} shots'
      + ('' if is_paper else '  (not the paper run -> no reference values)'))
check('macro-F1 exact simulation (%)', 100 * f1(y, pe), P['f1_exact'], 0.01)
check('macro-F1 IBM hardware (%)', 100 * f1(y, ph), P['f1_hw'], 0.01)
check('  95% CI low (%)', 100 * np.percentile(bh, 2.5), P['f1_hw_lo'], 0.01)
check('  95% CI high (%)', 100 * np.percentile(bh, 97.5), P['f1_hw_hi'], 0.01)
check('hardware - exact (pp)', 100 * (f1(y, ph) - f1(y, pe)), P['delta_pp'], 0.01)
check('  95% CI low (pp)', 100 * np.percentile(d, 2.5), P['delta_lo'], 0.01)
check('  95% CI high (pp)', 100 * np.percentile(d, 97.5), P['delta_hi'], 0.01)
check('patient-clustered McNemar p', clustered_mcnemar(ph == y, pe == y, g), P['p_clustered'], 1e-3)
check('bootstrap p', p_boot, P['p_boot'], 1e-3)
check('agreement with exact predictions (%)', 100 * (ph == pe).mean(), None, 0)
check('attenuation slope', slope, P['slope'], 1e-3)
check('Pearson r (measured vs exact <Y>)', r, P['pearson_r'], 1e-3)
if os.path.exists(f'{R}/hw_summary.json'):
    s = json.load(open(f'{R}/hw_summary.json'))
    same = abs(s['delta_f1_hw_minus_exact_pp'] - 100 * (f1(y, ph) - f1(y, pe))) < 1e-6 and abs(s['attenuation_slope'] - slope) < 1e-6
    print('  matches hw_summary.json written by the notebook:', 'PASS' if same else 'FAIL'); ok &= same
info = next((f'{R}/{f}' for f in ['hw_info_main.json', 'hw_info.json'] if os.path.exists(f'{R}/{f}')), None)
if info:
    i = json.load(open(info))
    print(f"  backend {i['backend']}, two-qubit gates {i['two_qubit_gates']}, 2q depth {i['depth_2q']}, "
          f"median 2q error {i['median_2q_error']:.4g}, median readout error {i['median_readout_error']:.4g}")

# ---------------------------------------------------------------------------- 3. mitigated run
# ---------------------------------------------------------------------------- 4. raw IBM job results (optional)
def check_raw():
    global ok
    import glob
    files = sorted(glob.glob(f'{R}/ibm_jobs/job-*-result.json'))
    if not files:
        print('\n4. no raw IBM job results in ibm_jobs/ -> skipped'); return
    import subprocess, tempfile
    # decode in a separate process: importing qiskit after torch can fail ("static TLS block")
    dec = ("import json,sys,numpy as np\n"
           "from qiskit_ibm_runtime import RuntimeDecoder\n"
           "r=json.load(open(sys.argv[1]),cls=RuntimeDecoder)[0].data\n"
           "np.save(sys.argv[2], r.c.to_bool_array(order='little') if hasattr(r,'c') else np.asarray(r.evs))\n")
    print('\n4. saved arrays vs raw results downloaded from IBM Quantum')
    for f in files:
        jid = os.path.basename(f)[4:-12]
        tmp = tempfile.mktemp(suffix='.npy')
        p = subprocess.run([sys.executable, '-c', dec, f, tmp], capture_output=True, text=True)
        if p.returncode != 0:
            print(f'  {jid}: could not decode (is qiskit-ibm-runtime installed?) -> skipped'); continue
        arr = np.load(tmp); os.remove(tmp)
        if arr.dtype == bool:                                          # Sampler: raw bits
            same = os.path.exists(f'{R}/hw_bits_main.npy') and np.array_equal(arr, np.load(f'{R}/hw_bits_main.npy').astype(bool))
            print(f'  {jid}: Sampler bits {arr.shape} identical to hw_bits_main.npy:', 'PASS' if same else 'FAIL')
            ok &= same
        else:                                                          # Estimator: mitigated <Y>
            ev = arr
            r_unm = np.corrcoef(ev.ravel(), EV_EXACT[:len(ev)].ravel())[0, 1]
            slope_j = float((ev * EV_EXACT[:len(ev)]).sum() / (EV_EXACT[:len(ev)] ** 2).sum())
            same = os.path.exists(f'{R}/hw_ev_mitigated.npy') and np.allclose(ev, np.load(f'{R}/hw_ev_mitigated.npy'))
            pm_j = head(np.clip(ev, -1, 1)).argmax(1)
            print(f'  {jid}: Estimator <Y> {ev.shape}; macro-F1 {100 * f1(y_all[:len(ev)], pm_j):.2f}%, '
                  f'attenuation {slope_j:.4f}, r {r_unm:.4f}' + ('  (= hw_ev_mitigated.npy)' if same else '  (another repetition)'))


if os.path.exists(f'{R}/hw_ev_mitigated.npy'):
    EV_MIT = np.load(f'{R}/hw_ev_mitigated.npy'); n_m = len(EV_MIT)
    ym, gm, EVm, RAWm = y_all[:n_m], g_all[:n_m], EV_EXACT[:n_m], EV_HW[:n_m]
    r_ = np.random.RandomState(99); ugm, invm = np.unique(gm, return_inverse=True)
    memm = [np.where(invm == k)[0] for k in range(len(ugm))]
    BM = [np.concatenate([memm[k] for k in r_.randint(0, len(ugm), len(ugm))]) for _ in range(args.n_boot)]
    pm, pu, px = head(np.clip(EV_MIT, -1, 1)).argmax(1), head(RAWm).argmax(1), head(EVm).argmax(1)
    print(f'\n3. mitigated run: {n_m} images, {len(ugm)} patients')
    ref = pd.read_csv(f'{R}/hw_mitigation_comparison.csv', index_col=0) if os.path.exists(f'{R}/hw_mitigation_comparison.csv') else None
    for label, pp, raw in [('exact', px, EVm), ('unmitigated', pu, RAWm), ('mitigated', pm, EV_MIT)]:
        bf = [f1(ym[i], pp[i]) for i in BM]
        rr = None
        if ref is not None:
            row = ref[ref.index.str.startswith({'exact': 'exact', 'unmitigated': 'hardware, unmitigated', 'mitigated': 'hardware, mitigated'}[label])]
            rr = row.iloc[0] if len(row) else None
        check(f'macro-F1 {label} (%)', 100 * f1(ym, pp), None if rr is None else 100 * rr.f1, 0.01)
        check(f'  95% CI ({label}) low / high (%)', 100 * np.percentile(bf, 2.5), None if rr is None else 100 * rr.f1_lo, 0.01)
        check('', 100 * np.percentile(bf, 97.5), None if rr is None else 100 * rr.f1_hi, 0.01)
        if label != 'exact':
            check(f'  attenuation slope ({label})', float((raw * EVm).sum() / (EVm ** 2).sum()), None if rr is None else rr.attenuation_slope, 1e-3)
    dm = np.array([f1(ym[i], pm[i]) - f1(ym[i], pu[i]) for i in BM])
    check('mitigated - unmitigated (pp)', 100 * (f1(ym, pm) - f1(ym, pu)), None, 0)
    check('  95% CI low / high (pp)', 100 * np.percentile(dm, 2.5), None, 0)
    check('', 100 * np.percentile(dm, 97.5), None, 0)
    check('  patient-clustered McNemar p', clustered_mcnemar(pm == ym, pu == ym, gm), None, 0)
else:
    print('\n3. hw_ev_mitigated.npy not found -> mitigated run not checked')

check_raw()
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
