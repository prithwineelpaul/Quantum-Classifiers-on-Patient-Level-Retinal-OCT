#!/usr/bin/env python
# Identical to the script written by Cell 23 of notebook 02 (kept here so it can be read and run on its own).
"""Runs the trained hybrid VQC on IBM hardware in its own process (avoids TLS clashes with torch/sklearn).

Modes
  check          verify the circuit, connect, transpile, record calibration (no QPU time)
  test           Sampler, first --test-n images x --test-shots shots, dynamical decoupling only
  main           Sampler, all images x --main-shots shots, dynamical decoupling only (unmitigated reference)
  mit_test       Estimator, first --test-n images: TREX readout mitigation + gate twirling + DD
  mitigated      Estimator, all (or --mit-n) images: TREX + gate twirling + DD   [--zne adds zero-noise extrapolation]
  resume         fetch any earlier job by --job-id (its mode is looked up in the saved job files)

Reads  RESULTS/hw_inputs.npz (ANG, QW_NP, EV_EXACT)
Writes RESULTS/hw_bits_<mode>.npy (Sampler) or RESULTS/hw_ev_<mode>.npy (Estimator), hw_info_<mode>.json, ibm_job_<mode>.json"""
import os, sys, json, glob, argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['check', 'test', 'main', 'mit_test', 'mitigated', 'resume'])
ap.add_argument('--job-id', default='')
ap.add_argument('--backend', default='ibm_kingston')
ap.add_argument('--fake', action='store_true', help='use local FakeTorino instead of IBM (no QPU time)')
ap.add_argument('--results', default='/kaggle/working/results' if os.path.isdir('/kaggle/working') else 'results')
ap.add_argument('--test-n', type=int, default=20)
ap.add_argument('--test-shots', type=int, default=256)
ap.add_argument('--main-shots', type=int, default=1024)
ap.add_argument('--mit-n', type=int, default=0, help='images for the mitigated run (0 = all)')
ap.add_argument('--mit-shots', type=int, default=1024)
ap.add_argument('--zne', action='store_true', help='also apply zero-noise extrapolation (resilience level 2; ~3x QPU time)')
args = ap.parse_args()
R = args.results

z = np.load(f'{R}/hw_inputs.npz')
ANG, QW_NP, EV_EXACT = z['ANG'], z['QW_NP'], z['EV_EXACT']
C, L, n, _ = QW_NP.shape
W = C * n
RANGES = [(l % (n - 1)) + 1 for l in range(L)]


def build(measure):
    # C parallel n-qubit sub-circuits, PennyLane conventions; measure=True adds the Y-basis readout
    a = ParameterVector('a', W)
    qc = QuantumCircuit(W, W) if measure else QuantumCircuit(W)
    for c in range(C):
        q = [c * n + i for i in range(n)]
        for i in range(n):
            qc.ry(a[c * n + i], q[i])                                   # AngleEmbedding(rotation='Y')
        for l in range(L):
            for i in range(n):
                phi, theta, omega = QW_NP[c, l, i]
                qc.rz(phi, q[i]); qc.ry(theta, q[i]); qc.rz(omega, q[i])  # Rot(phi, theta, omega)
            for i in range(n):
                qc.cx(q[i], q[(i + RANGES[l]) % n])                     # StronglyEntanglingLayers ring
    if measure:
        for j in range(W):
            qc.sdg(j); qc.h(j)                                          # |+i> -> |0>
        qc.measure(range(W), range(W))
    return qc, a


qc, a = build(measure=True)

# ---- noiseless check against the PyTorch model (no QPU time)
for i in range(3):
    b = qc.assign_parameters(dict(zip(a, ANG[i])))
    bits = StatevectorSampler(seed=i).run([b], shots=20000).result()[0].data.c.to_bool_array(order='little')
    err = np.abs((1 - 2 * bits.mean(0)) - EV_EXACT[i]).max()
    print(f'check image {i}: max |sampled - exact| = {err:.3f}')
    if err > 0.05:
        sys.exit('CIRCUIT DOES NOT MATCH THE TRAINED MODEL -- stop, do not submit')
print('circuit verified against the PyTorch model')

# ---- which kind of job is this?
job_meta = {}
if args.mode == 'resume':
    for f in glob.glob(f'{R}/ibm_job_*.json'):
        m = json.load(open(f))
        if m.get('job_id') == args.job_id:
            job_meta = m
    if not job_meta:
        print('job id not found in saved job files -> assuming a Sampler "main" job')
        job_meta = dict(mode='main', primitive='sampler')
primitive = job_meta.get('primitive') or ('estimator' if args.mode in ('mit_test', 'mitigated') else 'sampler')
tag = job_meta.get('mode', args.mode)

# ---- backend
if args.fake:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend, service = FakeTorino(), None
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get('IBM_QUANTUM_TOKEN', '')
    if not token:
        sys.exit('IBM_QUANTUM_TOKEN not set in the environment')
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token,
                                   instance=os.environ.get('IBM_QUANTUM_INSTANCE') or None)
    print('QPUs available:', [x.name for x in service.backends(simulator=False)])
    backend = service.backend(args.backend)
    st = backend.status()
    print(f'{backend.name}: {backend.num_qubits} qubits | operational {st.operational} | queue {st.pending_jobs}')
    if not st.operational and args.mode not in ('check', 'resume'):
        sys.exit('backend not operational right now -- try later')

# ---- transpile once; mitigated runs reuse the physical qubits of the unmitigated main run
init_layout = None
_main_info = next((f for f in [f'{R}/hw_info_main.json', f'{R}/hw_info.json'] if os.path.exists(f)), None)
if primitive == 'estimator' and _main_info:
    prev = json.load(open(_main_info))
    if prev.get('backend') == backend.name and len(prev.get('physical_qubits', [])) == W:
        init_layout = prev['physical_qubits']
        print('using the physical qubits of the main run:', init_layout)
pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42, initial_layout=init_layout)
circ, a_c = (qc, a) if primitive == 'sampler' else build(measure=False)
isa = pm.run(circ)
two_q = {k: v for k, v in isa.count_ops().items() if k in ('cz', 'ecr', 'cx')}
layout = list(isa.layout.final_index_layout())
gname = next(iter(two_q), None); tgt = backend.target; e2, ero = [], []
for ins in isa.data:
    if ins.operation.name == gname:
        p = tgt[gname].get(tuple(isa.find_bit(x).index for x in ins.qubits))
        if p is not None and p.error is not None:
            e2.append(p.error)
for qb in layout:
    p = tgt['measure'].get((qb,)) if 'measure' in tgt else None
    if p is not None and p.error is not None:
        ero.append(p.error)
info = dict(backend=backend.name, primitive=primitive, logical_cnots=C * L * n, two_qubit_gates=two_q,
            depth=isa.depth(), depth_2q=isa.depth(lambda ins: ins.operation.num_qubits == 2), physical_qubits=layout,
            median_2q_error=float(np.median(e2)) if e2 else None,
            median_readout_error=float(np.median(ero)) if ero else None)
print('transpiled:', info)
json.dump(info, open(f'{R}/hw_info_{tag}.json', 'w'), indent=2, default=str)   # one record per step
if tag == 'main':
    json.dump(info, open(f'{R}/hw_info.json', 'w'), indent=2, default=str)  # the unmitigated main run is the reference
VALS = ANG[:, [list(a_c).index(p) for p in isa.parameters]]

if args.mode == 'check':
    print(f'check only, nothing submitted. Main run: {len(ANG)} images x {args.main_shots} shots '
          f'= {len(ANG) * args.main_shots:,} shots.')
    sys.exit(0)

# ---- submit or resume
if args.mode == 'resume':
    if not args.job_id or service is None:
        sys.exit('resume needs --job-id and a real IBM connection')
    job = service.job(args.job_id)
    print(f'resuming {primitive} job {args.job_id} (mode {tag})')
else:
    if args.mode == 'test':
        idx, shots = np.arange(min(args.test_n, len(VALS))), args.test_shots
    elif args.mode == 'main':
        idx, shots = np.arange(len(VALS)), args.main_shots
    elif args.mode == 'mit_test':
        idx, shots = np.arange(min(args.test_n, len(VALS))), args.mit_shots
    else:
        idx, shots = np.arange(args.mit_n if args.mit_n else len(VALS)), args.mit_shots
    if primitive == 'sampler':
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots
        if not args.fake:
            sampler.options.dynamical_decoupling.enable = True
        job = sampler.run([(isa, VALS[idx])])
        settings = dict(dynamical_decoupling=not args.fake)
    else:
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
        est = Estimator(mode=backend)
        est.options.default_shots = shots
        if not args.fake:
            est.options.resilience_level = 2 if args.zne else 1          # 1 = TREX readout mitigation
            est.options.resilience.measure_mitigation = True              # TREX
            est.options.twirling.enable_gates = True                      # Pauli twirling of 2-qubit gates
            est.options.twirling.enable_measure = True                    # measurement twirling (part of TREX)
            est.options.dynamical_decoupling.enable = True
            est.options.dynamical_decoupling.sequence_type = 'XpXm'
            if args.zne:
                est.options.resilience.zne_mitigation = True
        obs = [[SparsePauliOp.from_sparse_list([('Y', [j], 1.0)], num_qubits=W).apply_layout(isa.layout)
                for j in range(W)]]                                       # nested list -> observables shape (1, W)
        job = est.run([(isa, obs, VALS[idx][:, None, :])])                # values (N, 1, P) -> evs (N, W)
        settings = dict(TREX=not args.fake, gate_twirling=not args.fake, dynamical_decoupling=not args.fake,
                        zne=bool(args.zne and not args.fake))
    json.dump(dict(job_id=job.job_id(), mode=args.mode, primitive=primitive, shots=shots, n=int(len(idx)),
                   backend=backend.name, settings=settings),
              open(f'{R}/ibm_job_{args.mode}.json', 'w'), indent=2)
    print(f'JOB ID: {job.job_id()}  ({primitive}, {len(idx)} images x {shots} shots, {settings}) '
          f'-> saved to {R}/ibm_job_{args.mode}.json', flush=True)

res = job.result()                                                      # waits in the queue
if primitive == 'sampler':
    BITS = res[0].data.c.to_bool_array(order='little')                  # (N, shots, W); index j = logical qubit j
    np.save(f'{R}/hw_bits_{tag}.npy', BITS)
    ev = 1 - 2 * BITS.mean(1)
    print(f'saved {R}/hw_bits_{tag}.npy  shape {BITS.shape}')
else:
    ev = np.asarray(res[0].data.evs, dtype=float).reshape(-1, W)        # mitigated <Y_j> per image
    std = np.asarray(res[0].data.stds, dtype=float).reshape(-1, W)
    np.save(f'{R}/hw_ev_{tag}.npy', ev); np.save(f'{R}/hw_ev_std_{tag}.npy', std)
    print(f'saved {R}/hw_ev_{tag}.npy  shape {ev.shape}')
ref = EV_EXACT[:len(ev)]
print(f'hardware vs exact <Y>: Pearson r = {np.corrcoef(ev.ravel(), ref.ravel())[0, 1]:.3f}, '
      f'MAE = {np.abs(ev - ref).mean():.3f}, attenuation slope = {(ev * ref).sum() / (ref ** 2).sum():.3f}')
try:
    print('QPU usage (s):', job.usage())
except Exception:
    pass
