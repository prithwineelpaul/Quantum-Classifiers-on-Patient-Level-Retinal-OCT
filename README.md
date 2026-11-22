# No robust quantum advantage on patient-level retinal OCT: a controlled benchmark

Code and result files for the paper

> **No Robust Quantum Advantage for Variational and Kernel Quantum Classifiers on Patient-Level Retinal OCT: A Controlled Benchmark with IBM Hardware Validation** — Prithwineel Paul *et al.* (submitted to *Quantum Machine Intelligence*).

The study asks whether a quantum model adds anything on top of a strong classical image backbone for four-class retinal OCT
classification (Kermany OCT2017: NORMAL, CNV, DME, DRUSEN), when the comparison is controlled: the same frozen features, the
same patient-grouped folds, a locked patient-disjoint test set, and a classical control with **exactly** the same number of
trainable parameters for every quantum model. It covers variational heads (2-layer VQC, data re-uploading with L = 3 and 6),
IQP fidelity and projected quantum-kernel SVMs, the geometric difference of Huang et al. (2021), simulated shot and
depolarising noise, execution of the trained VQC on IBM hardware (`ibm_kingston`) with and without error mitigation, and
external validation on two independent datasets.

**Result in one line:** no quantum model beat its matched classical control on cross-validation, the locked test set, the
hardware run or the external data; the differences fall inside a ±1.5 pp equivalence margin.

---

## Repository layout

```
notebooks/
  01_backbone_vqc_controls.ipynb          run A: split, backbone, 2-layer VQC vs controls, data efficiency, noise
  02_benchmark_hardware_external.ipynb    run B: re-uploading VQCs, quantum kernels, geometric difference, locked test
                                          run C: IBM hardware (unmitigated + mitigated), external validation
src/ibm_run.py                            the Qiskit runner used for the hardware jobs (notebook 02 writes the same file)
scripts/verify_split.py                   recompute the patient-level split and the locked-test hash (no GPU)
scripts/verify_cv_tests.py                recompute the corrected resampled t-tests from the saved per-fold results
scripts/verify_hardware.py                recompute every hardware number from the raw IBM measurement bits
results/run_A, run_B, run_C               result files of the three reported runs (see results/README.md)
data/README.md                            how to obtain the three datasets
requirements.txt
```

## Where each result of the paper comes from

| Paper | Run | Notebook / cell | Output file(s) in the results folder |
|---|---|---|---|
| Table 1 (patient-level partition), locked-test hash | all | 01 / Cell 3 (02 / Cell 6 is identical) | `split_summary.csv`, `split_manifest.json` |
| Simulator agreement with PennyLane | all | 01 / Cell 8; 02 / Cells 11, 12, 16 | printed |
| Table 2 (parameter counts) | B | 02 / Cell 13 | `benchmark_param_counts.csv` |
| Table 4, Fig. 2 (CV and locked test, seven pre-declared pairs) | B | 02 / Cells 15, 17, 19, 20, 21 | `benchmark_cv_heads.csv`, `benchmark_cv_kernels.csv`, `benchmark_cv_tests.csv`, `benchmark_test_metrics.csv`, `benchmark_test_comparisons.csv`, `benchmark_verdicts.md` |
| Fig. 3 (data efficiency) | A | 01 / Cell 13 | `data_efficiency.csv` |
| Fig. 4, Tables A1–A2 (kernel concentration, geometric difference) | B | 02 / Cells 16, 18 | `benchmark_kernel_concentration.csv`, `benchmark_geometric_difference.csv` |
| Simulated noise (shots, depolarising; image noise) | A | 01 / Cells 16, 17 | `quantum_noise.csv`, `image_noise_robustness.csv` |
| Table 5, Fig. 5, Table A4 (IBM hardware, mitigation, job list) | C | 02 / Cells 22–26 | `hw_*.npy`, `hw_info_main.json`, `hw_results.csv`, `hw_summary.json`, `hw_mitigation_comparison.csv`, `hw_mitigation_tests.csv` |
| Table 6 (external validation) | C | 02 / Cells 27, 28 | `external_validation.csv`, `external_validation_tests.csv` |
| External validation: single-model robustness, per-class F1 (exploratory) | C | 02 / Cell 29 | `external_robustness_single_models.csv`, `external_per_class_f1.csv`, `external_prediction_distribution.csv` |
| Table A3 (run A locked / official test) | A | 01 / Cell 14 | `final_test_metrics.csv`, `final_test_comparisons.csv` |

All runs use the same patient partition (paper, Table 3). Runs B and C each fine-tuned their own backbone; run A
(notebook 01) was run in run C's results folder, so it reused run C's backbone checkpoint (`reuse_backbone = True`). Run B is
notebook 02 with `RUN_BENCHMARK = True`; run C is notebook 02 with `RUN_BENCHMARK = False`, `RUN_EXTERNAL = True` and the IBM
steps. Notebooks 01 and 02 reuse a backbone checkpoint or `features.npz` found in their results folder; give a run its own
results folder (`RESULTS_DIR`) to fine-tune an independent backbone instead.

## Three levels of verification

### 1. Check the reported numbers without re-training (minutes, CPU)

```bash
pip install -r requirements.txt
# the split and the locked-test hash (needs only the Kermany images)
python scripts/verify_split.py /path/to/OCT2017 --manifest results/run_C/split_manifest.json
# expected: locked-test SHA-256 b735ec67e3e4531ba2b911a0aa3221810d19b4c5aa1f91973c6927540a804961

# CV significance tests, from the saved per-fold results
python scripts/verify_cv_tests.py --results results/run_A     # notebook 01 (run A)
python scripts/verify_cv_tests.py --results results/run_B     # notebook 02 (run B: paper Table 4)
python scripts/verify_cv_tests.py --results results/run_C     # notebook 02 (run C replication)

# all hardware numbers, from the raw IBM measurement bits (also re-simulates the circuit in plain numpy)
python scripts/verify_hardware.py --results results/run_C
```

Each script prints `PASS`/`FAIL` per check and exits non-zero if anything fails. Running notebook 02 with the default
`IBM_STEP = 'none'` and the `results/run_C/` files in its results folder re-runs the hardware analysis cells (22–26) on the saved
hardware data without contacting IBM.

### 2. Re-run the pipeline end to end (GPU)

1. Get the data (`data/README.md`).
2. Open the notebook on **Kaggle** (add the `paultimothymooney/kermany2018` dataset as input, GPU T4/P100) or **Google
   Colab** (GPU runtime; add a `KAGGLE_API_TOKEN` secret so Cell 0 can download the dataset; results go to
   `MyDrive/oct_qml/results`). Locally, set `OCT_ROOT=/path/to/OCT2017` and `RESULTS_DIR=/path/to/results`.
3. Run all cells.

| Notebook | T4 runtime (approx.) |
|---|---|
| 01 | 1.5–2 h (backbone ≈ 40 min, CV ≈ 30 min, noise experiments ≈ 20 min) |
| 02, `RUN_BENCHMARK = True` | 1.5–2.5 h (+ backbone if no features are found) |
| 02, external validation only | ≈ 20 min |

The split, the locked-test hash and the parameter counts are deterministic. Network training uses cuDNN autotuning and
mixed precision, so re-trained models differ from ours at the level of GPU non-determinism (typically a few tenths of a
percentage point of macro-F1 per model); the conclusions do not depend on this. Each notebook ends by writing
`environment_01.txt` / `environment_02.txt` (Python, CUDA, GPU and `pip freeze`) to the results folder.

### 3. Smoke test (minutes, CPU)

`FAST_DEV_RUN=1` runs every cell on a tiny subset with 1 backbone epoch, 3 folds and 1 seed. It checks that the pipeline
runs; the numbers are meaningless.

```bash
FAST_DEV_RUN=1 OCT_ROOT=/path/to/OCT2017 RESULTS_DIR=/tmp/smoke jupyter nbconvert --to notebook --execute notebooks/01_backbone_vqc_controls.ipynb
FAST_DEV_RUN=1 OCT_ROOT=/path/to/OCT2017 RESULTS_DIR=/tmp/smoke jupyter nbconvert --to notebook --execute notebooks/02_benchmark_hardware_external.ipynb
```

## IBM hardware

The hardware section never contacts IBM unless `IBM_STEP` (Cell 4 of notebook 02) is changed from `'none'`.
To repeat the experiment you need an IBM Quantum account: store the API key as a secret named `IBM_QUANTUM_TOKEN`
(Kaggle: *Add-ons → Secrets*; Colab: key icon) and, optionally, the instance CRN as `IBM_QUANTUM_INSTANCE`.
Never put the key in the notebook. The steps are `'check'` (verifies the Qiskit circuit against the PyTorch model and
transpiles; no QPU time) → `'test'` (20 images × 256 shots) → `'main'` (300 images × 1,024 shots, Sampler, dynamical
decoupling only) → `'mit_test'` → `'mitigated'` (Estimator: TREX readout mitigation, Pauli twirling of the two-qubit gates,
dynamical decoupling). `'resume'` with `IBM_JOB_ID` fetches a finished job. `IBM_FAKE = True` runs the same steps on the
local `FakeTorino` noise model with no account.

Qiskit runs in a separate process (`ibm_run.py`) because importing it next to PyTorch/scikit-learn on Kaggle fails with
*"cannot allocate memory in static TLS block"*. Calibration and queue state change daily, so a new hardware run will
differ from ours in detail; the raw bits of our runs are in `results/run_C/` for exact re-analysis.

Disclosure kept from the paper: the mitigated run reused the physical qubits of the unmitigated run as the initial layout,
and the transpiler then produced 136 instead of 109 two-qubit gates, so the mitigated circuit was deeper than the
unmitigated one.

## External validation data

Duke/Srinivasan 2014 (Spectralis, 45 volumes) and OCTDL (Optovue). OCTDL classes are mapped as NO → NORMAL, DME → DME,
early and intermediate AMD → DRUSEN; late AMD and the other diseases are excluded. Kermany images are excluded by folder and
by file-name pattern, and the notebook asserts that no external image is a Kermany file. See `data/README.md`.

## Citation

See `CITATION.cff`. Please also cite the datasets (Kermany et al., *Cell* 2018; Srinivasan et al., *Biomed. Opt. Express*
2014; Kulyabin et al., *Sci. Data* 2024).

## Licence

Code: MIT (see `LICENSE`). The datasets are not redistributed here and keep their own licences.
