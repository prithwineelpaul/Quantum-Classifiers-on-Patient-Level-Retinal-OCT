# Result files

One folder per run (paper, Table 3). All runs use the same patient-level partition; runs B and C fine-tuned their own backbones, and run A reused run C's backbone.

| Folder | Run | Platform | Produced by |
|---|---|---|---|
| `run_A/` | A: VQC vs classical controls, data efficiency, simulated noise | Colab T4, backbone of run C | notebook 01 |
| `run_B/` | B: seven pre-declared pairs, kernels, geometric difference (Table 4, Fig. 2, Fig. 4) | Colab T4, own backbone | notebook 02, `RUN_BENCHMARK = True`, `RUN_EXTERNAL = False` |
| `run_C/` | C: IBM hardware, error mitigation, external validation (Tables 5, 6, A4, Fig. 5) | Colab T4 + IBM | notebook 02 |

`run_C/` also contains a complete replication of the run-B benchmark on run C's backbone (`benchmark_*` files).
It is consistent with run B: every locked-test difference lies within the ±1.5 pp margin, and no quantum model beats its matched control.

## Checking the numbers

```bash
python scripts/verify_split.py /path/to/OCT2017 --manifest results/run_C/split_manifest.json
python scripts/verify_hardware.py --results results/run_C
python scripts/verify_cv_tests.py --results results/run_A     # notebook 01 CV tests
python scripts/verify_cv_tests.py --results results/run_B     # notebook 02 CV tests (paper, Table 4)
python scripts/verify_cv_tests.py --results results/run_C     # replication
```

With `qiskit-ibm-runtime` installed, `verify_hardware.py` also confirms that `hw_bits_main.npy` is bit-for-bit identical to
IBM's own record of the main job (`run_C/ibm_jobs/job-dbc2fds24mrc73cilusg-result.json`, downloaded from the IBM Quantum
dashboard), and re-analyses mitigated repetition 1 from its raw result (`job-dbddemvi0auc73cqt4kg-result.json`).

## Files

| File | Written by | Content |
|---|---|---|
| `split_manifest.json`, `split_summary.csv` | split cell | file names of every pool; Table 1 |
| `config.json` | settings cell | all hyper-parameters of the run |
| `backbone_history.csv`, `backbone_training.png` | backbone cell | backbone training curve |
| `cv_runs.csv`, `cv_tests.csv`, `data_efficiency.csv`, `final_test_metrics.csv`, `final_test_comparisons.csv`, `quantum_noise.csv`, `image_noise_robustness.csv`, `param_counts.csv`, `summary.json` | notebook 01 | run A |
| `benchmark_param_counts.csv`, `benchmark_cv_heads.csv`, `benchmark_cv_kernels.csv`, `benchmark_kernel_selected_params.csv`, `benchmark_cv_tests.csv`, `benchmark_kernel_concentration.csv`, `benchmark_geometric_difference.csv`, `benchmark_test_metrics.csv`, `benchmark_test_comparisons.csv`, `benchmark_verdicts.md`, `*.png` | notebook 02, `RUN_BENCHMARK` | runs B (and C replication) |
| `hw_inputs.npz` | hardware cell 1 | circuit angles, trained circuit weights, exact ⟨Y⟩ for the 300 images |
| `hw_heads_fold0_seed0.pt`, `hw_images.csv` | hardware cell 1 | the trained head and the 300 images sent to the QPU |
| `hw_bits_main.npy` | `ibm_run.py main` | raw measurement bits of the main job, 300 × 1024 × 20 |
| `hw_bits_test.npy` | `ibm_run.py test` | raw bits of the last test job (20 × 256 × 20) |
| `hw_ev_mitigated.npy`, `hw_ev_std_mitigated.npy` | `ibm_run.py mitigated` | mitigated ⟨Y⟩ and standard errors of the last mitigated job (repetition 2, `dbddkchgv2cc73do7fr0`) |
| `hw_ev_mit_test.npy`, `hw_ev_std_mit_test.npy` | `ibm_run.py mit_test` | mitigated test job (20 images) |
| `hw_info_main.json`, `hw_info_mitigated.json`, `hw_info.json` | `ibm_run.py` | backend, physical qubits, gate counts, calibration medians |
| `ibm_job_*.json` | `ibm_run.py` | job ID and settings of the last job of each step |
| `ibm_jobs/job-<id>-result.json` | IBM Quantum dashboard (download) | IBM's raw results of the main job and of mitigated repetition 1 |
| `hw_results.csv`, `hw_summary.json`, `hw_results.png`, `hw_paper_paragraph.txt` | hardware analysis cell | Table 5, Fig. 5a |
| `hw_mitigation_comparison.csv`, `hw_mitigation_tests.csv`, `hw_mitigation.png`, `hw_mitigation_paragraph.txt` | mitigation cell | Fig. 5b |
| `external_validation.csv`, `external_validation_tests.csv`, `external_confusion.png`, `external_paragraph.txt` | external validation cell | Table 6 |
| `external_robustness_single_models.csv`, `external_per_class_f1.csv`, `external_prediction_distribution.csv` | exploratory cell | single-model and per-class analysis of Section 6.6 |
| `environment_*.txt` | last cell | Python, CUDA, GPU and `pip freeze` of the run |

## Not stored in git

Large binaries are excluded by `.gitignore`: `backbone_resnet18d_ra2_in1k.pt` (45 MB), `features.npz` (27 MB),
`ext_features_*.npz`, `ext_head_ensembles.pt`. They are only needed to skip re-training; attach them to a GitHub release
or a Zenodo record if you want to share them.
