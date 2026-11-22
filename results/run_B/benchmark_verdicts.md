| Quantum model | Matched classical | CV delta macro-F1 (pp) | CV p (Holm) | Locked-test delta (pp) [95% CI] | McNemar p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| hybrid_vqc | classical_matched | -1.82 (0/15) | 0.0341 | +0.08 [-0.32, +0.52] | 1 | classical better in CV only |
| reupload_L3 | classical_deep_2 | -1.22 (0/15) | 0.0506 | -0.29 [-0.74, +0.19] | 0.557 | no detectable difference |
| reupload_L6 | classical_deep_4 | -0.98 (0/15) | 0.0341 | -0.53 [-1.22, +0.03] | 0.0469 | classical better in CV only |
| qsvm_fidelity_q8 | rbf_svm_pca8 | -0.04 (3/5) | 1 | -0.19 [-0.82, +0.36] | 1 | no detectable difference |
| qsvm_projected_q8 | rbf_svm_pca8 | +0.18 (2/5) | 1 | -0.39 [-0.96, +0.11] | 1 | no detectable difference |
| qsvm_fidelity_q12 | rbf_svm_pca12 | +0.70 (3/5) | 1 | +0.35 [-0.34, +1.12] | 1 | no detectable difference |
| qsvm_projected_q12 | rbf_svm_pca12 | +0.86 (3/5) | 1 | +0.55 [-0.23, +1.31] | 0.586 | no detectable difference |
