| Model | Trainable middle params | CV macro-F1 (mean +- sd) | Locked test macro-F1 [95% CI] | Official test macro-F1 [95% CI] |
|---|---|---|---|---|
| ResNet end-to-end | - | - | 88.36 [84.02-91.55] | 96.58 [95.20-97.88] |
| Logistic regression | - | 89.44 +- 1.12 | 88.61 [84.18-91.75] | 98.25 [97.13-99.17] |
| Bottleneck only | 0 | 89.72 +- 1.51 | 88.73 [84.10-91.95] | 99.38 [98.74-99.89] |
| Classical, 120 params | 120 | 89.85 +- 1.48 | 89.03 [84.44-92.19] | 99.38 [98.74-99.89] |
| VQC, frozen random | 0 | 87.16 +- 1.82 | 88.02 [83.40-91.34] | 99.38 [98.83-99.80] |
| VQC, trained (120 params) | 120 | 87.92 +- 1.61 | 88.49 [83.96-91.72] | 99.48 [98.98-99.89] |
