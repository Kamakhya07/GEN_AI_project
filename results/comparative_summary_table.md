| Condition | Accuracy | Macro-F1 | PSNR (dB) | SSIM | Mean Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| Exp A: Clean Reference | 0.773 | 0.673 | — | — | 197.03 ms |
| Exp B: Blurred (Baseline) | 0.318 | 0.308 | 28.09 | 1.7032 | 197.03 ms |
| Exp C: Classical Restored | 0.273 | 0.213 | 27.77 | 1.6648 | 198.2 ms |
| Exp D: GAN Restored (Ours) | 0.273 | 0.220 | 15.94 | 1.7422 | 357.62 ms |
| Exp E: Blur-Augmented Classifier | 0.591 | 0.434 | — | — | 197.03 ms |