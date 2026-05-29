# AMC Modulation Classifier

Deep learning-based Automatic Modulation Classification (AMC) using a CNN architecture in MATLAB R2026a. Replicates and extends the [MathWorks Modulation Classification with Deep Learning](https://www.mathworks.com/help/comm/ug/modulation-classification-with-deep-learning.html) example with novel contributions targeting IEEE GLOBECOM/ICC 2027.

---

## Overview

This project builds a deep learning pipeline that classifies radio modulation schemes directly from raw IQ samples — no feature engineering required. The classifier learns to distinguish modulation types by treating IQ signals as a 2-channel 1D input to a CNN.

```
Random Bits → Modulator → Pulse Shaping (SRRC, sps=8) → Channel → Frame [2×1024] → CNN → Label
```

**Channel Impairments:** AWGN (SNR controlled) · Rician multipath (3 paths, K=4) · Clock offset (±5 ppm) · IQ imbalance (±5% gain, ±2° phase)

---

## CNN Architecture

| Stage | Layer | Output Shape |
|---|---|---|
| Input | IQ frame (I/Q channels) | 1024×2 |
| Block 1 | Conv1D(16, k=8) → BN → ReLU → MaxPool | 512×16 |
| Block 2 | Conv1D(32, k=8) → BN → ReLU → MaxPool | 256×32 |
| Block 3 | Conv1D(48, k=8) → BN → ReLU → MaxPool | 128×48 |
| Block 4 | Conv1D(64, k=8) → BN → ReLU → MaxPool | 64×64 |
| Block 5 | Conv1D(32, k=8) → BN → ReLU → GlobalAvgPool | 32 |
| Output | FC(11) → Softmax | 11 |

**Total layers:** 23 · **Parameters:** ~58,500 · **Optimizer:** Adam · **LR:** 0.001 · **Epochs:** 10 · **Batch size:** 256 · **Loss:** Cross-entropy · **GPU:** GTX 1650 Ti

---

## Dataset

### Baseline — 11 Classes

`BPSK · QPSK · 8PSK · 16QAM · 64QAM · PAM4 · GFSK · CPFSK · B-FM · DSB-AM · SSB-AM`

| Split | Frames |
|---|---|
| Total | 11,000 |
| Train | 8,800 |
| Test | 2,200 |

### Extended — 15 Classes

Adds `PI/2-BPSK · 256QAM · LTE-OFDM · WLAN-OFDM` via MathWorks 5G/LTE/WLAN toolboxes.

| Split | Frames |
|---|---|
| Total | 15,000 |
| Train | 12,000 |
| Test | 3,000 |

**Key parameters:** `sps=8` · `spf=1024` · `fs=200 kHz` · `SNR=30 dB` · `1000 frames/class`

---

## Experiments & Results

### Summary

| Experiment | Classes | Accuracy | Train Time |
|---|---|---|---|
| Baseline (30 dB) | 11 | **99.77%** | 16m 04s |
| IQ Imbalance Augmentation | 11 | **99.91%** | 12m 43s |
| Extended Dataset | 15 | **99.40%** | 11m 00s |

### SNR Sweep (Baseline Model)

| SNR (dB) | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| Accuracy (%) | 9.09 | 9.09 | 13.27 | 43.41 | 74.86 | 87.05 | 98.50 | 99.18 | 99.77 |

> **Insight:** Sharp performance transition occurs between 0–10 dB; accuracy saturates above 20 dB.

### Notable Observations

- **Baseline confusion:** 16QAM↔64QAM, PAM4→B-FM at low SNR
- **IQ Augmentation:** ±5% gain / ±2° phase imbalance during training improved robustness and reduced train time
- **Extended (15-class):** Minimal accuracy drop vs. baseline despite adding complex waveforms (LTE-OFDM, WLAN-OFDM)

---

## File Structure

```
AMC-Modulation-Classifier/
├── main_amc.m                          # Main script (13-section pipeline)
├── helperModClassCNN.m                 # CNN architecture definition
├── helperModClassFrameGenerator.m      # IQ frame generation with channel impairments
├── helperModClassGetModulator.m        # Modulator objects for each class
├── helperModClassSplitData.m           # Train/test splitting utility
├── helperModClassPlotScores.m          # Confusion matrix & accuracy plots
└── README.md
```

**Script sections:** `1-Parameters · 2-Modulations · 3-Channel · 4-FrameGen · 5-Split · 6-CNN · 7-TrainOpts · 8-Train · 9-Eval · 10-SNRSweep · 11-IQImbalance · 12-Extended · 13-Plots`

---

## Environment

| Component | Details |
|---|---|
| MATLAB | R2026a |
| GPU | NVIDIA GTX 1650 Ti (4 GB) |
| OS | Windows 10 |
| Toolboxes | Communications · Deep Learning · Signal Processing · Parallel Computing · 5G · LTE · WLAN |

---

## Novel Contributions

1. **SNR robustness evaluation** across −10 to 30 dB with fine-grained sweep
2. **IQ imbalance modeling** as a training augmentation strategy for real-world hardware imperfections
3. **Unified AMC across 5G/LTE/WiFi** waveforms using native MATLAB toolbox integrations (15-class extension)


---

## Author

**Shlok Pandey**
