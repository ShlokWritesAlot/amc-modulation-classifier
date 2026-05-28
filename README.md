# AMC Modulation Classifier
Deep learning-based Automatic Modulation Classification (AMC) using CNN and ResNet architectures in MATLAB R2026a.  
Replicates and extends the [MathWorks Modulation Classification with Deep Learning](https://in.mathworks.com/help/comm/ug/modulation-classification-with-deep-learning.html) example with novel contributions targeting IEEE GLOBECOM/ICC 2027.

---

## Overview
This project builds a deep learning pipeline that classifies radio modulation schemes directly from raw IQ samples — no feature engineering required. The classifier learns to distinguish between modulation types by treating the IQ signal as a 2-channel 1D image.

---

## Pipeline
```text
Random Bits → Modulator → Pulse Shaping → Channel → Frame [2×1024] → CNN → Label
```

Channel impairments applied per frame:
- AWGN (controlled by SNR)
- Rician multipath (3 delayed paths, K=4)
- Clock offset (±5 ppm)
- IQ Imbalance (±5% gain, ±2° phase) — *Extension 2*

---

## Modulation Types

### Baseline (11 types)

| Type    | Category    | Bits/Symbol |
|---------|-------------|-------------|
| BPSK    | Digital PSK | 1           |
| QPSK    | Digital PSK | 2           |
| 8PSK    | Digital PSK | 3           |
| 16QAM   | Digital QAM | 4           |
| 64QAM   | Digital QAM | 6           |
| PAM4    | Digital PAM | 2           |
| GFSK    | Digital FSK | 2           |
| CPFSK   | Digital FSK | 2           |
| B-FM    | Analog FM   | —           |
| DSB-AM  | Analog AM   | —           |
| SSB-AM  | Analog AM   | —           |

### Extended *(coming soon)*
- π/2-BPSK, 256QAM via 5G Toolbox
- LTE OFDM via LTE Toolbox
- 802.11 OFDM via WLAN Toolbox

---

## Extensions

| # | Extension                          | Status      |
|---|------------------------------------|-------------|
| 1 | SNR Sweep (−10 to 30 dB)           | In progress |
| 2 | IQ Imbalance channel impairment    | In progress |
| 3 | Extended modulation types          | In progress |
| 4 | ResNet architecture vs baseline CNN| In progress |

---

## Environment

| Item       | Detail                                                                                        |
|------------|-----------------------------------------------------------------------------------------------|
| MATLAB     | R2026a                                                                                        |
| GPU        | NVIDIA GeForce GTX 1650 Ti (4 GB)                                                             |
| Toolboxes  | Communications, Deep Learning, Signal Processing, Parallel Computing, 5G, LTE, WLAN          |

---

## Key Parameters

| Parameter          | Value   | Meaning                            |
|--------------------|---------|------------------------------------|
| `sps`              | 8       | Samples per symbol                 |
| `spf`              | 1024    | Samples per frame (CNN input)      |
| `fs`               | 200e3   | Sample rate (200 kHz)              |
| `SNR`              | 30      | Signal-to-noise ratio (dB)         |
| `numFramesPerModType` | 1000 | Frames per class (11,000 total)   |

---

## Architecture — Baseline CNN

```text
Input: complex IQ sequence [2 × 1024]
Conv1D(16)  → BN → ReLU → MaxPool
Conv1D(32)  → BN → ReLU → MaxPool
Conv1D(48)  → BN → ReLU → MaxPool
Conv1D(64)  → BN → ReLU → MaxPool
Conv1D(32)  → BN → ReLU → GlobalAvgPool
FC(11)      → Softmax → Predicted Label
```

| Property              | Value         |
|-----------------------|---------------|
| Total layers          | 23            |
| Learnable parameters  | 58,500        |
| Optimizer             | Adam          |
| Learning rate         | 0.001         |
| Epochs                | 10            |
| Batch size            | 256           |
| Loss function         | Cross-entropy |

---

## Script Structure

| Block | Description                          | Status      |
|-------|--------------------------------------|-------------|
| 1     | Parameters                           | Done        |
| 2     | Modulation types + bits per symbol   | Done        |
| 3     | Channel setup (AWGN, Rician, Clock)  | Done        |
| 4     | Frame generator loop (11,000 frames) | Done        |
| 5     | Train/test split (80/20)             | Done        |
| 6     | Build CNN architecture               | Done        |
| 7     | Training options (Adam, 10 epochs, GPU) | Done     |
| 8     | Train network                        | Done        |
| 9     | Evaluate on test set + confusion matrix | Done     |
| 10    | SNR sweep (−10 to 30 dB)            | In progress |
| 11    | IQ Imbalance impairment              | In progress |
| 12    | Extended modulation types            | In progress |
| 13    | ResNet architecture                  | In progress |
| 14    | Final comparison plots               | In progress |

---

## Results

| Experiment          | Accuracy | SNR          |
|---------------------|----------|--------------|
| Baseline CNN        | 99.77%   | 30 dB        |
| CNN SNR Sweep       | —        | −10 to 30 dB |
| CNN + IQ Imbalance  | —        | 30 dB        |
| Extended Mod Types  | —        | 30 dB        |
| ResNet vs CNN       | —        | 30 dB        |

---

## File Structure

```
main_amc.m                          — main script (all blocks)
helperModClassCNN.m                 — MathWorks helper: CNN architecture
helperModClassFrameGenerator.m      — MathWorks helper: frame slicing
helperModClassGetModulator.m        — MathWorks helper: modulator handles
helperModClassSplitData.m           — MathWorks helper: data splitting
helperModClassPlotScores.m          — MathWorks helper: score plotting
README.md
```

---

## Target Publication
IEEE GLOBECOM 2027 / IEEE ICC 2027

---

## Author
**Shlok Pandey** — Intern, Professor Darak's Lab, Manipal University Jaipur