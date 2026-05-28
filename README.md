
**Channel impairments applied per frame:**
- AWGN (controlled by SNR)
- Rician multipath (3 delayed paths, K=4)
- Clock offset (±5 ppm)
- IQ Imbalance (±5% gain, ±2° phase) ← extension

---

## Modulation Types

### Baseline (11 types)
| Type | Category | Bits/Symbol |
|---|---|---|
| BPSK | Digital PSK | 1 |
| QPSK | Digital PSK | 2 |
| 8PSK | Digital PSK | 3 |
| 16QAM | Digital QAM | 4 |
| 64QAM | Digital QAM | 6 |
| PAM4 | Digital PAM | 2 |
| GFSK | Digital FSK | 2 |
| CPFSK | Digital FSK | 2 |
| B-FM | Analog FM | — |
| DSB-AM | Analog AM | — |
| SSB-AM | Analog AM | — |

### Extended (coming soon)
- π/2-BPSK, 256QAM (5G Toolbox)
- LTE OFDM (LTE Toolbox)
- 802.11 OFDM / Wi-Fi (WLAN Toolbox)

---

## Extensions (over baseline)

| # | Extension | Status |
|---|---|---|
| 1 | SNR Sweep (-10 to 30 dB) | 🔲 In progress |
| 2 | IQ Imbalance channel impairment | 🔲 In progress |
| 3 | Extended modulation types (5G/LTE/WLAN) | 🔲 In progress |
| 4 | ResNet architecture vs baseline CNN | 🔲 In progress |

---

## Environment

| Item | Detail |
|---|---|
| MATLAB | R2026a |
| GPU | NVIDIA GeForce GTX 1650 Ti (4 GB) |
| Toolboxes | Communications, Deep Learning, Signal Processing, Parallel Computing, 5G, LTE, WLAN |

---

## Key Parameters

```matlab
sps = 8;                    % samples per symbol
spf = 1024;                 % samples per frame (CNN input size)
fs  = 200e3;                % sample rate (200 kHz)
SNR = 30;                   % dB (swept from -10 to 30 in Extension 1)
numFramesPerModType = 1000; % frames per class (11,000 total)
