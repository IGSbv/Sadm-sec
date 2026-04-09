# SADM-SEC
## Spatially Aware Directional Modulation — Secure
### Physical Layer Security System | 8-Element ULA

---

```
=================================================================
  SADM-SEC  |  Spatial Aware Directional Modulation - Secure
  Physical Layer Security System  |  8-Antenna ULA
  BECE304L Analog Communication Systems  |  VIT Chennai
=================================================================
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [File Structure](#3-file-structure)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [Module Descriptions](#6-module-descriptions)
7. [Running Each Phase](#7-running-each-phase)
8. [BECE304L Syllabus Alignment](#8-bece304l-syllabus-alignment)
9. [Key Results](#9-key-results)
10. [Outputs](#10-outputs)
11. [GNU Radio Integration](#11-gnu-radio-integration)
12. [References](#12-references)

---

## 1. Project Overview

SADM-SEC is a Python simulation of a **Physical Layer Security (PLS)** system that uses an 8-element Uniform Linear Array (ULA) to transmit information securely to an intended receiver (Bob) while rendering any eavesdropper (Eve) unable to recover the signal.

### Core Idea

The transmit signal is:

```
X(t) = w · m(t) · √Ps  +  P_AN · n(t) · √Pan
```

Where:
- `w` = MRT beamforming weight vector steered toward Bob
- `P_AN` = Null-space projector — ensures AN is **zero at Bob, maximum at Eve**
- `m(t)` = message signal (any modulation scheme)
- `n(t)` = complex Gaussian noise vector

### What Makes It Secure

The key property is `P_AN · a(θ_Bob) ≈ 0` (machine precision: ~1.1e-16).  
Bob receives clean signal + array gain. Eve receives signal + full AN power.

### BECE304L Alignment

This project covers **Modules 2, 3, 4, 6, and 7** of the BECE304L syllabus:

| Module | Topic | SADM-SEC Implementation |
|--------|-------|------------------------|
| 2 | AM — single tone, BW, power, efficiency | AM generated, transmitted through SADM, demodulated |
| 3 | DSB-SC, SSB-SC, VSB | All generated via balanced/Hilbert method, transmitted through SADM |
| 4 | FM — narrow/wideband, Carson's rule | FM-NB (β=0.5) and FM-WB (β=5) implemented + discriminator demod |
| 6 | Noise Figure, Noise Temperature, FOM | Full `noise_analysis.py` module computing all Module 6 metrics |
| 7 | Sampling theorem | 8 kHz / 48 kHz sampling rates justified by Nyquist |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SADM-SEC SYSTEM                          │
│                                                                  │
│  Bob sends pilot ──► Alice estimates DOA (Root-MUSIC or MLP)    │
│                                  │                               │
│                          w = a(θ_Bob)/‖a‖                       │
│                          P_AN = E_n · E_n^H                     │
│                                  │                               │
│  Message m(t) ──► X = w·m·√Ps + P_AN·n·√Pan ──► [8 antennas]  │
│                                  │                               │
│              ┌───────────────────┼──────────────────┐            │
│              ▼                                       ▼            │
│         Bob (30°)                              Eve (-45°)        │
│    a_Bob^H · P_AN ≈ 0                    a_Eve^H · P_AN ≠ 0    │
│    SNR = +69 dB                           SNR = -13 dB          │
│    Demodulates cleanly                    Receives noise         │
└─────────────────────────────────────────────────────────────────┘
```

### DOA Estimation Pipeline

```
Pilot snapshots (N×K)
        │
        ├──► Root-MUSIC ──► Covariance R ──► Eigendecompose ──► Polynomial roots ──► θ_est
        │
        └──► MLP NN ──► Flatten Re/Im(R) ──► 128 features ──► model.predict() ──► θ_est
                                │
                          (Pre-trained on sadm_training_data.npz)
                          (Saved as ml_doa_model.pkl)
```

---

## 3. File Structure

```
SADM_CORE/
│
├── spatial_logic.py              # Core mathematical engine
│   ├── steering_vector()         # ULA steering vector a(θ)
│   ├── beamforming_weights()     # MRT weights w = a/‖a‖
│   ├── noise_projection_matrix() # P_AN null-space projector
│   ├── sadm_transmit()           # Transmit matrix X = w·m + P_AN·n
│   ├── virtual_channel()         # Channel model + thermal noise
│   ├── compute_snr_analytical()  # Analytical SNR at any angle
│   ├── secrecy_rate()            # Wyner wiretap Cs formula
│   ├── root_music_doa()          # Root-MUSIC estimator
│   ├── generate_pilot_ping()     # Uplink pilot simulation
│   ├── ml_doa_estimate()         # MLP NN inference
│   └── SADMTracker               # Real-time tracking class
│
├── noise_analysis.py             # Module 6 alignment engine
│   ├── noise_figure()            # NF = SNR_ch - SNR_out (dB)
│   ├── noise_temperature()       # Te = T0·(F-1)
│   ├── figure_of_merit_sadm()    # FOM for SADM system
│   ├── fom_am/dsb/ssb/fm()      # Textbook FOM formulas
│   ├── fom_vs_snr_sweep()        # FOM comparison sweep
│   ├── nf_vs_angle_sweep()       # NF spatial profile
│   └── print_noise_budget()      # Full Module 6 table printout
│
├── virtual_channel.py            # Channel simulation + ZMQ live mode
│   ├── ZMQAntennaSubscriber      # GNU Radio ZMQ subscriber
│   ├── run_simulation()          # Static + moving target simulation
│   └── run_zmq_live()            # Live GNU Radio mode
│
├── visualization.py              # Publication-quality plots
│   ├── plot_beam_pattern()       # Polar beam pattern
│   ├── plot_fom_comparison()     # FOM vs AM/DSB/FM [Module 6]
│   ├── plot_nf_vs_angle()        # NF spatial profile [Module 6]
│   ├── plot_moving_target()      # Tracking + SNR timeseries
│   ├── plot_an_heatmap()         # 2D AN power heat map
│   └── render_all()              # Master dashboard render
│
├── main.py                       # Master entry point
│
├── modulation_comparison.py      # All 6 schemes compared
│   ├── gen_am/dsb/ssb/fm/sadm() # Signal generators
│   ├── plot_signals()            # Waveform + spectrum figure
│   ├── plot_metrics()            # Metrics comparison figure
│   └── generate_pdf()            # PDF report
│
├── sadm_through_modulations.py   # Modulations through SADM channel
│   ├── sadm_transmit_modulated() # Transmit any modulation via SADM
│   ├── demodulate()              # Scheme-specific demodulators
│   ├── compute_metrics()         # SNR, fidelity, BER, Cs
│   └── generate_report_1()       # Report 1 PDF
│
├── sadm_literature_comparison.py # Benchmark against 5 papers
│   ├── compute_our_results()     # Our metrics at paper operating points
│   ├── plot_literature()         # 6-panel comparison figure
│   └── generate_report_2()       # Report 2 PDF
│
├── final_report.py               # Final comprehensive PDF report
│
├── realtime_monitor.py           # Live interactive GUI dashboard
│   ├── Demo mode (--demo)        # No hardware, sliders fully live
│   └── Live mode                 # Connects to GNU Radio ZMQ ports
│
├── sadm_gnuradio_block.py        # GNU Radio Embedded Python Block
├── generate_grc.py               # GRC flowgraph generator
├── sadm_flowgraph.grc            # GNU Radio Companion flowgraph
├── ml_doa_model.pkl              # Pre-trained MLP model
├── sadm_training_data.npz        # Training dataset for MLP
├── requirements.txt              # Python dependencies
│
└── outputs/                      # All generated figures and reports
    ├── sadm_plots.png            # Main 5-panel dashboard
    ├── modulation_signals.png    # Waveform + spectrum comparison
    ├── modulation_metrics.png    # Metrics bar charts + radar
    ├── sadm_through_signals.png  # Received + demodulated waveforms
    ├── sadm_through_metrics.png  # Through-SADM metrics
    ├── sadm_literature_plots.png # Literature comparison plots
    ├── modulation_report.pdf     # Standalone modulation comparison report
    ├── sadm_report_1.pdf         # Report 1: Modulations through SADM
    ├── sadm_report_2.pdf         # Report 2: Literature comparison
    └── SADM_SEC_Final_Report.pdf # Final comprehensive project report
```

---

## 4. Installation

### Requirements

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
scipy
scikit-learn
matplotlib
pyzmq
gnuradio          # only needed for GNU Radio live mode
```

> **Note:** `gnuradio` is only required if you want to run the live GNU Radio mode. All simulation phases work without it.

### Verify Installation

```bash
python3 main.py --phase math
```

You should see:
```
[1] Steering vector at Bob  (30.0 deg): ...
[2] Null-space check  ||P_AN . a_Bob|| = 1.11e-16  (should -> 0)
[3] Root-MUSIC DOA estimate: ~30.00 deg  (true: 30.0 deg)
[4] SNR at Bob (30.0 deg) : +69.03 dB
    SNR at Eve (-45.0 deg) : -13.37 dB
    Secrecy Rate            : 22.8666 bits/s/Hz
[5] ML Tracker converged to: ~30.00 deg  (true: 30.0 deg)
  All self-tests passed [OK]
```

---

## 5. Quick Start

### Run everything (all phases)

```bash
python3 main.py
```

### Run with Root-MUSIC only (no ML)

```bash
python3 main.py --no-ml
```

### Run individual phases

```bash
python3 main.py --phase math    # Phase 1: mathematical self-test
python3 main.py --phase noise   # Phase 2: Module 6 noise analysis
python3 main.py --phase sim     # Phase 3: static simulation
python3 main.py --phase track   # Phase 4: moving target tracking
python3 main.py --phase viz     # Phase 5: generate all plots
```

### Generate full modulation comparison

```bash
python3 modulation_comparison.py
```

### Transmit modulations through SADM and get Report 1

```bash
python3 sadm_through_modulations.py
```

### Generate literature comparison (Report 2)

```bash
python3 sadm_literature_comparison.py
```

### Generate final project report

```bash
python3 final_report.py
```

### Run interactive real-time monitor (demo mode)

```bash
python3 realtime_monitor.py --demo
```

---

## 6. Module Descriptions

### `spatial_logic.py` — Core Math Engine

The mathematical backbone of the entire system. Self-contained, no ZMQ required.

**Key functions:**

```python
# Compute the steering vector for angle theta
a = steering_vector(angle_deg=30.0)

# MRT beamforming weights toward Bob
w = beamforming_weights(bob_angle_deg=30.0)

# Null-space AN projector — zero at Bob
P_AN = noise_projection_matrix(bob_angle_deg=30.0)

# Transmit matrix: X = w*m + P_AN*n
X = sadm_transmit(message_samples, bob_angle_deg=30.0, snr_signal_db=20, snr_noise_db=10)

# Receive at any angle (includes path loss + thermal noise)
y = virtual_channel(X, rx_angle_deg=30.0, path_loss_db=60, thermal_noise_db=-30)

# Analytical SNR at any angle (no noise realization)
snr = compute_snr_analytical(rx_angle=30.0, bob_angle=30.0, snr_signal_db=20, snr_noise_db=10)

# Wyner secrecy capacity
Cs = secrecy_rate(snr_bob_db=69.03, snr_eve_db=-13.37)

# Root-MUSIC DOA estimate
angles = root_music_doa(snapshots, n_sources=1)

# Real-time tracking class
tracker = SADMTracker(bob_initial_angle=0.0, alpha=0.3, use_ml=True)
est_angle = tracker.update(pilot_snapshots)
X = tracker.transmit(message_samples)
```

### `noise_analysis.py` — Module 6 Framework

Implements all Module 6 metrics and textbook FOM formulas.

```python
# Noise Figure at any receiver angle
nf_db, nf_lin = noise_figure(rx_angle_deg, bob_angle_deg, snr_signal_db, snr_noise_db)

# Noise Temperature
Te = noise_temperature(nf_lin)   # Te = T0*(F-1), T0=290K

# Figure of Merit — SADM system
fom_db, snr_out_db, snr_ch_db = figure_of_merit_sadm(rx_angle, bob_angle, ...)

# Textbook FOM formulas (Module 6)
fom_am(m=1.0)         # = m²/2 / (1 + m²/2) = 0.333 = -4.77 dB
fom_dsb_sc()          # = 1.0 = 0 dB (baseline)
fom_ssb_sc()          # = 1.0 = 0 dB
fom_fm(beta=5)        # = 3β²(β+1) = 450 = +26.5 dB

# Print full Module 6 noise budget table
print_noise_budget(bob_angle_deg=30.0, eve_angle_deg=-45.0)
```

### `SADMTracker` — Real-Time Tracking

```python
tracker = SADMTracker(
    bob_initial_angle=30.0,   # starting estimate
    n_antennas=8,
    alpha=0.3,                # smoothing factor [0,1] — higher = faster tracking
    use_ml=True               # True = MLP NN, False = Root-MUSIC
)

# Update with new pilot data → returns smoothed angle estimate
new_angle = tracker.update(pilot_snapshots)   # shape (N, K)

# Transmit signal using current tracked angle
X = tracker.transmit(message_samples, snr_signal_db=20, snr_noise_db=10)
```

---

## 7. Running Each Phase

### Phase 1 — Mathematical Self-Test

Tests all core math functions and verifies the null-space property.

```bash
python3 main.py --phase math
```

Expected key results:
- `||P_AN · a_Bob|| ≈ 1.1e-16` (machine precision null)
- Root-MUSIC estimate within ±0.1° of true angle
- SNR Bob ≈ +69 dB, SNR Eve ≈ -13 dB
- Secrecy Rate ≈ 22.87 bits/s/Hz

### Phase 2 — Module 6 Noise Analysis

Prints the full BECE304L Module 6 noise budget table.

```bash
python3 main.py --phase noise
```

Output includes:
- Section A: System parameters
- Section B: Bob vs Eve — NF, Noise Temperature, FOM side by side
- Section C: Textbook FOM comparison table (AM / DSB-SC / SSB-SC / FM / SADM)

### Phase 3 — Static Simulation

Runs 8 blocks with Bob fixed at 30°, Eve at -45°.

```bash
python3 main.py --phase sim          # ML mode (default)
python3 main.py --phase sim --no-ml  # Root-MUSIC mode
```

### Phase 4 — Moving Target Tracking

Bob sweeps from -60° to +60° over 20 blocks. Tracker follows dynamically.

```bash
python3 main.py --phase track
python3 main.py --phase track --no-ml
```

### Phase 5 — Visualization

Renders the full 5-panel dashboard to `outputs/sadm_plots.png`.

```bash
python3 main.py --phase viz
```

Panels generated:
1. Polar beam pattern (message beam + AN pattern)
2. Figure of Merit comparison — SADM vs AM/DSB-SC/FM (Module 6)
3. Noise Figure vs Angle — spatial NF profile (Module 6)
4. Moving target tracking (angle tracking + SNR timeseries)
5. Artificial Noise 2D heat map

---

## 8. BECE304L Syllabus Alignment

### Module 2 — Amplitude Modulation

The AM signal `s(t) = Ac[1 + m·cos(2πfm·t)]cos(2πfc·t)` is generated with m=1,
transmitted through SADM-SEC, and demodulated via envelope detector at Bob.

**Key result:** Transmission efficiency η = 33.3% (carrier power wasted).  
FOM = m²/2 / (1 + m²/2) = -4.77 dB.

### Module 3 — DSB-SC and SSB-SC

DSB-SC: `s(t) = m(t)cos(2πfc·t)` — produced via balanced modulator equivalent.  
SSB-SC: `s(t) = m(t)cos(wct) - m̂(t)sin(wct)` — Hilbert (phase-shift) method.

**Key insight:** The MRT beamformer `w·s(t)` is structurally identical to a DSB-SC
modulator in the spatial domain. DSB-SC and SSB-SC are the most natural modulations
for SADM-SEC — they pass through with near-zero distortion (fidelity > 99%).

### Module 4 — FM

FM-NB (β=0.5): BW = 3 kHz by Carson's rule. FOM = 3×0.25×1.5 = 1.125 — marginal.  
FM-WB (β=5.0): BW = 12 kHz. FOM = 3×25×6 = 450 = +26.5 dB — massive SNR gain.

**Key insight:** FM phase is preserved at Bob because AN lies in his null space.
The FM discriminator operates correctly despite the SADM channel.

### Module 6 — Noise in Communication Systems

The `noise_analysis.py` module directly implements:

| Formula | Code |
|---------|------|
| `NF = SNR_in - SNR_out (dB)` | `noise_figure()` |
| `Te = T0(F-1)` | `noise_temperature()` |
| `FOM_AM = m²/2 / (1 + m²/2)` | `fom_am(m)` |
| `FOM_DSB = 1` | `fom_dsb_sc()` |
| `FOM_FM = 3β²(β+1)` | `fom_fm(beta)` |
| `FOM_SADM = SNR_out - SNR_ch` | `figure_of_merit_sadm()` |

Run `python3 main.py --phase noise` for the full BECE304L-formatted table output.

### Module 7 — Sampling Theorem

All simulations use:
- Fs = 48,000 Hz (simulation), 8,000 Hz (pilot tone)
- fm = 440 Hz (main) / 1,000 Hz (comparison)
- Nyquist satisfied: Fs >> 2·fm in all cases
- Block size = 1024 samples → ~128 ms duration at 8 kHz

---

## 9. Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Bob Output SNR | +69.03 dB | Analytical, Signal=20dB, AN=10dB |
| Eve Output SNR | -13.37 dB | At -45°, same parameters |
| Secrecy Rate Cs | 22.87 bits/s/Hz | Wyner wiretap model |
| Null-space error | 1.11e-16 | ‖P_AN · a_Bob‖ |
| Array gain | +9.03 dB | 10·log10(8) |
| Noise Figure @ Bob | -9.03 dB | Module 6 — improvement |
| Noise Figure @ Eve | +73.37 dB | Module 6 — severe degradation |
| FOM @ Bob | +9.03 dB | Better than DSB-SC, below FM-WB |
| FOM @ Eve | -73.37 dB | Far below AM (worst textbook) |
| Root-MUSIC RMSE | 0.03° | 50 trials, SNR=15dB, 256 snapshots |
| MLP NN RMSE | 2.41° | Pre-trained MLPRegressor |
| Best modulation through SADM | SSB-SC | 99.2% fidelity, +17.7 dB SNR |
| Eve fidelity (all schemes) | < 10% | Across all 5 modulations |

---

## 10. Outputs

All outputs are written to `outputs/`:

| File | Generator | Contents |
|------|-----------|----------|
| `sadm_plots.png` | `main.py --phase viz` | 5-panel dashboard |
| `modulation_signals.png` | `modulation_comparison.py` | Waveform + spectrum (6 schemes) |
| `modulation_metrics.png` | `modulation_comparison.py` | BW, η, FOM, NF bar charts |
| `sadm_through_signals.png` | `sadm_through_modulations.py` | Transmitted → Received → Demodulated |
| `sadm_through_metrics.png` | `sadm_through_modulations.py` | Fidelity, SNR, BER, radar |
| `sadm_literature_plots.png` | `sadm_literature_comparison.py` | 6-panel literature benchmark |
| `modulation_report.pdf` | `modulation_comparison.py` | Scheme comparison report |
| `sadm_report_1.pdf` | `sadm_through_modulations.py` | Modulations through SADM |
| `sadm_report_2.pdf` | `sadm_literature_comparison.py` | Literature benchmark |
| `SADM_SEC_Final_Report.pdf` | `final_report.py` | Complete 15-page final report |

---

## 11. GNU Radio Integration

The project includes full GNU Radio integration for hardware deployment.

### Flowgraph

Open `sadm_flowgraph.grc` in GNU Radio Companion. The flowgraph connects:

```
Audio Source → DSB-SC Modulator → [in0] SADM Block → ZMQ PUB Sink (×8, ports 5555–5562)
Noise Source              →       [in1] SADM Block
```

### GRC Python Block

`sadm_gnuradio_block.py` is a `gr.sync_block` with:
- **Inputs:** 2 (message stream, noise stream)
- **Outputs:** 8 (one per antenna element)
- **Parameters:** `bob_angle`, `snr_sig_db`, `snr_an_db`

Drop this into GRC as an Embedded Python Block.

### Live Monitor

Connect the ZMQ PUB sinks (ports 5555–5562) then run:

```bash
python3 realtime_monitor.py          # live mode
python3 realtime_monitor.py --demo   # demo mode (no hardware)
```

Interactive sliders control:
- Bob angle: -90° → +90°
- Signal power: 5 → 35 dB
- AN power: 0 → 30 dB

---

## 12. References

```
[1]  M. P. Daly and J. T. Bernhard, "Directional Modulation Technique for
     Phased Arrays," IEEE Trans. Antennas Propagat., vol. 57, no. 9, 2009.

[2]  Y. Ding and V. Fusco, "A Vector Approach for the Analysis and Synthesis
     of Directional Modulation Transmitters," IEEE Trans. Antennas Propagat.,
     vol. 61, no. 12, 2013.

[3]  N. Valliappan, A. Lozano, R. W. Heath Jr., "Antenna Subset Modulation
     for Secure Millimeter-Wave Communications," IEEE Trans. Commun., 2013.

[4]  S. Hu, F. Deng, J. Xu, "Robust Synthesis Scheme for Secure Directional
     Modulation in the Multibeam Satellite System," IEEE Access, 2016.

[5]  H. Shi, W. Li, J. Hu, L. Hanzo, "Directional Modulation Aided Secure
     Multi-User OFDM Networks," IEEE Trans. Veh. Technol., vol. 67, 2018.

[6]  A. D. Wyner, "The Wire-Tap Channel," Bell Syst. Tech. J., 1975.

[7]  R. Schmidt, "Multiple emitter location and signal parameter estimation,"
     IEEE Trans. Antennas Propagat., vol. 34, no. 3, 1986.

[8]  S. Haykin, Communication Systems, 5th ed., Wiley, 2019.

[9]  G. Kennedy, B. Davis, Electronic Communication Systems, 6th ed.,
     McGraw-Hill Education, 2017.
```

---

## Acknowledgements

Course: BECE304L — Analog Communication Systems  
Institution: VIT Chennai  
Syllabus version: 1.0 (Approved by Academic Council No. 66, 16-06-2022)

---

*SADM-SEC — Physical Layer Security through Spatial Null-Space Artificial Noise Projection*
#   S a d m - s e c  
 #   S a d m - s e c  
 #   S a d m - s e c  
 