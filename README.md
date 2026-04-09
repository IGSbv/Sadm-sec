# SADM-SEC — Modulation Comparison Suite
**BECE304L Analog Communication Systems | VIT Chennai**  
Modules 2, 3, 4 & 6 — AM, DSB-SC, SSB-SC, FM vs Same Schemes + DM Beamforming (N=8)

---

## Overview

This project compares five standard analog modulation schemes against their
SADM-SEC-enhanced counterparts (`+DM`), and includes the full SADM-SEC system
as a physical-layer security reference.

| Category | Schemes |
|---|---|
| Traditional | AM, DSB-SC, SSB-SC, FM-NB (β=0.5), FM-WB (β=5) |
| Traditional + DM | AM+DM, DSB-SC+DM, SSB-SC+DM, FM-NB+DM, FM-WB+DM |
| Reference | SADM-SEC (N=8 ULA, Bob @ 30°, Eve @ −45°) |

**What `+DM` means:** The existing modulated signal is passed through the
full SADM-SEC beamformer — MRT steering toward Bob *and* null-space AN
injection — giving coherent array gain (+9.03 dB) at the cost of the AN
power overhead (−9.1% η).

---

## File Structure

```
sadm_sec_modulation/
│
├── README.md                          ← this file
├── spatial_logic.py                   ← ULA math engine (steering vectors, AN projector, SNR)
├── noise_analysis.py                  ← FOM / NF formulas for all schemes + SNR sweep
├── modulation_comparision_updated.py  ← main simulation + plotting + PDF report
│
└── outputs/
    ├── modulation_signals.png         ← time-domain + spectrum + Eve signal + beam pattern
    ├── modulation_metrics.png         ← 6-panel metrics dashboard
    └── modulation_report.pdf          ← full A4 PDF report
```

---

## Quick Start

### Requirements
```
pip install numpy scipy matplotlib reportlab
```

### Run
```bash
python modulation_comparision_updated.py
```

Outputs are written to `outputs/`.

---

## Module Descriptions

### `spatial_logic.py`
Core SADM-SEC array processing engine.

| Function | Description |
|---|---|
| `steering_vector(θ)` | N×1 normalised ULA manifold vector |
| `beamforming_weights(θ)` | MRT weights = `a(θ_Bob)` |
| `noise_projection_matrix(θ)` | `P_AN = I − a·a†` — null-space AN projector |
| `compute_snr_analytical(...)` | Analytical Bob / Eve SNR (dB) |
| `secrecy_rate(snr_bob, snr_eve)` | Shannon secrecy capacity Cs (bits/s/Hz) |
| `sadm_transmit(...)` | Full N×L transmit matrix X(t) |
| `virtual_channel(X, θ)` | Receive combiner at arbitrary angle |

**Key physics:**
- Bob: `y_Bob = a†(θ_B)·w·s·√P_s + 0` — AN cancelled (null space)
- Eve: `y_Eve = AF·s·√P_s + a†(θ_E)·P_AN·n·√P_AN` — AN not cancelled

### `noise_analysis.py`
Figure of Merit and Noise Figure formulas (Module 6).

| Function | Formula |
|---|---|
| `fom_am(m)` | `(m²/2) / (1 + m²/2)` |
| `fom_dsb_sc()` | `1.0` |
| `fom_ssb_sc()` | `1.0` |
| `fom_fm(β)` | `3β²(β+1)` |
| `figure_of_merit_sadm(...)` | `10·log10(N)` at Bob |
| `fom_vs_snr_sweep(...)` | FOM curves vs channel SNR for Bob and Eve |

---

## Simulation Parameters

| Parameter | Value | Reference |
|---|---|---|
| Message frequency f_m | 1 000 Hz | Modules 2, 4 |
| Carrier frequency f_c | 10 000 Hz | Modules 2, 3, 4 |
| Sample rate Fs | 48 000 Hz | Module 7 (Nyquist) |
| AM modulation index m | 1.0 | Module 2 |
| FM narrowband index β | 0.5 | Module 4 |
| FM wideband index β | 5.0 | Module 4 |
| SADM array size N | 8 elements (ULA) | Course Project |
| SADM signal power | +20 dB | Module 6 |
| SADM artificial noise | +10 dB | Module 6 |
| Bob angle | 30° | Course Project |
| Eve angle | −45° | Course Project |
| Thermal noise floor | −40 dBW | Calibrated to SNR targets |

---

## Numerical Results (Table 1)

| Scheme | BW (kHz) | η (%) | FOM (dB) | NF (dB) | Array Gain | Secrecy |
|---|---|---|---|---|---|---|
| AM | 2.0 | 33.3 | −4.77 | +4.77 | — | No |
| AM+DM | 2.0 | 30.3 | +4.26 | −4.26 | +9.0 dB | No |
| DSB-SC | 2.0 | 100 | +0.00 | −0.00 | — | No |
| DSB-SC+DM | 2.0 | 90.9 | +9.03 | −9.03 | +9.0 dB | No |
| SSB-SC | 1.0 | 100 | +0.00 | −0.00 | — | No |
| SSB-SC+DM | 1.0 | 90.9 | +9.03 | −9.03 | +9.0 dB | No |
| FM-NB | 3.0 | 100 | +0.51 | −0.51 | — | No |
| FM-NB+DM | 3.0 | 90.9 | +9.54 | −9.54 | +9.0 dB | No |
| FM-WB | 12.0 | 100 | +26.53 | −26.53 | — | No |
| FM-WB+DM | 12.0 | 90.9 | +35.56 | −35.56 | +9.0 dB | No |
| **SADM-SEC** | **2.0** | **90.9** | **+9.03** | **−9.03** | **+9.0 dB** | **Yes (spatial)** |

**SADM-SEC only:** SNR_Bob = +69 dB, SNR_Eve = −13.4 dB, Cs ≈ 22.9 bits/s/Hz

---

## Key Design Decisions

**Why do `+DM` variants have η < 100%?**  
`+DM` means the full SADM-SEC system is applied — MRT beamforming *and*
null-space AN injection. AN consumes `P_AN / (P_s + P_AN) = 9.1%` of the
power budget regardless of which modulation scheme carries the signal.
Traditional (non-DM) schemes transmit no AN → η unchanged.

**Why does Eve's waveform look like noise?**  
`gen_sadm_eve()` computes `a†(θ_Eve)·X(t)` where X is the full transmit
matrix (signal + AN). The signal leakage at Eve is `|AF(−45°, 30°)|² · P_s`
— tiny due to off-axis array factor. The AN term `P_AN · ‖P_AN · a_Eve‖²`
is not cancelled (null only at Bob) and dominates completely. Eve's output
SNR ≈ −13.4 dB.

**Why is the beam pattern shown?**  
`|a†(θ)·w|²` peaks at Bob (30°). `a†(θ)·P_AN·P_AN†·a(θ)` is zero at Bob
and maximum at all other angles — these two together make the spatial
security mechanism visually explicit.

---

*BECE304L Analog Communication Systems | VIT Chennai | Modules 2, 3, 4, 6 | SADM-SEC Course Project*
