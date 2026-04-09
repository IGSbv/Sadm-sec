"""
=============================================================================
SADM-SEC  |  noise_analysis.py
=============================================================================
MODULE 6 ALIGNMENT — Noise in Communication Systems  (BECE304L)
=============================================================================

This module reframes the SADM-SEC system in the standard Module 6 framework:

  1. Noise Figure (NF)
       F = SNR_in / SNR_out  (linear)
       NF = SNR_in_dB - SNR_out_dB  (dB)

  2. Noise Temperature
       T_e = T_0 * (F - 1),   T_0 = 290 K  (IEEE reference)

  3. Figure of Merit (FOM)
       FOM = SNR_output / SNR_channel_reference
       Compared against textbook results for AM, DSB-SC, SSB-SC, FM.

  4. SNR Budget Table
       Bob (legitimate) vs Eve (eavesdropper) — in Module 6 table format.

  5. FOM vs SNR_channel Sweep
       Shows how SADM advantage holds across the full SNR operating range.

Key insight mapped to syllabus:
  - The ULA beamformer (w) acts as a spatial DSB-SC modulator.
  - The null-space AN projector acts as a spatial noise injector for Eve.
  - Bob's FOM exceeds FM wideband due to array gain (N=8 → +9 dB).
  - Eve's FOM drops below conventional AM, demonstrating physical-layer security
    through deliberate noise figure degradation at the eavesdropper.

=============================================================================
"""

import numpy as np
from spatial_logic import (
    N_ANTENNAS, D_OVER_LAMBDA,
    compute_snr_analytical, secrecy_rate,
    steering_vector, beamforming_weights, noise_projection_matrix
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

T0 = 290.0        # IEEE reference noise temperature (Kelvin)
kB = 1.38e-23     # Boltzmann constant (J/K)

# Thermal floor used consistently across spatial_logic.py
THERMAL_FLOOR = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _channel_snr_db(snr_signal_db: float) -> float:
    """
    Reference channel SNR (dB) — signal power vs thermal noise only.
    No array gain, no artificial noise. Single-antenna baseline.
    """
    sig_pow = 10 ** (snr_signal_db / 10)
    return 10 * np.log10(sig_pow / THERMAL_FLOOR)


# ─────────────────────────────────────────────────────────────────────────────
#  1. NOISE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def noise_figure(rx_angle_deg: float,
                 bob_angle_deg: float,
                 snr_signal_db: float = 20.0,
                 snr_noise_db: float  = 10.0,
                 n_antennas: int = N_ANTENNAS) -> tuple:
    """
    Compute Noise Figure (NF) at a given receiver angle.

    Definition (Module 6):
        F   = SNR_in / SNR_out                  (linear)
        NF  = SNR_in_dB - SNR_out_dB            (dB)

    Physical interpretation in SADM-SEC:
        Bob  → AN lies in his null space → SNR_out ≈ SNR_in + array gain
               → NF_Bob ≈ -10·log10(N)  (negative = improvement, not degradation)
        Eve  → full AN power hits her   → SNR_out << SNR_in
               → NF_Eve >> 0 dB         (severe degradation)

    Parameters
    ----------
    rx_angle_deg  : float – receiver angle to evaluate
    bob_angle_deg : float – angle at which beam is steered
    snr_signal_db : float – transmit signal power (dB)
    snr_noise_db  : float – artificial noise power (dB)
    n_antennas    : int

    Returns
    -------
    nf_db  : float – Noise Figure in dB
    nf_lin : float – Noise Figure (linear, F)
    """
    snr_ch_db  = _channel_snr_db(snr_signal_db)
    snr_out_db = compute_snr_analytical(rx_angle_deg, bob_angle_deg,
                                        snr_signal_db, snr_noise_db,
                                        n_antennas)
    nf_db  = snr_ch_db - snr_out_db
    nf_lin = 10 ** (nf_db / 10)
    return nf_db, nf_lin


# ─────────────────────────────────────────────────────────────────────────────
#  2. NOISE TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────

def noise_temperature(nf_lin: float) -> float:
    """
    Effective Noise Temperature (Module 6):
        T_e = T_0 * (F - 1)

    For SADM-Eve, F >> 1  →  T_e >> T_0  (very hot receiver)
    For SADM-Bob, F < 1   →  T_e < 0 K  (mathematical artifact of array gain
                                          exceeding single-antenna reference)
    """
    return T0 * (nf_lin - 1)


# ─────────────────────────────────────────────────────────────────────────────
#  3. FIGURE OF MERIT — SADM-SEC
# ─────────────────────────────────────────────────────────────────────────────

def figure_of_merit_sadm(rx_angle_deg: float,
                          bob_angle_deg: float,
                          snr_signal_db: float = 20.0,
                          snr_noise_db: float  = 10.0,
                          n_antennas: int = N_ANTENNAS) -> tuple:
    """
    Figure of Merit for SADM system at a given receiver angle (dB).

    FOM = SNR_output_dB - SNR_channel_dB

    Interpretation:
        FOM > 0 dB → system improves SNR (Bob, via array gain)
        FOM < 0 dB → system degrades SNR (Eve, via AN injection)

    Returns
    -------
    fom_db        : float – Figure of Merit (dB)
    snr_out_db    : float – Output SNR at rx_angle
    snr_ch_db     : float – Reference channel SNR
    """
    snr_ch_db  = _channel_snr_db(snr_signal_db)
    snr_out_db = compute_snr_analytical(rx_angle_deg, bob_angle_deg,
                                        snr_signal_db, snr_noise_db,
                                        n_antennas)
    fom_db = snr_out_db - snr_ch_db
    return fom_db, snr_out_db, snr_ch_db


# ─────────────────────────────────────────────────────────────────────────────
#  4. TEXTBOOK FOM — Standard Analog Systems  (Module 6)
# ─────────────────────────────────────────────────────────────────────────────

def fom_am(modulation_index: float = 1.0) -> float:
    """
    Figure of Merit for conventional AM (Module 6).
        η   = (m²/2) / (1 + m²/2)   ← transmission efficiency
        FOM = η / (1 + η)  ≡  m²/2 / (1 + m²)

    For m=1:  FOM = 1/3  (-4.77 dB)  — worst among all analog systems.
    Carrier power is wasted, reducing effective SNR at the detector.
    """
    eta = (modulation_index**2 / 2) / (1.0 + modulation_index**2 / 2)
    return eta  # linear


def fom_dsb_sc() -> float:
    """
    Figure of Merit for DSB-SC (Module 6).
    FOM = 1  (0 dB) — no carrier waste, full power in sidebands.
    This is the textbook baseline against which FM is compared.
    """
    return 1.0


def fom_ssb_sc() -> float:
    """
    Figure of Merit for SSB-SC (Module 6).
    FOM = 1  (0 dB) — same as DSB-SC in terms of SNR performance.
    Advantage over DSB-SC is half the bandwidth, not better SNR.
    """
    return 1.0


def fom_fm(beta: float) -> float:
    """
    Figure of Merit for FM with a single modulating tone (Module 6).
        FOM = 3 β² (β + 1)

    For wide-band FM (β = 5):  FOM = 3×25×6 = 450  (+26.5 dB)
    FM trades bandwidth for SNR — the wider the bandwidth, the better the FOM.

    Parameters
    ----------
    beta : float – FM modulation index  (Δf / f_m)
    """
    return 3.0 * beta**2 * (beta + 1.0)


def fom_theoretical_array(n_antennas: int) -> float:
    """
    Theoretical FOM for SADM-Bob in the absence of AN at Bob's receiver.
    Due to coherent array combining: FOM = N  (N-fold SNR improvement).
    For N=8: FOM = 8  (+9.03 dB)
    """
    return float(n_antennas)


# ─────────────────────────────────────────────────────────────────────────────
#  5. FOM vs SNR_channel SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def fom_vs_snr_sweep(bob_angle_deg: float  = 30.0,
                     eve_angle_deg: float   = -45.0,
                     snr_signal_range       = None,
                     snr_noise_db: float    = 10.0,
                     n_antennas: int        = N_ANTENNAS) -> dict:
    """
    Sweep transmit signal power and compute FOM for all systems.

    Returns a dict of arrays suitable for plotting a Module 6 comparison figure:
        snr_ch   : reference channel SNR values (x-axis)
        fom_bob  : SADM-SEC Bob FOM (dB)
        fom_eve  : SADM-SEC Eve FOM (dB)
        fom_am   : conventional AM FOM (dB), m=1, constant
        fom_dsb  : DSB-SC FOM (dB), constant = 0
        fom_fm   : FM FOM (dB), β=5, constant (bandwidth limited)
        fom_arr  : theoretical array FOM (dB), constant = 10·log10(N)
    """
    if snr_signal_range is None:
        snr_signal_range = np.linspace(-10, 30, 100)

    snr_ch_arr, fom_bob_arr, fom_eve_arr = [], [], []

    for snr_sig in snr_signal_range:
        snr_ch = _channel_snr_db(snr_sig)
        snr_b  = compute_snr_analytical(bob_angle_deg, bob_angle_deg,
                                        snr_sig, snr_noise_db, n_antennas)
        snr_e  = compute_snr_analytical(eve_angle_deg, bob_angle_deg,
                                        snr_sig, snr_noise_db, n_antennas)
        snr_ch_arr.append(snr_ch)
        fom_bob_arr.append(snr_b - snr_ch)
        fom_eve_arr.append(snr_e - snr_ch)

    snr_ch_arr = np.array(snr_ch_arr)

    # Textbook systems — FOM is constant (independent of SNR_channel)
    fom_am_db   = 10 * np.log10(fom_am(1.0))
    fom_dsb_db  = 10 * np.log10(fom_dsb_sc())
    fom_fm_db   = 10 * np.log10(fom_fm(beta=5))
    fom_arr_db  = 10 * np.log10(fom_theoretical_array(n_antennas))

    return {
        "snr_ch"    : snr_ch_arr,
        "fom_bob"   : np.array(fom_bob_arr),
        "fom_eve"   : np.array(fom_eve_arr),
        "fom_am"    : np.full_like(snr_ch_arr, fom_am_db),
        "fom_dsb"   : np.full_like(snr_ch_arr, fom_dsb_db),
        "fom_fm"    : np.full_like(snr_ch_arr, fom_fm_db),
        "fom_array" : np.full_like(snr_ch_arr, fom_arr_db),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6. NOISE FIGURE vs ANGLE SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def nf_vs_angle_sweep(bob_angle_deg: float = 30.0,
                      snr_signal_db: float  = 20.0,
                      snr_noise_db: float   = 10.0,
                      n_antennas: int       = N_ANTENNAS,
                      n_pts: int            = 180) -> tuple:
    """
    Compute Noise Figure (NF) across all receiver angles.

    Demonstrates the spatial selectivity of the SADM noise injection:
        - Deep null in NF at Bob's angle (NF << 0 dB)
        - Large NF everywhere else (Eve is degraded regardless of her position)

    Returns
    -------
    angles : np.ndarray – angles swept (degrees)
    nf_db  : np.ndarray – Noise Figure at each angle (dB)
    """
    angles = np.linspace(-80, 80, n_pts)
    nf_db  = np.zeros(n_pts)
    for i, theta in enumerate(angles):
        nf_db[i], _ = noise_figure(theta, bob_angle_deg,
                                   snr_signal_db, snr_noise_db, n_antennas)
    return angles, nf_db


# ─────────────────────────────────────────────────────────────────────────────
#  7. SNR BUDGET TABLE  (Module 6 — print format)
# ─────────────────────────────────────────────────────────────────────────────

def print_noise_budget(bob_angle_deg: float = 30.0,
                       eve_angle_deg: float  = -45.0,
                       snr_signal_db: float  = 20.0,
                       snr_noise_db: float   = 10.0,
                       n_antennas: int       = N_ANTENNAS) -> dict:
    """
    Print a complete Module 6 noise budget in textbook format.

    Sections:
      A. System parameters
      B. Bob vs Eve — NF, Noise Temperature, FOM
      C. Textbook FOM comparison table  (AM / DSB-SC / SSB-SC / FM / SADM)
    """
    snr_bob = compute_snr_analytical(bob_angle_deg, bob_angle_deg,
                                     snr_signal_db, snr_noise_db, n_antennas)
    snr_eve = compute_snr_analytical(eve_angle_deg, bob_angle_deg,
                                     snr_signal_db, snr_noise_db, n_antennas)
    snr_ch  = _channel_snr_db(snr_signal_db)
    Cs      = secrecy_rate(snr_bob, snr_eve)

    # Noise Figure
    nf_bob_db, nf_bob_lin = noise_figure(bob_angle_deg, bob_angle_deg,
                                         snr_signal_db, snr_noise_db, n_antennas)
    nf_eve_db, nf_eve_lin = noise_figure(eve_angle_deg, bob_angle_deg,
                                         snr_signal_db, snr_noise_db, n_antennas)

    # Noise Temperature
    te_bob = noise_temperature(nf_bob_lin)
    te_eve = noise_temperature(nf_eve_lin)

    # FOM
    fom_bob_db = snr_bob - snr_ch
    fom_eve_db = snr_eve - snr_ch

    # Textbook systems
    systems = [
        ("AM  (m = 1)",             fom_am(1.0),              "Carrier wasted; FOM < 1"),
        ("DSB-SC",                  fom_dsb_sc(),             "Textbook baseline  (0 dB)"),
        ("SSB-SC",                  fom_ssb_sc(),             "Same SNR as DSB-SC; half BW"),
        ("FM  (beta = 5, wideband)",   fom_fm(beta=5),           "BW traded for SNR"),
        ("FM  (beta = 2, narrowband)", fom_fm(beta=2),           "Lower BW, lower FOM"),
        ("SADM-Bob  (N = 8)",       10**(fom_bob_db/10),      "Array gain + null-space AN"),
        ("SADM-Eve  (N = 8)",       10**(fom_eve_db/10),      "AN degrades eavesdropper"),
    ]

    W = 70
    print("\n" + "="*W)
    print("  SADM-SEC  |  MODULE 6 NOISE ANALYSIS  (BECE304L Alignment)")
    print("="*W)

    # Section A
    print("\n  [A] SYSTEM PARAMETERS")
    print(f"      Carrier Frequency       : 2.4 GHz (ISM Band)")
    print(f"      Array Size              : N = {n_antennas} antenna elements (ULA, d/lambda = 0.5)")
    print(f"      Array Gain (coherent)   : {10*np.log10(n_antennas):.2f} dB")
    print(f"      Signal Power (Tx)       : {snr_signal_db:+.1f} dB")
    print(f"      AN Power (Tx)           : {snr_noise_db:+.1f} dB")
    print(f"      Reference Channel SNR   : {snr_ch:+.2f} dB  (thermal noise only)")
    print(f"      Reference Temp T0       : {T0:.0f} K  (IEEE 60268)")

    # Section B
    print(f"\n  [B] BOB vs EVE - MODULE 6 METRICS")
    hdr = f"  {'Metric':<28} {'Bob @ {:.0f} deg'.format(bob_angle_deg):>18} {'Eve @ {:.0f} deg'.format(eve_angle_deg):>18}"
    print("  " + "-"*(W-2))
    print(hdr)
    print("  " + "-"*(W-2))
    print(f"  {'Output SNR (dB)':<28} {snr_bob:>+18.2f} {snr_eve:>+18.2f}")
    print(f"  {'Channel SNR (dB)':<28} {snr_ch:>+18.2f} {snr_ch:>+18.2f}")
    print(f"  {'Noise Figure NF (dB)':<28} {nf_bob_db:>+18.2f} {nf_eve_db:>+18.2f}")
    print(f"  {'Noise Temperature Te (K)':<28} {te_bob:>+18.1f} {te_eve:>+18.1f}")
    print(f"  {'Figure of Merit (dB)':<28} {fom_bob_db:>+18.2f} {fom_eve_db:>+18.2f}")
    print("  " + "-"*(W-2))
    print(f"  {'Secrecy Rate Cs (bits/s/Hz)':<28} {Cs:>+18.4f}")

    # Section C
    print(f"\n  [C] TEXTBOOK FOM COMPARISON  (Module 6 - Figure of Merit)")
    print("  " + "-"*(W-2))
    print(f"  {'System':<28} {'FOM (linear)':>12} {'FOM (dB)':>10}   Notes")
    print("  " + "-"*(W-2))
    for name, fom_lin, note in systems:
        fom_db = 10 * np.log10(abs(fom_lin) + 1e-30)
        sign   = "+" if fom_lin >= 0 else ""
        marker = " <" if "SADM" in name else ""
        print(f"  {name:<28} {fom_lin:>12.4f} {fom_db:>+10.2f}   {note}{marker}")
    print("="*W)

    return {
        "snr_bob"     : snr_bob,    "snr_eve"     : snr_eve,
        "snr_channel" : snr_ch,
        "nf_bob_db"   : nf_bob_db,  "nf_eve_db"   : nf_eve_db,
        "te_bob"      : te_bob,     "te_eve"       : te_eve,
        "fom_bob_db"  : fom_bob_db, "fom_eve_db"   : fom_eve_db,
        "secrecy_rate": Cs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = print_noise_budget()
