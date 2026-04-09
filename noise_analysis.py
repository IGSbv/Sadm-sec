"""
=============================================================================
SADM-SEC  |  noise_analysis.py (Unified)
=============================================================================
MODULE 6 ALIGNMENT — Noise in Communication Systems  (BECE304L)
Includes all sweep functions required for Phase 5 Visualization.
"""

import numpy as np
from spatial_logic import (
    N_ANTENNAS, compute_snr_analytical, secrecy_rate,
    THERMAL_NOISE_DB
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

T0 = 290.0        # IEEE reference noise temperature (Kelvin)
THERMAL_FLOOR = 10 ** (THERMAL_NOISE_DB / 10)  # Align with spatial_logic.py

def _channel_snr_db(snr_signal_db: float) -> float:
    """Reference channel SNR (dB) — signal power vs thermal noise only."""
    sig_pow = 10 ** (snr_signal_db / 10)
    return 10 * np.log10(sig_pow / THERMAL_FLOOR)

# ─────────────────────────────────────────────────────────────────────────────
#  1. NOISE FIGURE & TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────

def noise_figure(rx_angle_deg: float,
                 bob_angle_deg: float,
                 snr_signal_db: float = 20.0,
                 snr_noise_db: float  = 10.0,
                 n_antennas: int = N_ANTENNAS) -> tuple:
    snr_ch_db  = _channel_snr_db(snr_signal_db)
    snr_out_db = compute_snr_analytical(rx_angle_deg, bob_angle_deg,
                                        snr_signal_db, snr_noise_db,
                                        n_antennas)
    nf_db  = snr_ch_db - snr_out_db
    nf_lin = 10 ** (nf_db / 10)
    return nf_db, nf_lin

def noise_temperature(nf_lin: float) -> float:
    return T0 * (nf_lin - 1)

# ─────────────────────────────────────────────────────────────────────────────
#  2. TEXTBOOK FOM — Standard Analog Systems  (Module 6)
# ─────────────────────────────────────────────────────────────────────────────

def fom_am(modulation_index: float = 1.0) -> float:
    eta = (modulation_index**2 / 2) / (1.0 + modulation_index**2 / 2)
    return eta

def fom_dsb_sc() -> float:
    return 1.0

def fom_ssb_sc() -> float:
    return 1.0

def fom_fm(beta: float) -> float:
    return 3.0 * beta**2 * (beta + 1.0)

def fom_theoretical_array(n_antennas: int) -> float:
    return float(n_antennas)

# ─────────────────────────────────────────────────────────────────────────────
#  3. SWEEP FUNCTIONS (Required for Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE OF MERIT — SADM-SEC
# ─────────────────────────────────────────────────────────────────────────────

def figure_of_merit_sadm(rx_angle_deg: float,
                          bob_angle_deg: float,
                          snr_signal_db: float = 20.0,
                          snr_noise_db: float  = None,
                          n_antennas: int = N_ANTENNAS) -> tuple:
    """
    Figure of Merit for SADM system at a given receiver angle (dB).
    """
    # CRITICAL FIX: If no explicit noise floor is provided, force it to scale 
    # proportionally with the signal (e.g., 10 dB below signal power).
    if snr_noise_db is None:
        snr_noise_db = snr_signal_db - 10.0
        
    snr_ch_db  = _channel_snr_db(snr_signal_db)
    
    # Strict kwargs to prevent positional hijacking
    snr_out_db = compute_snr_analytical(
        rx_angle_deg=rx_angle_deg, 
        bob_deg=bob_angle_deg,
        signal_pow_db=snr_signal_db, 
        an_pow_db=snr_noise_db,
        n_antennas=n_antennas
    )
    fom_db = snr_out_db - snr_ch_db
    return fom_db, snr_out_db, snr_ch_db

def fom_vs_snr_sweep(bob_angle_deg: float  = 30.0,
                     eve_angle_deg: float   = -45.0,
                     snr_signal_range       = None,
                     snr_noise_db: float    = 10.0,
                     n_antennas: int        = N_ANTENNAS) -> dict:
    if snr_signal_range is None:
        # Sweeping transmit signal power from -10 to 30 dBW
        snr_signal_range = np.linspace(-10, 30, 100)

    snr_ch_arr, fom_bob_arr, fom_eve_arr = [], [], []

    for snr_sig in snr_signal_range:
        snr_ch = _channel_snr_db(snr_sig)
        
        # Keep AN proportional to signal
        current_an_pow_db = snr_sig - 10.0

        # STRICT KWARGS to prevent silent positional hijacking
        snr_b  = compute_snr_analytical(
            rx_angle_deg=bob_angle_deg, 
            bob_deg=bob_angle_deg,
            signal_pow_db=snr_sig, 
            an_pow_db=current_an_pow_db, 
            n_antennas=n_antennas
        )
        snr_e  = compute_snr_analytical(
            rx_angle_deg=eve_angle_deg, 
            bob_deg=bob_angle_deg,
            signal_pow_db=snr_sig, 
            an_pow_db=current_an_pow_db, 
            n_antennas=n_antennas
        )
        
        snr_ch_arr.append(snr_ch)
        fom_bob_arr.append(snr_b - snr_ch)
        fom_eve_arr.append(snr_e - snr_ch)

    snr_ch_arr = np.array(snr_ch_arr)
    
    # Textbook baseline limits
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

def nf_vs_angle_sweep(bob_angle_deg: float = 30.0,
                      snr_signal_db: float  = 20.0,
                      snr_noise_db: float   = 10.0,
                      n_antennas: int       = N_ANTENNAS,
                      n_pts: int            = 180) -> tuple:
    """Computes Noise Figure across all receiver angles."""
    angles = np.linspace(-80, 80, n_pts)
    nf_db  = np.zeros(n_pts)
    for i, theta in enumerate(angles):
        nf_db[i], _ = noise_figure(theta, bob_angle_deg,
                                   snr_signal_db, snr_noise_db, n_antennas)
    return angles, nf_db

# ─────────────────────────────────────────────────────────────────────────────
#  4. SNR BUDGET TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_noise_budget(bob_angle_deg: float = 30.0,
                       eve_angle_deg: float  = -45.0,
                       snr_signal_db: float  = 20.0,
                       snr_noise_db: float   = 10.0,
                       n_antennas: int       = N_ANTENNAS) -> dict:
    
    snr_bob = compute_snr_analytical(bob_angle_deg, bob_angle_deg, snr_signal_db, snr_noise_db, n_antennas)
    snr_eve = compute_snr_analytical(eve_angle_deg, bob_angle_deg, snr_signal_db, snr_noise_db, n_antennas)
    snr_ch  = _channel_snr_db(snr_signal_db)
    
    print("\n" + "="*70)
    print("  SADM-SEC  |  MODULE 6 NOISE ANALYSIS  (BECE304L Alignment)")
    print("="*70)
    print(f"  Bob SNR: {snr_bob:.2f} dB  |  Eve SNR: {snr_eve:.2f} dB")
    print("="*70)
    return {}

if __name__ == "__main__":
    print_noise_budget()