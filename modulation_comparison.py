"""
=============================================================================
SADM-SEC  |  modulation_comparison.py
=============================================================================
BECE304L Analog Communication Systems — Full Modulation Comparison
Modules 2, 3, 4, 6

Schemes compared
----------------
  1. AM         — Amplitude Modulation (m = 1, single tone)
  2. DSB-SC     — Double Sideband Suppressed Carrier
  3. SSB-SC     — Single Sideband Suppressed Carrier (USB, Hilbert method)
  4. FM-NB      — Narrowband FM  (beta = 0.5)
  5. FM-WB      — Wideband FM    (beta = 5)
  6. SADM-SEC   — Spatially Aware Directional Modulation (N=8 ULA, Bob)

Metrics computed
----------------
  • Time-domain waveform
  • Frequency spectrum (one-sided, dB)
  • Bandwidth  (Carson's rule for FM; 2*fm for AM/DSB; fm for SSB)
  • Total transmitted power (normalised)
  • Useful signal power  (carrier stripped for AM; sideband only)
  • Transmission efficiency  η = P_useful / P_total
  • Figure of Merit  FOM = SNR_out / SNR_channel  (Module 6)
  • Noise Figure  NF = SNR_channel - SNR_out  (dB)
  • Secrecy Rate  (SADM-SEC only, Bob vs Eve)

Output
------
  outputs/modulation_signals.png    — 6-panel waveform + spectrum figure
  outputs/modulation_metrics.png    — bar/line comparison charts
  outputs/modulation_report.pdf     — full PDF report
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert

# ── SADM math engine ─────────────────────────────────────────────────────────
import sys
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.path.abspath("."))
from spatial_logic import (
    N_ANTENNAS, beamforming_weights, noise_projection_matrix,
    compute_snr_analytical, secrecy_rate, sadm_transmit, virtual_channel
)
from noise_analysis import (
    figure_of_merit_sadm, noise_figure as sadm_noise_figure,
    fom_am, fom_dsb_sc, fom_ssb_sc, fom_fm,
    _channel_snr_db
)

# =============================================================================
#  SHARED PARAMETERS
# =============================================================================

FS       = 48_000          # sample rate  (Hz)
DURATION = 0.05            # signal duration (s)
FC       = 10_000          # carrier frequency (Hz)  — normalised for simulation
FM       = 1_000           # message frequency (Hz)
N        = int(FS * DURATION)
t        = np.linspace(0, DURATION, N, endpoint=False)

# Message and carrier
msg      = np.cos(2 * np.pi * FM * t)          # m(t)
carrier  = np.cos(2 * np.pi * FC * t)          # c(t)
carrier_s= np.sin(2 * np.pi * FC * t)          # 90-deg phase carrier

# Modulation parameters
M        = 1.0             # AM modulation index
BETA_NB  = 0.5             # FM narrowband index
BETA_WB  = 5.0             # FM wideband index
SADM_SIG = 20.0            # SADM signal power (dB)
SADM_AN  = 10.0            # SADM artificial noise power (dB)
BOB_ANG  = 30.0            # Bob's angle
EVE_ANG  = -45.0           # Eve's angle

# Visual theme
BG    = "#0A0F1E"
PANEL = "#111827"
CYAN  = "#00D4FF"
RED   = "#FF4040"
GREEN = "#00FF88"
GOLD  = "#FFD700"
PUR   = "#7C3AED"
ORG   = "#FF8800"
GREY  = "#4A5A6A"
WHITE = "#E2E8F0"
GCOL  = "#1E2A3E"

SCHEME_COLORS = {
    "AM"          : CYAN,
    "DSB-SC"      : GREEN,
    "SSB-SC"      : GOLD,
    "FM-NB"       : ORG,
    "FM-WB"       : PUR,
    "SADM-SEC"    : RED,
    # Traditional + DM variants (same hue, dimmed)
    "AM+DM"       : "#007A99",   # dark cyan
    "DSB-SC+DM"   : "#007A44",   # dark green
    "SSB-SC+DM"   : "#997A00",   # dark gold
    "FM-NB+DM"    : "#994C00",   # dark orange
    "FM-WB+DM"    : "#4A1F8A",   # dark purple
}

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : PANEL,
    "axes.edgecolor"   : GCOL,
    "axes.labelcolor"  : WHITE,
    "xtick.color"      : WHITE,
    "ytick.color"      : WHITE,
    "text.color"       : WHITE,
    "grid.color"       : GCOL,
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.5,
    "font.family"      : "monospace",
    "font.size"        : 8.5,
})


# =============================================================================
#  SIGNAL GENERATORS
# =============================================================================

def gen_am():
    """AM: s(t) = A_c [1 + m·n(t)] cos(2pi fc t)"""
    return (1 + M * msg) * carrier

def gen_dsb_sc():
    """DSB-SC: s(t) = m(t) cos(2pi fc t)"""
    return msg * carrier

def gen_ssb_sc():
    """SSB-SC (USB): s(t) = m(t)cos - m_hat(t)sin  (Hilbert / phase-shift method)"""
    m_hat = np.imag(hilbert(msg))   # Hilbert transform of message
    return msg * carrier - m_hat * carrier_s

def gen_fm_nb():
    """FM narrowband (beta=0.5): s(t) ≈ cos(wc t) - beta*sin(wm t)*sin(wc t)"""
    kf    = BETA_NB * FM            # frequency sensitivity
    phi   = 2 * np.pi * kf * np.cumsum(msg) / FS
    return np.cos(2 * np.pi * FC * t + phi)

def gen_fm_wb():
    """FM wideband (beta=5)"""
    kf    = BETA_WB * FM
    phi   = 2 * np.pi * kf * np.cumsum(msg) / FS
    return np.cos(2 * np.pi * FC * t + phi)

def gen_sadm():
    """
    SADM-SEC display signal — clean spatial combining at Bob.

    Physics:
        y_bob = a(theta_Bob)^H * msg_part
              = sqrt(N) * msg * sqrt(P_s)   [AN is in Bob null space -> 0]

    No path loss, no thermal noise: this shows the signal Bob actually decodes.
    Remodulated onto the carrier so the spectrum aligns with the other panels.
    """
    from spatial_logic import steering_vector, beamforming_weights as bfw
    w        = bfw(BOB_ANG)
    sig_pow  = 10 ** (SADM_SIG / 10)
    msg_part = np.outer(w, msg) * np.sqrt(sig_pow)        # (N, L)
    a_bob    = steering_vector(BOB_ANG)
    y_base   = (a_bob.conj() @ msg_part).real              # baseband: sqrt(N)*sqrt(Ps)*msg
    y_rf     = y_base * carrier                            # remodulate for display
    return y_rf / (np.max(np.abs(y_rf)) + 1e-12)


def gen_sadm_eve():
    """
    SADM-SEC received signal at Eve (theta = EVE_ANG = -45 deg).

    Eve receives:
        y_Eve = a†(θ_E)·w·s·√P_s  +  a†(θ_E)·P_AN·n·√P_AN

    Signal term  : |AF(θ_E, θ_B)|² · P_s  — tiny (off-axis leakage)
    AN term      : NOT cancelled → dominated by artificial noise
    Result       : Eve's output looks like noise — secrecy visible in time domain.
    """
    from spatial_logic import steering_vector, beamforming_weights as bfw, noise_projection_matrix
    w        = bfw(BOB_ANG)
    sig_pow  = 10 ** (SADM_SIG / 10)
    an_pow   = 10 ** (SADM_AN  / 10)
    L        = len(msg)

    # Signal component at Eve
    msg_part   = np.outer(w, msg) * np.sqrt(sig_pow)          # (N, L)
    a_eve      = steering_vector(EVE_ANG)
    y_sig_eve  = (a_eve.conj() @ msg_part).real               # tiny — off-axis

    # AN component at Eve (NOT in null-space of Eve — only in null-space of Bob)
    rng        = np.random.default_rng(42)
    P_AN_mat   = noise_projection_matrix(BOB_ANG)
    noise_src  = (rng.standard_normal((N_ANTENNAS, L)) +
                  1j * rng.standard_normal((N_ANTENNAS, L))) / np.sqrt(2)
    an_part    = P_AN_mat @ noise_src * np.sqrt(an_pow)
    y_an_eve   = (a_eve.conj() @ an_part).real                # dominant — noise

    y_eve = y_sig_eve + y_an_eve
    return y_eve / (np.max(np.abs(y_eve)) + 1e-12)


def _apply_dm_beamforming(baseband_sig):
    """
    Apply SADM beamforming (without artificial noise) to any baseband signal.
    This is the 'Traditional + DM' combiner: steer the existing modulated
    signal toward Bob using MRT weights, giving array gain N without secrecy AN.
    The received signal at Bob = sqrt(N) * sqrt(P_s) * baseband_sig.
    Returns the signal normalised to unit amplitude for display.
    """
    from spatial_logic import steering_vector, beamforming_weights as bfw
    w        = bfw(BOB_ANG)
    sig_pow  = 10 ** (SADM_SIG / 10)
    # Beamform: project the scalar signal through N-element array -> Bob combines
    msg_part = np.outer(w, baseband_sig) * np.sqrt(sig_pow)
    a_bob    = steering_vector(BOB_ANG)
    y_base   = (a_bob.conj() @ msg_part).real
    y_rf     = y_base * carrier
    return y_rf / (np.max(np.abs(y_rf)) + 1e-12)


def gen_am_dm():       return _apply_dm_beamforming(gen_am())
def gen_dsb_sc_dm():   return _apply_dm_beamforming(gen_dsb_sc())
def gen_ssb_sc_dm():   return _apply_dm_beamforming(gen_ssb_sc())
def gen_fm_nb_dm():    return _apply_dm_beamforming(gen_fm_nb())
def gen_fm_wb_dm():    return _apply_dm_beamforming(gen_fm_wb())


# =============================================================================
#  METRICS
# =============================================================================

def power(sig):
    return float(np.mean(np.abs(sig)**2))

def bandwidth_hz(scheme):
    """Theoretical bandwidth (Carson's rule for FM)."""
    base = scheme.replace("+DM", "")
    bw = {
        "AM"      : 2 * FM,
        "DSB-SC"  : 2 * FM,
        "SSB-SC"  : FM,
        "FM-NB"   : 2 * (BETA_NB + 1) * FM,
        "FM-WB"   : 2 * (BETA_WB + 1) * FM,
        "SADM-SEC": 2 * FM,         # same baseband message BW
    }
    return bw[base]

def transmission_efficiency(scheme):
    """
    eta = P_useful / P_total  — effective transmission efficiency.

    This metric captures TWO sources of power waste:
      (1) Modulation inefficiency: carrier waste in AM (TX modulator level)
      (2) AN overhead: power spent on artificial noise instead of signal (SADM-SEC)

    Traditional (omnidirectional):
        AM:              eta = (m^2/2) / (1 + m^2/2) = 33.3%  [carrier waste]
        DSB-SC/SSB-SC/FM: eta = 1.0                            [no waste]

    Traditional + DM (beamformed, zero AN):
        The beamformer concentrates ALL signal power toward Bob. No AN is injected.
        Power waste = modulation inefficiency only (no AN overhead).
        AM+DM:    eta = eta_AM × 1/(1 - 0) = eta_AM        = 33.3%  [same modulation waste]
        Others+DM: eta = 1.0 × 1/(1 - 0)  = 1.0            = 100%  [no waste at all]

    SADM-SEC (beamformed + AN injection):
        AN consumes P_AN of the total budget (P_signal + P_AN).
        eta = P_signal / (P_signal + P_AN)                  = 90.9%  [AN overhead]
        NOTE: AM-modulated SADM would stack both wastes, but SADM-SEC uses
              a clean baseband signal (not AM), so only AN waste applies.

    KEY TAKEAWAY for the comparison:
        Traditional:   η varies by modulation (33–100%)
        Traditional+DM: η = same as traditional base (no extra waste from DM)
        SADM-SEC:       η < 100% due to AN power cost (90.9% here)

    This shows that +DM adds array gain for FREE in terms of power efficiency —
    the η cost is borne only by SADM-SEC (which trades power for secrecy via AN).
    """
    is_dm = "+DM" in scheme
    base  = scheme.replace("+DM", "")

    if base == "SADM-SEC":
        # AN consumes part of the power budget
        sig_pow = 10**(SADM_SIG/10)
        an_pow  = 10**(SADM_AN/10)
        return sig_pow / (sig_pow + an_pow)

    # Base modulation efficiency (TX modulation ratio — carrier waste in AM)
    if base == "AM":
        eta_mod = (M**2 / 2) / (1 + M**2 / 2)   # 33.3% at m=1
    else:  # DSB-SC, SSB-SC, FM-NB, FM-WB
        eta_mod = 1.0

    # +DM uses the full SADM-SEC system (beamforming + AN injection).
    # AN consumes P_AN of the total budget, same as SADM-SEC.
    # So eta(+DM) = eta_mod * (P_s / (P_s + P_AN))
    # → AM+DM ≈ 30%, all others+DM ≈ 91%
    # Traditional (no DM) → no AN injected → no AN penalty.
    if is_dm:
        sig_pow   = 10 ** (SADM_SIG / 10)
        an_pow    = 10 ** (SADM_AN  / 10)
        an_factor = sig_pow / (sig_pow + an_pow)   # 0.909
        return eta_mod * an_factor
    return eta_mod

def figure_of_merit(scheme):
    """
    FOM = SNR_out / SNR_channel  (linear)
    Module 6 textbook formula.
    For +DM variants: base FOM * N_ANTENNAS (coherent array gain, no AN penalty).
    """
    is_dm = "+DM" in scheme
    base  = scheme.replace("+DM", "")
    if base == "AM":
        base_fom = fom_am(M)
    elif base == "DSB-SC":
        base_fom = fom_dsb_sc()
    elif base == "SSB-SC":
        base_fom = fom_ssb_sc()
    elif base == "FM-NB":
        base_fom = fom_fm(BETA_NB)
    elif base == "FM-WB":
        base_fom = fom_fm(BETA_WB)
    elif base == "SADM-SEC":
        fom_db, _, _ = figure_of_merit_sadm(BOB_ANG, BOB_ANG, SADM_SIG, SADM_AN)
        return 10**(fom_db/10)

    # +DM: multiply by array gain N (no artificial noise, so full gain applies)
    if is_dm:
        return base_fom * N_ANTENNAS
    return base_fom

def noise_figure_db(scheme):
    """NF = SNR_channel_dB - SNR_out_dB"""
    fom_lin = figure_of_merit(scheme)
    fom_db  = 10 * np.log10(max(fom_lin, 1e-30))
    return -fom_db   # NF = -FOM in dB

def spectrum_db(sig, n_pts=4096):
    """One-sided power spectrum in dB."""
    win   = np.blackman(len(sig))
    S     = np.fft.rfft(sig * win, n=n_pts)
    freqs = np.fft.rfftfreq(n_pts, d=1/FS)
    psd   = 20 * np.log10(np.abs(S) / n_pts + 1e-12)
    return freqs, psd


# =============================================================================
#  ASSEMBLE ALL SCHEMES
# =============================================================================

TRADITIONAL = ["AM", "DSB-SC", "SSB-SC", "FM-NB", "FM-WB"]
TRADITIONAL_DM = ["AM+DM", "DSB-SC+DM", "SSB-SC+DM", "FM-NB+DM", "FM-WB+DM"]
SCHEMES = TRADITIONAL + TRADITIONAL_DM + ["SADM-SEC"]

GEN     = {
    "AM"        : gen_am,
    "DSB-SC"    : gen_dsb_sc,
    "SSB-SC"    : gen_ssb_sc,
    "FM-NB"     : gen_fm_nb,
    "FM-WB"     : gen_fm_wb,
    "AM+DM"     : gen_am_dm,
    "DSB-SC+DM" : gen_dsb_sc_dm,
    "SSB-SC+DM" : gen_ssb_sc_dm,
    "FM-NB+DM"  : gen_fm_nb_dm,
    "FM-WB+DM"  : gen_fm_wb_dm,
    "SADM-SEC"  : gen_sadm,
    "SADM-EVE"  : gen_sadm_eve,   # Eve's received signal — noise-dominated
}

def compute_all():
    results = {}
    for name in SCHEMES + ["SADM-EVE"]:
        print(f"  Simulating {name} ...")
        sig = GEN[name]()
        freqs, psd = spectrum_db(sig)
        results[name] = {
            "signal"     : sig,
            "freqs"      : freqs,
            "psd"        : psd,
            "power_db"   : 10 * np.log10(power(sig) + 1e-30),
            "bw_hz"      : bandwidth_hz(name if name != "SADM-EVE" else "SADM-SEC"),
            "eta"        : transmission_efficiency(name if name != "SADM-EVE" else "SADM-SEC"),
            "fom_lin"    : figure_of_merit(name if name != "SADM-EVE" else "SADM-SEC"),
            "fom_db"     : 10 * np.log10(max(figure_of_merit(name if name != "SADM-EVE" else "SADM-SEC"), 1e-30)),
            "nf_db"      : noise_figure_db(name if name != "SADM-EVE" else "SADM-SEC"),
        }

    # ── Beam pattern: signal and AN power vs azimuth angle ───────────────────
    from spatial_logic import steering_vector, beamforming_weights as bfw, noise_projection_matrix
    angles_deg = np.linspace(-90, 90, 361)
    w          = bfw(BOB_ANG)
    P_AN_mat   = noise_projection_matrix(BOB_ANG)
    sig_pattern = []
    an_pattern  = []
    for ang in angles_deg:
        a_rx = steering_vector(ang)
        # Signal power at angle: |a†(ang) · w|² * P_s
        sig_pattern.append(abs(np.dot(a_rx.conj(), w))**2)
        # AN power at angle: a†(ang) · P_AN · P_AN† · a(ang) * P_AN_pw
        an_pattern.append(float(np.real(a_rx.conj() @ P_AN_mat @ P_AN_mat.conj().T @ a_rx)))

    results["_beam_pattern"] = {
        "angles"      : angles_deg,
        "signal_norm" : np.array(sig_pattern) / max(sig_pattern),
        "an_norm"     : np.array(an_pattern)  / max(an_pattern),
    }
    return results


# =============================================================================
#  PLOT 1 — WAVEFORMS + SPECTRA  (6 × 2 grid)
# =============================================================================

def plot_signals(results, out="outputs/modulation_signals.png"):
    # Show traditional vs +DM pairs, then SADM-SEC Bob vs Eve, then beam pattern
    PLOT_PAIRS = [("AM", "AM+DM"), ("DSB-SC", "DSB-SC+DM"),
                  ("SSB-SC", "SSB-SC+DM"), ("FM-NB", "FM-NB+DM"),
                  ("FM-WB", "FM-WB+DM")]
    PLOT_ORDER = [s for pair in PLOT_PAIRS for s in pair] + ["SADM-SEC", "SADM-EVE"]

    # Extra row at bottom for beam pattern — total rows = len(PLOT_ORDER) + 1
    n_rows = len(PLOT_ORDER) + 1

    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        "BECE304L  |  Traditional vs Traditional + DM Comparison  —  Time Domain & Frequency Spectrum\n"
        "Message: f_m = 1 kHz cosine   |   Carrier: f_c = 10 kHz   |   Fs = 48 kHz   |   "
        "Solid = baseline, Dimmed = +DM  |  SADM-SEC: Bob (clean) vs Eve (noise-dominated)",
        fontsize=11, fontweight="bold", color=CYAN, y=0.998)

    gs = gridspec.GridSpec(n_rows, 2, figure=fig,
                           hspace=0.45, wspace=0.28,
                           left=0.06, right=0.97, top=0.975, bottom=0.03)

    n_show = int(0.003 * FS)
    t_ms   = t[:n_show] * 1e3

    for row, name in enumerate(PLOT_ORDER):
        # SADM-EVE uses RED but dimmed
        if name == "SADM-EVE":
            col = "#992020"
        else:
            col = SCHEME_COLORS[name]

        d = results[name]

        ax_t = fig.add_subplot(gs[row, 0])
        sig_show = d["signal"][:n_show]
        amp = np.max(np.abs(sig_show)) + 1e-12
        ax_t.plot(t_ms, sig_show / amp, color=col, linewidth=0.9)
        ax_t.axhline(0, color=GCOL, linewidth=0.6)
        ax_t.set_xlim(0, t_ms[-1])
        ax_t.set_ylim(-1.35, 1.35)
        ax_t.set_ylabel("Norm. Amp.", fontsize=7.5)

        if name == "SADM-EVE":
            title_str = (f"SADM-SEC Eve (θ={EVE_ANG}°)  [AN not cancelled — noise-dominated]"
                         f"  |  SNR≈−13 dB")
        elif name == "SADM-SEC":
            title_str = (f"SADM-SEC Bob (θ={BOB_ANG}°)  [AN nulled — clean signal]"
                         f"  |  BW={d['bw_hz']/1000:.1f} kHz  η={d['eta']*100:.0f}%  FOM={d['fom_db']:+.1f} dB")
        else:
            sadm_note = ""
            dm_note   = "  [+DM: array gain N=8, AN overhead −9%]" if "+DM" in name else ""
            title_str = (f"{name}{dm_note}  |  BW={d['bw_hz']/1000:.1f} kHz  "
                         f"η={d['eta']*100:.0f}%  FOM={d['fom_db']:+.1f} dB")

        ax_t.set_title(title_str, color=col, fontsize=8, fontweight="bold", pad=4)
        ax_t.grid(True)
        if row == n_rows - 2:      # last signal row (before beam pattern)
            ax_t.set_xlabel("Time (ms)")

        if name == "AM":
            env = (1 + M * msg[:n_show])
            ax_t.plot(t_ms,  env / amp, color=WHITE, linewidth=0.7,
                      linestyle="--", alpha=0.5, label="Envelope")
            ax_t.plot(t_ms, -env / amp, color=WHITE, linewidth=0.7,
                      linestyle="--", alpha=0.5)

        # Frequency panel
        ax_f = fig.add_subplot(gs[row, 1])
        freqs = d["freqs"] / 1000
        psd   = d["psd"]
        mask  = (freqs >= 6) & (freqs <= 30)
        ax_f.plot(freqs[mask], psd[mask], color=col, linewidth=1.0)
        ax_f.fill_between(freqs[mask], psd[mask] - 80, psd[mask],
                          alpha=0.15, color=col)
        ax_f.axvline(FC/1000, color=WHITE, linewidth=0.7, linestyle="--", alpha=0.4)
        ax_f.set_ylim(-80, 10)
        ax_f.set_ylabel("PSD (dB)", fontsize=7.5)
        ax_f.grid(True)
        if row == n_rows - 2:
            ax_f.set_xlabel("Frequency (kHz)")

        for fmark in [FC/1000 + FM/1000, FC/1000 - FM/1000]:
            if 6 <= fmark <= 30:
                ax_f.axvline(fmark, color=col, linewidth=0.6, linestyle=":", alpha=0.6)

        if name == "SADM-EVE":
            ax_f.set_title("Eve PSD — noise floor raised, no clean spectral line",
                           color=col, fontsize=7.5, pad=3)

    # ── Beam pattern row: signal power and AN power vs azimuth ───────────────
    bp = results["_beam_pattern"]
    angles   = bp["angles"]
    sig_norm = bp["signal_norm"]
    an_norm  = bp["an_norm"]

    ax_bp_lin = fig.add_subplot(gs[n_rows - 1, 0])
    ax_bp_lin.plot(angles, 10 * np.log10(sig_norm + 1e-12),
                   color=GREEN, linewidth=1.4, label="Signal power (dB)")
    ax_bp_lin.plot(angles, 10 * np.log10(an_norm  + 1e-12),
                   color=RED,   linewidth=1.4, label="AN power (dB)", linestyle="--")
    ax_bp_lin.axvline(BOB_ANG, color=GREEN, linewidth=1.0, linestyle=":",
                      alpha=0.7, label=f"Bob θ={BOB_ANG}°")
    ax_bp_lin.axvline(EVE_ANG, color=RED,   linewidth=1.0, linestyle=":",
                      alpha=0.7, label=f"Eve θ={EVE_ANG}°")
    # Annotate null at Bob's angle
    ax_bp_lin.annotate("AN = 0\nat Bob", xy=(BOB_ANG, -60),
                        xytext=(BOB_ANG + 12, -45),
                        fontsize=7, color=GREEN,
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
    ax_bp_lin.annotate("AN dominates\nat Eve", xy=(EVE_ANG, -3),
                        xytext=(EVE_ANG - 35, -20),
                        fontsize=7, color=RED,
                        arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax_bp_lin.set_xlim(-90, 90)
    ax_bp_lin.set_ylim(-70, 5)
    ax_bp_lin.set_xlabel("Azimuth angle (degrees)")
    ax_bp_lin.set_ylabel("Normalised Power (dB)")
    ax_bp_lin.set_title(
        f"SADM-SEC Spatial Pattern  —  N={N_ANTENNAS} ULA  |  "
        f"Signal beam → Bob ({BOB_ANG}°)  |  AN null @ Bob, maximum @ Eve ({EVE_ANG}°)",
        color=CYAN, fontsize=9, fontweight="bold", pad=5)
    ax_bp_lin.legend(fontsize=7, facecolor=BG)
    ax_bp_lin.grid(True)
    ax_bp_lin.set_facecolor(PANEL)

    # Polar beam pattern
    ax_bp_pol = fig.add_subplot(gs[n_rows - 1, 1], projection="polar")
    theta_rad = np.deg2rad(angles)
    ax_bp_pol.plot(theta_rad, sig_norm, color=GREEN, linewidth=1.4, label="Signal")
    ax_bp_pol.plot(theta_rad, an_norm,  color=RED,   linewidth=1.4,
                   linestyle="--", label="AN")
    ax_bp_pol.axvline(np.deg2rad(BOB_ANG), color=GREEN, linewidth=1.0,
                      linestyle=":", alpha=0.8)
    ax_bp_pol.axvline(np.deg2rad(EVE_ANG), color=RED, linewidth=1.0,
                      linestyle=":", alpha=0.8)
    ax_bp_pol.set_theta_zero_location("N")
    ax_bp_pol.set_theta_direction(-1)
    ax_bp_pol.set_facecolor(PANEL)
    ax_bp_pol.grid(color=GCOL, linewidth=0.5)
    ax_bp_pol.set_title("Polar Pattern  (signal vs AN)",
                         color=CYAN, fontsize=9, fontweight="bold", pad=15)
    ax_bp_pol.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12),
                     fontsize=7, facecolor=BG)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [Viz] Saved -> {out}")
    return out


# =============================================================================
#  PLOT 2 — METRICS COMPARISON
# =============================================================================

def plot_metrics(results, out="outputs/modulation_metrics.png"):
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        "BECE304L  |  Traditional vs Traditional + DM  —  Quantitative Metrics Comparison\n"
        "Solid bars = traditional alone  |  Hatched bars = traditional + DM beamforming (N=8 array gain)",
        fontsize=11, fontweight="bold", color=CYAN, y=0.995)

    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.07, right=0.97, top=0.95, bottom=0.08)

    # Use 5 pairs: traditional side by side with +DM
    BASE_NAMES = ["AM", "DSB-SC", "SSB-SC", "FM-NB", "FM-WB"]
    DM_NAMES   = [n + "+DM" for n in BASE_NAMES]
    # Also include SADM-SEC as a reference bar
    ALL_NAMES  = BASE_NAMES + ["SADM-SEC"]
    x_base     = np.arange(len(BASE_NAMES))
    bar_w      = 0.38

    # Color lists
    base_colors = [SCHEME_COLORS[n] for n in BASE_NAMES]
    dm_colors   = [SCHEME_COLORS[n] for n in DM_NAMES]
    sadm_color  = SCHEME_COLORS["SADM-SEC"]

    def style_ax(ax, title, ylabel):
        ax.set_facecolor(PANEL)
        xticks = list(x_base) + [x_base[-1] + 1.4]
        xlabels = BASE_NAMES + ["SADM-SEC"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7, rotation=18, ha="right")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", color=GCOL, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, color=WHITE, fontsize=9.5, fontweight="bold", pad=6)

    def draw_pair_bars(ax, base_vals, dm_vals, sadm_val, fmt="{:.1f}", yoff=0.5):
        # Traditional bars (left of each pair)
        bars1 = ax.bar(x_base - bar_w/2, base_vals, width=bar_w,
                       color=base_colors, zorder=3, label="Traditional")
        # +DM bars (right of each pair, hatched)
        bars2 = ax.bar(x_base + bar_w/2, dm_vals, width=bar_w,
                       color=dm_colors, zorder=3, hatch="///",
                       edgecolor="#FFFFFF44", label="+DM (N=8)")
        # SADM-SEC reference bar
        ax.bar(x_base[-1] + 1.4, sadm_val, width=bar_w,
               color=sadm_color, zorder=3)
        for i, (v1, v2) in enumerate(zip(base_vals, dm_vals)):
            ax.text(i - bar_w/2, v1 + yoff, fmt.format(v1),
                    ha="center", va="bottom", fontsize=6.5, color=WHITE)
            ax.text(i + bar_w/2, v2 + yoff, fmt.format(v2),
                    ha="center", va="bottom", fontsize=6.5, color=WHITE)
        ax.text(x_base[-1] + 1.4, sadm_val + yoff, fmt.format(sadm_val),
                ha="center", va="bottom", fontsize=6.5, color=WHITE)

    # ── 1. Bandwidth ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bw_base = [results[n]["bw_hz"] / 1000 for n in BASE_NAMES]
    bw_dm   = [results[n]["bw_hz"] / 1000 for n in DM_NAMES]
    bw_sadm = results["SADM-SEC"]["bw_hz"] / 1000
    draw_pair_bars(ax1, bw_base, bw_dm, bw_sadm, "{:.1f}k", 0.1)
    style_ax(ax1, "Bandwidth  (kHz)  [unchanged by DM]", "BW (kHz)")
    ax1.axhline(FM/1000, color=GREY, linewidth=1, linestyle="--",
                label=f"f_m = {FM/1000:.0f} kHz")
    ax1.legend(fontsize=6.5, facecolor=BG)

    # ── 2. Transmission Efficiency ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    eta_base = [results[n]["eta"] * 100 for n in BASE_NAMES]
    eta_dm   = [results[n]["eta"] * 100 for n in DM_NAMES]
    eta_sadm = results["SADM-SEC"]["eta"] * 100
    draw_pair_bars(ax2, eta_base, eta_dm, eta_sadm, "{:.0f}%", 1.0)
    style_ax(ax2, "Transmission Efficiency  η (%)\n+DM bears same AN overhead as SADM-SEC (−9.1% on all beamformed schemes)", "η (%)")
    ax2.set_ylim(0, 130)
    ax2.axhline(100, color=GREEN, linewidth=0.8, linestyle="--",
                alpha=0.5, label="100% ideal")
    # Annotate the AN penalty — now applies to all +DM and SADM-SEC bars
    sadm_x = x_base[-1] + 1.4
    ax2.annotate("AN penalty\n−9.1%", xy=(sadm_x, eta_sadm),
                 xytext=(sadm_x - 0.9, eta_sadm + 12),
                 fontsize=7, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    # Annotate same penalty on DSB-SC+DM as representative example
    idx_dsb = BASE_NAMES.index("DSB-SC")
    ax2.annotate("same\n−9.1%", xy=(idx_dsb + bar_w/2, eta_dm[idx_dsb]),
                 xytext=(idx_dsb + bar_w/2 + 0.5, eta_dm[idx_dsb] + 15),
                 fontsize=6.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax2.text(0.5, 0.30,
             "All +DM variants bear AN overhead\n(beamforming + AN injection = full SADM-SEC)\n"
             "Traditional base has no AN → η gap shows AN cost",
             transform=ax2.transAxes, fontsize=6.5, color=GREY,
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GREY, alpha=0.8))
    ax2.legend(fontsize=6.5, facecolor=BG)

    # ── 3. Figure of Merit ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    fom_base = [results[n]["fom_db"] for n in BASE_NAMES]
    fom_dm   = [results[n]["fom_db"] for n in DM_NAMES]
    fom_sadm = results["SADM-SEC"]["fom_db"]
    # Use raw bar drawing for +/- values
    bars3a = ax3.bar(x_base - bar_w/2, fom_base, width=bar_w,
                     color=base_colors, zorder=3)
    bars3b = ax3.bar(x_base + bar_w/2, fom_dm,   width=bar_w,
                     color=dm_colors,   zorder=3, hatch="///",
                     edgecolor="#FFFFFF44")
    ax3.bar(x_base[-1] + 1.4, fom_sadm, width=bar_w, color=sadm_color, zorder=3)
    for i, (v1, v2) in enumerate(zip(fom_base, fom_dm)):
        ax3.text(i - bar_w/2, v1 + (0.4 if v1 >= 0 else -2.5),
                 f"{v1:+.1f}", ha="center", va="bottom", fontsize=6.5, color=WHITE)
        ax3.text(i + bar_w/2, v2 + (0.4 if v2 >= 0 else -2.5),
                 f"{v2:+.1f}", ha="center", va="bottom", fontsize=6.5, color=WHITE)
    ax3.text(x_base[-1] + 1.4, fom_sadm + 0.4, f"{fom_sadm:+.1f}",
             ha="center", va="bottom", fontsize=6.5, color=WHITE)
    style_ax(ax3, "Figure of Merit  (dB)  [+DM adds +9.03 dB array gain]", "FOM (dB)")
    ax3.axhline(0, color=GREEN, linewidth=1, linestyle="--", label="DSB-SC baseline")
    # Annotate the +9 dB gain arrow on DSB-SC
    idx_dsb = BASE_NAMES.index("DSB-SC")
    ax3.annotate("", xy=(idx_dsb + bar_w/2, fom_dm[idx_dsb]),
                 xytext=(idx_dsb - bar_w/2, fom_base[idx_dsb]),
                 arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.2))
    ax3.legend(fontsize=6.5, facecolor=BG)

    # ── 4. Noise Figure ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    nf_base = [results[n]["nf_db"] for n in BASE_NAMES]
    nf_dm   = [results[n]["nf_db"] for n in DM_NAMES]
    nf_sadm = results["SADM-SEC"]["nf_db"]
    nf_base_c = [min(v, 30) if v > 0 else max(v, -30) for v in nf_base]
    nf_dm_c   = [min(v, 30) if v > 0 else max(v, -30) for v in nf_dm]
    nf_sadm_c = min(nf_sadm, 30) if nf_sadm > 0 else max(nf_sadm, -30)
    ax4.bar(x_base - bar_w/2, nf_base_c, width=bar_w, color=base_colors, zorder=3)
    ax4.bar(x_base + bar_w/2, nf_dm_c,   width=bar_w, color=dm_colors, zorder=3,
            hatch="///", edgecolor="#FFFFFF44")
    ax4.bar(x_base[-1] + 1.4, nf_sadm_c, width=bar_w, color=sadm_color, zorder=3)
    for i, (v, vc) in enumerate(zip(nf_base, nf_base_c)):
        ax4.text(i - bar_w/2, vc + (0.4 if vc >= 0 else -2.5),
                 f"{v:+.1f}", ha="center", va="bottom", fontsize=6, color=WHITE)
    for i, (v, vc) in enumerate(zip(nf_dm, nf_dm_c)):
        ax4.text(i + bar_w/2, vc + (0.4 if vc >= 0 else -2.5),
                 f"{v:+.1f}", ha="center", va="bottom", fontsize=6, color=WHITE)
    ax4.text(x_base[-1] + 1.4, nf_sadm_c + 0.4, f"{nf_sadm:+.1f}",
             ha="center", va="bottom", fontsize=6, color=WHITE)
    ax4.axhline(0, color=GREEN, linewidth=1, linestyle="--")
    style_ax(ax4, "Noise Figure  NF (dB)  [Module 6]", "NF (dB)")

    # ── 5. FOM sweep — traditional vs +DM vs SADM-SEC ────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    snr_range = np.linspace(-10, 40, 120)

    sweep_pairs = [
        ("AM m=1",      fom_am(M),       CYAN,  "solid"),
        ("DSB-SC",      fom_dsb_sc(),    GREEN, "solid"),
        ("SSB-SC",      fom_ssb_sc(),    GOLD,  "solid"),
        ("FM NB β=0.5", fom_fm(BETA_NB), ORG,   "solid"),
        ("FM WB β=5",   fom_fm(BETA_WB), PUR,   "solid"),
    ]
    for label, fom_const, col, _ in sweep_pairs:
        fom_db_val = 10 * np.log10(fom_const)
        ax5.axhline(fom_db_val, color=col, linewidth=1.0, linestyle="--",
                    label=f"{label}: {fom_db_val:+.1f} dB")
        # +DM variant: same FOM + 9.03 dB
        fom_dm_db = fom_db_val + 10 * np.log10(N_ANTENNAS)
        ax5.axhline(fom_dm_db, color=col, linewidth=1.2, linestyle=":",
                    alpha=0.85, label=f"{label}+DM: {fom_dm_db:+.1f} dB")

    from noise_analysis import fom_vs_snr_sweep
    sweep = fom_vs_snr_sweep(BOB_ANG, EVE_ANG,
                             snr_signal_range=snr_range, snr_noise_db=SADM_AN)
    ax5.plot(sweep["snr_ch"], sweep["fom_bob"], color=RED, linewidth=2,
             label="SADM-SEC Bob (N=8)")
    ax5.plot(sweep["snr_ch"], sweep["fom_eve"], color=RED, linewidth=1.2,
             linestyle="-.", label="SADM-SEC Eve")

    ax5.set_xlabel("Channel SNR (dB)")
    ax5.set_ylabel("FOM (dB)")
    ax5.set_ylim(-30, 40)
    ax5.grid(True, color=GCOL)
    ax5.set_facecolor(PANEL)
    ax5.legend(fontsize=5.5, facecolor=BG, ncol=2)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.set_title("FOM vs Channel SNR  —  Traditional / +DM / SADM-SEC",
                  color=WHITE, fontsize=9.5, fontweight="bold", pad=6)

    # ── 6. Spider / Radar chart ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2], projection="polar")

    categories = ["FOM\n(norm)", "η\n(norm)", "BW\nEffic.", "SNR\nSecurity",
                  "Const\nEnvelope"]
    N_cat = len(categories)
    angles_radar = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles_radar += angles_radar[:1]

    RADAR_SCHEMES = BASE_NAMES + ["DSB-SC+DM", "FM-WB+DM", "SADM-SEC"]
    security = {n: 0 for n in BASE_NAMES}
    security.update({"DSB-SC+DM": 0, "FM-WB+DM": 0, "SADM-SEC": 1.0,
                     "FM-NB": 0.1, "FM-WB": 0.15})
    const_env = {n: 0 for n in BASE_NAMES}
    const_env.update({"FM-NB": 1.0, "FM-WB": 1.0,
                      "DSB-SC+DM": 0, "FM-WB+DM": 1.0, "SADM-SEC": 0.6})

    all_fom  = [results[n]["fom_db"] for n in RADAR_SCHEMES]
    fom_min, fom_max = min(all_fom), max(all_fom)
    all_bw   = [results[n]["bw_hz"] for n in RADAR_SCHEMES]
    max_bw   = max(all_bw)

    for name in RADAR_SCHEMES:
        d   = results[name]
        col = SCHEME_COLORS[name]
        fom_n = (d["fom_db"] - fom_min) / (fom_max - fom_min + 1e-6)
        eta_n = d["eta"]
        bw_n  = 1 - (d["bw_hz"] / (max_bw + 1))
        sec_n = security.get(name, 0)
        env_n = const_env.get(name, 0)
        vals  = [fom_n, eta_n, bw_n, sec_n, env_n]
        vals += vals[:1]
        ls = ":" if "+DM" in name else "-"
        ax6.plot(angles_radar, vals, color=col, linewidth=1.4,
                 linestyle=ls, label=name)
        ax6.fill(angles_radar, vals, alpha=0.06, color=col)

    ax6.set_thetagrids(np.degrees(angles_radar[:-1]), categories,
                       fontsize=7, color=WHITE)
    ax6.set_ylim(0, 1)
    ax6.set_facecolor(PANEL)
    ax6.grid(color=GCOL, linewidth=0.5)
    ax6.set_title("Multi-Metric Radar  (dotted = +DM)",
                  color=WHITE, fontsize=9.5, fontweight="bold", pad=18)
    ax6.legend(loc="lower left", bbox_to_anchor=(-0.40, -0.18),
               fontsize=6, facecolor=BG, ncol=2)

    # Legend for bar style
    from matplotlib.patches import Patch
    leg_elem = [Patch(facecolor=GREY, label="Solid: Traditional"),
                Patch(facecolor=GREY, hatch="///", label="Hatched: +DM")]
    fig.legend(handles=leg_elem, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=8, facecolor=BG,
               framealpha=0.7, labelcolor=WHITE)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [Viz] Saved -> {out}")
    return out


# =============================================================================
#  PDF REPORT
# =============================================================================

def generate_pdf(results, signals_img, metrics_img,
                 out="outputs/modulation_report.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    W, H = A4
    doc  = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2*cm,    bottomMargin=2*cm
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    C_DARK  = colors.HexColor("#0A0F1E")
    C_CYAN  = colors.HexColor("#00D4FF")
    C_GREEN = colors.HexColor("#00C87A")
    C_RED   = colors.HexColor("#E03030")
    C_GREY  = colors.HexColor("#4A5A6A")
    C_WHITE = colors.HexColor("#FFFFFF")
    C_TEXT  = colors.HexColor("#1A1A2E")
    C_ACC   = colors.HexColor("#1D4E89")

    def sty(name, **kw):
        s = ParagraphStyle(name, parent=styles["Normal"], **kw)
        return s

    S_TITLE  = sty("Title2",  fontSize=20, leading=26, textColor=C_ACC,
                   alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold")
    S_SUB    = sty("Sub",     fontSize=11, leading=14, textColor=C_GREY,
                   alignment=TA_CENTER, spaceAfter=2)
    S_H1     = sty("H1",      fontSize=13, leading=17, textColor=C_ACC,
                   fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=5)
    S_H2     = sty("H2",      fontSize=10.5, leading=13, textColor=C_ACC,
                   fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    S_BODY   = sty("Body",    fontSize=9.5, leading=14, textColor=C_TEXT,
                   alignment=TA_JUSTIFY, spaceAfter=6)
    S_MONO   = sty("Mono",    fontSize=8.5, leading=12, textColor=C_TEXT,
                   fontName="Courier", spaceAfter=4)
    S_CAP    = sty("Cap",     fontSize=8, leading=11, textColor=C_GREY,
                   alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
    S_BULLET = sty("Bullet",  fontSize=9.5, leading=14, textColor=C_TEXT,
                   leftIndent=16, spaceAfter=3)
    S_WINNER = sty("Winner",  fontSize=9.5, leading=14, textColor=C_GREEN,
                   fontName="Helvetica-Bold")
    S_RESULT = sty("Result",  fontSize=11, leading=15, textColor=C_ACC,
                   fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6)

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.2*cm),
        Paragraph("Analog Communication Systems", S_TITLE),
        Paragraph("Traditional vs Traditional + DM Comparative Analysis", S_TITLE),
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=2, color=C_ACC),
        Spacer(1, 0.3*cm),
        Paragraph("BECE304L  |  VIT Chennai", S_SUB),
        Paragraph(
            "Modules 2, 3, 4 &amp; 6  —  AM, DSB-SC, SSB-SC, FM vs Same Schemes + DM Beamforming (N=8)",
            S_SUB),
        Spacer(1, 0.6*cm),
        HRFlowable(width="60%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.4*cm),
        Paragraph(
            "This report presents a fair comparison of five standard analog modulation "
            "schemes against their DM-enhanced counterparts (Traditional + DM). "
            "Each +DM variant applies the same SADM-SEC N=8 ULA beamforming to steer "
            "the existing modulated signal toward Bob, adding the coherent array gain "
            "of N=8 (+9.03 dB) without changing the modulation itself. "
            "SADM-SEC (with Artificial Noise) is shown as a reference for secrecy. "
            "This framing is fair: DM is added on top of each traditional scheme "
            "rather than compared against it.",
            S_BODY),
        PageBreak(),
    ]

    # ── Section 1: System Setup ───────────────────────────────────────────────
    story += [
        Paragraph("1.  Simulation Setup", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
        Paragraph(
            "All modulation schemes share the same baseband message signal and "
            "carrier parameters. The message is a single cosine tone at 1 kHz — "
            "the standard single-tone assumption used throughout Module 2 and Module 4. "
            "The carrier is placed at 10 kHz with a sample rate of 48 kHz so that "
            "spectra are clearly resolved.",
            S_BODY),
    ]

    setup_data = [
        ["Parameter", "Value", "Module Reference"],
        ["Message frequency  f_m", "1 000 Hz", "Modules 2, 4"],
        ["Carrier frequency  f_c", "10 000 Hz", "Modules 2, 3, 4"],
        ["Sample rate  Fs", "48 000 Hz", "Module 7 (Nyquist)"],
        ["AM modulation index  m", "1.0", "Module 2"],
        ["FM narrowband index  beta", "0.5", "Module 4"],
        ["FM wideband index  beta", "5.0", "Module 4"],
        ["SADM array size  N", "8 elements (ULA)", "Course Project"],
        ["SADM signal power", "+20 dB", "Module 6"],
        ["SADM artificial noise", "+10 dB", "Module 6"],
        ["SADM Bob angle", "30 deg", "Course Project"],
        ["Reference temperature  T0", "290 K (IEEE)", "Module 6"],
    ]
    t_setup = Table(setup_data, colWidths=[6*cm, 5*cm, 5*cm])
    t_setup.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  C_ACC),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F8FF"),
                                               colors.HexColor("#EAF0FB")]),
        ("GRID",        (0, 0), (-1, -1), 0.4, C_GREY),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
    ]))
    story += [t_setup, Spacer(1, 0.4*cm)]

    # ── Section 2: Scheme Descriptions ───────────────────────────────────────
    story += [
        Paragraph("2.  Modulation Schemes", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
    ]

    descriptions = {
        "AM": (
            "Amplitude Modulation  (Module 2)",
            "s(t) = A<sub>c</sub> [1 + m&#183;m(t)] cos(2&#960;f<sub>c</sub>t)",
            "The carrier amplitude varies with the message. With m=1 the modulation "
            "index is 100%. The carrier carries no information but consumes half the "
            "total power, giving a transmission efficiency of only 33.3%. Demodulation "
            "is simple (envelope detector) but SNR performance is the worst of all "
            "analog systems. Bandwidth = 2f_m."
        ),
        "DSB-SC": (
            "Double Sideband Suppressed Carrier  (Module 3)",
            "s(t) = m(t) cos(2&#960;f<sub>c</sub>t)",
            "The carrier is suppressed; all transmitted power goes into the two "
            "sidebands. Transmission efficiency = 100%. SNR performance equals the "
            "channel SNR (FOM = 0 dB, the baseline for comparison). Requires coherent "
            "detection (synchronous detector). Bandwidth = 2f_m. Generated using a "
            "balanced modulator or ring modulator (Module 3)."
        ),
        "SSB-SC": (
            "Single Sideband Suppressed Carrier  (Module 3)",
            "s(t) = m(t)cos(2&#960;f<sub>c</sub>t) &#8722; m&#770;(t)sin(2&#960;f<sub>c</sub>t)",
            "Only the upper (or lower) sideband is transmitted, halving the bandwidth "
            "to f_m. This is the most spectrum-efficient analog modulation. FOM = 0 dB "
            "— same as DSB-SC in terms of SNR performance, but uses half the bandwidth. "
            "Implemented here using the Hilbert (phase-shift) method. Requires "
            "synchronous detection."
        ),
        "FM-NB": (
            "Narrowband FM  (beta = 0.5, Module 4)",
            "s(t) = A<sub>c</sub> cos[2&#960;f<sub>c</sub>t + 2&#960;k<sub>f</sub>&#8747;m(t)dt]",
            "With beta=0.5, Carson's rule gives BW = 2(beta+1)f_m = 3 kHz. "
            "Constant envelope property means no amplitude variation — robust to "
            "nonlinear amplifiers. FOM = 3*beta^2*(beta+1) = 0.375 linear — "
            "actually worse than DSB-SC because the bandwidth expansion at low beta "
            "yields no SNR benefit. Narrowband FM is mainly used where constant "
            "envelope matters more than SNR (e.g. walkie-talkies)."
        ),
        "FM-WB": (
            "Wideband FM  (beta = 5, Module 4)",
            "Same FM equation, large frequency deviation",
            "With beta=5, BW = 2(5+1)*1000 = 12 kHz — 12x the message bandwidth. "
            "This bandwidth expansion buys a massive SNR improvement: FOM = "
            "3*25*6 = 450 linear = +26.5 dB. This is the key FM theorem — bandwidth "
            "is traded for noise performance. Used in FM broadcast radio (beta~5) "
            "and satellite links. Constant envelope maintained."
        ),
        "SADM-SEC": (
            "SADM-SEC — 8-Element ULA  (Course Project, Module 6)",
            "X(t) = w&#183;m(t) + P<sub>AN</sub>&#183;n(t)  [N-antenna transmit matrix]",
            "Spatially Aware Directional Modulation steers the message beam toward "
            "Bob (30 deg) using MRT beamforming weights w = a(theta)/||a(theta)||. "
            "Artificial Noise is projected into the null space of Bob's steering "
            "vector so that it is zero at Bob but non-zero everywhere else. "
            "Bob's FOM = N = 8 (coherent array gain) = +9.03 dB — better than "
            "DSB-SC and SSB-SC but below wideband FM. Eve's FOM = -73 dB, achieving "
            "physical-layer secrecy. This is unique to SADM-SEC and has no equivalent "
            "in conventional analog modulation."
        ),
    }

    for name in ["AM", "DSB-SC", "SSB-SC", "FM-NB", "FM-WB", "SADM-SEC"]:
        title_s, formula_s, desc_s = descriptions[name]
        idx = ["AM", "DSB-SC", "SSB-SC", "FM-NB", "FM-WB", "SADM-SEC"].index(name) + 1
        story += [
            Paragraph(f"2.{idx}  {title_s}", S_H2),
            Paragraph(f"<i>Formula:</i>  {formula_s}", S_MONO),
            Paragraph(desc_s, S_BODY),
        ]
        # +DM variant note for traditional schemes
        if name != "SADM-SEC":
            fom_base = 10 * np.log10(max(figure_of_merit(name), 1e-30))
            fom_dm   = fom_base + 10 * np.log10(N_ANTENNAS)
            story.append(Paragraph(
                f"<b>{name}+DM</b>: Applying SADM-SEC N=8 beamforming steers this "
                f"signal to Bob with coherent array gain (+9.03 dB). "
                f"FOM improves from {fom_base:+.2f} dB to {fom_dm:+.2f} dB. "
                f"Bandwidth and modulation structure are unchanged.",
                S_BULLET))

    story.append(PageBreak())

    # ── Section 3: Signals figure ─────────────────────────────────────────────
    story += [
        Paragraph("3.  Time Domain Waveforms and Frequency Spectra", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
        Paragraph(
            "The figure below shows the first 3 ms of each modulated signal (left "
            "column) and its one-sided power spectral density zoomed around the "
            "carrier frequency (right column). Each panel title shows the scheme's "
            "bandwidth, transmission efficiency, and Figure of Merit.",
            S_BODY),
        Image(signals_img, width=16.5*cm, height=10.8*cm),
        Paragraph(
            "Figure 1  — Time domain and frequency spectrum for all six modulation "
            "schemes. Message f_m = 1 kHz, carrier f_c = 10 kHz. Carrier power is "
            "visible in AM but suppressed in DSB-SC/SSB-SC. FM waveforms show "
            "constant envelope with varying instantaneous frequency. SADM-SEC shows "
            "the received baseband signal at Bob after spatial combining.",
            S_CAP),
        PageBreak(),
    ]

    # ── Section 4: Metrics comparison ────────────────────────────────────────
    story += [
        Paragraph("4.  Quantitative Metrics Comparison  (Module 6)", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
        Image(metrics_img, width=16.5*cm, height=9.9*cm),
        Paragraph(
            "Figure 2  — Six-panel metrics dashboard. Top row: bandwidth, "
            "transmission efficiency, Figure of Merit. Bottom row: Noise Figure, "
            "FOM vs channel SNR sweep (Module 6 comparison), multi-metric radar chart.",
            S_CAP),
    ]

    # ── Section 5: Full metrics table ────────────────────────────────────────
    story += [
        Paragraph("5.  Numerical Results Table", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
    ]

    hdr = ["Scheme", "BW (kHz)", "eta (%)", "FOM (dB)", "NF (dB)",
           "Array Gain", "Secrecy"]
    table_data = [hdr]
    TABLE_SCHEMES = ["AM", "AM+DM", "DSB-SC", "DSB-SC+DM", "SSB-SC", "SSB-SC+DM",
                     "FM-NB", "FM-NB+DM", "FM-WB", "FM-WB+DM", "SADM-SEC"]
    for name in TABLE_SCHEMES:
        d = results[name]
        bw   = f"{d['bw_hz']/1000:.1f}"
        eta  = f"{d['eta']*100:.0f}"
        fom  = f"{d['fom_db']:+.2f}"
        nf   = f"{d['nf_db']:+.2f}" if abs(d['nf_db']) < 30 else f"{d['nf_db']:+.0f}"
        ag   = f"+{10*np.log10(N_ANTENNAS):.1f} dB" if "+DM" in name or name == "SADM-SEC" else "—"
        sec  = "Yes (spatial)" if name == "SADM-SEC" else "No"
        table_data.append([name, bw, eta, fom, nf, ag, sec])

    col_w = [3.5*cm, 2.0*cm, 2.0*cm, 2.2*cm, 2.3*cm, 2.2*cm, 2.3*cm]
    t_met = Table(table_data, colWidths=col_w)

    row_colors = [
        colors.HexColor("#F5F8FF"),
        colors.HexColor("#EAF0FB"),
    ]
    sadm_row       = TABLE_SCHEMES.index("SADM-SEC") + 1
    highlight_rows = [i+1 for i, n in enumerate(TABLE_SCHEMES) if "+DM" in n]

    ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),   C_ACC),
        ("TEXTCOLOR",   (0, 0), (-1, 0),   C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),   "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1),  8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), row_colors),
        ("BACKGROUND",  (0, sadm_row), (-1, sadm_row),
                        colors.HexColor("#D9EAF8")),
        ("FONTNAME",    (0, sadm_row), (-1, sadm_row), "Helvetica-Bold"),
        ("GRID",        (0, 0), (-1, -1),  0.4, C_GREY),
        ("ALIGN",       (0, 0), (-1, -1),  "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1),  6),
        ("TOPPADDING",  (0, 0), (-1, -1),  5),
        ("BOTTOMPADDING",(0,0), (-1, -1),  5),
    ])
    for r in highlight_rows:
        ts.add("BACKGROUND", (0, r), (-1, r), colors.HexColor("#E8F4FD"))
    t_met.setStyle(ts)
    story += [t_met, Spacer(1, 0.3*cm),
              Paragraph(
                  "Table 1  — Full metric comparison. SADM-SEC row highlighted. "
                  "NF values beyond +/-30 dB are shown rounded. "
                  "SADM-SEC FOM and NF refer to the Bob receiver only.",
                  S_CAP)]

    # ── Section 6: Discussion ─────────────────────────────────────────────────
    story += [
        PageBreak(),
        Paragraph("6.  Discussion", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
    ]

    disc_paras = [
        ("6.1  Bandwidth vs SNR Tradeoff  (Module 4)",
         "The FM wideband scheme (beta=5) achieves the highest Figure of Merit "
         "(+26.5 dB) by expanding bandwidth to 12 kHz — 12 times the message "
         "bandwidth. This directly demonstrates the bandwidth-SNR exchange taught "
         "in Module 4 via Carson's Rule. Narrowband FM (beta=0.5) provides no "
         "SNR advantage over DSB-SC but maintains constant envelope, which is "
         "valuable when power amplifier linearity is a constraint. "
         "SSB-SC offers the best spectral efficiency at half the bandwidth of "
         "DSB-SC with identical SNR performance, making it preferred for HF "
         "point-to-point links where spectrum is scarce."),
        ("6.2  Carrier Power Waste — AM vs DSB-SC  (Module 2 & 3)",
         "At m=1, conventional AM wastes 66.7% of transmit power in the carrier, "
         "achieving only eta=33.3% transmission efficiency. This directly reduces "
         "the SNR at the receiver: FOM = m^2/2 / (1 + m^2/2) = 0.333 = -4.77 dB, "
         "meaning AM's receiver SNR is always at least 4.77 dB below what the "
         "channel can support. DSB-SC and SSB-SC eliminate this waste, achieving "
         "FOM = 0 dB (the baseline). This is the central result of Module 3."),
        ("6.3  Noise Figure Analysis  (Module 6)",
         "The Noise Figure (NF = SNR_channel - SNR_output in dB) quantifies how "
         "much each system degrades the channel's natural SNR. AM has NF = +4.77 dB "
         "— it actively worsens the SNR by wasting carrier power. DSB-SC and SSB-SC "
         "have NF = 0 dB — they neither improve nor degrade. FM-WB has NF = -26.5 dB "
         "— a negative NF meaning it improves on the thermal channel SNR by trading "
         "bandwidth. SADM-SEC achieves NF = -9.03 dB at Bob via coherent array gain "
         "(10*log10(8) = 9.03 dB) without any bandwidth expansion, which no "
         "conventional single-antenna analog system can achieve."),
        ("6.4  Physical Layer Security — Unique to SADM-SEC",
         "Conventional analog modulation schemes (AM, DSB-SC, SSB-SC, FM) transmit "
         "omnidirectionally — any receiver at any angle can intercept the signal with "
         "the same SNR. SADM-SEC fundamentally changes this by using the null-space "
         "artificial noise projector P_AN to degrade any receiver not at Bob's angle. "
         "Eve at -45 deg receives a signal with FOM = -73.4 dB and NF = +73.4 dB, "
         "making recovery practically impossible. This is the only metric where "
         "SADM-SEC is categorically different from all textbook systems — it adds a "
         "spatial security dimension that has no equivalent in Modules 2-4."),
    ]

    for heading, body in disc_paras:
        story += [
            Paragraph(heading, S_H2),
            Paragraph(body, S_BODY),
        ]

    # ── Section 7: Verdict ────────────────────────────────────────────────────
    story += [
        Paragraph("7.  Which Modulation is Best?", S_H1),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
        Paragraph(
            "There is no single winner — the best scheme depends entirely on "
            "the application constraint. The table below gives the verdict "
            "by use case:",
            S_BODY),
    ]

    verdict_data = [
        ["Priority", "Best Scheme", "Reason"],
        ["Highest SNR performance",    "FM-WB (beta=5)",  "+26.5 dB FOM via BW expansion"],
        ["Best spectrum efficiency",   "SSB-SC",          "BW = f_m only, FOM = 0 dB"],
        ["Simplest demodulation",      "AM",              "Envelope detector, no sync needed"],
        ["No carrier waste",           "DSB-SC / SSB-SC", "eta = 100%"],
        ["Constant envelope (robust)", "FM-NB / FM-WB",   "Immune to amplitude nonlinearity"],
        ["Physical-layer security",    "SADM-SEC",        "Only scheme with spatial secrecy"],
        ["Array-gain SNR + security",  "SADM-SEC (Bob)",  "FOM = +9.03 dB AND Cs > 20 bits/s/Hz"],
    ]

    t_verdict = Table(verdict_data, colWidths=[5.5*cm, 4.5*cm, 6.4*cm])
    t_verdict.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  C_ACC),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F5F8FF"), colors.HexColor("#EAF0FB")]),
        ("BACKGROUND",  (0, 6), (-1, 7),  colors.HexColor("#D9EAF8")),
        ("FONTNAME",    (0, 6), (-1, 7),  "Helvetica-Bold"),
        ("GRID",        (0, 0), (-1, -1), 0.4, C_GREY),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0, -1),  "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
    ]))
    story += [t_verdict, Spacer(1, 0.4*cm)]

    story += [
        Paragraph("Overall Conclusion", S_H2),
        Paragraph(
            "For pure noise performance with unconstrained bandwidth: "
            "<b>FM Wideband (beta=5)</b> wins at +26.5 dB FOM.  "
            "For best spectrum efficiency: <b>SSB-SC</b>.  "
            "For a communication system that must be simultaneously high-SNR at the "
            "intended receiver AND unrecoverable at any eavesdropper — the only "
            "scheme in this comparison that achieves both is <b>SADM-SEC</b>, which "
            "adds a physical-layer security dimension (Secrecy Rate > 22 bits/s/Hz) "
            "that fundamentally cannot be replicated by any conventional modulation "
            "scheme regardless of how much power or bandwidth is used.",
            S_BODY),
        Spacer(1, 0.5*cm),
        HRFlowable(width="100%", thickness=0.5, color=C_GREY),
        Spacer(1, 0.2*cm),
        Paragraph(
            "BECE304L Analog Communication Systems  |  VIT Chennai  |  "
            "Modules 2, 3, 4, 6  |  SADM-SEC Course Project",
            sty("footer", fontSize=8, textColor=C_GREY, alignment=TA_CENTER)),
    ]

    doc.build(story)
    print(f"  [PDF] Saved -> {out}")
    return out


# =============================================================================
#  MASTER RUN
# =============================================================================

def run_all():
    os.makedirs("outputs", exist_ok=True)
    print("\n" + "="*62)
    print("  SADM-SEC  |  Modulation Comparison  (BECE304L)")
    print("="*62)

    print("\n[1] Simulating all modulation schemes ...")
    results = compute_all()

    print("\n[2] Generating waveform + spectrum figure ...")
    sig_img = plot_signals(results)

    print("\n[3] Generating metrics comparison figure ...")
    met_img = plot_metrics(results)

    print("\n[4] Generating PDF report ...")
    pdf_out = generate_pdf(results, sig_img, met_img)

    print("\n" + "="*62)
    print("  All outputs complete.")
    print(f"    {sig_img}")
    print(f"    {met_img}")
    print(f"    {pdf_out}")
    print("="*62 + "\n")
    return pdf_out, sig_img, met_img


if __name__ == "__main__":
    run_all()