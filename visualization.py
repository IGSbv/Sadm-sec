"""
=============================================================================
SADM-SEC  |  visualization.py
=============================================================================
Comprehensive visualization suite — aligned with BECE304L Module 6.
Produces 5 publication-quality plots:

  Fig 1 - Antenna Array Beam Pattern (polar)
  Fig 2 - Figure of Merit Comparison  [MODULE 6]
            SADM-Bob / SADM-Eve vs AM / DSB-SC / SSB-SC / FM
  Fig 3 - Noise Figure (NF) vs Receiver Angle  [MODULE 6]
            Spatial NF profile — null at Bob, maximum at Eve
  Fig 4 - Moving Target Tracking (angle + SNR timeseries)
  Fig 5 - Artificial Noise Null-Space Heat Map

Run:
    python3 visualization.py

=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # non-interactive, saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

from spatial_logic import (
    N_ANTENNAS, steering_vector, beamforming_weights,
    noise_projection_matrix, sadm_transmit, virtual_channel,
    compute_snr, compute_snr_analytical, secrecy_rate, generate_pilot_ping,
    root_music_doa, SADMTracker
)
from noise_analysis import (
    fom_vs_snr_sweep, nf_vs_angle_sweep,
    fom_am, fom_dsb_sc, fom_ssb_sc, fom_fm, fom_theoretical_array,
    print_noise_budget
)


# -----------------------------------------------------------------------------
#  SHARED STYLE
# -----------------------------------------------------------------------------

STYLE = {
    "bob_color"  : "#00D4FF",    # cyan
    "eve_color"  : "#FF4040",    # red
    "sec_color"  : "#00FF88",    # green
    "bg_dark"    : "#0A0F1E",
    "bg_panel"   : "#111827",
    "grid_color" : "#1E2A3E",
    "text_color" : "#E2E8F0",
    "accent"     : "#7C3AED",
}

plt.rcParams.update({
    "figure.facecolor"  : STYLE["bg_dark"],
    "axes.facecolor"    : STYLE["bg_panel"],
    "axes.edgecolor"    : STYLE["grid_color"],
    "axes.labelcolor"   : STYLE["text_color"],
    "xtick.color"       : STYLE["text_color"],
    "ytick.color"       : STYLE["text_color"],
    "text.color"        : STYLE["text_color"],
    "grid.color"        : STYLE["grid_color"],
    "grid.linestyle"    : "--",
    "grid.linewidth"    : 0.5,
    "font.family"       : "monospace",
    "font.size"         : 9,
})


def fig_title(ax, text):
    ax.set_title(text, color=STYLE["text_color"],
                 fontsize=11, fontweight="bold", pad=10)


# -----------------------------------------------------------------------------
#  FIG 1 — BEAM PATTERN  (polar)
# -----------------------------------------------------------------------------

def plot_beam_pattern(ax_polar, bob_angle: float = 30.0):
    """
    Polar beam pattern showing:
      - Message beam (steered to Bob)
      - Artificial Noise pattern (null at Bob)
    """
    angles_deg = np.linspace(-90, 90, 720)
    angles_rad = np.deg2rad(angles_deg)

    w    = beamforming_weights(bob_angle)
    P_AN = noise_projection_matrix(bob_angle)

    msg_gain  = np.zeros(len(angles_deg))
    an_gain   = np.zeros(len(angles_deg))

    for i, θ in enumerate(angles_deg):
        a = steering_vector(θ)
        msg_gain[i] = np.abs(w.conj() @ a) ** 2
        an_gain[i]  = np.real(a.conj() @ P_AN @ a)

    # Normalise to dB, clip at -40 dB
    def to_db(g):
        return np.maximum(10 * np.log10(g / (np.max(g) + 1e-30) + 1e-10), -40)

    msg_db = to_db(msg_gain)
    an_db  = to_db(an_gain)

    # Convert to polar (shift so -90..90 maps to full circle display)
    # Map -90 to 90 degrees directly to the polar coordinates
    # Polar plots in Matplotlib usually treat 0 as East (Right)
    # We want 0 to be North (Top)
    theta_plot = np.deg2rad(angles_deg) 

    # Ensure the plot spans the correct range
    ax_polar.set_theta_zero_location("N")  # Sets 0 deg to the top
    ax_polar.set_theta_direction(-1)       # Clockwise
    ax_polar.set_thetamin(-90)
    ax_polar.set_thetamax(90)

    r_msg = (msg_db + 40) / 40    # normalise [0,1]
    r_an  = (an_db  + 40) / 40

    ax_polar.plot(theta_plot, r_msg, color=STYLE["bob_color"],
                  linewidth=2, label="Message Beam")
    ax_polar.fill(theta_plot, r_msg, alpha=0.15, color=STYLE["bob_color"])
    ax_polar.plot(theta_plot, r_an,  color=STYLE["eve_color"],
                  linewidth=1.5, linestyle="--", label="Artificial Noise")
    ax_polar.fill(theta_plot, r_an,  alpha=0.1,  color=STYLE["eve_color"])

    # Annotate Bob and Eve
    bob_r    = np.interp(bob_angle, angles_deg, r_msg)
    bob_t    = np.deg2rad(bob_angle)
    ax_polar.annotate(f"Bob\n{bob_angle} deg",
                      xy=(bob_t, bob_r),
                      xytext=(bob_t + 0.3, bob_r + 0.15),
                      color=STYLE["bob_color"], fontsize=8,
                      arrowprops=dict(arrowstyle="->",
                                      color=STYLE["bob_color"], lw=1))

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rlabel_position(135)
    ax_polar.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax_polar.set_yticklabels(["-30dB", "-20dB", "-10dB", "0dB"],
                              fontsize=7)
    ax_polar.set_thetamin(-90)
    ax_polar.set_thetamax(90)
    ax_polar.set_facecolor(STYLE["bg_panel"])
    ax_polar.grid(True, color=STYLE["grid_color"], linewidth=0.5)
    ax_polar.legend(loc="lower left", fontsize=8,
                    facecolor=STYLE["bg_dark"], edgecolor=STYLE["grid_color"])
    fig_title(ax_polar, "Array Beam Pattern  (ULA, N=8)")


# -----------------------------------------------------------------------------
#  FIG 2 — FIGURE OF MERIT COMPARISON  (Module 6 — BECE304L)
# -----------------------------------------------------------------------------

def plot_fom_comparison(ax, bob_angle: float = 30.0, eve_angle: float = -45.0):
    """
    Figure of Merit (FOM) comparison — MODULE 6 alignment.

    Plots FOM (dB) vs reference channel SNR for:
      • SADM-Bob  — benefits from N=8 array gain (+9 dB above DSB-SC baseline)
      • SADM-Eve  — degraded by AN injection (well below conventional AM)
      • AM (m=1)  — textbook FOM = -4.77 dB
      • DSB-SC    — textbook baseline at 0 dB
      • SSB-SC    — same as DSB-SC
      • FM β=5    — wideband FM at +26.5 dB (trades bandwidth for SNR)

    The y-axis is FOM = SNR_output - SNR_channel  (dB).
    A positive FOM means the system improves on single-antenna thermal SNR.
    """
    data = fom_vs_snr_sweep(bob_angle, eve_angle)
    x = data["snr_ch"]

    # SADM curves — vary with SNR (AN power is fixed, so FOM compresses at high SNR)
    ax.plot(x, data["fom_bob"],  color=STYLE["bob_color"], linewidth=2.5,
            label=f"SADM-Bob  (N=8, θ={bob_angle:.0f}°)")
    ax.fill_between(x, data["fom_bob"], data["fom_dsb"],
                    where=(data["fom_bob"] > data["fom_dsb"]),
                    alpha=0.12, color=STYLE["bob_color"])

    ax.plot(x, data["fom_eve"],  color=STYLE["eve_color"], linewidth=2.5,
            linestyle="-.", label=f"SADM-Eve  (θ={eve_angle:.0f}°)")
    ax.fill_between(x, data["fom_eve"], data["fom_am"],
                    where=(data["fom_eve"] < data["fom_am"]),
                    alpha=0.10, color=STYLE["eve_color"])

    # Textbook system baselines — horizontal reference lines
    ax.axhline(10 * np.log10(fom_fm(beta=5)),   color="#FFD700", linewidth=1.4,
               linestyle="--", label="FM  β=5  (+26.5 dB)")
    ax.axhline(10 * np.log10(fom_dsb_sc()),     color=STYLE["sec_color"], linewidth=1.4,
               linestyle="--", label="DSB-SC  (0 dB baseline)")
    ax.axhline(10 * np.log10(fom_ssb_sc()),     color="#88FFCC", linewidth=0.8,
               linestyle=":")
    ax.axhline(10 * np.log10(fom_am(1.0)),      color="#FF8844", linewidth=1.4,
               linestyle="--", label="AM  m=1  (-4.77 dB)")

    # Theoretical array gain ceiling for Bob
    arr_db = 10 * np.log10(fom_theoretical_array(N_ANTENNAS))
    ax.axhline(arr_db, color=STYLE["bob_color"], linewidth=0.8, linestyle=":",
               alpha=0.6, label=f"Array gain limit  ({arr_db:.1f} dB)")

    # Labels on the reference lines
    x_label = x[-1] - (x[-1] - x[0]) * 0.02
    ax.text(x_label, 10 * np.log10(fom_fm(5))  + 1.0, "FM  β=5",
            color="#FFD700", fontsize=7, ha="right")
    ax.text(x_label, 10 * np.log10(fom_dsb_sc()) + 1.0, "DSB-SC",
            color=STYLE["sec_color"], fontsize=7, ha="right")
    ax.text(x_label, 10 * np.log10(fom_am(1.0)) - 2.0, "AM  m=1",
            color="#FF8844", fontsize=7, ha="right")

    ax.set_xlabel("Reference Channel SNR  (dB)")
    ax.set_ylabel("Figure of Merit  (dB)")
    ax.grid(True)
    ax.legend(fontsize=7.5, facecolor=STYLE["bg_dark"],
              edgecolor=STYLE["grid_color"], loc="upper left")
    fig_title(ax,
              "Figure of Merit — SADM-SEC vs AM / DSB-SC / FM  (Module 6)")


# -----------------------------------------------------------------------------
#  FIG 3 — NOISE FIGURE vs ANGLE  (Module 6 — BECE304L)
# -----------------------------------------------------------------------------

def plot_nf_vs_angle(ax, bob_angle: float = 30.0, eve_angle: float = -45.0):
    """
    Noise Figure (NF) as a function of receiver angle — MODULE 6 alignment.

    NF = SNR_channel_dB - SNR_output_dB

    Key features to observe:
      • Deep negative NF at Bob's angle → array gain improves on thermal SNR
      • Large positive NF everywhere else → AN injection severely degrades SNR
      • NF is spatially selective: SADM acts as a spatial noise figure equaliser
    """
    angles, nf_db = nf_vs_angle_sweep(bob_angle)

    # Colour-code: low NF = good (Bob zone), high NF = bad (Eve zone)
    ax.plot(angles, nf_db, color=STYLE["acc"] if "acc" in STYLE else STYLE["accent"],
            linewidth=2, zorder=3)

    # Shade secure (low NF) and insecure (high NF) regions relative to 0 dB
    ax.fill_between(angles, nf_db, 0,
                    where=(nf_db < 0),
                    alpha=0.18, color=STYLE["bob_color"],
                    label="NF < 0 dB  (array gain region — Bob)")
    ax.fill_between(angles, nf_db, 0,
                    where=(nf_db > 0),
                    alpha=0.12, color=STYLE["eve_color"],
                    label="NF > 0 dB  (AN-degraded region — Eve)")

    # 0 dB reference = single-antenna performance (DSB-SC baseline)
    ax.axhline(0, color=STYLE["sec_color"], linewidth=1.2, linestyle="--",
               label="0 dB  (DSB-SC / single-antenna baseline)")

    # Mark textbook reference lines
    am_nf  = -10 * np.log10(fom_am(1.0))        # NF of AM system equivalent
    ax.axhline(am_nf, color="#FF8844", linewidth=0.9, linestyle=":",
               label=f"AM equiv. NF  ({am_nf:.1f} dB)")

    # Annotate Bob and Eve
    bob_nf = float(np.interp(bob_angle, angles, nf_db))
    eve_nf = float(np.interp(eve_angle, angles, nf_db))

    ax.annotate(f"Bob\nNF = {bob_nf:.1f} dB",
                xy=(bob_angle, bob_nf),
                xytext=(bob_angle + 12, bob_nf - 8),
                color=STYLE["bob_color"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=STYLE["bob_color"], lw=1))
    ax.annotate(f"Eve\nNF = {eve_nf:.1f} dB",
                xy=(eve_angle, eve_nf),
                xytext=(eve_angle - 5, eve_nf - 10),
                color=STYLE["eve_color"], fontsize=8, ha="right",
                arrowprops=dict(arrowstyle="->", color=STYLE["eve_color"], lw=1))

    ax.set_xlabel("Receiver Angle  (degrees)")
    ax.set_ylabel("Noise Figure NF  (dB)")
    ax.grid(True)
    ax.legend(fontsize=7.5, facecolor=STYLE["bg_dark"],
              edgecolor=STYLE["grid_color"])
    fig_title(ax, "Noise Figure vs Angle  (Module 6  |  NF = SNR_ch − SNR_out)")


# -----------------------------------------------------------------------------
#  FIG 3 — MOVING TARGET  (tracking + SNR timeseries)
# -----------------------------------------------------------------------------

def plot_moving_target(ax_angle, ax_snr,
                       n_steps: int = 40,
                       eve_angle: float = -45.0,
                       use_ml: bool = True): # Default to True for your project
    """
    Bob sweeps from -60 deg to +60 deg.
    Top subplot:   true vs estimated angle using ML.
    Bottom subplot: Bob SNR vs Eve SNR over time.
    """
    true_angles = np.linspace(-60, 60, n_steps)
    est_angles  = np.zeros(n_steps)
    snr_bobs    = np.zeros(n_steps)
    snr_eves    = np.zeros(n_steps)

    # Tracker now uses the ML flag!
    # Use higher alpha (0.7) for moving targets to track faster
    tracker = SADMTracker(bob_initial_angle=-60, alpha=0.7, use_ml=use_ml)
    L       = 512
    t       = np.linspace(0, 1, L)
    message = np.sin(2 * np.pi * 440 * t * L / 8000)

    for i, true_θ in enumerate(true_angles):
        pilot      = generate_pilot_ping(true_θ, n_snapshots=512,
                                          snr_pilot_db=15)
        est_angles[i] = tracker.update(pilot)

        X     = tracker.transmit(message)

        # Use analytical SNR - empirical compute_snr has noise realization mismatch
        snr_bobs[i] = compute_snr_analytical(true_θ, est_angles[i], 20.0, 10.0)
        snr_eves[i] = compute_snr_analytical(eve_angle, est_angles[i], 20.0, 10.0)

    steps = np.arange(n_steps)

    # -- Angle tracking ----------------------------------------------------
    ax_angle.plot(steps, true_angles, color=STYLE["bob_color"],
                  linewidth=2, label="True Angle")
    
    # Updated label to credit the Neural Network
    label_str = "MLP Neural Net Estimate" if use_ml else "Root-MUSIC Estimate"
    ax_angle.plot(steps, est_angles,  color=STYLE["accent"],
                  linewidth=1.5, linestyle="--", label=label_str)
    
    ax_angle.fill_between(steps, true_angles, est_angles,
                          alpha=0.2, color=STYLE["accent"],
                          label="Tracking Error")
    ax_angle.set_ylabel("Angle ( deg)")
    ax_angle.set_ylim(-75, 75)
    ax_angle.grid(True)
    ax_angle.legend(fontsize=8, facecolor=STYLE["bg_dark"],
                    edgecolor=STYLE["grid_color"])
    
    # Updated title
    title_str = "Moving Target Tracking (ML-Powered)" if use_ml else "Moving Target Tracking (Root-MUSIC)"
    fig_title(ax_angle, title_str)

    # -- SNR timeseries ----------------------------------------------------
    ax_snr.plot(steps, snr_bobs, color=STYLE["bob_color"],
                linewidth=2, label="Bob SNR")
    ax_snr.plot(steps, snr_eves, color=STYLE["eve_color"],
                linewidth=2, linestyle="--", label=f"Eve SNR ({eve_angle} deg)")
    ax_snr.fill_between(steps, snr_eves, snr_bobs,
                        where=(snr_bobs >= snr_eves),
                        alpha=0.12, color=STYLE["sec_color"],
                        label="Secrecy Margin")
    ax_snr.axhline(0, color=STYLE["grid_color"], linewidth=0.8)
    ax_snr.set_xlabel("Time Step")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.grid(True)
    ax_snr.legend(fontsize=8, facecolor=STYLE["bg_dark"],
                  edgecolor=STYLE["grid_color"])
    fig_title(ax_snr, "SNR Comparison During Movement")


# -----------------------------------------------------------------------------
#  FIG 4 — ARTIFICIAL NOISE HEAT MAP
# -----------------------------------------------------------------------------

def plot_an_heatmap(ax, bob_angle: float = 30.0, n_pts: int = 150):
    """
    2-D heat map showing AN power as a function of
    (Tx steering angle, Rx observation angle).
    The null (dark stripe) traces the diagonal where Tx=Rx=Bob.
    """
    tx_angles = np.linspace(-80, 80, n_pts)
    rx_angles = np.linspace(-80, 80, n_pts)
    Z         = np.zeros((n_pts, n_pts))

    for i, tx in enumerate(tx_angles):
        P_AN = noise_projection_matrix(tx)
        for j, rx in enumerate(rx_angles):
            a     = steering_vector(rx)
            Z[j, i] = np.real(a.conj() @ P_AN @ a)

    # Custom colormap: black -> purple -> white
    cmap = LinearSegmentedColormap.from_list(
        "sadm",
        ["#000000", "#0A0F1E", STYLE["accent"], "#FF4040", "#FFFFFF"],
        N=512
    )

    im = ax.imshow(
        10 * np.log10(Z / np.max(Z) + 1e-10),
        extent=[-80, 80, -80, 80],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-40, vmax=0
    )
    plt.colorbar(im, ax=ax, label="AN Power (dB)", shrink=0.8)

    # Mark Bob's null diagonal
    ax.plot([-80, 80], [-80, 80], color=STYLE["bob_color"],
            linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(bob_angle, color=STYLE["bob_color"], linewidth=1.5,
               label=f"Bob Tx angle ({bob_angle} deg)")
    ax.axhline(bob_angle, color=STYLE["bob_color"], linewidth=1.5)
    ax.plot(bob_angle, bob_angle, "o", color=STYLE["bob_color"],
            markersize=10, zorder=5)
    ax.annotate("Bob's\nNull Zone",
                xy=(bob_angle, bob_angle),
                xytext=(bob_angle + 15, bob_angle + 15),
                color=STYLE["bob_color"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=STYLE["bob_color"]))

    ax.set_xlabel("Tx Steering Angle ( deg)")
    ax.set_ylabel("Rx Observation Angle ( deg)")
    ax.legend(fontsize=8, facecolor=STYLE["bg_dark"],
              edgecolor=STYLE["grid_color"])
    fig_title(ax, "Artificial Noise Power Map  (dB, null at Bob)")


# -----------------------------------------------------------------------------
#  MASTER RENDER
# -----------------------------------------------------------------------------

def render_all(output_path: str = "outputs/sadm_plots.png", use_ml: bool = True):
    """
    Render the full SADM-SEC dashboard.

    Layout (3 rows × 3 columns):
    ┌──────────────────┬──────────────────────────┬──────────────────────┐
    │  Fig 1           │  Fig 2  [MODULE 6]        │  Fig 3  [MODULE 6]   │
    │  Beam Pattern    │  Figure of Merit           │  Noise Figure        │
    │  (polar)         │  SADM vs AM/DSB-SC/FM      │  NF vs Angle         │
    ├──────────────────┴──────────────────┬─────────┴──────────────────────┤
    │  Fig 4 — Moving Target Tracking     │  Fig 5 — AN Heat Map            │
    │  (angle tracking + SNR timeseries)  │                                 │
    └─────────────────────────────────────┴─────────────────────────────────┘
    """
    import os

    fig = plt.figure(figsize=(22, 18), facecolor=STYLE["bg_dark"])
    fig.suptitle(
        "SADM-SEC  ·  Physical Layer Security Dashboard\n"
        "Spatially Aware Directional Modulation  |  Module 6: Noise Analysis  (BECE304L)",
        fontsize=13, fontweight="bold", color=STYLE["bob_color"], y=0.99
    )

    # Outer grid: 2 rows
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.44,
        wspace=0.34,
        left=0.06, right=0.97,
        top=0.94, bottom=0.06
    )

    # Row 0: beam pattern | FOM comparison | NF vs angle
    ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_fom   = fig.add_subplot(gs[0, 1])
    ax_nf    = fig.add_subplot(gs[0, 2])

    # Row 1: tracking (2 stacked) | heat map
    gs_track = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, :2], hspace=0.45)
    ax_angle = fig.add_subplot(gs_track[0])
    ax_snr   = fig.add_subplot(gs_track[1])
    ax_heat  = fig.add_subplot(gs[1, 2])

    print("[Viz] Rendering beam pattern ...")
    plot_beam_pattern(ax_polar, bob_angle=30.0)

    print("[Viz] Rendering Figure of Merit comparison  [Module 6] ...")
    plot_fom_comparison(ax_fom, bob_angle=30.0, eve_angle=-45.0)

    print("[Viz] Rendering Noise Figure vs Angle  [Module 6] ...")
    plot_nf_vs_angle(ax_nf, bob_angle=30.0, eve_angle=-45.0)

    print("[Viz] Rendering moving target ...")
    plot_moving_target(ax_angle, ax_snr, n_steps=40, eve_angle=-45.0, use_ml=use_ml)

    print("[Viz] Rendering AN heat map ...")
    plot_an_heatmap(ax_heat, bob_angle=30.0)

    # Module 6 label badge
    fig.text(0.01, 0.01,
             "Figs 2 & 3: Module 6 — Noise Figure · Figure of Merit · Noise Temperature",
             ha="left", va="bottom", fontsize=7.5, color="#4A8FBF",
             style="italic")

    # Watermark
    fig.text(0.99, 0.01,
             "SADM-SEC  |  BECE304L Analog Communication Systems",
             ha="right", va="bottom",
             fontsize=7, color="#2A3A5E")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg_dark"])
    plt.close(fig)
    print(f"[Viz] Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    render_all()
