"""
SADM-SEC | realtime_monitor.py  (v3 — Windows/Linux compatible)
================================================================
BUGS FIXED:
  FIX1: GridSpecFromSubplotSpec does not accept left/right kwargs
        → replaced with figure.add_axes([rect]) for slider panel
  FIX2: Bar heights crash on negative SNR → offset baseline (50 dB)
  FIX3: Polar theta convention wrong → Bob marker misplaced
  FIX4: params_received flag froze angle tracking permanently
  FIX5: sadm_beamformer missing set_snr_sig_db/set_snr_an_db callbacks
  FIX6: Heat map redrawn every frame → flicker and wasted CPU

INTERACTIVE CONTROLS (sliders at bottom):
  ● Bob Angle     −90 → +90°
  ● Signal Power    5 → 35 dB
  ● AN Power        0 → 30 dB
  ● [Reset] button

USAGE:
  python realtime_monitor.py --demo    # no hardware, sliders fully live
  python realtime_monitor.py           # connect to GNU Radio ports 5555-5562
================================================================
"""

import sys, os, time
import numpy as np

# ── matplotlib backend (try Qt first, fall back to Tk, then Agg) ─────────────
import matplotlib
for _be in ("Qt5Agg", "QtAgg", "TkAgg"):
    try:
        matplotlib.use(_be)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close()          # test the backend actually opens
        break
    except Exception:
        continue
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("[WARN] No interactive backend found - using Agg (no live window)")

import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

# ── ZMQ ──────────────────────────────────────────────────────────────────────
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

# ── Spatial math ─────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.path.abspath("."))
from spatial_logic import (
    N_ANTENNAS, steering_vector, beamforming_weights,
    noise_projection_matrix, SADMTracker,
    compute_snr_analytical, secrecy_rate, generate_pilot_ping,
)

# ── Theme ─────────────────────────────────────────────────────────────────────
C_BG   = "#0A0F1E"
C_PAN  = "#111827"
C_CYAN = "#00D4FF"
C_GRN  = "#00FF88"
C_RED  = "#FF4040"
C_AMB  = "#FFB300"
C_TXT  = "#E2E8F0"
C_DIM  = "#4A5A6A"
C_GRID = "#1E2A3E"

EVE_ANGLE = -45.0
HIST_LEN  = 120
BLOCK_SZ  = 1024


# =============================================================================
#  ZMQ helpers
# =============================================================================
class ZMQSubscriber:
    def __init__(self, base=5555, host="127.0.0.1", n=N_ANTENNAS):
        self.n   = n
        self.ctx = zmq.Context()
        self.subs = []
        for i in range(n):
            s = self.ctx.socket(zmq.SUB)
            s.connect(f"tcp://{host}:{base+i}")
            s.setsockopt_string(zmq.SUBSCRIBE, "")
            s.setsockopt(zmq.RCVTIMEO, 200)
            self.subs.append(s)
        print(f"[ZMQ] Antenna ports {base}–{base+n-1}")

    def receive(self):
        X = np.zeros((self.n, BLOCK_SZ), dtype=np.complex64)
        for i, s in enumerate(self.subs):
            try:
                raw = s.recv()
                arr = np.frombuffer(raw, dtype=np.complex64)
                ln  = min(len(arr), BLOCK_SZ)
                X[i, :ln] = arr[:ln]
            except zmq.Again:
                return None
        return X

    def close(self):
        for s in self.subs: s.close()
        self.ctx.term()


class ParamSubscriber:
    def __init__(self, port=5563, host="127.0.0.1"):
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{host}:{port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.RCVTIMEO, 50)
        print(f"[ZMQ] Param port {port}")

    def receive(self):
        try:
            import json
            return json.loads(self.sub.recv().decode())
        except Exception:
            return None

    def close(self):
        self.sub.close()
        self.ctx.term()


# =============================================================================
#  Real-Time Plotter
# =============================================================================
class RealtimePlotter:
    """
    6-panel real-time dashboard + 3 interactive sliders.

    Layout (figure coords, top=1, bottom=0):
      ┌──────────┬──────────┬──────────┐  top 43%
      │ Polar    │ SNR bars │ Metrics  │
      ├──────────┼──────────┼──────────┤  mid 43%
      │ Angle    │ SNR hist │ Heat map │
      ├──────────┴──────────┴──────────┤  bottom 14%
      │     Sliders (add_axes rects)   │
      └────────────────────────────────┘
    """

    # Slider rects [left, bottom, width, height] in figure coords
    _SL_ANGLE = [0.07, 0.04, 0.22, 0.035]
    _SL_SIG   = [0.37, 0.04, 0.22, 0.035]
    _SL_AN    = [0.67, 0.04, 0.22, 0.035]
    _BTN_RST  = [0.92, 0.04, 0.06, 0.035]

    def __init__(self, bob_init=30.0, sig_init=20.0, an_init=10.0, demo=False):
        plt.ion()
        self.demo = demo

        # ── State ────────────────────────────────────────────────────────────
        self.bob_angle     = bob_init
        self.snr_signal_db = sig_init
        self.snr_noise_db  = an_init
        self.snr_bob  = 0.0
        self.snr_eve  = 0.0
        self.sec_rate = 0.0
        self.grc_live = False
        self._heat_angle = None   # track when heat map needs redraw

        self.angle_hist = [bob_init]
        self.snrb_hist  = [0.0]
        self.snre_hist  = [0.0]
        self.sec_hist   = [0.0]

        self.tracker = SADMTracker(bob_initial_angle=bob_init, alpha=0.3, use_ml=True)

        # ── Figure ───────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(17, 10), facecolor=C_BG)
        self.fig.suptitle(
            "SADM-SEC  |  Real-Time Physical Layer Security Monitor",
            fontsize=13, fontweight="bold", color=C_CYAN, y=0.99)

        # Two rows of 3 panels each; leave bottom 15% for sliders
        gs_top = gridspec.GridSpec(
            1, 3,
            figure=self.fig,
            left=0.06, right=0.97,
            top=0.93, bottom=0.53,
            wspace=0.32
        )
        gs_mid = gridspec.GridSpec(
            1, 3,
            figure=self.fig,
            left=0.06, right=0.97,
            top=0.49, bottom=0.16,
            wspace=0.32
        )

        self.ax_pol  = self.fig.add_subplot(gs_top[0], projection="polar")
        self.ax_snrb = self.fig.add_subplot(gs_top[1])
        self.ax_met  = self.fig.add_subplot(gs_top[2])
        self.ax_ang  = self.fig.add_subplot(gs_mid[0])
        self.ax_sh   = self.fig.add_subplot(gs_mid[1])
        self.ax_heat = self.fig.add_subplot(gs_mid[2])

        for ax in [self.ax_snrb, self.ax_met, self.ax_ang, self.ax_sh, self.ax_heat]:
            ax.set_facecolor(C_PAN)
        self.ax_pol.set_facecolor(C_PAN)

        self._build_polar()
        self._build_snr_bars()
        self._build_metrics()
        self._build_angle_hist()
        self._build_snr_hist()
        self._draw_heat()
        self._build_sliders()       # FIX1: uses add_axes with rect, not GridSpecFromSubplotSpec

        self.fig.canvas.draw()
        print("[Monitor] Window ready - drag the sliders at the bottom")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel builders
    # ─────────────────────────────────────────────────────────────────────────
    def _build_polar(self):
        ax = self.ax_pol
        # FIX3: E origin + direction +1 → 0°=East, 30° = upper-right (correct)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.tick_params(colors=C_DIM, labelsize=7)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["-30", "-20", "-10", "0 dB"], fontsize=7, color=C_DIM)
        ax.set_title("Array Beam Pattern", color=C_TXT, fontsize=10, pad=6)

        ths = np.linspace(-90, 90, 360)
        rm, ra = self._polar_data(self.bob_angle, ths)
        rad = np.deg2rad(ths)
        self._pl_msg, = ax.plot(rad, rm, C_CYAN, lw=2,   label="Msg beam")
        self._pl_an,  = ax.plot(rad, ra, C_RED,  lw=1.5, ls="--", label="AN")
        self._pl_bob, = ax.plot([np.deg2rad(self.bob_angle)], [1.07],
                                "o", color=C_GRN, ms=11, label="Bob", zorder=5)
        self._pl_eve, = ax.plot([np.deg2rad(EVE_ANGLE)], [1.07],
                                "v", color=C_RED, ms=10,  label="Eve", zorder=5)
        ax.legend(loc="lower left", fontsize=7, facecolor=C_BG,
                  labelcolor=C_TXT, framealpha=0.7)

    def _polar_data(self, bob, ths):
        w = beamforming_weights(bob)
        P = noise_projection_matrix(bob)
        mg = np.array([np.abs(w.conj() @ steering_vector(t))**2 for t in ths])
        ag = np.array([np.real(steering_vector(t).conj() @ P @ steering_vector(t)) for t in ths])
        def norm(g):
            db = np.maximum(10*np.log10(g/(np.max(g)+1e-30)+1e-10), -40)
            return (db + 40) / 40
        return norm(mg), norm(ag)

    def _build_snr_bars(self):
        ax = self.ax_snrb
        # FIX2: offset so bar heights are always positive
        self._off = 50.0
        ax.set_facecolor(C_PAN)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 140)          # covers −50 to +90 dB after +50 offset
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Bob", "Eve"], fontsize=11, color=C_TXT)
        ax.set_title("Live SNR", color=C_TXT, fontsize=10)
        ax.tick_params(colors=C_DIM)
        ax.grid(axis="y", color=C_GRID, lw=0.5)
        ax.axhline(self._off, color=C_AMB, lw=1.2, ls="--", alpha=0.7)

        # Y-axis labels in real dB
        real_db = np.arange(-50, 91, 20)
        ax.set_yticks(real_db + self._off)
        ax.set_yticklabels([f"{v:+d}" for v in real_db], color=C_DIM, fontsize=8)
        ax.set_ylabel("dB", color=C_DIM, fontsize=8)
        ax.text(0.5, -0.08, "Dashed = 0 dB", ha="center",
                transform=ax.transAxes, color=C_DIM, fontsize=7)

        self._bbar = ax.bar([0], [self._off], color=C_CYAN, width=0.5, alpha=0.85)
        self._ebar = ax.bar([1], [self._off], color=C_RED,  width=0.5, alpha=0.85)
        self._lbob = ax.text(0, 132, "", ha="center", color=C_CYAN, fontsize=10, fontweight="bold")
        self._leve = ax.text(1, 132, "", ha="center", color=C_RED,  fontsize=10, fontweight="bold")

    def _build_metrics(self):
        ax = self.ax_met
        ax.axis("off")
        ax.set_title("System Metrics", color=C_TXT, fontsize=10)
        self._mtxt = ax.text(
            0.04, 0.95, "", transform=ax.transAxes,
            fontsize=9.5, color=C_TXT, family="monospace", va="top", linespacing=1.7)
        self._badge = ax.text(
            0.5, 0.07, "◉  SECURE", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=C_GRN,
            ha="center", va="center",
            bbox=dict(fc="#002A10", ec=C_GRN, lw=2.5, pad=7, boxstyle="round,pad=0.4"))

    def _build_angle_hist(self):
        ax = self.ax_ang
        ax.set_facecolor(C_PAN)
        ax.set_ylim(-95, 95)
        ax.set_title("Angle Tracking", color=C_TXT, fontsize=10)
        ax.set_ylabel("Angle (°)", color=C_TXT, fontsize=9)
        ax.set_xlabel("Samples", color=C_TXT, fontsize=9)
        ax.tick_params(colors=C_DIM)
        ax.grid(color=C_GRID, lw=0.5)
        ax.axhline(EVE_ANGLE, color=C_RED, lw=0.8, ls=":", alpha=0.6,
                   label=f"Eve {EVE_ANGLE:.0f}°")
        self._al_est,  = ax.plot([], [], C_CYAN, lw=1.5, label="Estimated")
        self._al_true, = ax.plot([], [], C_GRN,  lw=1.0, ls="--", alpha=0.7,
                                  label="True (demo)")
        ax.legend(fontsize=7.5, facecolor=C_BG, labelcolor=C_TXT)

    def _build_snr_hist(self):
        ax = self.ax_sh
        ax.set_facecolor(C_PAN)
        ax.set_ylim(-55, 85)
        ax.set_title("SNR History", color=C_TXT, fontsize=10)
        ax.set_ylabel("SNR (dB)", color=C_TXT, fontsize=9)
        ax.set_xlabel("Samples",  color=C_TXT, fontsize=9)
        ax.tick_params(colors=C_DIM)
        ax.grid(color=C_GRID, lw=0.5)
        ax.axhline(0, color=C_AMB, lw=0.8, ls="--", alpha=0.6, label="0 dB")
        self._bl, = ax.plot([], [], C_CYAN, lw=1.5, label="Bob SNR")
        self._el, = ax.plot([], [], C_RED,  lw=1.5, label="Eve SNR")
        self._sfill = ax.fill_between([], [], [], color=C_GRN, alpha=0.1)
        ax.legend(fontsize=7.5, facecolor=C_BG, labelcolor=C_TXT)

    def _draw_heat(self):
        """Redraw AN null-space heat map. Called only when angle changes > 2°."""
        ax = self.ax_heat
        ax.clear()
        ax.set_facecolor(C_PAN)

        tx = np.linspace(-90, 90, 36)
        rx = np.linspace(-90, 90, 36)
        Z  = np.zeros((36, 36))
        for i, ta in enumerate(tx):
            P = noise_projection_matrix(ta)
            for j, ra in enumerate(rx):
                a = steering_vector(ra)
                Z[j, i] = np.real(a.conj() @ P @ a)

        im = ax.imshow(
            10*np.log10(Z / (np.max(Z)+1e-10) + 1e-10),
            extent=[-90, 90, -90, 90], origin="lower",
            aspect="auto", cmap="plasma", vmin=-40, vmax=0)
        self.fig.colorbar(im, ax=ax, label="AN (dB)", shrink=0.8)
        ax.plot([-90,90], [-90,90], "w--", lw=0.7, alpha=0.4)
        ax.axvline(self.bob_angle, color=C_GRN, lw=1.8, ls="--", alpha=0.85)
        ax.axhline(self.bob_angle, color=C_GRN, lw=1.8, ls="--", alpha=0.85)
        ax.axvline(EVE_ANGLE,      color=C_RED,  lw=1.2, ls=":",  alpha=0.7)
        ax.axhline(EVE_ANGLE,      color=C_RED,  lw=1.2, ls=":",  alpha=0.7)
        ax.plot([self.bob_angle], [self.bob_angle],
                "o", color=C_GRN, ms=9, zorder=5, label="Bob null")
        ax.set_xlabel("Tx Steer (°)",    color=C_TXT, fontsize=9)
        ax.set_ylabel("Rx Observe (°)",  color=C_TXT, fontsize=9)
        ax.set_title("AN Null-Space Map", color=C_TXT, fontsize=10)
        ax.tick_params(colors=C_DIM)
        ax.legend(fontsize=7.5, facecolor=C_BG, labelcolor=C_TXT, loc="upper left")
        self._heat_angle = self.bob_angle

    # ─────────────────────────────────────────────────────────────────────────
    # FIX1: Sliders via add_axes (rect list) — no GridSpecFromSubplotSpec kwargs
    # ─────────────────────────────────────────────────────────────────────────
    def _build_sliders(self):
        # Labels above the sliders
        self.fig.text(0.18, 0.115, "Bob Angle (°)",
                      ha="center", color=C_TXT, fontsize=9)
        self.fig.text(0.48, 0.115, "Signal Power (dB)",
                      ha="center", color=C_TXT, fontsize=9)
        self.fig.text(0.78, 0.115, "AN Power (dB)",
                      ha="center", color=C_TXT, fontsize=9)
        self.fig.text(0.5, 0.005,
                      "Eve fixed at −45°  |  N=8 ULA  |  Green=SECURE  Red=LEAKED",
                      ha="center", color=C_DIM, fontsize=8)

        # Add slider axes using rect [left, bottom, width, height]
        ax_ang = self.fig.add_axes(self._SL_ANGLE, facecolor=C_PAN)
        ax_sig = self.fig.add_axes(self._SL_SIG,   facecolor=C_PAN)
        ax_an  = self.fig.add_axes(self._SL_AN,    facecolor=C_PAN)
        ax_btn = self.fig.add_axes(self._BTN_RST,  facecolor=C_PAN)

        self.s_ang = Slider(ax_ang, "", -90, 90,
                            valinit=self.bob_angle, color=C_CYAN, track_color="#1A2A3A")
        self.s_sig = Slider(ax_sig, "",   5, 35,
                            valinit=self.snr_signal_db, color=C_GRN, track_color="#1A2A3A")
        self.s_an  = Slider(ax_an,  "",   0, 30,
                            valinit=self.snr_noise_db,  color=C_RED,  track_color="#1A2A3A")

        for sl, col in [(self.s_ang, C_CYAN), (self.s_sig, C_GRN), (self.s_an, C_RED)]:
            sl.valtext.set_color(col)
            sl.valtext.set_fontsize(9)

        self.btn_rst = Button(ax_btn, "Reset", color=C_PAN, hovercolor="#2A3A5A")
        self.btn_rst.label.set_color(C_TXT)

        self.s_ang.on_changed(self._sl_angle)
        self.s_sig.on_changed(self._sl_sig)
        self.s_an.on_changed(self._sl_an)
        self.btn_rst.on_clicked(self._sl_reset)

    def _sl_angle(self, val):
        self.bob_angle = float(val)
        self.tracker.angle = self.bob_angle
        self.tracker._update_weights()
        self._heat_angle = None          # force heat redraw

    def _sl_sig(self, val):
        self.snr_signal_db = float(val)

    def _sl_an(self, val):
        self.snr_noise_db = float(val)

    def _sl_reset(self, _event):
        self.s_ang.reset()
        self.s_sig.reset()
        self.s_an.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # Main update (called every loop iteration)
    # ─────────────────────────────────────────────────────────────────────────
    def update(self, X=None, true_angle=None, grc_params=None):
        """
        X           : (N,L) complex from ZMQ, or None
        true_angle  : ground truth for demo (green reference line)
        grc_params  : dict from param_publisher {'bob_angle','snr_signal_db','snr_noise_db'}
        """
        # FIX4: accept GRC params only if they agree within 2° of slider
        if grc_params and not self.demo:
            grc_ang = grc_params.get("bob_angle", self.bob_angle)
            if abs(grc_ang - self.s_ang.val) <= 2.0:
                self.bob_angle     = grc_ang
                self.snr_signal_db = grc_params.get("snr_signal_db", self.snr_signal_db)
                self.snr_noise_db  = grc_params.get("snr_noise_db",  self.snr_noise_db)
                self.grc_live      = True

        # DOA tracking from ZMQ data (live mode, when GRC not controlling)
        if X is not None and not self.grc_live:
            try:
                est = self.tracker.update(X)
                self.bob_angle = est
                self.s_ang.set_val(est)
            except Exception:
                pass

        # Compute SNR and secrecy rate
        self.snr_bob  = compute_snr_analytical(
            self.bob_angle, self.bob_angle, self.snr_signal_db, self.snr_noise_db)
        self.snr_eve  = compute_snr_analytical(
            EVE_ANGLE,      self.bob_angle, self.snr_signal_db, self.snr_noise_db)
        self.sec_rate = secrecy_rate(self.snr_bob, self.snr_eve)

        # History
        self.angle_hist.append(true_angle if true_angle is not None else self.bob_angle)
        self.snrb_hist.append(self.snr_bob)
        self.snre_hist.append(self.snr_eve)
        self.sec_hist.append(self.sec_rate)
        for lst in [self.angle_hist, self.snrb_hist, self.snre_hist, self.sec_hist]:
            if len(lst) > HIST_LEN:
                del lst[:-HIST_LEN]

        # Refresh panels
        self._rf_polar()
        self._rf_snr_bars()
        self._rf_metrics()
        self._rf_angle_hist(true_angle)
        self._rf_snr_hist()
        if self._heat_angle is None or abs(self._heat_angle - self.bob_angle) > 2.0:
            self._draw_heat()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ─────────────────────────────────────────────────────────────────────────
    # Panel refresh helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _rf_polar(self):
        ths = np.linspace(-90, 90, 360)
        rm, ra = self._polar_data(self.bob_angle, ths)
        rad = np.deg2rad(ths)
        self._pl_msg.set_data(rad, rm)
        self._pl_an.set_data(rad,  ra)
        self._pl_bob.set_data([np.deg2rad(self.bob_angle)], [1.07])

    def _rf_snr_bars(self):
        # FIX2: guaranteed positive heights via offset
        self._bbar[0].set_height(max(0.5, self.snr_bob + self._off))
        self._ebar[0].set_height(max(0.5, self.snr_eve + self._off))
        self._bbar[0].set_color(C_CYAN if self.snr_bob >= 0 else C_AMB)
        self._ebar[0].set_color(C_RED  if self.snr_eve <= 5 else C_AMB)
        self._lbob.set_text(f"{self.snr_bob:+.1f} dB")
        self._leve.set_text(f"{self.snr_eve:+.1f} dB")

    def _rf_metrics(self):
        n_sec = sum(s > 0 for s in self.sec_hist)
        pct   = 100 * n_sec / max(1, len(self.sec_hist))
        src   = "[GRC Live]" if self.grc_live else ("[Demo]" if self.demo else "[Standalone]")
        self._mtxt.set_text(
            f" {src}\n"
            f" {'─'*27}\n"
            f" Bob Angle    {self.bob_angle:+7.1f} °\n"
            f" Eve Angle    {EVE_ANGLE:+7.1f} °\n"
            f" Signal Pwr   {self.snr_signal_db:+7.1f} dB\n"
            f" AN Power     {self.snr_noise_db:+7.1f} dB\n"
            f" {'─'*27}\n"
            f" SNR Bob      {self.snr_bob:+7.2f} dB\n"
            f" SNR Eve      {self.snr_eve:+7.2f} dB\n"
            f" SNR Gap      {self.snr_bob-self.snr_eve:+7.2f} dB\n"
            f" {'─'*27}\n"
            f" Secrecy      {self.sec_rate:+7.4f} b/s/Hz\n"
            f" Secure %%     {pct:7.1f} %%\n"
            f" Samples      {len(self.angle_hist):7d}\n"
        )
        if self.sec_rate > 0:
            self._badge.set_text("◉  SECURE")
            self._badge.set_color(C_GRN)
            self._badge.get_bbox_patch().set(ec=C_GRN, fc="#002A10")
        else:
            self._badge.set_text("✕  LEAKED")
            self._badge.set_color(C_RED)
            self._badge.get_bbox_patch().set(ec=C_RED, fc="#1A0000")

    def _rf_angle_hist(self, true_angle):
        t = np.arange(len(self.angle_hist))
        self._al_est.set_data(t, self.angle_hist)
        if true_angle is not None:
            self._al_true.set_data(t, [true_angle]*len(t))
        else:
            self._al_true.set_data([], [])
        self.ax_ang.set_xlim(0, max(HIST_LEN, len(t)))

    def _rf_snr_hist(self):
        t = np.arange(len(self.snrb_hist))
        self._bl.set_data(t, self.snrb_hist)
        self._el.set_data(t, self.snre_hist)
        try:
            self._sfill.remove()
        except Exception:
            pass
        self._sfill = self.ax_sh.fill_between(
            t, self.snre_hist, self.snrb_hist,
            where=[b > e for b, e in zip(self.snrb_hist, self.snre_hist)],
            color=C_GRN, alpha=0.12)
        self.ax_sh.set_xlim(0, max(HIST_LEN, len(t)))


# =============================================================================
#  Demo mode
# =============================================================================
def run_demo(save_path=None):
    print("=" * 65)
    print("  SADM-SEC  |  Real-Time Monitor  [DEMO MODE]")
    print("  Moving target  -60 deg -> +60 deg -> -60 deg  (200 steps)")
    print("  >  Drag the sliders at the bottom to change live")
    print("  Ctrl+C to stop")
    print("=" * 65)

    plotter = RealtimePlotter(bob_init=-60.0, sig_init=20.0, an_init=10.0, demo=True)
    tracker = SADMTracker(bob_initial_angle=-60.0, alpha=0.5, use_ml=True)

    n_steps = 200
    true_angs = np.concatenate([
        np.linspace(-60, 60, n_steps // 2),
        np.linspace( 60, -60, n_steps // 2),
    ])

    try:
        for i, true in enumerate(true_angs):
            pilot = generate_pilot_ping(true, n_snapshots=512, snr_pilot_db=15.0)
            est   = tracker.update(pilot)

            # Honour slider if user has moved it significantly
            sl_val = plotter.s_ang.val
            if abs(sl_val - est) > 5.0:
                est = sl_val
                plotter.bob_angle = sl_val
            else:
                plotter.bob_angle = est
                plotter.s_ang.set_val(est)

            plotter.update(X=None, true_angle=true)

            print(f"\r  [{i+1:03d}/{n_steps}]  "
                  f"True {true:+6.1f} deg  Est {est:+6.1f} deg  "
                  f"Err {abs(true-est):4.1f} deg  "
                  f"C_s {plotter.sec_rate:.3f}  "
                  f"Bob {plotter.snr_bob:+.1f}  Eve {plotter.snr_eve:+.1f}",
                  end="", flush=True)

            plt.pause(0.06)

        print("\n\n  [Loop done - window stays open. Sliders still live.]")
        plt.ioff()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plotter.fig.savefig(save_path, dpi=150, facecolor=C_BG)
            print(f"  [Saved → {save_path}]")
        plt.show()

    except KeyboardInterrupt:
        print("\n  [Stopped by user]")
    finally:
        plt.close("all")


# =============================================================================
#  Live GNU Radio mode
# =============================================================================
def run_live():
    if not ZMQ_AVAILABLE:
        print("[ERROR] pyzmq not installed.  Run:  pip install pyzmq")
        sys.exit(1)

    print("=" * 65)
    print("  SADM-SEC  |  Real-Time Monitor  [GNU RADIO LIVE]")
    print("  Antenna data  : ports 5555–5562")
    print("  Param updates : port 5563")
    print("  GRC sliders and matplotlib sliders both work live")
    print("  Ctrl+C to stop")
    print("=" * 65)

    sub     = ZMQSubscriber()
    psub    = ParamSubscriber()
    plotter = RealtimePlotter(demo=False)

    try:
        n = 0
        while True:
            grc_p = psub.receive()
            X     = sub.receive()
            if X is not None:
                n += 1
            plotter.update(X=X, grc_params=grc_p)
            plt.pause(0.01)
            print(f"\r  [{n:05d}]  "
                  f"Angle {plotter.bob_angle:+7.2f}°  "
                  f"Bob {plotter.snr_bob:+7.2f} dB  "
                  f"Eve {plotter.snr_eve:+7.2f} dB  "
                  f"C_s {plotter.sec_rate:.4f}  "
                  f"{'SECURE' if plotter.sec_rate > 0 else 'LEAKED'}",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n  [Stopped]")
    finally:
        sub.close()
        psub.close()
        plt.close("all")
        print("  [Shutdown complete]")


# =============================================================================
#  Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SADM-SEC Real-Time Monitor v3")
    ap.add_argument("--demo", action="store_true",
                    help="Demo mode — no hardware required")
    ap.add_argument("--save", default=None,
                    help="(demo) Save final frame to this image path")
    args = ap.parse_args()

    if args.demo or not ZMQ_AVAILABLE:
        if not ZMQ_AVAILABLE and not args.demo:
            print("[INFO] pyzmq not found → falling back to demo mode")
        run_demo(save_path=args.save or "outputs/realtime_demo.png")
    else:
        run_live()
