"""
=============================================================================
SADM-SEC  |  sadm_gnuradio_block.py
=============================================================================
GNU Radio Embedded Python Block
Drop this file into GNU Radio Companion as an "Embedded Python Block".

Block spec
──────────
  Inputs  : 2
    [0] message  – complex baseband (DSB-SC audio), 1 sample/step
    [1] noise    – complex white Gaussian noise,    1 sample/step
  Outputs : 8  (one per antenna)

  Parameters (editable in GRC GUI)
    bob_angle   : float  – current steering angle (degrees)
    snr_sig_db  : float  – message signal power
    snr_an_db   : float  – artificial noise power

Integration steps in GRC
─────────────────────────
1.  Drag an "Embedded Python Block" from the block list.
2.  Paste the entire contents of this file into the code editor.
3.  Connect:
      Audio Source  → Multiply (DSB-SC)  → [in0] of this block
      Noise Source  →                    → [in1] of this block
      [out0..7]     → ZMQ PUB Sink × 8
4.  Set sample rate = 48000, vec_length = 64 (or match your block size).

=============================================================================
"""

import numpy as np
import os
import sys
from gnuradio import gr

# Attempt to import our math engine from the same directory.
# In GRC, add the sadm_sec/ folder to the GRC Python path via
# Tools → Options → General → "Additional Python Paths".
# Also try adding the script directory to path for embedded blocks.
try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _current_dir = os.path.abspath(".")
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from spatial_logic import (
        N_ANTENNAS, beamforming_weights,
        noise_projection_matrix, SADMTracker
    )
    _SPATIAL_OK = True
except ImportError as e:
    _SPATIAL_OK = False
    print(f"[SADM Block] Warning: Could not import spatial_logic: {e}")


class sadm_beamformer(gr.sync_block):
    """
    SADM Physical-Layer Security Block

    Takes 1 message stream + 1 noise stream and outputs
    N_ANTENNAS = 8 spatially shaped streams.
    """

    def __init__(self,
                 bob_angle: float = 30.0,
                 snr_sig_db: float = 20.0,
                 snr_an_db: float  = 10.0,
                 n_antennas: int   = 8):

        gr.sync_block.__init__(
            self,
            name="SADM Beamformer",
            in_sig  = [np.complex64, np.complex64],          # msg, noise
            out_sig = [np.complex64] * n_antennas             # 8 antennas
        )

        self.N          = n_antennas
        self.snr_sig    = 10 ** (snr_sig_db / 10)
        self.snr_an     = 10 ** (snr_an_db  / 10)

        if _SPATIAL_OK:
            self.tracker = SADMTracker(bob_initial_angle=bob_angle,
                                       n_antennas=n_antennas,
                                       alpha=0.3)
        else:
            # Fallback: pre-compute static weights
            theta = np.deg2rad(bob_angle)
            k     = np.arange(n_antennas)
            a     = np.exp(1j * np.pi * np.sin(theta) * k)
            self.w_static    = a / np.linalg.norm(a)
            self.P_AN_static = np.eye(n_antennas, dtype=complex)
            print("[SADM Block] spatial_logic.py not found - "
                  "using static beamforming fallback.")

    # ── Called by GRC framework to update the steering angle ─────────────────
    def set_bob_angle(self, angle_deg: float) -> None:
        if _SPATIAL_OK:
            self.tracker.angle = angle_deg
            self.tracker._update_weights()
        else:
            theta  = np.deg2rad(angle_deg)
            k      = np.arange(self.N)
            a      = np.exp(1j * np.pi * np.sin(theta) * k)
            self.w_static = a / np.linalg.norm(a)

    # ── Called by GRC framework to update signal power ──────────────────────────
    def set_snr_sig_db(self, snr_db: float) -> None:
        """Update the signal power ratio (in dB)."""
        self.snr_sig = 10 ** (snr_db / 10)

    # ── Called by GRC framework to update AN power ──────────────────────────────
    def set_snr_an_db(self, snr_db: float) -> None:
        """Update the artificial noise power ratio (in dB)."""
        self.snr_an = 10 ** (snr_db / 10)

    # ── Main processing function ──────────────────────────────────────────────
    def work(self, input_items, output_items):
        msg   = input_items[0]          # (L,) complex64
        noise = input_items[1]          # (L,) complex64
        L     = len(msg)

        if _SPATIAL_OK:
            w    = self.tracker.w                    # (N,)
            P_AN = self.tracker.P_AN                 # (N,N)
        else:
            w    = self.w_static
            P_AN = self.P_AN_static

        # Message stream → beamformed signal
        msg_part = np.outer(w, msg) * np.sqrt(self.snr_sig)     # (N, L)

        # Noise stream → projected artificial noise
        noise_mat = np.outer(np.ones(self.N), noise)             # (N, L)
        an_part   = (P_AN @ noise_mat) * np.sqrt(self.snr_an)   # (N, L)

        transmit = (msg_part + an_part).astype(np.complex64)     # (N, L)

        # Write to output ports
        for i in range(self.N):
            output_items[i][:L] = transmit[i]

        return L
