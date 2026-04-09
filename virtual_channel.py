"""
=============================================================================
SADM-SEC  |  virtual_channel.py
=============================================================================
Phase 3 - Virtual Channel & Performance Metrics
Simulates the wireless medium, computes Bob/Eve SNR, Secrecy Rate,
and optionally interfaces with GNU Radio via ZMQ (when available).

Run standalone:
    python3 virtual_channel.py

=============================================================================
"""

import sys
import time
import numpy as np
from typing import Optional

# === import our math engine ===
from spatial_logic import (
    N_ANTENNAS, steering_vector, beamforming_weights,
    noise_projection_matrix, sadm_transmit, virtual_channel,
    compute_snr, compute_snr_analytical, secrecy_rate,
    generate_pilot_ping, root_music_doa, SADMTracker
)

# === optional ZMQ (graceful fallback if not installed) ===
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


# -----------------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------------

ZMQ_BASE_PORT   = 5555          # Ports 5555 ... 5562  (8 antennas)
ZMQ_HOST        = "127.0.0.1"
BLOCK_SIZE      = 1024          # Samples per processing block
BOB_ANGLE       = 30.0          # Bob's initial angle (degrees)
EVE_ANGLE       = -45.0         # Eve's intercept angle (degrees)
SNR_SIGNAL_DB   = 20.0          # Transmit signal power
SNR_NOISE_DB    = 10.0          # Artificial Noise power
PATH_LOSS_DB    = 60.0          # Free-space path loss


# -----------------------------------------------------------------------------
#  ZMQ SUBSCRIBER  (real GNU Radio integration)
# -----------------------------------------------------------------------------

class ZMQAntennaSubscriber:
    """
    Subscribes to N ZMQ PUB sockets (one per antenna, from GRC flowgraph)
    and assembles data into the (N x L) matrix used by spatial_logic.
    """

    def __init__(self, n_antennas: int = N_ANTENNAS,
                 base_port: int = ZMQ_BASE_PORT,
                 host: str = ZMQ_HOST):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("pyzmq is not installed. "
                               "Run: pip install pyzmq")
        self.n    = n_antennas
        self.ctx  = zmq.Context()
        self.subs = []
        for i in range(n_antennas):
            sub = self.ctx.socket(zmq.SUB)
            sub.connect(f"tcp://{host}:{base_port + i}")
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            self.subs.append(sub)
        self.poller = zmq.Poller()
        for s in self.subs:
            self.poller.register(s, zmq.POLLIN)
        print(f"[ZMQ] Subscribed to ports "
              f"{base_port}-{base_port + n_antennas - 1}")

    def receive_block(self, block_size: int = BLOCK_SIZE,
                      timeout_ms: int = 2000) -> Optional[np.ndarray]:
        """
        Collect one block from all antennas.

        Returns np.ndarray (N, block_size) or None on timeout.
        """
        X = np.zeros((self.n, block_size), dtype=complex)
        for idx, sub in enumerate(self.subs):
            evts = dict(self.poller.poll(timeout_ms))
            if sub in evts:
                raw = sub.recv()
                arr = np.frombuffer(raw, dtype=np.complex64)
                length = min(len(arr), block_size)
                X[idx, :length] = arr[:length]
            else:
                print(f"[ZMQ] Timeout on antenna {idx}")
                return None
        return X

    def close(self):
        for s in self.subs:
            s.close()
        self.ctx.term()


# -----------------------------------------------------------------------------
#  PERFORMANCE METRICS PRINTER
# -----------------------------------------------------------------------------

def print_metrics(block_num: int,
                  bob_angle: float, eve_angle: float,
                  snr_bob: float, snr_eve: float) -> None:

    Cs = secrecy_rate(snr_bob, snr_eve)
    bar_bob = "#" * max(0, int((snr_bob + 10) / 3))
    bar_eve = "#" * max(0, int((snr_eve + 10) / 3))

    print(f"\n{'-'*58}")
    print(f"  Block #{block_num:04d}  |  Bob @ {bob_angle:+6.1f} deg  "
          f"  Eve @ {eve_angle:+6.1f} deg")
    print(f"{'-'*58}")
    print(f"  SNR Bob : {snr_bob:+7.2f} dB  {bar_bob}")
    print(f"  SNR Eve : {snr_eve:+7.2f} dB  {bar_eve}")
    print(f"  Secrecy : {Cs:.4f} bits/s/Hz "
          f"{'[OK] SECURE' if Cs > 0 else '[FAIL] LEAKED'}")
    print(f"{'-'*58}")


# -----------------------------------------------------------------------------
#  STANDALONE SIMULATION MODE  (no GNU Radio / ZMQ needed)
# -----------------------------------------------------------------------------

def run_simulation(n_blocks: int      = 20,
                   bob_angle: float   = BOB_ANGLE,
                   eve_angle: float   = EVE_ANGLE,
                   block_size: int    = BLOCK_SIZE,
                   moving: bool       = False,
                   use_ml: bool       = True) -> dict: # <-- Defaulted to True

    print("\n" + "="*58)
    print("  SADM-SEC  Virtual Channel Simulation")
    mode_desc = "MOVING TARGET" if moving else "STATIC TARGET"
    engine_desc = "NEURAL NETWORK" if use_ml else "ROOT-MUSIC"
    print(f"  Mode:   {mode_desc}")
    print(f"  Engine: {engine_desc}")
    print("="*58)

    # For moving target, initialize tracker at the starting angle (-60 deg)
    # to avoid large initial tracking error
    initial_angle = -60.0 if moving else bob_angle
    # Use higher alpha for moving targets (0.7) to track faster
    # Use lower alpha for static (0.4) for better noise filtering
    alpha = 0.7 if moving else 0.4
    tracker = SADMTracker(bob_initial_angle=initial_angle, alpha=alpha, use_ml=use_ml)
    
    # ... [rest of the function remains exactly as it was] ...

    snr_bobs, snr_eves, Cs_vals, est_angles = [], [], [], []
    t = np.linspace(0, 1, block_size)
    message_base = np.sin(2 * np.pi * 440 * t * block_size / 8000)

    for blk in range(n_blocks):

        if moving:
            true_angle = -60 + (120 * blk / max(n_blocks - 1, 1))
        else:
            true_angle = bob_angle

        # Bob sends a pilot ping -> tracker estimates angle (Using ML if enabled!)
        pilot = generate_pilot_ping(true_angle, n_snapshots=512,
                                    snr_pilot_db=15.0)
        est_angle = tracker.update(pilot)

        # Transmit with tracked angle
        X = tracker.transmit(message_base,
                             snr_signal_db=SNR_SIGNAL_DB,
                             snr_noise_db=SNR_NOISE_DB)

        snr_bob = compute_snr_analytical(true_angle,  est_angle,
                                         SNR_SIGNAL_DB, SNR_NOISE_DB)
        snr_eve = compute_snr_analytical(eve_angle,   est_angle,
                                         SNR_SIGNAL_DB, SNR_NOISE_DB)
        Cs      = secrecy_rate(snr_bob, snr_eve)

        snr_bobs.append(snr_bob)
        snr_eves.append(snr_eve)
        Cs_vals.append(Cs)
        est_angles.append(est_angle)

        print_metrics(blk + 1, true_angle, eve_angle, snr_bob, snr_eve)
        if moving:
            print(f"  Tracker:   true={true_angle:+6.1f} deg  "
                  f"est={est_angle:+6.1f} deg  "
                  f"err={abs(true_angle-est_angle):.2f} deg")

    # ... [Rest of summary print logic remains the same] ...
    return dict(snr_bob=snr_bobs, snr_eve=snr_eves,
                secrecy=Cs_vals, est_angles=est_angles)

# -----------------------------------------------------------------------------
#  ZMQ LIVE MODE  (requires GNU Radio flowgraph running)
# -----------------------------------------------------------------------------

def run_zmq_live(bob_angle: float = BOB_ANGLE,
                 eve_angle: float = EVE_ANGLE) -> None:
    """
    Live mode: receive antenna streams from GNU Radio over ZMQ,
    apply spatial combining, and print real-time metrics.

    Prerequisites
    -------------
    1. GNU Radio flowgraph (sadm_flowgraph.grc) must be running.
    2. ZMQ PUB sinks on ports 5555-5562 must be active.
    3. pyzmq must be installed.
    """
    if not ZMQ_AVAILABLE:
        print("[ERROR] pyzmq not found. Install with:  pip install pyzmq")
        print("[INFO]  Falling back to simulation mode.\n")
        run_simulation()
        return

    sub    = ZMQAntennaSubscriber()
    tracker = SADMTracker(bob_initial_angle=bob_angle, alpha=0.3)
    blk    = 0
    t_ref  = np.linspace(0, BLOCK_SIZE / 8000, BLOCK_SIZE)
    ref_msg = np.sin(2 * np.pi * 440 * t_ref * BLOCK_SIZE / 8000)

    print("[ZMQ] Waiting for GNU Radio data ...  (Ctrl-C to stop)\n")

    try:
        while True:
            X = sub.receive_block(BLOCK_SIZE)
            if X is None:
                print("[ZMQ] No data received, retrying ...")
                time.sleep(0.5)
                continue

            blk += 1
            # Update tracker with the received pilot (use first antenna as proxy)
            pilot = X                         # use live antenna data
            est_angle = tracker.update(pilot)

            y_bob = virtual_channel(X, est_angle,  path_loss_db=PATH_LOSS_DB)
            y_eve = virtual_channel(X, eve_angle,   path_loss_db=PATH_LOSS_DB)

            ref_b = virtual_channel(
                np.outer(beamforming_weights(est_angle), ref_msg), est_angle)
            snr_bob = compute_snr(y_bob, ref_b)
            ref_e   = virtual_channel(
                np.outer(beamforming_weights(est_angle), ref_msg), eve_angle)
            snr_eve = compute_snr(y_eve, ref_e)

            print_metrics(blk, est_angle, eve_angle, snr_bob, snr_eve)

    except KeyboardInterrupt:
        print("\n[ZMQ] Interrupted by user.")
    finally:
        sub.close()


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"

    if mode == "zmq":
        run_zmq_live()
    elif mode == "moving":
        run_simulation(n_blocks=24, moving=True)
    else:
        # Defaults to ML=True if running standalone simulation
        run_simulation(n_blocks=10, moving=False, use_ml=True)
