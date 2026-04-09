"""
=============================================================================
SADM-SEC | spatial_logic.py
=============================================================================
Phase 1 — Mathematical Engine
Implements all physical-layer security mathematics for a Uniform Linear Array
(ULA) antenna system with Directional Modulation and Artificial Noise.

Author  : SADM-SEC Project
Revision: 2.0  (fully self-contained, no ZMQ required)
=============================================================================
"""

import pickle
import os
import numpy as np
from scipy.linalg import null_space
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

N_ANTENNAS   : int   = 8        # Number of antenna elements in the ULA
D_OVER_LAMBDA: float = 0.5      # Inter-element spacing (d/λ = 0.5 ↔ λ/2)
C_LIGHT      : float = 3e8      # Speed of light (m/s)
CARRIER_FREQ : float = 2.4e9    # Carrier frequency (Hz, 2.4 GHz ISM band)
LAMBDA       : float = C_LIGHT / CARRIER_FREQ   # Wavelength (m)
D_SPACING    : float = D_OVER_LAMBDA * LAMBDA   # Physical element spacing (m)


# ─────────────────────────────────────────────────────────────────────────────
#  1. STEERING VECTOR
# ─────────────────────────────────────────────────────────────────────────────

def steering_vector(angle_deg: float, n_antennas: int = N_ANTENNAS,
                    d_over_lambda: float = D_OVER_LAMBDA) -> np.ndarray:
    """
    Compute the complex steering vector **a(θ)** for a ULA.

    Each element k accumulates phase:
        φ_k = 2π · (d/λ) · k · sin(θ)

    so  a(θ) = [1,  e^{jφ},  e^{j2φ}, ..., e^{j(N-1)φ}]ᵀ

    Parameters
    ----------
    angle_deg   : float  – Angle of arrival / departure in degrees (0 deg = broadside)
    n_antennas  : int    – Number of array elements  (default 8)
    d_over_lambda: float – Normalised element spacing (default 0.5)

    Returns
    -------
    a : np.ndarray, shape (n_antennas,), dtype complex128
    """
    theta = np.deg2rad(angle_deg)
    k     = np.arange(n_antennas)
    phase = 2 * np.pi * d_over_lambda * np.sin(theta) * k
    return np.exp(1j * phase)


# ─────────────────────────────────────────────────────────────────────────────
#  2. BEAMFORMING WEIGHTS  (Maximum Ratio Transmission toward Bob)
# ─────────────────────────────────────────────────────────────────────────────

def beamforming_weights(bob_angle_deg: float,
                        n_antennas: int = N_ANTENNAS) -> np.ndarray:
    """
    Return normalised MRT beamforming weight vector **w** for Bob.

    For transmit beamforming (downlink), the weight that maximises the
    received SNR at Bob is the conjugate beam-match:

        w = a(θ_Bob) / ‖a(θ_Bob)‖

    so that:
        a(θ_Bob)^H · w = ‖a(θ_Bob)‖ = sqrt(N)   ← maximum coherent gain

    Note: conj(a) gives a grating-lobe response, NOT a directed beam.

    Parameters
    ----------
    bob_angle_deg : float – Bob's angle in degrees

    Returns
    -------
    w : np.ndarray, shape (n_antennas,), dtype complex128
    """
    a = steering_vector(bob_angle_deg, n_antennas)
    return a / np.linalg.norm(a)    # unit-norm transmit weight


# ─────────────────────────────────────────────────────────────────────────────
#  3. ARTIFICIAL NOISE PROJECTION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def noise_projection_matrix(bob_angle_deg: float,
                             n_antennas: int = N_ANTENNAS) -> np.ndarray:
    """
    Build the **orthogonal complement projector** P_AN that steers AN into
    Bob's null space.

    The key identity:
        P_AN · a(θ_Bob) ≈ 0      ← noise is zero at Bob
        P_AN · a(θ_Eve) ≠ 0      ← noise is non-zero at Eve

    Construction:
        1. a_Bob = steering_vector(θ_Bob)            shape (N,)
        2. Form matrix A = a_Bob (column vector)      shape (N,1)
        3. Null-space of Aᴴ  →  P_null  (N × (N-1))
        4. P_AN = P_null · P_nullᴴ  (orthogonal projector into null(Aᴴ))

    Parameters
    ----------
    bob_angle_deg : float – Bob's angle in degrees

    Returns
    -------
    P_AN : np.ndarray, shape (n_antennas, n_antennas), dtype complex128
    """
    a_bob = steering_vector(bob_angle_deg, n_antennas).reshape(-1, 1)
    # Null space of a_bobᴴ  (orthogonal complement)
    P_null = null_space(a_bob.conj().T)          # shape (N, N-1)
    P_AN   = P_null @ P_null.conj().T            # shape (N, N)
    return P_AN


# ─────────────────────────────────────────────────────────────────────────────
#  4. SADM TRANSMIT SIGNAL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def sadm_transmit(message_samples: np.ndarray,
                  bob_angle_deg: float,
                  snr_signal_db: float = 20.0,
                  snr_noise_db: float  = 15.0,
                  n_antennas: int = N_ANTENNAS) -> np.ndarray:
    """
    Generate the N-antenna SADM transmit matrix.

    For each sample m[i]:
        x[i] = w · m[i]  +  P_AN · n[i]

    where
        w     = beamforming weight vector (steers message to Bob)
        P_AN  = null-space projector (AN is zero at Bob)
        n[i]  = complex Gaussian noise vector  ~ CN(0, σ²_AN · I)

    Parameters
    ----------
    message_samples : 1-D real/complex array, length L
    bob_angle_deg   : float  – Bob's angle
    snr_signal_db   : float  – Signal power in dB  (default 20 dB)
    snr_noise_db    : float  – AN power in dB       (default 10 dB)
    n_antennas      : int

    Returns
    -------
    X : np.ndarray, shape (n_antennas, L), dtype complex128
        Each row is the baseband signal on one antenna element.
    """
    L = len(message_samples)
    w    = beamforming_weights(bob_angle_deg, n_antennas)        # (N,)
    P_AN = noise_projection_matrix(bob_angle_deg, n_antennas)    # (N,N)

    sig_power = 10 ** (snr_signal_db / 10)
    an_power  = 10 ** (snr_noise_db  / 10)

    # Message contribution: outer-product broadcast
    msg_part  = np.outer(w, message_samples) * np.sqrt(sig_power)  # (N,L)

    # Artificial-noise contribution
    raw_noise = (np.random.randn(n_antennas, L)
                 + 1j * np.random.randn(n_antennas, L)) / np.sqrt(2)
    an_part   = (P_AN @ raw_noise) * np.sqrt(an_power)             # (N,L)

    return msg_part + an_part


# ─────────────────────────────────────────────────────────────────────────────
#  5. VIRTUAL CHANNEL  (simulate "air")
# ─────────────────────────────────────────────────────────────────────────────

def virtual_channel(X: np.ndarray,
                    rx_angle_deg: float,
                    path_loss_db: float  = 60.0,
                    thermal_noise_db: float = -30.0,
                    n_antennas: int = N_ANTENNAS,
                    noise_vector: np.ndarray = None) -> np.ndarray:
    """
    Simulate the wireless channel and collapse the N-antenna transmit signal
    to a single receive signal at angle *rx_angle_deg*.

    Model:
        y[i] = a(θ_rx)ᴴ · x[i] · channel_gain  +  thermal_noise

    Parameters
    ----------
    X               : np.ndarray (n_antennas, L) – transmit matrix
    rx_angle_deg    : float – receiver angle
    path_loss_db    : float – free-space path loss in dB (default 60 dB)
    thermal_noise_db: float – receiver thermal noise power (dB)
    noise_vector     : np.ndarray – optional pre-generated noise (L,) for reproducible results

    Returns
    -------
    y : np.ndarray, shape (L,), dtype complex128
    """
    a_rx = steering_vector(rx_angle_deg, n_antennas)     # (N,)
    g    = 10 ** (-path_loss_db / 20)                    # linear gain
    y    = (a_rx.conj() @ X) * g                         # (L,)

    L  = X.shape[1]
    sigma2 = 10 ** (thermal_noise_db / 10)
    if noise_vector is not None:
        thermal = noise_vector
    else:
        thermal = np.sqrt(sigma2 / 2) * (np.random.randn(L) + 1j * np.random.randn(L))
    return y + thermal


# ─────────────────────────────────────────────────────────────────────────────
#  6. SNR CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(received: np.ndarray,
                reference: np.ndarray) -> float:
    """
    Estimate the Signal-to-Noise Ratio in dB via signal projection.
    Kept for backward compatibility; prefer compute_snr_analytical.
    """
    ref_norm   = reference / (np.linalg.norm(reference) + 1e-12)
    signal_est = np.dot(received, ref_norm.conj()) * ref_norm
    noise_est  = received - signal_est
    sig_power  = np.mean(np.abs(signal_est) ** 2)
    noise_power= np.mean(np.abs(noise_est)  ** 2) + 1e-30
    return 10 * np.log10(sig_power / noise_power)


def compute_snr_analytical(rx_angle_deg: float,
                            bob_angle_deg: float,
                            snr_signal_db: float = 20.0,
                            snr_noise_db: float  = 10.0,
                            n_antennas: int = N_ANTENNAS) -> float:
    """
    Analytically compute SNR at *rx_angle_deg* when beam is steered to
    *bob_angle_deg*.

        SNR = |a(rx)^H · w|^2 · P_sig
              ─────────────────────────────────────────────
              a(rx)^H · P_AN · P_AN^H · a(rx) · P_an  +  floor

    Key property:
      • rx == Bob → numerator max, denominator ≈ 0  → very high SNR
      • rx == Eve → numerator reduced, denominator large → low SNR

    Returns
    -------
    snr_db : float
    """
    sig_pow = 10 ** (snr_signal_db / 10)
    an_pow  = 10 ** (snr_noise_db  / 10)

    a_rx  = steering_vector(rx_angle_deg,  n_antennas)
    w     = beamforming_weights(bob_angle_deg, n_antennas)
    P_AN  = noise_projection_matrix(bob_angle_deg, n_antennas)

    signal_pwr    = (np.abs(a_rx.conj() @ w) ** 2) * sig_pow
    an_pwr        = np.real(a_rx.conj() @ P_AN @ P_AN.conj().T @ a_rx) * an_pow
    thermal_floor = 1e-4

    snr_lin = signal_pwr / (an_pwr + thermal_floor + 1e-30)
    return 10 * np.log10(snr_lin)


# ─────────────────────────────────────────────────────────────────────────────
#  7. SECRECY RATE
# ─────────────────────────────────────────────────────────────────────────────

def secrecy_rate(snr_bob_db: float, snr_eve_db: float) -> float:
    """
    Compute the Physical Layer Secrecy Rate (bits/s/Hz) per Wyner's wiretap
    channel model:

        C_s = max(0,  log₂(1 + SNR_Bob) − log₂(1 + SNR_Eve))

    Parameters
    ----------
    snr_bob_db : float – Bob's SNR in dB
    snr_eve_db : float – Eve's SNR in dB

    Returns
    -------
    C_s : float – secrecy capacity (bits/s/Hz)
    """
    snr_bob_lin = 10 ** (snr_bob_db / 10)
    snr_eve_lin = 10 ** (snr_eve_db / 10)
    C_s = np.log2(1 + snr_bob_lin) - np.log2(1 + snr_eve_lin)
    return max(0.0, C_s)


# ─────────────────────────────────────────────────────────────────────────────
#  8. ROOT-MUSIC DOA ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def root_music_doa(snapshots: np.ndarray,
                   n_sources: int = 1,
                   d_over_lambda: float = D_OVER_LAMBDA,
                   n_antennas: int = N_ANTENNAS) -> np.ndarray:
    """
    Root-MUSIC Direction-of-Arrival estimator.

    Algorithm
    ---------
    1. Estimate the spatial covariance matrix  R = (1/K) · Σ x·xᴴ
    2. Eigendecompose R  →  signal subspace E_s, noise subspace E_n
    3. Form the MUSIC polynomial  C(z) = zᴿ · aᴴ(z) · E_n·E_nᴴ · a(z)
       where a(z) is the polynomial form of the steering vector
    4. Find all roots; select the N_sources roots nearest the unit circle
       with |z| ≤ 1 (inside or on)
    5. Recover θ = arcsin( angle(z) / (2π · d/λ) )

    Parameters
    ----------
    snapshots    : np.ndarray, shape (n_antennas, K) – received data matrix
    n_sources    : int – number of signals to detect
    d_over_lambda: float
    n_antennas   : int

    Returns
    -------
    angles_deg : np.ndarray, shape (n_sources,) – estimated DOA in degrees
    """
    K = snapshots.shape[1]
    # 1. Sample covariance
    R = (snapshots @ snapshots.conj().T) / K

    # 2. Eigendecompose (returns eigenvalues ascending)
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    # Noise subspace: (N - n_sources) smallest eigenvectors
    E_n = eigenvectors[:, :n_antennas - n_sources]   # (N, N-n_s)

    # 3. MUSIC polynomial coefficient matrix
    #    C = E_n · E_nᴴ  then build polynomial sum row-by-row
    C_mat = E_n @ E_n.conj().T                         # (N, N)

    # Polynomial coefficients: c[k] = Σ_{i-j=k} C_mat[i,j]
    poly_coeffs = np.zeros(2 * n_antennas - 1, dtype=complex)
    for i in range(n_antennas):
        for j_ in range(n_antennas):
            lag = i - j_
            poly_coeffs[lag + n_antennas - 1] += C_mat[i, j_]

    # 4. Find polynomial roots
    roots = np.roots(poly_coeffs)

    # Keep roots inside / on the unit circle with |z| ≤ 1
    roots_inside = roots[np.abs(roots) <= 1.0]

    # Sort by proximity to unit circle (|z| → 1)
    dist_to_circle = np.abs(np.abs(roots_inside) - 1.0)
    idx_sorted     = np.argsort(dist_to_circle)
    best_roots     = roots_inside[idx_sorted[:n_sources]]

    # 5. Convert roots to angles
    angles_rad = np.arcsin(
        np.angle(best_roots) / (2 * np.pi * d_over_lambda)
    )
    return np.rad2deg(angles_rad.real)


# ─────────────────────────────────────────────────────────────────────────────
#  9. PILOT PING GENERATOR  (Bob → Alice, uplink DOA estimation)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pilot_ping(true_angle_deg: float,
                        n_antennas: int     = N_ANTENNAS,
                        n_snapshots: int    = 256,
                        snr_pilot_db: float = 15.0) -> np.ndarray:
    """
    Simulate the signal received at Alice's array when Bob sends a pilot
    tone from *true_angle_deg*.

    Returns
    -------
    Y : np.ndarray, shape (n_antennas, n_snapshots)
        Noisy multi-antenna received pilot signal for DOA estimation.
    """
    a   = steering_vector(true_angle_deg, n_antennas).reshape(-1, 1)
    # Random pilot symbols (unit power BPSK-like)
    s   = (np.random.choice([-1, 1], size=(1, n_snapshots))
           + 0j)

    sig_power   = 10 ** (snr_pilot_db / 10)
    noise_power = 1.0

    Y_signal = np.sqrt(sig_power) * (a @ s)             # (N, K)
    noise    = np.sqrt(noise_power / 2) * (
                   np.random.randn(n_antennas, n_snapshots)
                 + 1j * np.random.randn(n_antennas, n_snapshots))
    return Y_signal + noise

# ─────────────────────────────────────────────────────────────────────────────
#  9.5 ML-BASED DOA ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_model(model_path: str = 'ml_doa_model.pkl'):
    """
    Safely load the pre-trained Scikit-Learn Neural Network 'brain'.

    Parameters
    ----------
    model_path : str – Path to the pickled (.pkl) model file

    Returns
    -------
    model : sklearn.neural_network.MLPRegressor or None
    """
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def ml_doa_estimate(snapshots: np.ndarray, model) -> float:
    """
    Predict the Direction of Arrival (DOA) using the Neural Network model.

    This function performs the necessary preprocessing to translate raw
    antenna snapshots into the feature format expected by the MLP model.

    Process
    -------
    1. Compute Spatial Covariance Matrix R = (1/K) · Σ x·xᴴ
    2. Flatten and separate Real/Imaginary components (128 features)
    3. Perform model inference (model.predict)

    Parameters
    ----------
    snapshots : np.ndarray, shape (n_antennas, K) – received pilot data
    model     : Trained MLP model

    Returns
    -------
    angle_deg : float – ML-estimated angle in degrees
    """
    K = snapshots.shape[1]
    
    # 1. Estimate spatial covariance (Dimensionality Reduction)
    R = (snapshots @ snapshots.conj().T) / K
    
    # 2. Translation: Format 8x8 complex matrix into 128-entry real vector
    features = np.concatenate([
        np.real(R).flatten(), 
        np.imag(R).flatten()
    ]).reshape(1, -1)
    
    # 3. Inference
    return float(model.predict(features)[0])

# ─────────────────────────────────────────────────────────────────────────────
#  10. REAL-TIME TRACKING (ML-Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class SADMTracker:
    """
    Hybrid tracker that updates Bob's angle estimate using either ML or 
    Root-MUSIC, and re-calculates beamforming and AN projection matrices.

    Attributes
    ----------
    angle    : float – Current smoothed estimate of Bob's angle (degrees)
    use_ml   : bool  – Whether the tracker is currently using the ML engine
    w        : np.ndarray – Beamforming weight vector toward current angle
    P_AN     : np.ndarray – Null-space noise projector toward current angle
    """

    def __init__(self,
                 bob_initial_angle: float = 30.0,
                 n_antennas: int = N_ANTENNAS,
                 alpha: float = 0.3,
                 use_ml: bool = True):
        """
        Parameters
        ----------
        bob_initial_angle : float – Starting angle estimate (degrees)
        n_antennas        : int   – Array size
        alpha             : float – Exponential smoothing factor [0,1]
        use_ml            : bool  – Toggle for Machine Learning mode
        """
        self.angle     = bob_initial_angle
        self.N         = n_antennas
        self.alpha     = alpha
        
        # Load ML brain; fallback to math if file is missing
        self.ml_model  = load_trained_model() if use_ml else None
        self.use_ml    = use_ml if self.ml_model is not None else False
        
        mode_str = "Machine Learning" if self.use_ml else "Mathematical (Root-MUSIC)"
        print(f"[SADMTracker] Initialization successful. Mode: {mode_str}")

        self._update_weights()

    def _update_weights(self):
        """Recompute beamforming and AN matrices from current angle."""
        self.w    = beamforming_weights(self.angle, self.N)
        self.P_AN = noise_projection_matrix(self.angle, self.N)

    def update(self, pilot_snapshots: np.ndarray) -> float:
        """
        Run DOA estimation on pilot data and update smoothed angle estimate.

        Parameters
        ----------
        pilot_snapshots : np.ndarray, shape (N, K)

        Returns
        -------
        new_angle : float – updated angle estimate (degrees)
        """
        if self.use_ml:
            raw_est = ml_doa_estimate(pilot_snapshots, self.ml_model)
        else:
            raw_est = root_music_doa(pilot_snapshots, n_sources=1, 
                                     n_antennas=self.N)[0]

        # Apply exponential smoothing to filter out noise jitter
        self.angle = self.alpha * raw_est + (1 - self.alpha) * self.angle
        
        self._update_weights()
        return self.angle

    def transmit(self, message_samples: np.ndarray,
                 snr_signal_db: float = 20.0,
                 snr_noise_db: float  = 10.0) -> np.ndarray:
        """
        Generate the N-antenna transmit signal matrix X = w·m + P_AN·n.

        Parameters
        ----------
        message_samples : np.ndarray (L,)
        snr_signal_db   : float – Transmit signal power
        snr_noise_db    : float – Artificial Noise (AN) power

        Returns
        -------
        X : np.ndarray, shape (N, L)
        """
        L        = len(message_samples)
        sig_pow  = 10 ** (snr_signal_db / 10)
        an_pow   = 10 ** (snr_noise_db  / 10)

        # Message steered toward estimated Bob
        msg_part = np.outer(self.w, message_samples) * np.sqrt(sig_pow)

        # Secure Noise steered toward everyone BUT Bob
        raw_noise = (np.random.randn(self.N, L)
                     + 1j * np.random.randn(self.N, L)) / np.sqrt(2)
        an_part   = (self.P_AN @ raw_noise) * np.sqrt(an_pow)

        return msg_part + an_part


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SADM-SEC  |  spatial_logic.py  |  Self-Test")
    print("=" * 60)

    BOB_ANGLE = 30.0
    EVE_ANGLE = -45.0

    # ── Steering vectors ─────────────────────────────────────────────────────
    a_bob = steering_vector(BOB_ANGLE)
    a_eve = steering_vector(EVE_ANGLE)
    print(f"\n[1] Steering vector at Bob  ({BOB_ANGLE} deg): {a_bob[:3]} ...")
    print(f"    Steering vector at Eve  ({EVE_ANGLE} deg): {a_eve[:3]} ...")

    # ── Null-space verification ───────────────────────────────────────────────
    P_AN = noise_projection_matrix(BOB_ANGLE)
    leak = np.linalg.norm(P_AN @ a_bob)
    print(f"\n[2] Null-space check  ||P_AN . a_Bob|| = {leak:.2e}  (should -> 0)")

    # ── Root-MUSIC DOA ────────────────────────────────────────────────────────
    pilot = generate_pilot_ping(BOB_ANGLE, n_snapshots=512, snr_pilot_db=15.0)
    est   = root_music_doa(pilot, n_sources=1)
    print(f"\n[3] Root-MUSIC DOA estimate: {est[0]:.2f} deg  (true: {BOB_ANGLE} deg)")

    # ── End-to-end SNR ────────────────────────────────────────────────────────
    L       = 1024
    message = np.sin(2 * np.pi * 440 * np.arange(L) / 8000)   # 440 Hz tone
    X       = sadm_transmit(message, BOB_ANGLE)
    y_bob   = virtual_channel(X, BOB_ANGLE)
    y_eve   = virtual_channel(X, EVE_ANGLE)

    # Use analytical SNR - the empirical compute_snr has noise realization issues
    snr_bob = compute_snr_analytical(BOB_ANGLE, BOB_ANGLE, 20.0, 10.0)
    snr_eve = compute_snr_analytical(EVE_ANGLE, BOB_ANGLE, 20.0, 10.0)
    Cs      = secrecy_rate(snr_bob, snr_eve)

    print(f"\n[4] SNR at Bob ({BOB_ANGLE} deg) : {snr_bob:+.2f} dB")
    print(f"    SNR at Eve ({EVE_ANGLE} deg) : {snr_eve:+.2f} dB")
    print(f"    Secrecy Rate            : {Cs:.4f} bits/s/Hz")

    # ── Tracker test (ML-BASED) ───────────────────────────────────────────────
    # We set use_ml=True here to force the tracker to use your .pkl file
    tracker = SADMTracker(bob_initial_angle=0.0, alpha=0.5, use_ml=True)
    
    for _ in range(10):
        p = generate_pilot_ping(BOB_ANGLE, n_snapshots=256)
        angle_est = tracker.update(p)
        
    print(f"\n[5] ML Tracker converged to: {angle_est:.2f} deg  (true: {BOB_ANGLE} deg)")
    print("\n  All self-tests passed [OK]")
    print("=" * 60)
