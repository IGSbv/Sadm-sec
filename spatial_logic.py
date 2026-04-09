"""
=============================================================================
SADM-SEC | spatial_logic.py (Unified Architecture)
=============================================================================
Consolidates analytical models with empirical/ZMQ fallbacks for full pipeline execution.
"""

import pickle
import os
import numpy as np

# ── Array Constants ──────────────────────────────────────────────────────────
N_ANTENNAS       = 8          
D_HALF_LAMBDA    = 0.5        
THERMAL_NOISE_DB = -40.0   

# ── 1. Steering & Beamforming ────────────────────────────────────────────────

def steering_vector(theta_deg: float) -> np.ndarray:
    """N×1 ULA steering vector (unit-normalised)."""
    theta = np.deg2rad(theta_deg)
    n     = np.arange(N_ANTENNAS)
    return np.exp(1j * np.pi * n * np.sin(theta)) / np.sqrt(N_ANTENNAS)

def beamforming_weights(theta_deg: float) -> np.ndarray:
    """MRT weights (conjugate match for ULA)."""
    return steering_vector(theta_deg)

def noise_projection_matrix(theta_deg: float) -> np.ndarray:
    """Null-space projector: P_AN = I - a(theta)a†(theta)."""
    a = steering_vector(theta_deg).reshape(-1, 1)
    I = np.eye(N_ANTENNAS, dtype=complex)
    return I - a @ a.conj().T

def array_factor(theta_deg: float, bob_deg: float) -> complex:
    """AF(theta, theta_B) = a†(theta) · a(theta_B)."""
    return complex(np.dot(steering_vector(theta_deg).conj(), steering_vector(bob_deg)))

# ── 2. Transmit & Channel Simulation ─────────────────────────────────────────

def sadm_transmit(
        signal:        np.ndarray,
        bob_deg:       float,
        signal_pow_db: float = 20.0,
        an_pow_db:     float = 10.0,
        rng:           np.random.Generator = None
) -> np.ndarray:
    """Generates the N-antenna transmit matrix X."""
    if rng is None:
        rng = np.random.default_rng()

    Ps  = 10 ** (signal_pow_db / 10)
    Pan = 10 ** (an_pow_db     / 10)
    L   = len(signal)

    w        = beamforming_weights(bob_deg)
    sig_part = np.outer(w, signal) * np.sqrt(Ps)

    P_AN_mat = noise_projection_matrix(bob_deg)
    noise    = (rng.standard_normal((N_ANTENNAS, L)) +
                1j * rng.standard_normal((N_ANTENNAS, L))) / np.sqrt(2)
    an_part  = (P_AN_mat @ noise) * np.sqrt(Pan)

    return sig_part + an_part

def virtual_channel(X: np.ndarray, theta_deg: float, path_loss_db: float = 0.0) -> np.ndarray:
    """Simulates the wireless medium including path loss."""
    a = steering_vector(theta_deg)
    g = 10 ** (-path_loss_db / 20)
    return np.real(a.conj() @ X) * g

# ── 3. Analysis Engine (Analytical & Empirical) ──────────────────────────────

def compute_snr_analytical(
        rx_angle_deg:     float,
        bob_deg:          float,
        signal_pow_db:    float = 20.0,
        an_pow_db:        float = 10.0,
        n_antennas:       int   = N_ANTENNAS,
        thermal_noise_db: float = THERMAL_NOISE_DB
) -> float:
    """Analytical SNR calculation for stable FOM reporting."""
    Ps     = 10 ** (signal_pow_db / 10)
    Pan    = 10 ** (an_pow_db / 10)
    sigma2 = 10 ** (thermal_noise_db / 10)

    af = array_factor(rx_angle_deg, bob_deg)
    # Apply N_ANTENNAS parameter directly
    sig_power = n_antennas * (abs(af) ** 2) * Ps

    P_AN_mat = noise_projection_matrix(bob_deg)
    a_rx     = steering_vector(rx_angle_deg)
    # Apply N_ANTENNAS parameter directly
    an_power = n_antennas * Pan * float(np.real(a_rx.conj() @ P_AN_mat @ P_AN_mat.conj().T @ a_rx))

    snr_lin = sig_power / (sigma2 + an_power + 1e-20)
    return 10 * np.log10(max(snr_lin, 1e-15))

def compute_snr(received: np.ndarray, reference: np.ndarray) -> float:
    """Empirical SNR calculation via signal projection (Legacy/ZMQ)."""
    ref_norm   = reference / (np.linalg.norm(reference) + 1e-12)
    signal_est = np.dot(received, ref_norm.conj()) * ref_norm
    noise_est  = received - signal_est
    sig_power  = np.mean(np.abs(signal_est) ** 2)
    noise_power= np.mean(np.abs(noise_est)  ** 2) + 1e-30
    return 10 * np.log10(sig_power / noise_power)

def secrecy_rate(snr_bob_db: float, snr_eve_db: float) -> float:
    """Shannon physical-layer secrecy capacity."""
    snr_bob = 10 ** (snr_bob_db / 10)
    snr_eve = 10 ** (snr_eve_db / 10)
    return max(0.0, np.log2(1.0 + snr_bob) - np.log2(1.0 + snr_eve))

# ── 4. Tracking & DOA Estimation ─────────────────────────────────────────────

def generate_pilot_ping(
        true_angle_deg: float, 
        n_antennas: int = N_ANTENNAS, 
        n_snapshots: int = 256, 
        snr_pilot_db: float = 15.0
) -> np.ndarray:
    """Simulates Bob's uplink pilot transmission."""
    a = steering_vector(true_angle_deg).reshape(-1, 1)
    s = (np.random.choice([-1, 1], size=(1, n_snapshots)) + 0j)
    
    sig_power = 10 ** (snr_pilot_db / 10)
    Y_signal = np.sqrt(sig_power) * (a @ s)
    noise = np.sqrt(0.5) * (np.random.randn(n_antennas, n_snapshots) + 1j * np.random.randn(n_antennas, n_snapshots))
    return Y_signal + noise

def root_music_doa(snapshots: np.ndarray, n_sources: int = 1) -> np.ndarray:
    """Mathematical DOA estimation using Root-MUSIC."""
    K = snapshots.shape[1]
    R = (snapshots @ snapshots.conj().T) / K
    _, eigenvectors = np.linalg.eigh(R)
    En = eigenvectors[:, :N_ANTENNAS - n_sources]
    C_mat = En @ En.conj().T
    poly_coeffs = np.zeros(2 * N_ANTENNAS - 1, dtype=complex)
    for i in range(N_ANTENNAS):
        for j in range(N_ANTENNAS):
            poly_coeffs[i - j + N_ANTENNAS - 1] += C_mat[i, j]
    roots = np.roots(poly_coeffs)
    roots_inside = roots[np.abs(roots) <= 1.01]
    best_roots = roots_inside[np.argsort(np.abs(np.abs(roots_inside) - 1.0))[:n_sources]]
    return np.rad2deg(np.arcsin(np.angle(best_roots) / np.pi).real)

def ml_doa_estimate(snapshots: np.ndarray, model) -> float:
    """Neural Network inference for Bob's angle."""
    K = snapshots.shape[1]
    R = (snapshots @ snapshots.conj().T) / K
    features = np.concatenate([np.real(R).flatten(), np.imag(R).flatten()]).reshape(1, -1)
    return float(model.predict(features)[0])

class SADMTracker:
    """Hybrid tracker handling stateful beamforming and real-time updates."""
    def __init__(self, bob_initial_angle: float = 30.0, alpha: float = 0.3, use_ml: bool = True):
        self.angle = bob_initial_angle
        self.alpha = alpha
        self.ml_model = self._load_model() if use_ml else None
        self.use_ml = use_ml if self.ml_model is not None else False
        self._update_weights()

    def _load_model(self):
        if os.path.exists('ml_doa_model.pkl'):
            with open('ml_doa_model.pkl', 'rb') as f: return pickle.load(f)
        return None

    def _update_weights(self):
        self.w = beamforming_weights(self.angle)
        self.P_AN = noise_projection_matrix(self.angle)

    def update(self, pilot_snapshots: np.ndarray) -> float:
        raw_est = ml_doa_estimate(pilot_snapshots, self.ml_model) if self.use_ml else root_music_doa(pilot_snapshots)[0]
        self.angle = self.alpha * raw_est + (1 - self.alpha) * self.angle
        self._update_weights()
        return self.angle

    def transmit(self, message, snr_signal_db=20.0, snr_noise_db=10.0):
        """Transmits using currently tracked weights."""
        return sadm_transmit(message, self.angle, snr_signal_db, snr_noise_db)