"""
Fourier error-spectrum diagnostic for the Kawahara PINN.

This script loads a trained Kawahara PINN model and computes the Hann-windowed
spatial power spectra of the exact solution and of the PINN error at the final
time. It is intended to reproduce the spectral-error analysis reported in the
paper.

Expected model file:
    models/kawahara_standard_pinn_model.pt

Run from the repository root:
    python diagnostics/kawahara_error_spectrum.py
"""

from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 0. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results" / "raw" / "kawahara"
FIGURES_DIR = ROOT_DIR / "figures" / "generated" / "kawahara"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CANDIDATES = [
    MODELS_DIR / "kawahara_standard_pinn_model.pt",
    ROOT_DIR / "src" / "kawahara_standard_pinn_model.pt",
    ROOT_DIR / "kawahara_standard_pinn_model.pt",
]


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
USE_INPUT_NORMALIZATION = True

L = 20.0
T_max = 2.0

N_FFT = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. EXACT KAWAHARA SOLITON
# ============================================================

def exact_kawahara(x, t, x0=-5.0):
    """
    Exact solitary wave for:

        u_t + u*u_x + u_xxx - u_xxxxx = 0

    u(x,t) = 105/169 * sech^4(k * (x - v*t - x0)),
    where v = 36/169 and k = 1/(2*sqrt(13)).
    """
    v = 36.0 / 169.0
    amp = 105.0 / 169.0
    k = 1.0 / (2.0 * math.sqrt(13.0))

    arg = k * (x - v * t - x0)
    return amp * (1.0 / torch.cosh(arg)) ** 4


# ============================================================
# 3. PINN ARCHITECTURE
# ============================================================

class Kawahara_PINN(nn.Module):
    def __init__(self, width=256, depth=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if USE_INPUT_NORMALIZATION:
            x_in = x / L
            t_in = 2.0 * t / T_max - 1.0
        else:
            x_in = x
            t_in = t

        inp = torch.cat([x_in, t_in], dim=1)
        return self.net(inp)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n".join(f"  - {p}" for p in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a trained Kawahara model.\n"
        "Run the Kawahara training script first:\n\n"
        "    python src/pinn_kawahara.py\n\n"
        "Expected one of:\n"
        f"{candidates}"
    )


model_path = find_model_path()

model = Kawahara_PINN(width=256, depth=4).to(device)

try:
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
except TypeError:
    state_dict = torch.load(model_path, map_location=device)

model.load_state_dict(state_dict)
model.eval()

print("=" * 80)
print("Kawahara Fourier error-spectrum diagnostic")
print(f"Device     : {device}")
print(f"Model file : {model_path}")
print("=" * 80)


# ============================================================
# 5. SPECTRAL DIAGNOSTICS
# ============================================================

def spectral_centroid(k, power):
    """
    Spectral centroid: sum(k * P(k)) / sum(P(k)).
    """
    return float(np.sum(k * power) / (np.sum(power) + 1e-30))


def compute_error_spectrum(N=N_FFT):
    """
    Compute Hann-windowed spectra of the exact solution and PINN error
    at the final time T_max.
    """
    x = torch.linspace(-L, L, N).view(-1, 1).to(device)
    t = torch.full_like(x, T_max)

    with torch.no_grad():
        u_pred = model(x, t).detach().cpu().numpy().reshape(-1)
        u_exact = exact_kawahara(x, t).detach().cpu().numpy().reshape(-1)

    error = u_pred - u_exact

    # Hann window suppresses boundary artifacts from the truncated solitary wave.
    window = np.hanning(N)

    u_exact_w = u_exact * window
    error_w = error * window

    dx = (2.0 * L) / (N - 1)

    k = np.fft.rfftfreq(N, d=dx) * 2.0 * np.pi
    power_exact = np.abs(np.fft.rfft(u_exact_w)) ** 2
    power_error = np.abs(np.fft.rfft(error_w)) ** 2

    return k, power_error, power_exact


k_fft, power_error, power_exact = compute_error_spectrum(N=N_FFT)

centroid_error = spectral_centroid(k_fft, power_error)
centroid_exact = spectral_centroid(k_fft, power_exact)
centroid_ratio = centroid_error / (centroid_exact + 1e-30)

high_k_mask = k_fft > (2.0 * centroid_exact)

high_k_fraction_error = float(
    np.sum(power_error[high_k_mask]) / (np.sum(power_error) + 1e-30)
)

high_k_fraction_exact = float(
    np.sum(power_exact[high_k_mask]) / (np.sum(power_exact) + 1e-30)
)

print("\n--- Error spectrum diagnostics ---")
print(f"Spectral centroid of error    : k = {centroid_error:.6f}")
print(f"Spectral centroid of solution : k = {centroid_exact:.6f}")
print(f"Ratio error/solution          : {centroid_ratio:.6f}")
print(f"High-k energy fraction error  : {high_k_fraction_error:.6f}")
print(f"High-k energy fraction solut. : {high_k_fraction_exact:.6f}")
print("High-k is defined as k > 2 * solution centroid.")


# ============================================================
# 6. SAVE METRICS
# ============================================================

metrics_path = RESULTS_DIR / "kawahara_error_spectrum_metrics.txt"

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("Kawahara Fourier error-spectrum diagnostic\n")
    f.write(f"Model file                         = {model_path}\n")
    f.write(f"N_FFT                              = {N_FFT}\n")
    f.write(f"Spectral centroid of error         = {centroid_error:.10f}\n")
    f.write(f"Spectral centroid of solution      = {centroid_exact:.10f}\n")
    f.write(f"Ratio error/solution               = {centroid_ratio:.10f}\n")
    f.write(f"High-k energy fraction error       = {high_k_fraction_error:.10f}\n")
    f.write(f"High-k energy fraction solution    = {high_k_fraction_exact:.10f}\n")


# ============================================================
# 7. SAVE FIGURE
# ============================================================

power_error_norm = power_error / (power_error.max() + 1e-30)
power_exact_norm = power_exact / (power_exact.max() + 1e-30)

plt.figure(figsize=(8, 5))
plt.semilogy(k_fft, power_exact_norm, linewidth=2, label="Exact solution spectrum")
plt.semilogy(k_fft, power_error_norm, linestyle="--", linewidth=2, label="Error spectrum")
plt.axvline(
    centroid_exact,
    linestyle=":",
    alpha=0.8,
    label=f"Solution centroid k={centroid_exact:.2f}",
)
plt.axvline(
    centroid_error,
    linestyle=":",
    alpha=0.8,
    label=f"Error centroid k={centroid_error:.2f}",
)
plt.xlabel("Wavenumber $k$")
plt.ylabel("Normalized power spectrum")
plt.title("Kawahara: spectral content of error vs solution")
plt.xlim(0, 10)
plt.ylim(1e-12, 2.0)
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

figure_path = FIGURES_DIR / "kawahara_error_spectrum_hann.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved outputs:")
print(f"  - {metrics_path}")
print(f"  - {figure_path}")
