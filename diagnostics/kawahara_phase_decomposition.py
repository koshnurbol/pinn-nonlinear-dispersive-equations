"""
Phase-shift and spectral-control diagnostic for the Kawahara PINN.

This script loads a trained Kawahara PINN model and checks whether the observed
high-wavenumber error content can be explained by a simple rigid phase shift.
It computes the final-time error before and after removing the optimal phase
shift and compares their spatial Fourier spectra.

Expected model file:
    models/kawahara_standard_pinn_model.pt

Run from the repository root:
    python diagnostics/kawahara_phase_decomposition.py
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

N_GRID = 4000
APPLY_HANN_WINDOW = True

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
print("Kawahara phase-shift and spectral-control diagnostic")
print(f"Device     : {device}")
print(f"Model file : {model_path}")
print("=" * 80)


# ============================================================
# 5. FINAL-TIME PROFILES
# ============================================================

with torch.no_grad():
    x = torch.linspace(-L, L, N_GRID).view(-1, 1).to(device)
    t = torch.full_like(x, T_max)

    u_pred = model(x, t).detach().cpu().numpy().reshape(-1)
    u_exact = exact_kawahara(x, t).detach().cpu().numpy().reshape(-1)

x_grid = x.detach().cpu().numpy().reshape(-1)


# ============================================================
# 6. PEAK-BASED INITIAL PHASE-SHIFT ESTIMATE
# ============================================================

i_pred = int(np.argmax(u_pred))
i_exact = int(np.argmax(u_exact))

peak_shift = float(x_grid[i_pred] - x_grid[i_exact])

amp_pred = float(u_pred[i_pred])
amp_exact = float(u_exact[i_exact])
amp_rel_error = abs(amp_pred - amp_exact) / (abs(amp_exact) + 1e-30)

print("\n--- Peak, amplitude, and initial phase estimate ---")
print(f"Exact peak amplitude : A = {amp_exact:.8f} at x = {x_grid[i_exact]:+.6f}")
print(f"PINN peak amplitude  : A = {amp_pred:.8f} at x = {x_grid[i_pred]:+.6f}")
print(f"Relative amplitude error : {100.0 * amp_rel_error:.6f}%")
print(f"Peak-based phase shift   : dx = {peak_shift:+.6f}")


# ============================================================
# 7. L2-OPTIMAL RIGID PHASE SHIFT
# ============================================================

shift_candidates = np.linspace(peak_shift - 0.5, peak_shift + 0.5, 2001)

best_shift = peak_shift
best_err = np.inf

with torch.no_grad():
    x_torch = torch.linspace(-L, L, N_GRID).view(-1, 1).to(device)
    t_torch = torch.full_like(x_torch, T_max)

    for shift in shift_candidates:
        u_exact_shifted = (
            exact_kawahara(x_torch - shift, t_torch)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        err = np.linalg.norm(u_pred - u_exact_shifted) / (
            np.linalg.norm(u_exact_shifted) + 1e-30
        )

        if err < best_err:
            best_err = float(err)
            best_shift = float(shift)

with torch.no_grad():
    u_exact_shifted = (
        exact_kawahara(x_torch - best_shift, t_torch)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

print(f"L2-optimal phase shift    : s = {best_shift:+.6f}")


# ============================================================
# 8. ERROR BEFORE AND AFTER PHASE REMOVAL
# ============================================================

err_raw = u_pred - u_exact
err_dephased = u_pred - u_exact_shifted

l2_raw = np.linalg.norm(err_raw) / (np.linalg.norm(u_exact) + 1e-30)
l2_dephased = np.linalg.norm(err_dephased) / (
    np.linalg.norm(u_exact_shifted) + 1e-30
)

phase_error_fraction = 1.0 - l2_dephased / (l2_raw + 1e-30)

print("\n--- Relative L2 errors before and after phase removal ---")
print(f"Raw relative L2 error        : {l2_raw:.8e}")
print(f"Dephased relative L2 error   : {l2_dephased:.8e}")
print(f"Phase-removable error share  : {100.0 * phase_error_fraction:.4f}%")


# ============================================================
# 9. SPECTRA BEFORE AND AFTER PHASE REMOVAL
# ============================================================

dx_fft = (2.0 * L) / (N_GRID - 1)
k_fft = np.fft.rfftfreq(N_GRID, d=dx_fft) * 2.0 * np.pi

if APPLY_HANN_WINDOW:
    window = np.hanning(N_GRID)
else:
    window = np.ones(N_GRID)

u_exact_w = u_exact * window
err_raw_w = err_raw * window
err_dephased_w = err_dephased * window

P_exact = np.abs(np.fft.rfft(u_exact_w)) ** 2
P_raw = np.abs(np.fft.rfft(err_raw_w)) ** 2
P_dephased = np.abs(np.fft.rfft(err_dephased_w)) ** 2


def spectral_centroid(k, power):
    return float(np.sum(k * power) / (np.sum(power) + 1e-30))


c_exact = spectral_centroid(k_fft, P_exact)
c_raw = spectral_centroid(k_fft, P_raw)
c_dephased = spectral_centroid(k_fft, P_dephased)

high_k_mask = k_fft > (2.0 * c_exact)

high_k_exact = float(np.sum(P_exact[high_k_mask]) / (np.sum(P_exact) + 1e-30))
high_k_raw = float(np.sum(P_raw[high_k_mask]) / (np.sum(P_raw) + 1e-30))
high_k_dephased = float(
    np.sum(P_dephased[high_k_mask]) / (np.sum(P_dephased) + 1e-30)
)

print("\n--- Spectral centroids ---")
print(f"Solution centroid            : k = {c_exact:.8f}")
print(f"Raw error centroid           : k = {c_raw:.8f}  ratio = {c_raw / (c_exact + 1e-30):.4f}")
print(f"Dephased error centroid      : k = {c_dephased:.8f}  ratio = {c_dephased / (c_exact + 1e-30):.4f}")
print(f"Centroid reduction after dephasing : {100.0 * (1.0 - c_dephased / (c_raw + 1e-30)):.4f}%")

print("\n--- High-k energy fractions ---")
print(f"Solution high-k fraction     : {high_k_exact:.8f}")
print(f"Raw error high-k fraction    : {high_k_raw:.8f}")
print(f"Dephased high-k fraction     : {high_k_dephased:.8f}")
print("High-k is defined as k > 2 * solution centroid.")


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics_path = RESULTS_DIR / "kawahara_phase_decomposition_metrics.txt"

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("Kawahara phase-shift and spectral-control diagnostic\n")
    f.write(f"Model file                              = {model_path}\n")
    f.write(f"N_GRID                                  = {N_GRID}\n")
    f.write(f"Apply Hann window                       = {APPLY_HANN_WINDOW}\n")
    f.write(f"Exact peak amplitude                    = {amp_exact:.12e}\n")
    f.write(f"PINN peak amplitude                     = {amp_pred:.12e}\n")
    f.write(f"Relative amplitude error                = {amp_rel_error:.12e}\n")
    f.write(f"Peak-based phase shift                  = {peak_shift:.12e}\n")
    f.write(f"L2-optimal phase shift                  = {best_shift:.12e}\n")
    f.write(f"Raw relative L2 error                   = {l2_raw:.12e}\n")
    f.write(f"Dephased relative L2 error              = {l2_dephased:.12e}\n")
    f.write(f"Phase-removable error share             = {phase_error_fraction:.12e}\n")
    f.write(f"Solution spectral centroid              = {c_exact:.12e}\n")
    f.write(f"Raw error spectral centroid             = {c_raw:.12e}\n")
    f.write(f"Dephased error spectral centroid        = {c_dephased:.12e}\n")
    f.write(f"Raw centroid ratio                      = {c_raw / (c_exact + 1e-30):.12e}\n")
    f.write(f"Dephased centroid ratio                 = {c_dephased / (c_exact + 1e-30):.12e}\n")
    f.write(f"Solution high-k fraction                = {high_k_exact:.12e}\n")
    f.write(f"Raw error high-k fraction               = {high_k_raw:.12e}\n")
    f.write(f"Dephased error high-k fraction          = {high_k_dephased:.12e}\n")

spectra_path = RESULTS_DIR / "kawahara_phase_decomposition_spectra.csv"

P_exact_norm = P_exact / (P_exact.max() + 1e-30)
P_raw_norm = P_raw / (P_raw.max() + 1e-30)
P_dephased_norm = P_dephased / (P_dephased.max() + 1e-30)

np.savetxt(
    spectra_path,
    np.column_stack([k_fft, P_exact_norm, P_raw_norm, P_dephased_norm]),
    delimiter=",",
    header="k,exact_solution_spectrum,raw_error_spectrum,dephased_error_spectrum",
    comments="",
)


# ============================================================
# 11. SAVE FIGURE
# ============================================================

plt.figure(figsize=(8, 5))
plt.semilogy(k_fft, P_exact_norm, linewidth=2, label="Exact solution")
plt.semilogy(k_fft, P_raw_norm, linestyle="--", linewidth=2, label="Error, raw")
plt.semilogy(k_fft, P_dephased_norm, linestyle="-.", linewidth=2, label="Error, phase removed")

plt.axvline(c_exact, linestyle=":", alpha=0.8, label=f"Solution centroid k={c_exact:.2f}")
plt.axvline(c_raw, linestyle=":", alpha=0.8, label=f"Raw error centroid k={c_raw:.2f}")
plt.axvline(c_dephased, linestyle=":", alpha=0.8, label=f"Dephased error centroid k={c_dephased:.2f}")

plt.xlabel("Wavenumber $k$")
plt.ylabel("Normalized power spectrum")
plt.title("Kawahara: error spectrum before/after phase removal")
plt.xlim(0, 10)
plt.ylim(1e-12, 2.0)
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

figure_path = FIGURES_DIR / "kawahara_error_spectrum_dephased.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved outputs:")
print(f"  - {metrics_path}")
print(f"  - {spectra_path}")
print(f"  - {figure_path}")
