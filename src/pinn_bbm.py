import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from pathlib import Path

# ============================================================
# 0. ABLATION FLAGS
# ============================================================

SEED = 2026

USE_INPUT_NORMALIZATION = True
USE_ENERGY_LOSS = False
USE_LBFGS = True

RUN_NAME = f"bbm_norm_{USE_INPUT_NORMALIZATION}_energy_{USE_ENERGY_LOSS}_lbfgs_{USE_LBFGS}_seed_{SEED}"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = ROOT_DIR / "results" / "raw" / "bbm"
FIGURES_DIR = ROOT_DIR / "figures" / "generated" / "bbm"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. EXACT BBM SOLITON
# ============================================================

def exact_bbm(x, t, c=0.5, x0=-10.0):
    """
    Exact solitary wave for:

        u_t + u_x + u*u_x - u_xxt = 0

    u(x,t) = 3c sech^2( 1/2 sqrt(c/(c+1)) * (x - (c+1)t - x0) )
    """
    c_tensor = torch.tensor(c, dtype=x.dtype, device=x.device)

    speed = c + 1.0
    k = 0.5 * torch.sqrt(c_tensor / (c_tensor + 1.0))

    arg = k * (x - speed * t - x0)

    return 3.0 * c * (1.0 / torch.cosh(arg)) ** 2


# ============================================================
# 2. PINN ARCHITECTURE
# ============================================================

class BBM_PINN(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
# 3. SETTINGS
# ============================================================

L = 20.0
T_max = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BBM_PINN(width=128, depth=4).to(device)

batch_size = 1000
lbfgs_points = 2000

lambda_ic = 10.0
lambda_bc = 1.0
lambda_energy = 1.0 if USE_ENERGY_LOSS else 0.0

adam_epochs = 12000
adam_lr = 0.002
lbfgs_max_iter = 300

print("=" * 80)
print(f"RUN NAME: {RUN_NAME}")
print(f"Device: {device}")
print(f"USE_INPUT_NORMALIZATION = {USE_INPUT_NORMALIZATION}")
print(f"USE_ENERGY_LOSS         = {USE_ENERGY_LOSS}")
print(f"USE_LBFGS               = {USE_LBFGS}")
print("PDE residual: F = u_t + u_x + u*u_x - u_xxt")
print("=" * 80)


# ============================================================
# 4. AD UTILITIES
# ============================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


def bbm_residual_pinn(x, t):
    """
    Standard PINN residual for BBM:

        F = u_t + u_x + u*u_x - u_xxt

    All derivatives are computed by AD.
    """
    u = model(x, t)

    u_t = grad(u, t)

    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    # Mixed derivative:
    # u_xxt = d/dt(u_xx)
    u_xxt = grad(u_xx, t)

    residual = u_t + u_x + u * u_x - u_xxt

    return residual


# ============================================================
# 5. ENERGY
# ============================================================
# BBM invariant:
# E(t) = integral (u^2 + u_x^2) dx

x_energy = torch.linspace(-L, L, 400).view(-1, 1).to(device)
dx = x_energy[1] - x_energy[0]

# E0 requires u_x, therefore AD is needed
x_energy_e0 = x_energy.clone().detach().requires_grad_(True)
t_energy_e0 = torch.zeros_like(x_energy_e0)

u0 = exact_bbm(x_energy_e0, t_energy_e0)
u0_x = grad(u0, x_energy_e0)

E0 = torch.sum((u0 ** 2 + u0_x ** 2)) * dx
E0 = E0.detach()


def compute_energy_pinn(t_scalar):
    """
    E(t) = integral (u^2 + u_x^2) dx
    """
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=device)

    x_curr = x_energy.clone().detach().requires_grad_(True)
    t_curr = torch.full_like(x_curr, t_scalar)

    u_curr = model(x_curr, t_curr)
    u_x_curr = grad(u_curr, x_curr)

    E_curr = torch.sum((u_curr ** 2 + u_x_curr ** 2)) * dx

    return E_curr


# ============================================================
# 6. ADAM TRAINING
# ============================================================

optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=3000, gamma=0.5)

history = {
    "total": [],
    "pde": [],
    "ic": [],
    "bc": [],
    "energy": []
}

start_time = time.time()

for epoch in range(adam_epochs):
    optimizer_adam.zero_grad()

    # ------------------------------
    # PDE collocation points
    # ------------------------------
    t_col = (torch.rand(batch_size, 1, device=device) * T_max).requires_grad_(True)
    x_col = ((torch.rand(batch_size, 1, device=device) * 2 * L) - L).requires_grad_(True)

    residual = bbm_residual_pinn(x_col, t_col)

    # Здесь решается PDE:
    # F = u_t + u_x + u*u_x - u_xxt
    loss_pde = torch.mean(residual ** 2)

    # ------------------------------
    # Initial condition
    # ------------------------------
    t_ic = torch.zeros(batch_size, 1, device=device)
    x_ic = (torch.rand(batch_size, 1, device=device) * 2 * L) - L

    u_ic_pred = model(x_ic, t_ic)
    u_ic_exact = exact_bbm(x_ic, t_ic)

    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

    # ------------------------------
    # Boundary condition
    # ------------------------------
    t_bc = torch.rand(batch_size, 1, device=device) * T_max

    x_left = -L * torch.ones(batch_size, 1, device=device)
    x_right = L * torch.ones(batch_size, 1, device=device)

    u_left_pred = model(x_left, t_bc)
    u_right_pred = model(x_right, t_bc)

    loss_bc = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)

    # ------------------------------
    # Energy loss
    # ------------------------------
    if USE_ENERGY_LOSS:
        t_e = torch.rand(3, 1, device=device) * T_max

        loss_energy = 0.0
        for i in range(3):
            E_curr = compute_energy_pinn(t_e[i, 0])
            loss_energy = loss_energy + (E_curr - E0) ** 2

        loss_energy = loss_energy / 3.0
    else:
        loss_energy = torch.tensor(0.0, device=device)

    # ------------------------------
    # Total loss
    # ------------------------------
    loss_total = (
        loss_pde
        + lambda_ic * loss_ic
        + lambda_bc * loss_bc
        + lambda_energy * loss_energy
    )

    loss_total.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer_adam.step()
    scheduler.step()

    history["total"].append(loss_total.item())
    history["pde"].append(loss_pde.item())
    history["ic"].append(loss_ic.item())
    history["bc"].append(loss_bc.item())
    history["energy"].append(loss_energy.item())

    if epoch % 1000 == 0:
        print(
            f"Adam {epoch:5d} | "
            f"PDE: {loss_pde.item():.5e} | "
            f"IC: {loss_ic.item():.5e} | "
            f"BC: {loss_bc.item():.5e} | "
            f"E: {loss_energy.item():.5e} | "
            f"Total: {loss_total.item():.5e}"
        )


# ============================================================
# 7. L-BFGS REFINEMENT
# ============================================================

if USE_LBFGS:
    print("\nStarting L-BFGS refinement for BBM...")

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=100
    )

    t_col_f = (torch.rand(lbfgs_points, 1, device=device) * T_max).requires_grad_(True)
    x_col_f = ((torch.rand(lbfgs_points, 1, device=device) * 2 * L) - L).requires_grad_(True)

    t_ic_f = torch.zeros(lbfgs_points, 1, device=device)
    x_ic_f = (torch.rand(lbfgs_points, 1, device=device) * 2 * L) - L

    t_bc_f = torch.rand(lbfgs_points, 1, device=device) * T_max
    x_left_f = -L * torch.ones(lbfgs_points, 1, device=device)
    x_right_f = L * torch.ones(lbfgs_points, 1, device=device)

    t_e_f = torch.rand(3, 1, device=device) * T_max

    lbfgs_iter = 0

    def closure():
        global lbfgs_iter

        optimizer_lbfgs.zero_grad()

        # PDE loss
        residual = bbm_residual_pinn(x_col_f, t_col_f)

        # Здесь решается PDE в L-BFGS:
        # F = u_t + u_x + u*u_x - u_xxt
        loss_pde = torch.mean(residual ** 2)

        # IC loss
        u_ic_pred = model(x_ic_f, t_ic_f)
        u_ic_exact = exact_bbm(x_ic_f, t_ic_f)

        loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

        # BC loss
        u_left_pred = model(x_left_f, t_bc_f)
        u_right_pred = model(x_right_f, t_bc_f)

        loss_bc = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)

        # Energy loss
        if USE_ENERGY_LOSS:
            loss_energy_f = 0.0
            for i in range(3):
                E_curr = compute_energy_pinn(t_e_f[i, 0])
                loss_energy_f = loss_energy_f + (E_curr - E0) ** 2

            loss_energy_f = loss_energy_f / 3.0
        else:
            loss_energy_f = torch.tensor(0.0, device=device)

        loss_total = (
            loss_pde
            + lambda_ic * loss_ic
            + lambda_bc * loss_bc
            + lambda_energy * loss_energy_f
        )

        loss_total.backward()

        lbfgs_iter += 1

        if lbfgs_iter % 50 == 0:
            print(
                f"L-BFGS {lbfgs_iter:4d} | "
                f"PDE: {loss_pde.item():.5e} | "
                f"IC: {loss_ic.item():.5e} | "
                f"BC: {loss_bc.item():.5e} | "
                f"E: {loss_energy_f.item():.5e} | "
                f"Total: {loss_total.item():.5e}"
            )

        return loss_total

    optimizer_lbfgs.step(closure)

training_time = time.time() - start_time


# ============================================================
# 8. METRICS
# ============================================================

def relative_l2(u_pred, u_exact, mask=None):
    if mask is not None:
        u_pred = u_pred[mask]
        u_exact = u_exact[mask]

    return (torch.norm(u_pred - u_exact, 2) / torch.norm(u_exact, 2)).item()


def evaluate_final_time_l2():
    model.eval()

    with torch.no_grad():
        x_test = torch.linspace(-L, L, 1000).view(-1, 1).to(device)
        t_test = torch.full_like(x_test, T_max)

        u_pred = model(x_test, t_test)
        u_exact = exact_bbm(x_test, t_test)

        l2_error = torch.norm(u_pred - u_exact, 2) / torch.norm(u_exact, 2)

    return l2_error.item()


def evaluate_full_and_interior_l2():
    model.eval()

    with torch.no_grad():
        x_diag = torch.linspace(-L, L, 1000).view(-1, 1).to(device)

        margin = 4.0
        interior = (x_diag.squeeze() > -L + margin) & (x_diag.squeeze() < L - margin)

        t_fin = torch.full_like(x_diag, T_max)

        u_pred = model(x_diag, t_fin)
        u_exact = exact_bbm(x_diag, t_fin)

        err_full = relative_l2(u_pred, u_exact)
        err_int = relative_l2(u_pred, u_exact, interior)

    return err_full, err_int, margin


def evaluate_space_time_l2(Nx=600, Nt=80):
    model.eval()

    with torch.no_grad():
        x_grid = torch.linspace(-L, L, Nx).view(-1, 1).to(device)
        t_grid = torch.linspace(0.0, T_max, Nt).view(-1, 1).to(device)

        errors = []
        exacts = []

        for tv in t_grid:
            t_col = torch.full_like(x_grid, tv.item())

            u_pred = model(x_grid, t_col)
            u_exact = exact_bbm(x_grid, t_col)

            errors.append((u_pred - u_exact).reshape(-1))
            exacts.append(u_exact.reshape(-1))

        errors = torch.cat(errors)
        exacts = torch.cat(exacts)

        st_l2 = torch.norm(errors, 2) / torch.norm(exacts, 2)

    return st_l2.item()


def evaluate_energy_drift():
    model.eval()

    # Нужно AD для u_x, поэтому no_grad не используем
    t_energy_diag = torch.linspace(0.0, T_max, 80)
    energy_values = []

    for tv in t_energy_diag:
        E_curr = compute_energy_pinn(tv.to(device))
        energy_values.append(E_curr.detach().item())

    energy_values = np.array(energy_values)
    E0_np = E0.item()

    rel_energy_drift = np.abs(energy_values - E0_np) / np.abs(E0_np)

    max_energy_drift = np.max(rel_energy_drift)
    mean_energy_drift = np.mean(rel_energy_drift)

    return max_energy_drift, mean_energy_drift, t_energy_diag, rel_energy_drift


def evaluate_pde_loss_fixed(N_eval=3000):
    """
    Final PDE residual loss on new random points.
    Here u_xxt is again computed by AD.
    """
    model.eval()

    t_eval = (torch.rand(N_eval, 1, device=device) * T_max).requires_grad_(True)
    x_eval = ((torch.rand(N_eval, 1, device=device) * 2 * L) - L).requires_grad_(True)

    residual = bbm_residual_pinn(x_eval, t_eval)
    loss_pde_eval = torch.mean(residual ** 2)

    return loss_pde_eval.item()


final_l2 = evaluate_final_time_l2()
err_full, err_int, margin = evaluate_full_and_interior_l2()
space_time_l2 = evaluate_space_time_l2(Nx=600, Nt=80)
max_energy_drift, mean_energy_drift, t_energy_diag, rel_energy_drift = evaluate_energy_drift()
final_pde_eval = evaluate_pde_loss_fixed(N_eval=3000)

print("\n================ BBM ablation metrics ================")
print(f"Run name                   : {RUN_NAME}")
print(f"Seed                       : {SEED}")
print(f"Input normalization        : {USE_INPUT_NORMALIZATION}")
print(f"Energy loss                : {USE_ENERGY_LOSS}")
print(f"L-BFGS                     : {USE_LBFGS}")
print(f"Training time seconds      : {training_time:.2f}")
print(f"Final-time relative L2     : {final_l2:.6e}")
print(f"Full-domain L2 at T={T_max}: {err_full:.6e}")
print(f"Interior L2 at T={T_max}   : {err_int:.6e}  margin={margin}")
print(f"Space-time L2              : {space_time_l2:.6e}")
print(f"Final PDE eval loss        : {final_pde_eval:.6e}")
print(f"Max energy drift           : {max_energy_drift:.6e}")
print(f"Mean energy drift          : {mean_energy_drift:.6e}")
print("======================================================")


# ============================================================
# 9. SAVE METRICS TO CSV
# ============================================================

csv_file = RESULTS_DIR / "bbm_ablation_results.csv"

row = {
    "run_name": RUN_NAME,
    "seed": SEED,
    "input_normalization": USE_INPUT_NORMALIZATION,
    "energy_loss": USE_ENERGY_LOSS,
    "lbfgs": USE_LBFGS,
    "training_time": training_time,
    "final_l2": final_l2,
    "full_l2": err_full,
    "interior_l2": err_int,
    "space_time_l2": space_time_l2,
    "pde_eval_loss": final_pde_eval,
    "max_energy_drift": max_energy_drift,
    "mean_energy_drift": mean_energy_drift
}

try:
    with open(csv_file, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
except FileExistsError:
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

print(f"\nMetrics appended to {csv_file}")


# ============================================================
# 10. SAVE PLOTS
# ============================================================

# L2(t)
model.eval()

with torch.no_grad():
    x_diag = torch.linspace(-L, L, 1000).view(-1, 1).to(device)
    interior = (x_diag.squeeze() > -L + margin) & (x_diag.squeeze() < L - margin)

    t_grid = torch.linspace(0.0, T_max, 80)

    l2_full_t = []
    l2_int_t = []

    for tv in t_grid:
        t_col = torch.full_like(x_diag, tv.item())

        u_pred = model(x_diag, t_col)
        u_exact = exact_bbm(x_diag, t_col)

        l2_full_t.append(relative_l2(u_pred, u_exact))
        l2_int_t.append(relative_l2(u_pred, u_exact, interior))


plt.figure(figsize=(8, 5))
plt.plot(t_grid.cpu().numpy(), l2_full_t, "r-o", ms=3, label="Full domain")
plt.plot(t_grid.cpu().numpy(), l2_int_t, "b-s", ms=3, label=f"Interior (|x|<{L-margin})")
plt.xlabel("t")
plt.ylabel("Relative $L_2$ error")
plt.title(f"BBM ablation: {RUN_NAME}")
plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{RUN_NAME}_l2_time.png", dpi=150)
plt.show()


# Energy drift
plt.figure(figsize=(8, 5))
plt.plot(t_energy_diag.cpu().numpy(), rel_energy_drift, "k-o", ms=3)
plt.xlabel("t")
plt.ylabel("Relative energy drift")
plt.title(f"BBM energy drift: {RUN_NAME}")
plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{RUN_NAME}_energy_drift.png", dpi=150)
plt.show()


# Profiles
x_test = torch.linspace(-L, L, 1000).view(-1, 1).to(device)

plt.figure(figsize=(12, 6))

times_to_plot = [0.0, T_max / 2.0, T_max]
colors = ["blue", "green", "red"]

with torch.no_grad():
    for i, t_val in enumerate(times_to_plot):
        t_plot = torch.full_like(x_test, t_val)

        u_pred_plot = model(x_test, t_plot)
        u_exact_plot = exact_bbm(x_test, t_plot)

        plt.plot(
            x_test.cpu(),
            u_exact_plot.cpu(),
            color=colors[i],
            linestyle="dashed",
            linewidth=2,
            label=f"Exact t={t_val:.2f}"
        )

        plt.plot(
            x_test.cpu(),
            u_pred_plot.cpu(),
            color=colors[i],
            linewidth=2,
            label=f"PINN t={t_val:.2f}"
        )

plt.title(f"BBM PINN ablation: {RUN_NAME}, L2={final_l2:.4e}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{RUN_NAME}_profiles.png", dpi=150)
plt.show()
