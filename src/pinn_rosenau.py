import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import csv
import gc


# ============================================================
# 0. GLOBAL SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Если хочешь быстрее проверить за час, оставь FAST_MODE = True.
# Для финального запуска можно поставить FAST_MODE = False.
FAST_MODE = True

SAVE_PLOTS = False   # если хочешь сохранять графики для каждого эксперимента, поставь True

# Rosenau--KdV domain
L = 20.0
T_MAX = 2.0

# Architecture
WIDTH = 256
DEPTH = 4

# Loss weights
LAMBDA_IC = 20.0
LAMBDA_BC = 1.0

# Training settings
if FAST_MODE:
    BATCH_SIZE = 500
    LBFGS_POINTS = 700
    ADAM_EPOCHS = 5000
    LBFGS_MAX_ITER = 200
    N_ENERGY_GRID = 300
    N_PDE_EVAL = 1000
else:
    BATCH_SIZE = 800
    LBFGS_POINTS = 1200
    ADAM_EPOCHS = 10000
    LBFGS_MAX_ITER = 500
    N_ENERGY_GRID = 400
    N_PDE_EVAL = 1500

ADAM_LR = 0.002


# ============================================================
# 1. EXPERIMENT LIST
# ============================================================

EXPERIMENTS = [
    {
        "case": "R0",
        "name": "norm_energy_lbfgs",
        "seed": 42,
        "use_norm": True,
        "use_energy": True,
        "use_lbfgs": True,
    },
    {
        "case": "R1",
        "name": "no_norm_energy_lbfgs",
        "seed": 42,
        "use_norm": False,
        "use_energy": True,
        "use_lbfgs": True,
    },
    {
        "case": "R2",
        "name": "norm_no_energy_lbfgs",
        "seed": 42,
        "use_norm": True,
        "use_energy": False,
        "use_lbfgs": True,
    },
    {
        "case": "R3",
        "name": "norm_energy_no_lbfgs",
        "seed": 42,
        "use_norm": True,
        "use_energy": True,
        "use_lbfgs": False,
    },
]


# ============================================================
# 2. EXACT ROSENAU--KDV SOLITON
# ============================================================

def exact_rosenau_kdv(x, t, x0=-5.0):
    """
    Exact solitary wave for:

        u_t + u_x + u*u_x + u_xxx + u_xxxxt = 0

    u(x,t) = A sech^4(k * (x - c*t - x0))

    Parameters are obtained for the +u_xxxxt convention.
    """
    sqrt313 = math.sqrt(313.0)

    A = 35.0 * (sqrt313 - 13.0) / 312.0
    c = (13.0 + sqrt313) / 26.0
    k = math.sqrt(2.0 * sqrt313 - 26.0) / 24.0

    arg = k * (x - c * t - x0)
    return A * (1.0 / torch.cosh(arg)) ** 4


# ============================================================
# 3. MODEL
# ============================================================

class RosenauKdV_PINN(nn.Module):
    def __init__(self, width=256, depth=4, use_norm=True):
        super().__init__()

        self.use_norm = use_norm

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
        if self.use_norm:
            x_in = x / L
            t_in = 2.0 * t / T_MAX - 1.0
        else:
            x_in = x
            t_in = t

        inp = torch.cat([x_in, t_in], dim=1)
        return self.net(inp)


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


def rosenau_residual_pinn(model, x, t):
    """
    Standard PINN residual for Rosenau--KdV:

        F = u_t + u_x + u*u_x + u_xxx + u_xxxxt

    All derivatives are computed by AD.
    """
    u = model(x, t)

    # Time derivative
    u_t = grad(u, t)

    # Spatial derivatives
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    u_xxx = grad(u_xx, x)
    u_xxxx = grad(u_xxx, x)

    # Mixed derivative:
    # u_xxxxt = d/dt(u_xxxx)
    u_xxxxt = grad(u_xxxx, t)

    residual = u_t + u_x + u * u_x + u_xxx + u_xxxxt

    return residual


# ============================================================
# 5. ENERGY
# ============================================================
# Rosenau--KdV invariant used here:
# E(t) = integral (u^2 + u_xx^2) dx

def build_energy_grid():
    x_energy = torch.linspace(-L, L, N_ENERGY_GRID).view(-1, 1).to(DEVICE)
    dx = x_energy[1] - x_energy[0]

    # E0 requires second derivative of the exact solution
    x_e0 = x_energy.clone().detach().requires_grad_(True)
    t_e0 = torch.zeros_like(x_e0)

    u0 = exact_rosenau_kdv(x_e0, t_e0)
    u0_x = grad(u0, x_e0)
    u0_xx = grad(u0_x, x_e0)

    E0 = torch.sum((u0 ** 2 + u0_xx ** 2)) * dx
    E0 = E0.detach()

    return x_energy, dx, E0


def compute_energy_pinn(model, x_energy, dx, t_scalar):
    """
    E(t) = integral (u^2 + u_xx^2) dx
    """
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=DEVICE)

    x_curr = x_energy.clone().detach().requires_grad_(True)
    t_curr = torch.full_like(x_curr, t_scalar)

    u_curr = model(x_curr, t_curr)
    u_x_curr = grad(u_curr, x_curr)
    u_xx_curr = grad(u_x_curr, x_curr)

    E_curr = torch.sum((u_curr ** 2 + u_xx_curr ** 2)) * dx

    return E_curr


# ============================================================
# 6. METRIC FUNCTIONS
# ============================================================

def relative_l2(u_pred, u_exact, mask=None):
    if mask is not None:
        u_pred = u_pred[mask]
        u_exact = u_exact[mask]

    return (torch.norm(u_pred - u_exact, 2) / torch.norm(u_exact, 2)).item()


def evaluate_final_time_l2(model):
    model.eval()

    with torch.no_grad():
        x_test = torch.linspace(-L, L, 1000).view(-1, 1).to(DEVICE)
        t_test = torch.full_like(x_test, T_MAX)

        u_pred = model(x_test, t_test)
        u_exact = exact_rosenau_kdv(x_test, t_test)

        l2_error = torch.norm(u_pred - u_exact, 2) / torch.norm(u_exact, 2)

    return l2_error.item()


def evaluate_full_and_interior_l2(model):
    model.eval()

    with torch.no_grad():
        x_diag = torch.linspace(-L, L, 1000).view(-1, 1).to(DEVICE)

        margin = 4.0
        interior = (x_diag.squeeze() > -L + margin) & (x_diag.squeeze() < L - margin)

        t_fin = torch.full_like(x_diag, T_MAX)

        u_pred = model(x_diag, t_fin)
        u_exact = exact_rosenau_kdv(x_diag, t_fin)

        err_full = relative_l2(u_pred, u_exact)
        err_int = relative_l2(u_pred, u_exact, interior)

    return err_full, err_int, margin


def evaluate_space_time_l2(model, Nx=600, Nt=80):
    model.eval()

    with torch.no_grad():
        x_grid = torch.linspace(-L, L, Nx).view(-1, 1).to(DEVICE)
        t_grid = torch.linspace(0.0, T_MAX, Nt).view(-1, 1).to(DEVICE)

        errors = []
        exacts = []

        for tv in t_grid:
            t_col = torch.full_like(x_grid, tv.item())

            u_pred = model(x_grid, t_col)
            u_exact = exact_rosenau_kdv(x_grid, t_col)

            errors.append((u_pred - u_exact).reshape(-1))
            exacts.append(u_exact.reshape(-1))

        errors = torch.cat(errors)
        exacts = torch.cat(exacts)

        st_l2 = torch.norm(errors, 2) / torch.norm(exacts, 2)

    return st_l2.item()


def evaluate_energy_drift(model, x_energy, dx, E0):
    model.eval()

    # Need AD for u_xx, so no torch.no_grad()
    t_energy_diag = torch.linspace(0.0, T_MAX, 80)
    energy_values = []

    for tv in t_energy_diag:
        E_curr = compute_energy_pinn(model, x_energy, dx, tv.to(DEVICE))
        energy_values.append(E_curr.detach().item())

    energy_values = np.array(energy_values)
    E0_np = E0.item()

    rel_energy_drift = np.abs(energy_values - E0_np) / np.abs(E0_np)

    max_energy_drift = np.max(rel_energy_drift)
    mean_energy_drift = np.mean(rel_energy_drift)

    return max_energy_drift, mean_energy_drift, t_energy_diag, rel_energy_drift


def evaluate_pde_loss_fixed(model):
    """
    Final PDE residual loss on new random points.
    Here u_xxxxt is again computed by AD.
    """
    model.eval()

    t_eval = (torch.rand(N_PDE_EVAL, 1, device=DEVICE) * T_MAX).requires_grad_(True)
    x_eval = ((torch.rand(N_PDE_EVAL, 1, device=DEVICE) * 2 * L) - L).requires_grad_(True)

    residual = rosenau_residual_pinn(model, x_eval, t_eval)
    loss_pde_eval = torch.mean(residual ** 2)

    return loss_pde_eval.item()


# ============================================================
# 7. PLOTTING
# ============================================================

def save_plots(model, run_name, final_l2, margin, t_energy_diag, rel_energy_drift):
    if not SAVE_PLOTS:
        return

    model.eval()

    # L2(t)
    with torch.no_grad():
        x_diag = torch.linspace(-L, L, 1000).view(-1, 1).to(DEVICE)
        interior = (x_diag.squeeze() > -L + margin) & (x_diag.squeeze() < L - margin)

        t_grid = torch.linspace(0.0, T_MAX, 80)

        l2_full_t = []
        l2_int_t = []

        for tv in t_grid:
            t_col = torch.full_like(x_diag, tv.item())

            u_pred = model(x_diag, t_col)
            u_exact = exact_rosenau_kdv(x_diag, t_col)

            l2_full_t.append(relative_l2(u_pred, u_exact))
            l2_int_t.append(relative_l2(u_pred, u_exact, interior))

    plt.figure(figsize=(8, 5))
    plt.plot(t_grid.cpu().numpy(), l2_full_t, "r-o", ms=3, label="Full domain")
    plt.plot(t_grid.cpu().numpy(), l2_int_t, "b-s", ms=3, label=f"Interior (|x|<{L-margin})")
    plt.xlabel("t")
    plt.ylabel("Relative $L_2$ error")
    plt.title(f"Rosenau-KdV ablation: {run_name}")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_name}_l2_time.png", dpi=150)
    plt.close()

    # Energy drift
    plt.figure(figsize=(8, 5))
    plt.plot(t_energy_diag.cpu().numpy(), rel_energy_drift, "k-o", ms=3)
    plt.xlabel("t")
    plt.ylabel("Relative energy drift")
    plt.title(f"Rosenau-KdV energy drift: {run_name}")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_name}_energy_drift.png", dpi=150)
    plt.close()

    # Profiles
    x_test = torch.linspace(-L, L, 1000).view(-1, 1).to(DEVICE)

    plt.figure(figsize=(12, 6))

    times_to_plot = [0.0, T_MAX / 2.0, T_MAX]
    colors = ["blue", "green", "red"]

    with torch.no_grad():
        for i, t_val in enumerate(times_to_plot):
            t_plot = torch.full_like(x_test, t_val)

            u_pred_plot = model(x_test, t_plot)
            u_exact_plot = exact_rosenau_kdv(x_test, t_plot)

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

    plt.title(f"Rosenau-KdV PINN: {run_name}, L2={final_l2:.4e}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{run_name}_profiles.png", dpi=150)
    plt.close()


# ============================================================
# 8. SINGLE EXPERIMENT RUNNER
# ============================================================

def run_experiment(config):
    case = config["case"]
    seed = config["seed"]
    use_norm = config["use_norm"]
    use_energy = config["use_energy"]
    use_lbfgs = config["use_lbfgs"]

    run_name = f"rosenau_{case}_{config['name']}_seed_{seed}"

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 90)
    print(f"STARTING EXPERIMENT: {run_name}")
    print(f"Device                  : {DEVICE}")
    print(f"FAST_MODE               : {FAST_MODE}")
    print(f"Input normalization     : {use_norm}")
    print(f"Energy loss             : {use_energy}")
    print(f"L-BFGS                  : {use_lbfgs}")
    print("PDE residual            : F = u_t + u_x + u*u_x + u_xxx + u_xxxxt")
    print("=" * 90)

    model = RosenauKdV_PINN(width=WIDTH, depth=DEPTH, use_norm=use_norm).to(DEVICE)

    x_energy, dx, E0 = build_energy_grid()

    lambda_energy = 1.0 if use_energy else 0.0

    optimizer_adam = optim.Adam(model.parameters(), lr=ADAM_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=2500, gamma=0.5)

    start_time = time.time()

    # ------------------------------
    # Adam training
    # ------------------------------
    for epoch in range(ADAM_EPOCHS):
        optimizer_adam.zero_grad()

        # PDE collocation
        t_col = (torch.rand(BATCH_SIZE, 1, device=DEVICE) * T_MAX).requires_grad_(True)
        x_col = ((torch.rand(BATCH_SIZE, 1, device=DEVICE) * 2 * L) - L).requires_grad_(True)

        residual = rosenau_residual_pinn(model, x_col, t_col)
        loss_pde = torch.mean(residual ** 2)

        # Initial condition
        t_ic = torch.zeros(BATCH_SIZE, 1, device=DEVICE)
        x_ic = (torch.rand(BATCH_SIZE, 1, device=DEVICE) * 2 * L) - L

        u_ic_pred = model(x_ic, t_ic)
        u_ic_exact = exact_rosenau_kdv(x_ic, t_ic)

        loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

        # Boundary condition
        t_bc = torch.rand(BATCH_SIZE, 1, device=DEVICE) * T_MAX

        x_left = -L * torch.ones(BATCH_SIZE, 1, device=DEVICE)
        x_right = L * torch.ones(BATCH_SIZE, 1, device=DEVICE)

        u_left_pred = model(x_left, t_bc)
        u_right_pred = model(x_right, t_bc)

        loss_bc = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)

        # Energy loss
        if use_energy:
            t_e = torch.rand(3, 1, device=DEVICE) * T_MAX

            loss_energy = 0.0
            for i in range(3):
                E_curr = compute_energy_pinn(model, x_energy, dx, t_e[i, 0])
                loss_energy = loss_energy + (E_curr - E0) ** 2

            loss_energy = loss_energy / 3.0
        else:
            loss_energy = torch.tensor(0.0, device=DEVICE)

        loss_total = (
            loss_pde
            + LAMBDA_IC * loss_ic
            + LAMBDA_BC * loss_bc
            + lambda_energy * loss_energy
        )

        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer_adam.step()
        scheduler.step()

        if epoch % 1000 == 0:
            print(
                f"{case} Adam {epoch:5d} | "
                f"PDE: {loss_pde.item():.5e} | "
                f"IC: {loss_ic.item():.5e} | "
                f"BC: {loss_bc.item():.5e} | "
                f"E: {loss_energy.item():.5e} | "
                f"Total: {loss_total.item():.5e}"
            )

    # ------------------------------
    # L-BFGS refinement
    # ------------------------------
    if use_lbfgs:
        print(f"\n{case}: Starting L-BFGS refinement...")

        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=LBFGS_MAX_ITER,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=100
        )

        t_col_f = (torch.rand(LBFGS_POINTS, 1, device=DEVICE) * T_MAX).requires_grad_(True)
        x_col_f = ((torch.rand(LBFGS_POINTS, 1, device=DEVICE) * 2 * L) - L).requires_grad_(True)

        t_ic_f = torch.zeros(LBFGS_POINTS, 1, device=DEVICE)
        x_ic_f = (torch.rand(LBFGS_POINTS, 1, device=DEVICE) * 2 * L) - L

        t_bc_f = torch.rand(LBFGS_POINTS, 1, device=DEVICE) * T_MAX
        x_left_f = -L * torch.ones(LBFGS_POINTS, 1, device=DEVICE)
        x_right_f = L * torch.ones(LBFGS_POINTS, 1, device=DEVICE)

        t_e_f = torch.rand(3, 1, device=DEVICE) * T_MAX

        lbfgs_iter = 0

        def closure():
            nonlocal lbfgs_iter

            optimizer_lbfgs.zero_grad()

            residual = rosenau_residual_pinn(model, x_col_f, t_col_f)
            loss_pde = torch.mean(residual ** 2)

            u_ic_pred = model(x_ic_f, t_ic_f)
            u_ic_exact = exact_rosenau_kdv(x_ic_f, t_ic_f)

            loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

            u_left_pred = model(x_left_f, t_bc_f)
            u_right_pred = model(x_right_f, t_bc_f)

            loss_bc = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)

            if use_energy:
                loss_energy_f = 0.0
                for i in range(3):
                    E_curr = compute_energy_pinn(model, x_energy, dx, t_e_f[i, 0])
                    loss_energy_f = loss_energy_f + (E_curr - E0) ** 2

                loss_energy_f = loss_energy_f / 3.0
            else:
                loss_energy_f = torch.tensor(0.0, device=DEVICE)

            loss_total = (
                loss_pde
                + LAMBDA_IC * loss_ic
                + LAMBDA_BC * loss_bc
                + lambda_energy * loss_energy_f
            )

            loss_total.backward()

            lbfgs_iter += 1

            if lbfgs_iter % 50 == 0:
                print(
                    f"{case} L-BFGS {lbfgs_iter:4d} | "
                    f"PDE: {loss_pde.item():.5e} | "
                    f"IC: {loss_ic.item():.5e} | "
                    f"BC: {loss_bc.item():.5e} | "
                    f"E: {loss_energy_f.item():.5e} | "
                    f"Total: {loss_total.item():.5e}"
                )

            return loss_total

        optimizer_lbfgs.step(closure)

    training_time = time.time() - start_time

    # ------------------------------
    # Metrics
    # ------------------------------
    final_l2 = evaluate_final_time_l2(model)
    err_full, err_int, margin = evaluate_full_and_interior_l2(model)
    space_time_l2 = evaluate_space_time_l2(model, Nx=600, Nt=80)
    max_energy_drift, mean_energy_drift, t_energy_diag, rel_energy_drift = evaluate_energy_drift(model, x_energy, dx, E0)
    final_pde_eval = evaluate_pde_loss_fixed(model)

    print("\n================ Rosenau-KdV ablation metrics ================")
    print(f"Case                       : {case}")
    print(f"Run name                   : {run_name}")
    print(f"Seed                       : {seed}")
    print(f"Input normalization        : {use_norm}")
    print(f"Energy loss                : {use_energy}")
    print(f"L-BFGS                     : {use_lbfgs}")
    print(f"Training time seconds      : {training_time:.2f}")
    print(f"Final-time relative L2     : {final_l2:.6e}")
    print(f"Full-domain L2 at T={T_MAX}: {err_full:.6e}")
    print(f"Interior L2 at T={T_MAX}   : {err_int:.6e}  margin={margin}")
    print(f"Space-time L2              : {space_time_l2:.6e}")
    print(f"Final PDE eval loss        : {final_pde_eval:.6e}")
    print(f"Max energy drift           : {max_energy_drift:.6e}")
    print(f"Mean energy drift          : {mean_energy_drift:.6e}")
    print("==============================================================")

    # ------------------------------
    # Save metrics to CSV
    # ------------------------------
    csv_file = "rosenau_kdv_ablation_results.csv"

    row = {
        "case": case,
        "run_name": run_name,
        "seed": seed,
        "input_normalization": use_norm,
        "energy_loss": use_energy,
        "lbfgs": use_lbfgs,
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

    save_plots(model, run_name, final_l2, margin, t_energy_diag, rel_energy_drift)

    # Cleanup
    del model
    del optimizer_adam
    del x_energy
    del E0
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# 9. RUN ALL EXPERIMENTS SEQUENTIALLY
# ============================================================

all_results = []

for config in EXPERIMENTS:
    result = run_experiment(config)
    all_results.append(result)

print("\n\n================ ALL ROSENAU-KDV RESULTS SUMMARY ================")

for row in all_results:
    print(
        f"{row['case']} | "
        f"norm={row['input_normalization']} | "
        f"energy={row['energy_loss']} | "
        f"lbfgs={row['lbfgs']} | "
        f"Final L2={row['final_l2']:.6e} | "
        f"ST L2={row['space_time_l2']:.6e} | "
        f"PDE={row['pde_eval_loss']:.6e} | "
        f"Mean drift={row['mean_energy_drift']:.6e} | "
        f"Time={row['training_time']:.2f}s"
    )

print("=================================================================")
print("Results saved to: rosenau_kdv_ablation_results.csv")