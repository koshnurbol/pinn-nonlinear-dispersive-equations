import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import time

torch.manual_seed(42)


# ==========================================
# 1. ТОЧНОЕ РЕШЕНИЕ КАВАХАРЫ
# ==========================================
def exact_kawahara(x, t, x0=-5.0):
    """
    Exact solitary wave for:
        u_t + u*u_x + u_xxx - u_xxxxx = 0
    """
    v = 36.0 / 169.0
    amp = 105.0 / 169.0
    k = 1.0 / (2.0 * math.sqrt(13.0))

    arg = k * (x - v * t - x0)
    return amp * (1.0 / torch.cosh(arg)) ** 4


# ==========================================
# 2. STANDARD PINN АРХИТЕКТУРА
# ==========================================
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

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        # Нормализация входов
        x_norm = x / L
        t_norm = 2.0 * t / T_max - 1.0

        inp = torch.cat([x_norm, t_norm], dim=1)
        return self.net(inp)


# ==========================================
# 3. НАСТРОЙКИ
# ==========================================
L = 20.0
T_max = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Kawahara_PINN(width=256, depth=4).to(device)

# ВНИМАНИЕ:
# Для u_xxxxx standard PINN очень тяжёлый.
# Если будет CUDA out of memory, уменьши batch_size и lbfgs_points.
batch_size = 1000
lbfgs_points = 1500

lambda_ic = 20.0
lambda_bc = 1.0
lambda_energy = 1.0

adam_epochs = 10000
adam_lr = 0.002
lbfgs_max_iter = 500

print(f"Запуск standard PINN для Кавахары на {device}")
print("PDE residual: F = u_t + u*u_x + u_xxx - u_xxxxx")


# ==========================================
# 4. AD-ПРОИЗВОДНЫЕ
# ==========================================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


def kawahara_residual_pinn(x, t):
    """
    Здесь standard PINN решает PDE Кавахары:

        F = u_t + u*u_x + u_xxx - u_xxxxx

    ВАЖНО:
    Все производные считаются через AD.
    """
    u = model(x, t)

    u_t = grad(u, t)

    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    u_xxx = grad(u_xx, x)
    u_xxxx = grad(u_xxx, x)
    u_xxxxx = grad(u_xxxx, x)

    residual = u_t + u * u_x + u_xxx - u_xxxxx

    return residual


# ==========================================
# 5. ЭНЕРГИЯ
# ==========================================
x_energy = torch.linspace(-L, L, 500).view(-1, 1).to(device)
dx = x_energy[1] - x_energy[0]

with torch.no_grad():
    u_initial_exact = exact_kawahara(x_energy, torch.zeros_like(x_energy))
    E0 = torch.sum(u_initial_exact ** 2) * dx


def compute_energy_pinn(t_scalar):
    if not torch.is_tensor(t_scalar):
        t_scalar = torch.tensor(t_scalar, device=device)

    t_curr = torch.full_like(x_energy, t_scalar)
    u_curr = model(x_energy, t_curr)

    E_curr = torch.sum(u_curr ** 2) * dx
    return E_curr


# ==========================================
# 6. ОБУЧЕНИЕ ADAM
# ==========================================
optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=2500, gamma=0.5)

history_pde = []
history_ic = []
history_bc = []
history_energy = []
history_total = []

start_time = time.time()

for epoch in range(adam_epochs):
    optimizer_adam.zero_grad()

    # ------------------------------
    # PDE collocation
    # ------------------------------
    t_col = (torch.rand(batch_size, 1, device=device) * T_max).requires_grad_(True)
    x_col = ((torch.rand(batch_size, 1, device=device) * 2 * L) - L).requires_grad_(True)

    residual = kawahara_residual_pinn(x_col, t_col)
    loss_pde = torch.mean(residual ** 2)

    # ------------------------------
    # Initial condition
    # ------------------------------
    t_ic = torch.zeros(batch_size, 1, device=device)
    x_ic = (torch.rand(batch_size, 1, device=device) * 2 * L) - L

    u_ic_pred = model(x_ic, t_ic)
    u_ic_exact = exact_kawahara(x_ic, t_ic)

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
    t_e = torch.rand(3, 1, device=device) * T_max

    loss_energy = 0.0
    for i in range(3):
        E_curr = compute_energy_pinn(t_e[i, 0])
        loss_energy = loss_energy + (E_curr - E0) ** 2

    loss_energy = loss_energy / 3.0

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

    history_pde.append(loss_pde.item())
    history_ic.append(loss_ic.item())
    history_bc.append(loss_bc.item())
    history_energy.append(loss_energy.item())
    history_total.append(loss_total.item())

    if epoch % 1000 == 0:
        print(
            f"Adam {epoch:5d} | "
            f"PDE: {loss_pde.item():.5e} | "
            f"IC: {loss_ic.item():.5e} | "
            f"BC: {loss_bc.item():.5e} | "
            f"E: {loss_energy.item():.5e} | "
            f"Total: {loss_total.item():.5e}"
        )


# ==========================================
# 7. L-BFGS ПОЛИРОВКА
# ==========================================
print("\nПолировка L-BFGS для standard PINN...")

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
    residual = kawahara_residual_pinn(x_col_f, t_col_f)
    loss_pde = torch.mean(residual ** 2)

    # IC loss
    u_ic_pred = model(x_ic_f, t_ic_f)
    u_ic_exact = exact_kawahara(x_ic_f, t_ic_f)

    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

    # BC loss
    u_left_pred = model(x_left_f, t_bc_f)
    u_right_pred = model(x_right_f, t_bc_f)

    loss_bc = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)

    # Energy loss
    loss_energy_f = 0.0
    for i in range(3):
        E_curr = compute_energy_pinn(t_e_f[i, 0])
        loss_energy_f = loss_energy_f + (E_curr - E0) ** 2

    loss_energy_f = loss_energy_f / 3.0

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

print("Обучение standard PINN завершено!")
print(f"Время обучения: {training_time:.2f} секунд")


# ==========================================
# 8. МЕТРИКИ
# ==========================================
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
        u_exact = exact_kawahara(x_test, t_test)

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
        u_exact = exact_kawahara(x_diag, t_fin)

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
            u_exact = exact_kawahara(x_grid, t_col)

            errors.append((u_pred - u_exact).reshape(-1))
            exacts.append(u_exact.reshape(-1))

        errors = torch.cat(errors)
        exacts = torch.cat(exacts)

        st_l2 = torch.norm(errors, 2) / torch.norm(exacts, 2)

    return st_l2.item()


def evaluate_energy_drift():
    model.eval()

    with torch.no_grad():
        t_energy_diag = torch.linspace(0.0, T_max, 80)
        energy_values = []

        for tv in t_energy_diag:
            t_curr = torch.full_like(x_energy, tv.item())
            u_curr = model(x_energy, t_curr)

            E_curr = torch.sum(u_curr ** 2) * dx
            energy_values.append(E_curr.item())

        energy_values = np.array(energy_values)
        E0_np = E0.item()

        rel_energy_drift = np.abs(energy_values - E0_np) / np.abs(E0_np)

        max_energy_drift = np.max(rel_energy_drift)
        mean_energy_drift = np.mean(rel_energy_drift)

    return max_energy_drift, mean_energy_drift, t_energy_diag, rel_energy_drift


def evaluate_pde_loss_fixed(N_eval=2000):
    """
    Финальная PDE residual loss на новых точках.
    ВНИМАНИЕ: здесь снова считается u_xxxxx через AD.
    """
    model.eval()

    t_eval = (torch.rand(N_eval, 1, device=device) * T_max).requires_grad_(True)
    x_eval = ((torch.rand(N_eval, 1, device=device) * 2 * L) - L).requires_grad_(True)

    residual = kawahara_residual_pinn(x_eval, t_eval)
    loss_pde_eval = torch.mean(residual ** 2)

    return loss_pde_eval.item()


final_l2 = evaluate_final_time_l2()
err_full, err_int, margin = evaluate_full_and_interior_l2()
space_time_l2 = evaluate_space_time_l2(Nx=600, Nt=80)
max_energy_drift, mean_energy_drift, t_energy_diag, rel_energy_drift = evaluate_energy_drift()
final_pde_eval = evaluate_pde_loss_fixed(N_eval=2000)

print("\n================ Kawahara standard PINN metrics ================")
print(f"Training time seconds       : {training_time:.2f}")
print(f"Final-time relative L2      : {final_l2:.6e}")
print(f"Full-domain L2 at T={T_max} : {err_full:.6e}")
print(f"Interior L2 at T={T_max}    : {err_int:.6e}  margin={margin}")
print(f"Space-time L2               : {space_time_l2:.6e}")
print(f"Final PDE eval loss         : {final_pde_eval:.6e}")
print(f"Max energy drift            : {max_energy_drift:.6e}")
print(f"Mean energy drift           : {mean_energy_drift:.6e}")
print("=================================================================")


# ==========================================================
# 8b. ФУРЬЕ-АНАЛИЗ СПЕКТРА ОШИБКИ  (ЭТАП 3)
#     Проверка тезиса о spectral bias:
#     где по волновым числам сосредоточена ошибка?
# ==========================================================
def error_spectrum(exact_fn, N=2048):
    """
    Считает пространственный спектр ошибки и точного решения при t = T_max.
    Использует глобальные L, T_max, device, model.
    Возвращает:
        k          - угловые волновые числа (rad/unit length)
        P_err      - сырой спектр мощности ошибки
        P_exact    - сырой спектр мощности точного решения
    """
    model.eval()
    with torch.no_grad():
        x = torch.linspace(-L, L, N).view(-1, 1).to(device)
        t = torch.full_like(x, T_max)
        u_pred = model(x, t).cpu().numpy().ravel()
        u_exact = exact_fn(x, t).cpu().numpy().ravel()

    err = u_pred - u_exact
    dx_fft = (2.0 * L) / (N - 1)

    P_err = np.abs(np.fft.rfft(err)) ** 2
    P_exact = np.abs(np.fft.rfft(u_exact)) ** 2
    k = np.fft.rfftfreq(N, d=dx_fft) * 2.0 * np.pi  # угловые волновые числа

    return k, P_err, P_exact


def spectral_centroid(k, P):
    """Центр тяжести спектра по волновому числу: sum(k*P)/sum(P)."""
    return float(np.sum(k * P) / (np.sum(P) + 1e-30))


# --- вычисление ---
k_fft, P_err, P_exact = error_spectrum(exact_kawahara, N=2048)

centroid_err = spectral_centroid(k_fft, P_err)
centroid_exact = spectral_centroid(k_fft, P_exact)
ratio = centroid_err / (centroid_exact + 1e-30)

# доля энергии ошибки в "высоких" частотах (k > 2*centroid_exact)
hi_mask = k_fft > (2.0 * centroid_exact)
hi_frac_err = float(np.sum(P_err[hi_mask]) / (np.sum(P_err) + 1e-30))
hi_frac_exact = float(np.sum(P_exact[hi_mask]) / (np.sum(P_exact) + 1e-30))

print("\n--- Error spectrum diagnostics (Kawahara) ---")
print(f"Spectral centroid of ERROR    : k = {centroid_err:.4f}")
print(f"Spectral centroid of SOLUTION : k = {centroid_exact:.4f}")
print(f"Ratio (error / solution)      : {ratio:.3f}")
print(f"High-k energy fraction ERROR  : {hi_frac_err:.4f}")
print(f"High-k energy fraction SOLUT. : {hi_frac_exact:.4f}")
print("  (high-k defined as k > 2 * solution centroid)")

# --- график (нормированные на максимум для сравнения формы) ---
P_err_n = P_err / (P_err.max() + 1e-30)
P_exact_n = P_exact / (P_exact.max() + 1e-30)

plt.figure(figsize=(8, 5))
plt.semilogy(k_fft, P_exact_n, 'k-', lw=2, label='Exact solution spectrum')
plt.semilogy(k_fft, P_err_n, 'r--', lw=2, label='Error spectrum')
plt.axvline(centroid_err, color='r', ls=':', alpha=0.7,
            label=f'Error centroid k={centroid_err:.2f}')
plt.axvline(centroid_exact, color='k', ls=':', alpha=0.7,
            label=f'Solution centroid k={centroid_exact:.2f}')
plt.xlabel("Wavenumber $k$")
plt.ylabel("Normalized power spectrum")
plt.title("Kawahara: spectral content of error vs solution")
plt.xlim(0, 10)
plt.ylim(1e-12, 2.0)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("kawahara_error_spectrum.png", dpi=150)
plt.show()

# дозапись метрик спектра в файл
with open("kawahara_standard_pinn_metrics.txt", "a", encoding="utf-8") as f:
    f.write("\n--- Error spectrum diagnostics ---\n")
    f.write(f"Spectral centroid of error    = {centroid_err:.6f}\n")
    f.write(f"Spectral centroid of solution = {centroid_exact:.6f}\n")
    f.write(f"Ratio error/solution          = {ratio:.6f}\n")
    f.write(f"High-k energy fraction error  = {hi_frac_err:.6f}\n")
    f.write(f"High-k energy fraction solut. = {hi_frac_exact:.6f}\n")


# ==========================================
# 9. ГРАФИКИ (профили, L2(t), energy drift)
# ==========================================
x_test = torch.linspace(-L, L, 1000).view(-1, 1).to(device)

# L2(t)
model.eval()

with torch.no_grad():
    t_grid = torch.linspace(0.0, T_max, 80)
    interior = (x_test.squeeze() > -L + margin) & (x_test.squeeze() < L - margin)

    l2_full_t = []
    l2_int_t = []

    for tv in t_grid:
        t_col = torch.full_like(x_test, tv.item())

        u_pred = model(x_test, t_col)
        u_exact = exact_kawahara(x_test, t_col)

        l2_full_t.append(relative_l2(u_pred, u_exact))
        l2_int_t.append(relative_l2(u_pred, u_exact, interior))


plt.figure(figsize=(8, 5))
plt.plot(t_grid.cpu().numpy(), l2_full_t, 'r-o', ms=3, label='Full domain')
plt.plot(t_grid.cpu().numpy(), l2_int_t, 'b-s', ms=3, label=f'Interior (|x|<{L-margin})')
plt.xlabel("t")
plt.ylabel("Relative $L_2$ error")
plt.title("Kawahara standard PINN: error evolution in time")
plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("kawahara_standard_pinn_l2_time.png", dpi=150)
plt.show()


# Energy drift
plt.figure(figsize=(8, 5))
plt.plot(t_energy_diag.cpu().numpy(), rel_energy_drift, 'k-o', ms=3)
plt.xlabel("t")
plt.ylabel("Relative energy drift")
plt.title("Kawahara standard PINN: energy drift")
plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("kawahara_standard_pinn_energy_drift.png", dpi=150)
plt.show()


# Profiles
plt.figure(figsize=(12, 6))

times_to_plot = [0.0, T_max / 2.0, T_max]
colors = ['blue', 'green', 'red']

with torch.no_grad():
    for i, t_val in enumerate(times_to_plot):
        t_plot = torch.full_like(x_test, t_val)

        u_pred_plot = model(x_test, t_plot)
        u_exact_plot = exact_kawahara(x_test, t_plot)

        plt.plot(
            x_test.cpu(),
            u_exact_plot.cpu(),
            color=colors[i],
            linestyle='dashed',
            linewidth=2,
            label=f'Exact t={t_val:.2f}'
        )

        plt.plot(
            x_test.cpu(),
            u_pred_plot.cpu(),
            color=colors[i],
            linewidth=2,
            label=f'Standard PINN t={t_val:.2f}'
        )

plt.title(f"Kawahara equation: standard PINN, L2 error={final_l2:.4e}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kawahara_standard_pinn_profiles.png", dpi=150)
plt.show()


# ==========================================
# 10. СОХРАНЕНИЕ
# ==========================================
torch.save(model.state_dict(), "kawahara_standard_pinn_model.pt")

print("\nМетрики и графики сохранены:")
print(" - kawahara_standard_pinn_metrics.txt")
print(" - kawahara_error_spectrum.png   <-- НОВЫЙ (Этап 3)")
print(" - kawahara_standard_pinn_l2_time.png")
print(" - kawahara_standard_pinn_energy_drift.png")
print(" - kawahara_standard_pinn_profiles.png")