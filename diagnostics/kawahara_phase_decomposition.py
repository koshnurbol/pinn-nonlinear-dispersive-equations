import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)

# ==========================================
# НАСТРОЙКИ (должны совпадать с обучением)
# ==========================================
L = 20.0
T_max = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# ТОЧНОЕ РЕШЕНИЕ
# ==========================================
def exact_kawahara(x, t, x0=-5.0):
    v = 36.0 / 169.0
    amp = 105.0 / 169.0
    k = 1.0 / (2.0 * math.sqrt(13.0))
    arg = k * (x - v * t - x0)
    return amp * (1.0 / torch.cosh(arg)) ** 4


# ==========================================
# АРХИТЕКТУРА (идентична обучающему скрипту)
# ==========================================
class Kawahara_PINN(nn.Module):
    def __init__(self, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        x_norm = x / L
        t_norm = 2.0 * t / T_max - 1.0
        inp = torch.cat([x_norm, t_norm], dim=1)
        return self.net(inp)


model = Kawahara_PINN(width=256, depth=4).to(device)
model.load_state_dict(torch.load("kawahara_standard_pinn_model.pt", map_location=device))
model.eval()
print("Модель загружена из kawahara_standard_pinn_model.pt")


# ==========================================
# 1. ВЫЧИСЛЕНИЕ ОПТИМАЛЬНОГО ФАЗОВОГО СДВИГА
#    Ищем сдвиг s, минимизирующий ||u_pred(x) - u_exact(x - s)||
#    через сравнение положений пиков (высокое разрешение).
# ==========================================
N = 4000
with torch.no_grad():
    x = torch.linspace(-L, L, N).view(-1, 1).to(device)
    t = torch.full_like(x, T_max)
    u_pred = model(x, t).cpu().numpy().ravel()
    u_exact = exact_kawahara(x, t).cpu().numpy().ravel()
xg = x.cpu().numpy().ravel()

# положение пиков
i_pred = np.argmax(u_pred)
i_exact = np.argmax(u_exact)
peak_shift = xg[i_pred] - xg[i_exact]
amp_pred = u_pred[i_pred]
amp_exact = u_exact[i_exact]

print("\n--- Пик / амплитуда / фаза ---")
print(f"Пик exact   : A={amp_exact:.5f} @ x={xg[i_exact]:.4f}")
print(f"Пик PINN    : A={amp_pred:.5f} @ x={xg[i_pred]:.4f}")
print(f"Ошибка амплитуды : {abs(amp_pred-amp_exact)/amp_exact*100:.4f}%")
print(f"Фазовый сдвиг    : dx = {peak_shift:+.4f}")

# уточняем сдвиг минимизацией L2 по сетке кандидатов вокруг peak_shift
shift_candidates = np.linspace(peak_shift - 0.5, peak_shift + 0.5, 2001)
best_shift, best_err = peak_shift, np.inf
for s in shift_candidates:
    with torch.no_grad():
        xs = torch.linspace(-L, L, N).view(-1, 1).to(device)
        ts = torch.full_like(xs, T_max)
        u_e_shift = exact_kawahara(xs - s, ts).cpu().numpy().ravel()
    e = np.linalg.norm(u_pred - u_e_shift) / np.linalg.norm(u_e_shift)
    if e < best_err:
        best_err, best_shift = e, s

print(f"Оптимальный сдвиг (L2-min): s = {best_shift:+.4f}")


# ==========================================
# 2. РАЗЛОЖЕНИЕ ОШИБКИ
#    err_raw      = u_pred - u_exact          (полная ошибка)
#    err_dephased = u_pred - u_exact(x - s*)  (после снятия фазы)
# ==========================================
with torch.no_grad():
    xs = torch.linspace(-L, L, N).view(-1, 1).to(device)
    ts = torch.full_like(xs, T_max)
    u_e_shift = exact_kawahara(xs - best_shift, ts).cpu().numpy().ravel()

err_raw = u_pred - u_exact
err_dephased = u_pred - u_e_shift

l2_raw = np.linalg.norm(err_raw) / np.linalg.norm(u_exact)
l2_dephased = np.linalg.norm(err_dephased) / np.linalg.norm(u_e_shift)

print("\n--- L2 ошибки до/после снятия фазы ---")
print(f"L2 полная ошибка        : {l2_raw:.4e}")
print(f"L2 после снятия фазы    : {l2_dephased:.4e}")
print(f"Доля ошибки от фазы     : {(1 - l2_dephased/l2_raw)*100:.1f}%")


# ==========================================
# 3. СПЕКТР ДО И ПОСЛЕ СНЯТИЯ ФАЗЫ
# ==========================================
dx_fft = (2.0 * L) / (N - 1)
k = np.fft.rfftfreq(N, d=dx_fft) * 2.0 * np.pi

P_exact = np.abs(np.fft.rfft(u_exact)) ** 2
P_raw = np.abs(np.fft.rfft(err_raw)) ** 2
P_deph = np.abs(np.fft.rfft(err_dephased)) ** 2


def centroid(P):
    return float(np.sum(k * P) / (np.sum(P) + 1e-30))


c_exact = centroid(P_exact)
c_raw = centroid(P_raw)
c_deph = centroid(P_deph)

print("\n--- Спектральные центроиды ---")
print(f"Центроид решения              : k = {c_exact:.4f}")
print(f"Центроид ошибки (raw)         : k = {c_raw:.4f}   ratio={c_raw/c_exact:.2f}")
print(f"Центроид ошибки (без фазы)    : k = {c_deph:.4f}   ratio={c_deph/c_exact:.2f}")
print(f"Снижение центроида ошибки     : {(1 - c_deph/c_raw)*100:.1f}%")

hi = k > 2 * c_exact
print(f"\nHigh-k доля энергии:")
print(f"  решение            : {np.sum(P_exact[hi])/np.sum(P_exact):.4f}")
print(f"  ошибка raw         : {np.sum(P_raw[hi])/np.sum(P_raw):.4f}")
print(f"  ошибка без фазы    : {np.sum(P_deph[hi])/np.sum(P_deph):.4f}")


# ==========================================
# 4. ГРАФИК
# ==========================================
P_exact_n = P_exact / (P_exact.max() + 1e-30)
P_raw_n = P_raw / (P_raw.max() + 1e-30)
P_deph_n = P_deph / (P_deph.max() + 1e-30)

plt.figure(figsize=(8, 5))
plt.semilogy(k, P_exact_n, 'k-', lw=2, label='Exact solution')
plt.semilogy(k, P_raw_n, 'r--', lw=2, label='Error (raw)')
plt.semilogy(k, P_deph_n, 'b-.', lw=2, label='Error (phase removed)')
plt.axvline(c_exact, color='k', ls=':', alpha=0.6)
plt.axvline(c_raw, color='r', ls=':', alpha=0.6)
plt.axvline(c_deph, color='b', ls=':', alpha=0.6)
plt.xlabel("Wavenumber $k$")
plt.ylabel("Normalized power spectrum")
plt.title("Kawahara: error spectrum before/after phase removal")
plt.xlim(0, 10)
plt.ylim(1e-12, 2.0)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("kawahara_error_spectrum_dephased.png", dpi=150)
plt.show()

print("\nГрафик сохранён: kawahara_error_spectrum_dephased.png")