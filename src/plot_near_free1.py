import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

hbar = 1.055e-34
e_0 = 1.6e-19
m_0 = 9.1e-31
a = 5.43e-10
V_0 = 1e-19


def V(x):
    x_mod = np.mod(x + a / 2, a) - a / 2
    if -a / 4 <= x_mod <= a / 4:
        Vx = V_0 * np.cos(2 * np.pi / a * x_mod)
    else:
        Vx = 0
    return Vx


# 计算势能的傅里叶系数
def v_fourier(g, epsabs=1e-12, epsrel=1e-12):
    L = a / 4

    def func(x):
        return V_0 / a * np.cos(2 * np.pi * x / a) * np.cos(g * x)

    result, _ = quad(func, -L, L, epsabs=epsabs, epsrel=epsrel)

    return result


"""先画右边部分"""

# 动量基底
N = 5  # 截断G的个数（-5到5共11个）
G_array = np.array([n * 2 * np.pi / a for n in range(-N, N + 1)])

N_k = 200
Delta_k = (2 * np.pi / a) / 2
k_center = np.pi / a
k_arr1 = np.linspace(k_center - Delta_k, k_center + Delta_k, N_k)
E_bands = np.zeros((2 * N + 1, N_k))

for idx, k in enumerate(k_arr1):
    # 构造哈密顿量
    H = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)
    for i, G_1 in enumerate(G_array):
        for j, G_2 in enumerate(G_array):
            if i == j:
                H[i, j] = hbar**2 / (2 * m_0) * (k + G_1) ** 2
            H[i, j] += v_fourier(G_1 - G_2)
    eigs = np.linalg.eigh(H)[0]
    eigs = np.asarray(eigs, dtype=float)
    E_bands[:, idx] = np.sort(eigs)

# 将能量移动到原点
E_bands -= 3.18e-20
E_eig1_1 = E_bands[0, :]
E_eig2_1 = E_bands[1, :]

# 近自由电子近似
G_0 = 2 * np.pi / a
V_G0 = v_fourier(G_0)
E_free_1 = hbar**2 * k_arr1**2 / (2 * m_0)

E_avg_1 = hbar**2 * (k_arr1**2 + (k_arr1 - G_0) ** 2) / (4 * m_0)
Delta_E_1 = hbar**2 * (k_arr1**2 - (k_arr1 - G_0) ** 2) / (4 * m_0)
E_m_1 = E_avg_1 - np.sqrt(Delta_E_1**2 + V_G0**2)
E_p_1 = E_avg_1 + np.sqrt(Delta_E_1**2 + V_G0**2)


"""再画左边部分"""

k_center = -np.pi / a
k_arr2 = np.linspace(k_center - Delta_k, k_center + Delta_k, N_k)
E_bands = np.zeros((2 * N + 1, N_k))

for idx, k in enumerate(k_arr2):
    # 构造哈密顿量
    H = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)
    for i, G_1 in enumerate(G_array):
        for j, G_2 in enumerate(G_array):
            if i == j:
                H[i, j] = hbar**2 / (2 * m_0) * (k + G_1) ** 2
            H[i, j] += v_fourier(G_1 - G_2)
    eigs = np.linalg.eigh(H)[0]
    eigs = np.asarray(eigs, dtype=float)
    E_bands[:, idx] = np.sort(eigs)

# 将能量移动到原点
E_bands -= 3.18e-20
E_eig1_2 = E_bands[0, :]
E_eig2_2 = E_bands[1, :]

# 近自由电子近似
G_0 = 2 * np.pi / a
V_G0 = v_fourier(G_0)
E_free_2 = hbar**2 * k_arr2**2 / (2 * m_0)

E_avg_2 = hbar**2 * (k_arr2**2 + (k_arr2 + G_0) ** 2) / (4 * m_0)
Delta_E_2 = hbar**2 * (k_arr2**2 - (k_arr2 + G_0) ** 2) / (4 * m_0)
E_m_2 = E_avg_2 - np.sqrt(Delta_E_2**2 + V_G0**2)
E_p_2 = E_avg_2 + np.sqrt(Delta_E_2**2 + V_G0**2)

# 画出前两个能带
plt.figure(figsize=(15, 5))

plt.plot(
    k_arr1[0 : N_k // 2] * a / np.pi,
    E_eig1_1[0 : N_k // 2] / e_0,
    label="Band 1 (特征根法)",
)
plt.plot(
    k_arr2[N_k // 2 :] * a / np.pi,
    E_eig1_2[N_k // 2 :] / e_0,
    label="_nolegend_",
    color="C0",
)
plt.plot(
    k_arr1[N_k // 2 :] * a / np.pi,
    E_eig2_1[N_k // 2 :] / e_0,
    label="Band 2 (特征根法)",
)
plt.plot(
    k_arr2[0 : N_k // 2] * a / np.pi,
    E_eig2_2[0 : N_k // 2] / e_0,
    label="_nolegend_",
    color="C1",
)

plt.plot(k_arr1 * a / np.pi, E_free_1 / e_0, "--", label="自由电子")
plt.plot(k_arr2 * a / np.pi, E_free_2 / e_0, "--", label="_nolegend_", color="C2")

plt.plot(
    k_arr1[0 : N_k // 2] * a / np.pi,
    E_m_1[0 : N_k // 2] / e_0,
    ":",
    label="Band 1 (近自由电子近似)",
)
plt.plot(
    k_arr2[N_k // 2 :] * a / np.pi,
    E_m_2[N_k // 2 :] / e_0,
    ":",
    label="_nolegend_",
    color="C3",
)
plt.plot(
    k_arr1[N_k // 2 :] * a / np.pi,
    E_p_1[N_k // 2 :] / e_0,
    ":",
    label="Band 2 (近自由电子近似)",
)
plt.plot(
    k_arr2[0 : N_k // 2] * a / np.pi,
    E_p_2[0 : N_k // 2] / e_0,
    ":",
    label="_nolegend_",
    color="C4",
)

plt.xlabel(r"k ($\frac{\pi}{a}$)")
plt.ylabel("E (eV)")
plt.title("布里渊区边界能带曲线对比")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(
#     "./assets/solid_physics_project_near_free1.png", dpi=300, bbox_inches="tight"
# )
plt.show()
