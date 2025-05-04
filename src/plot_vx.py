import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

a = 5.43e-10
V0 = 1e-19


def V(x):
    x_mod = np.mod(x + a / 2, a) - a / 2
    condition = (-a / 4 <= x_mod) & (x_mod <= a / 4)
    Vx = np.zeros_like(x)
    Vx[condition] = V0 * np.cos(2 * np.pi / a * x_mod[condition])
    return Vx


x = np.linspace(-3 * a / 2, 3 * a / 2, 1000)
y = V(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("势能分布曲线")
plt.xlabel(r"$x\; (\text{m})$")
plt.ylabel(r"$V(x)\; (\text{J})$")
plt.grid(True)
plt.tight_layout()
plt.savefig("./assets/solid_physics_project_vx.png", dpi=300, bbox_inches="tight")
plt.show()
