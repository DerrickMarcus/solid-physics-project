# 固体物理大作业

> 姓名：陈彦旭
>
> 班级：无24

## 1. 绘制势能分布曲线

先将位置坐标 $x$ 归一化到以原点为中心的第一个周期内，然后根据范围确定势能。对应函数大概如下：

```python
def V(x):
    x_mod = np.mod(x + a / 2, a) - a / 2
    condition = (-a / 4 <= x_mod) & (x_mod <= a / 4)
    Vx = np.zeros_like(x)
    Vx[condition] = V0 * np.cos(2 * np.pi / a * x_mod[condition])
    return Vx
```

运行 `src/plot_vx.py` 文件，画出三个周期内的势能分布曲线。每一个周期内，中心部分为上半平面余弦函数，左右两侧势能为0。

![solid_physics_project_vx1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/solid_physics_project_vx1.png)



## 2. 采用特征根法求解能带

参考如下方法：

![solid_physics_project_sspband_01](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/solid_physics_project_sspband_01.png)

在周期势场 $V(x+a)=V(x)$ 下，Bloch 函数为 $\psi(x)=e^{ikx}u_k(x),\,u_k(x+a)=u_k(x)$  。

以自由电子的平面波为完备正交基，把布洛赫函数展开为：
$$
\psi_k(x)=\sum_{G_n} C_{k,G_n}e^{i(k+G_n)x}
$$

其中 $G_n=\dfrac{2\pi n}{a}, n\in \mathbb{Z}$ ，为倒格矢的整数倍。

带入薛定谔方程，得到矩阵本征方程为：

$$
\sum_{G_l}H_{G_n,G_l}(k)C_{k,G_l}=E(k)C_{k,G_l}
$$

第 $n$ 行第 $l$ 列的矩阵元为：
$$
H_{nl}=\dfrac{\hbar^2}{2m}(k+G_n)^2\delta_{nl}+v(G_n-G_l)
$$
其中第一项代表动能，只有 $n=l$ 时才存在。第二项代表势能耦合，实际上是势能的傅里叶展开系数 $v(g), g=G_n-G_l$ 。
$$
v(g)=\frac{1}{a}\int_{-a/2}^{a/2}V(x)\exp(-igx)\mathrm{d}x\\
=\frac{1}{a}\int_{-a/2}^{a/2}V(x)\cos(gx)\mathrm{d}x\\
=\frac{V_0}{a}\int_{-a/4}^{a/4}\cos\frac{2\pi x}{a}\cos(gx)\mathrm{d}x\\
=\frac{V_0}{2a}\int_{-a/4}^{a/4}\left(\cos(\frac{2\pi x}{a}+gx)+\cos(\frac{2\pi x}{a}-gx)\right)\mathrm{d}x
$$

其中势能的最大值为 $V_0=10^{-19}\text{J}$ 。第二步的依据是势能 $V(x)$ 为偶函数。

最终有：
$$
v(g)=
\begin{cases}
\dfrac{V_0}{4}, & g=\pm\cfrac{2\pi}{a}\\
\dfrac{4\pi V_0}{4\pi^2-g^2a^2}\cos(\dfrac{ga}{4}), & g\ne\pm\cfrac{2\pi}{a}
\end{cases}
$$

由于该本征方程是无穷阶的，需要选取 $N$ 截断，只保留 $n=-N,...,N$ 共 $2N+1$ 个分量，求有限个 $G_n$ 情况下的本征方程。我截取的范围是 $N=5,[-5, 5]$ 内的11个分量，可以计算出11条能带。

对于第一布里渊区内的每一个波矢 $k \in [-\dfrac{\pi}{a},\dfrac{\pi}{a}]$ ，计算它对应的哈密顿量矩阵，该矩阵的本征值就是该波矢的前若干个能量本征值。各个 $k$ 值下的本征值排列即可得到简约布里渊区图景。

根据以上计算编写代码，运行 `src/plot_eigs.py` ，在简约布里渊区绘制出前5个能带为：

![solid_physics_project_eig_bands](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/solid_physics_project_eig_bands.png)

计算得到前5个能带的4个能隙为：

```text
Band 0 to Band 1 gap: 0.3084 eV
Band 1 to Band 2 gap: 0.1510 eV
Band 2 to Band 3 gap: 0.0046 eV
Band 3 to Band 4 gap: 0.1061 eV
```



## 3. 近自由电子近似

在布里渊区边界处 $k=+\dfrac{\pi}{a},k'=k-G_0=-\dfrac{\pi}{a}$ 时，即相差一个倒格矢，态 $\psi_k^0(x),\psi_{k'}^0(x)$ 能量相同，在周期形式作用下两个态耦合，发生简并微扰。因此在布里渊区边界附近，需要使用简并微扰论求解。

求解行列式：
$$
\begin{vmatrix}
E_k^0-E & V_n^* \\
V_n & E_{k'}^*-E
\end{vmatrix}=0
$$
得到两个能量本征值为：
$$
E_{\pm}=\frac{E_k^0+E_{k'}^0}{2}\pm\sqrt{\left(\frac{E_k^0-E_{k'}^0}{2}\right)^2+|V_n|^2}
$$
在布里渊区边界处，有 $k=+\dfrac{\pi}{a},k'=-\dfrac{\pi}{a},\;E_k^0=E_{k'}^0=E^0$ 得到：
$$
E_{\pm} =
\begin{cases}
E^0+|V_n| \\
E^0-|V_n|
\end{cases}
$$
能隙即为 $2|V_n|$ 。又因为 $V_n$ 为势能 $V(x)$ 展开为傅里叶级数的系数：
$$
V_n=\frac{1}{a}\int_{-a/2}^{a/2}V(x)\exp(-i\frac{2\pi nx}{a})\mathrm{d}x\\
=\frac{1}{a}\int_{-a/2}^{a/2}V(x)\cos\frac{2\pi nx}{a}\mathrm{d}x\\
=\frac{V_0}{a}\int_{-a/4}^{a/4}\cos\frac{2\pi x}{a}\cos\frac{2\pi nx}{a}\mathrm{d}x\\
=\frac{V_0}{2a}\int_{-a/4}^{a/4}\left(\cos\frac{2\pi x(1+n)}{a}+\cos\frac{2\pi x(1-n)}{a}\right)\mathrm{d}x
$$

得到：
$$
V_1=\frac{V_0}{4}\\
V_2=\frac{V_0}{3\pi}\\
V_3=0\\
V_4=-\frac{V_0}{15\pi}
$$
由此可得前4个能隙为：
$$
2|V_1|=0.5\times 10^{-19}\text{J}=0.3125\text{eV}\\
2|V_2|=\frac{2}{3\pi}\times 10^{-19}\text{J}\approx 0.1326\text{eV}\\
2|V_3|=0\\
2|V_4|=\frac{2}{15\pi}\times 10^{-19}\text{J}\approx 0.0265\text{eV}
$$



与特征根法求解的结果相比，前3个能隙相差不大，第4个能隙结果相差较大。



在第一布里渊区和第二布里渊区边界附近， $\Delta k=\pm\dfrac{1}{10}\dfrac{2\pi}{a}$ 范围内，绘制**扩展布里渊区图景**下的前两个能带曲线，将特征根法、近自由电子近似与自由电子的情况进行对比：

![solid_physics_project_near_free1](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo_image/images/solid_physics_project_near_free1.png)

可见，在近似自由电子近似下，两条能带曲线以自由电子在布里渊区边界处的能量值 $E^0$ 为中心，形成能级劈裂，且偏离值相同。但是在特征根求解下，两条能态曲线虽然也形成能级劈裂，但是中心值不再自由电子能量处，而是能量均高于自由电子能量。
