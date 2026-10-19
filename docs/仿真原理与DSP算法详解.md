# 光纤通信仿真原理与接收端 DSP 算法详解

本文档配合 `Python/optsim` 与 `MATLAB` 两套实现阅读，详细说明仿真链路每一级的物理模型与数学原理、接收端数字信号处理（DSP, Digital Signal Processing）算法链的具体流程，并给出关键步骤的公式推导、伪代码与对应的实际代码片段，供学习与二次开发参考。配套的分步教学 Notebook 见 `Python/notebooks/01_相干DP-QAM全流程教程.ipynb`。

> 记号约定：标量用斜体（如 $\beta_2$），复包络（complex envelope）用大写 $E$ 或 $x$，向量/矩阵用粗体。全文公式默认采用**工程约定**（engineering convention）$E(t)\cdot e^{+j\omega_0 t}$，即基带正频率对应更高的光频率（更短波长）；这与 `Agrawal` 教材中物理约定 $e^{-i\omega_0 t}$ 符号相反，使用时需注意。

---

## 目录

1. [总体架构与信号流](#1-总体架构与信号流)
2. [仿真基本约定](#2-仿真基本约定)
3. [发射端数字信号处理（Tx DSP）](#3-发射端数字信号处理tx-dsp)
4. [发射端光电器件模型](#4-发射端光电器件模型)
5. [光纤信道传输模型](#5-光纤信道传输模型)
6. [WDM 系统与 GN 非线性模型](#6-wdm-系统与-gn-非线性模型)
7. [接收端光电器件模型](#7-接收端光电器件模型)
8. [接收端 DSP 算法链详解](#8-接收端-dsp-算法链详解)
9. [IM-DD 与自相干接收](#9-im-dd-与自相干接收)
10. [性能指标](#10-性能指标)
11. [神经网络均衡器与学习型 DBP（简介）](#11-神经网络均衡器与学习型-dbp简介)
12. [参考文献](#12-参考文献)

---

## 1. 总体架构与信号流

单信道相干偏振复用（PDM, Polarization-Division Multiplexing）QAM（Quadrature Amplitude Modulation，正交幅度调制）系统的完整链路如下（对应 `examples/coherent_dp_qam.py` / `main_coherent_dp_qam.m`）：

```
比特 ──Tx DSP──► 基带波形 ──DAC──► 模拟驱动信号
                                        │
                        激光器(CW) ──► DP-IQ 调制器 ──► 功率放大器(EDFA)
                                                              │
                                                     光纤链路(N 跨段 SSFM+放大器)
                                                              │
                                         WSS 滤波 ◄── 偏振相关损耗(PDL)
                                              │
                        本振激光器 ──► 相干接收机(90°混频器+平衡探测器)
                                              │
                                            ADC
                                              │
                                          Rx DSP 链
                                              │
                                      判决 ──► BER/SNR/EVM/GMI
```

每一级都有对应的仿真函数（Python / MATLAB 对照）：

| 阶段 | 作用 | Python | MATLAB |
|---|---|---|---|
| 比特与映射 | 产生比特、QAM 映射、成帧、脉冲成形 | `tx.py` | `Tx/` |
| 电域器件 | DAC/ADC | `devices.py` | `Components/` |
| 光域器件 | 激光器、调制器、放大器、接收机 | `devices.py` | `Components/` |
| 光纤传输 | 色散、非线性、PMD、多跨链路 | `channel.py` | `Channel/` |
| 接收 DSP | 本文档第 8 节 | `dsp.py` | `DSP/` |
| 系统封装 | 把以上串成一条链 | `system.py` | `System/` |
| 性能指标 | BER/SNR/EVM/GMI | `metrics.py` | `Metrics/` |

系统级封装函数 `system.rx_dsp`（Python）/ `rx_dsp.m`（MATLAB）把第 8 节的全部步骤串联成一条完整的接收链，是理解整个 DSP 流程的**最佳入口**；本文档第 8 节逐段讲解它内部每一步调用的算法。

---

## 2. 仿真基本约定

| 项目 | 约定 | 说明 |
|---|---|---|
| 信号形状 | 双偏振信号为 $2\times N$ 复数组（行 = X/Y 偏振），时间沿最后一维 | `np.ndarray` 形状 `(2, N)` |
| 光场单位 | $\sqrt{\mathrm{W}}$，$\lvert E\rvert^2$ 为瞬时光功率 $[\mathrm W]$ | 功率量纲一致性是数值仿真最容易出错的地方 |
| 场约定 | 工程约定 $E(t)e^{+j\omega_0 t}$ | 决定 WDM 信道摆放方向、$\beta_3$ 符号、SRS 能量流动方向 |
| 单位 | 长度 km；衰减 $\alpha$ dB/km；色散 $D$ ps/(nm·km)；色散斜率 $S$ ps/(nm²·km)；非线性系数 $\gamma$ 1/(W·km)；波长 nm；其余均为 SI 单位 | 与光通信领域工程习惯一致 |
| 循环（周期）仿真 | 色散、滤波、分数时延、重采样全部通过 FFT 在频域完成，等价于把有限长信号当作周期信号处理 | 优点是无需过采样保护间隔、频域运算效率高；代价是边界效应会从末尾“卷回”开头 |
| 相干探测输出 | 相干接收机输出 $\propto R\cdot E_s\cdot E_{LO}^{*}$ | 本振（LO, Local Oscillator）频偏为 $+\Delta f$ 时，拍频后信号呈现 $-\Delta f$ 的频偏 |

**为什么要强调“循环仿真”？** 因为这决定了随机过程（激光相位噪声、残余频偏导致的载波相位轨迹）在仿真中该怎样处理：

- **色散/成形滤波/定时恢复**（block 处理）：信号首尾本来就应当“无缝衔接”，用循环卷积（`np.fft.fft`/`ifft`）天然满足边界条件，不会引入额外误差；
- **激光器相位噪声**：维纳过程 $\phi(t)=\phi(t-1/f_s)+\sqrt{2\pi\Delta\nu/f_s}\cdot\mathcal N(0,1)$ 本身不是周期的。为了与循环仿真相容，代码中对相位噪声做了 **Brownian bridge** 处理（`devices.laser` 中 `phi -= arange(1,n+1)/n*phi[-1]`），即强行让相位轨迹首尾相接，避免跳变经色散/滤波扩散成边缘误码；频偏同样被取整到频率网格 $f_s/n$ 的整数倍，保证相位轨迹整体连续；
- **载波相位跟踪（BPS/VV/导频辅助/周跳消除）**：这些算法的滑动平均窗口沿时间**不能**循环——因为频偏补偿后残余的相位轨迹不是周期信号，首尾强行平均会在记录末尾产生虚假的相位跳变。因此 `dsp._movmean` 使用 `mode='nearest'`（边界取最近值）而非 `mode='wrap'`。

---

## 3. 发射端数字信号处理（Tx DSP）

对应 `optsim/tx.py`、`system.tx_signal`。

### 3.1 比特产生

两种方式：

- **伪随机序列**（PRBS, Pseudo-Random Binary Sequence）：`tx.prbs(order, n)`，基于 Fibonacci 型线性反馈移位寄存器（LFSR, Linear Feedback Shift Register），多项式 $x^{\text{order}}+x^{\text{tap}}+1$；
- **均匀随机比特**：`tx.rand_bits(n, rng)`，仿真中默认使用，统计性质更接近真实业务流量，且避免 PRBS 周期性带来的虚假相关。

### 3.2 星座映射（Constellation Mapping）

采用 **Gray 映射**（Gray mapping，相邻符号仅相差 1 bit，最小化误码对误比特率的放大）的方形 QAM：

$$
c(\text{label}) = \mathrm{PAM}_{\sqrt M}(\text{label}\gg k) + j\cdot \mathrm{PAM}_{\sqrt M}(\text{label}\ \&\ (2^k-1)), \quad k=\tfrac12\log_2 M
$$

其中 PAM（Pulse Amplitude Modulation）电平按 Gray 码反向索引生成：

```python
# tx.py: pam(M), qam(M)
def pam(M):
    lv = np.empty(M)
    lv[_gray(M)] = 2*np.arange(M) - M + 1      # Gray 标签 -> 电平 ±1,±3,...
    return lv

def qam(M):
    k = int(np.log2(M)) // 2; L = 2**k
    p = pam(L)
    lab = np.arange(M)
    c = p[lab >> k] + 1j*p[lab & (L-1)]        # 高 k 位 -> I，低 k 位 -> Q
    return c / np.sqrt(np.mean(np.abs(c)**2))   # 归一化到单位平均功率
```

判决（硬判决，hard decision）即找最近欧氏距离的星座点：

$$
\hat{\text{label}} = \arg\min_m \lvert y - c_m\rvert^2
$$

### 3.3 比特级软信息：LLR 与概率整形

- **对数似然比**（LLR, Log-Likelihood Ratio）`tx.llr`：在高斯信道假设下，第 $i$ 位比特的 LLR 为

$$
L_i = \log\frac{\sum_{m: b_i(m)=0} p(m)\exp(-|y-c_m|^2/\sigma^2)}{\sum_{m: b_i(m)=1} p(m)\exp(-|y-c_m|^2/\sigma^2)}
$$

  代码中用 `np.logaddexp.reduce` 做数值稳定的 log-sum-exp，是计算编码调制可达速率（GMI，见第 10 节）的基础。

- **概率整形**（PCS, Probabilistic Constellation Shaping）`tx.pcs` / `tx.pcs_symbols`：按 Maxwell-Boltzmann 分布 $p(c)\propto e^{-\nu|c|^2}$ 对星座点赋予不同概率，外侧点出现概率更低。整形增益（shaping gain）来自香农限下高斯分布优于均匀分布的事实，典型可带来 ~0.3–0.5 dB 的有效 SNR 增益或等效的速率提升。本仓库实现的是**理想分布匹配**（按分布直接抽样），未实现恒定组合分布匹配器（CCDM, Constant Composition Distribution Matcher）等实际编码方案。

### 3.4 帧结构（Framing）

`tx.frame(data, train, pilot, period)` 构造如下帧结构：

```
┌──────────────┬────────────────────────────────────────────────┐
│   训练序列    │  数据块 1 (period-1 数据 + 1 导频) │ 数据块 2 │ ...│
│  train_len    │←───────────── period ─────────────→│          │
└──────────────┴────────────────────────────────────────────────┘
kind:  1,1,...,1   0,0,...,0,2   0,0,...,0,2 ...
```

- **训练序列**（training sequence）：已知符号，用于均衡器初始收敛（数据辅助 LMS）、也可用于消除相位模糊；
- **导频符号**（pilot symbol）：每 `pilot_period` 个符号插入 1 个，用于导频辅助的载波相位恢复（pilot-aided CPE）与周跳（cycle slip）检测，开销为 $1/\text{period}$（默认 $1/32\approx 3.1\%$）；
- `kind` 数组标记每个符号的类型（0=数据，1=训练，2=导频），后续所有需要“已知符号”的算法（均衡器训练、相位模糊消除）都依赖这个标记而不是硬编码位置，便于灵活调整帧结构。

### 3.5 脉冲成形（Pulse Shaping）

使用**根升余弦**（RRC, Root-Raised-Cosine）滤波器做 Nyquist 成形，频域响应：

$$
H_{\text{RC}}(f)=\begin{cases}
1, & |f|\le \dfrac{(1-\beta)R_s}{2}\\[4pt]
\dfrac12\Big[1+\cos\Big(\dfrac{\pi}{\beta R_s}\big(|f|-\tfrac{(1-\beta)R_s}{2}\big)\Big)\Big], & \dfrac{(1-\beta)R_s}{2}<|f|\le\dfrac{(1+\beta)R_s}{2}\\[4pt]
0, & \text{其他}
\end{cases}
\qquad H_{\text{RRC}}(f)=\sqrt{H_{\text{RC}}(f)}
$$

$\beta$ 为滚降系数（roll-off factor），决定频谱展宽与符号间干扰（ISI, Inter-Symbol Interference）抑制之间的折中。发射端用 RRC、接收端再做一次 RRC 匹配滤波（matched filter），级联后整体满足 Nyquist 第一准则（无 ISI 采样点）：

```python
# tx.py: pulse_shape / matched_filter
def pulse_shape(sym, sps, beta, kind='rrc'):
    x = upsample(sym, sps)                      # 插零上采样
    f = freq(x.shape[-1], sps)                   # 归一化到符号率的频率轴
    H = nyquist_response(f, 1.0, beta, root=True)
    return fft_filter(x, H * sps)                # 频域滤波 = 循环卷积

def matched_filter(x, sps, beta):
    return fft_filter(x, nyquist_response(freq(x.shape[-1], sps), 1.0, beta))
```

---

## 4. 发射端光电器件模型

对应 `optsim/devices.py`。

### 4.1 DAC（Digital-to-Analog Converter）

`devices.dac(x, fs, enob, bw, kind)`：

1. **有效位数量化**（ENOB, Effective Number Of Bits）：按每路 RMS 的若干倍设满量程，均匀量化后截幅：
   $$ \hat x = \mathrm{clip}\Big(\mathrm{round}\big(\tfrac{x}{\Delta}-\tfrac12\big)+\tfrac12,\ -2^{\text{ENOB}-1}+\tfrac12,\ 2^{\text{ENOB}-1}-\tfrac12\Big)\cdot \Delta,\quad \Delta=\frac{2\cdot\text{clip}\cdot\mathrm{RMS}(x)}{2^{\text{ENOB}}} $$
2. **模拟带宽限制**（含驱动放大器响应）：高斯/贝塞尔/巴特沃斯/砖墙型低通滤波器，3 dB 带宽为 `bw`。

### 4.2 激光器（CW Laser）

`devices.laser(n, fs, p_dbm, linewidth, f_offset, rin_db)` 建模三种损伤：

- **相位噪声**：维纳过程（Wiener process），线宽 $\Delta\nu$ 决定相位扩散速率

  $$\phi[k]=\phi[k-1]+\sqrt{2\pi\Delta\nu/f_s}\cdot n[k],\quad n[k]\sim\mathcal N(0,1)$$

  实现中减去线性趋势使首尾相位相接（见第 2 节说明）；
- **相对强度噪声**（RIN, Relative Intensity Noise）：功率起伏 $P(t)=P_0(1+\delta(t))$，$\delta$ 为高斯白噪声，单边 PSD 为 `rin_db` [dB/Hz]；
- **频率偏移**（frequency offset）：$e^{j2\pi f_{\text{offset}}t}$，用于 WDM 信道摆位、相干接收端的本振频偏模拟。

### 4.3 调制器

- **MZM**（Mach-Zehnder Modulator）：推挽结构，场传输函数为

  $$E_{\text{out}} = E_{\text{in}}\cos\!\Big(\frac{\pi(v+v_{\text{bias}})}{2V_\pi}\Big)$$

  - $v_{\text{bias}}=-V_\pi$（零点偏置，null bias）：用于相干 IQ 子调制器，场近似与驱动电压线性（小信号时 $\cos\to\sin\approx$ 线性）；
  - $v_{\text{bias}}=-V_\pi/2$（正交偏置，quadrature bias）：用于 IM-DD（Intensity Modulation/Direct Detection），功率 $\propto(1+\sin(\cdot))/2$ 近似线性。

- **IQ 调制器**：两个零点偏置 MZM 正交合成

  $$E_{\text{out}}=\frac{1}{2}\Big[g\cdot\mathrm{MZM}(u_I) + \frac{e^{j(\pi/2+\varphi_e)}}{g}\cdot\mathrm{MZM}(u_Q)\Big]$$

  其中 $\varphi_e$ 为正交误差（quadrature error，理想 90°），$g$ 来自 I/Q 增益不平衡（gain imbalance）。

- **DP-IQ 调制器**（Dual-Polarization IQ Modulator）：偏振分束器（PBS, Polarization Beam Splitter）将光场均分到两路 IQ 调制器，再经偏振合束器（PBC）合成双偏振信号。

### 4.4 光放大器与噪声

`devices.edfa(E, fs, gain_db/p_out_dbm, nf_db, f0)`：

- 增益可设为固定值，或自动功率控制（APC, Automatic Power Control，指定输出总功率）；
- **放大自发辐射噪声**（ASE, Amplified Spontaneous Emission）单偏振功率谱密度：

  $$S_{\text{ASE}} = n_{sp}\cdot h\nu\cdot(G-1),\qquad n_{sp}=\frac{\mathrm{NF}_{\text{lin}}}{2}$$

  $h$ 为普朗克常数，$\nu=f_0$ 为放大器工作频率（随波段变化），NF 为噪声系数（noise figure）。噪声以复高斯白噪声形式加到双偏振上。

---

## 5. 光纤信道传输模型

对应 `optsim/channel.py`。这是全仓库物理内容最重的部分。

### 5.1 控制方程：Manakov 方程

双折射随机快速变化（远快于符号周期）的光纤中，偏振复用信号的传播可以用**平均化后的耦合非线性薛定谔方程**（Manakov 方程）描述（Marcuse-Menyuk-Wai, 1997）：

$$
\frac{\partial \mathbf E}{\partial z}= -\frac{\alpha}{2}\mathbf E
-j\frac{\beta_2}{2}\frac{\partial^2\mathbf E}{\partial t^2}
+\frac{\beta_3}{6}\frac{\partial^3\mathbf E}{\partial t^3}
-j\gamma\frac{8}{9}\Big(1+\frac{j}{\omega_0}\frac{\partial}{\partial t}\Big)\Big\{\mathbf E\cdot\big[(1-f_R)|\mathbf E|^2+f_R\,(h_R\otimes|\mathbf E|^2)\big]\Big\}
$$

（单偏振输入时退化为标量非线性薛定谔方程 NLSE, Nonlinear Schrödinger Equation，去掉 $8/9$ 因子）各项物理意义：

| 项 | 物理效应 | 对应代码参数 |
|---|---|---|
| $-\alpha/2\,\mathbf E$ | 光纤损耗 | `fib['alpha']` [dB/km] |
| $-j\beta_2/2\,\partial_t^2$ | 群速度色散（GVD, Group Velocity Dispersion） | `fib['D']` [ps/(nm·km)] |
| $\beta_3/6\,\partial_t^3$ | 三阶色散 / 色散斜率 | `fib['S']` [ps/(nm²·km)] |
| $\gamma\frac{8}{9}|\mathbf E|^2\mathbf E$ 的瞬时部分 | **Kerr 非线性**：自相位调制 SPM、交叉相位调制 XPM、四波混频 FWM | `fib['gamma']` [1/(W·km)] |
| $f_R\,(h_R\otimes|\mathbf E|^2)$ | **受激拉曼散射**（SRS, Stimulated Raman Scattering）的延迟响应 | `fib['raman']` 开关，`fib['fR']` |
| $(1+j\omega_0^{-1}\partial_t)$ | **自陡峭**（self-steepening），对应时域脉冲前沿变陡 | `fib['steep']` 开关 |

**单信道仿真中只对该信道波形求解即包含 SPM（含信道内 IXPM/IFWM）**；若把多个波分复用信道合波后对**全场**求解，XPM 与 FWM 会在求解过程中自动产生——求解器不对信道间相互作用做任何近似，这是分步傅里叶法相对于微扰法（perturbation-based）的优势，代价是计算量大。

### 5.2 参数换算

$\beta_2,\beta_3$ 由工程单位 $D,S,\lambda$ 换算（`channel.beta2` / `beta3`）：

$$
\beta_2=-\frac{D\lambda^2}{2\pi c},\qquad
\beta_3=\Big(\frac{\lambda}{2\pi c}\Big)^2\big(\lambda^2 S+2\lambda D\big)
$$

非线性系数 $\gamma=\dfrac{2\pi n_2}{\lambda A_{\text{eff}}}$ 则由 `params.fiber_params` 根据光纤的非线性折射率 $n_2$ 与有效面积 $A_{\text{eff}}$（随波长缩放）计算（见第 6 节参数文件说明）。

### 5.3 对称分步傅里叶法（Symmetric Split-Step Fourier Method, SSFM）

数值求解 Manakov 方程的标准方法：把线性算子 $\hat D$ 与非线性算子 $\hat N$ 在小步长 $\Delta z$ 内交替作用，用**半步线性-整步非线性-半步线性**提高精度（二阶精度 $O(\Delta z^3)$，优于简单分步的 $O(\Delta z^2)$）：

$$
\mathbf E(z+\Delta z)\approx \exp\!\Big(\frac{\Delta z}{2}\hat D\Big)\exp\!\Big(\Delta z\,\hat N\Big)\exp\!\Big(\frac{\Delta z}{2}\hat D\Big)\,\mathbf E(z)
$$

- **线性步**在频域完成（一次 FFT）：$\hat D\leftrightarrow -\alpha/2 - j(\beta_2\omega^2/2+\beta_3\omega^3/6)$；
- **非线性步**：
  - 若只有纯 Kerr 效应（无 SRS、无自陡峭），非线性算子退化为**瞬时相位旋转**，可解析求解（无需数值积分）：
    $$\mathbf E\leftarrow \mathbf E\cdot\exp\big(-j\gamma\tfrac89|\mathbf E|^2\Delta z\big)$$
  - 若开启 SRS 或自陡峭，非线性算子含时间导数与卷积，用 **4 阶 Runge-Kutta（RK4）** 数值积分子步。

伪代码（对应 `channel.ssfm`）：

```python
def ssfm(E, fs, fib, nsteps=None, phi_max=None):
    Hh = exp(linear_operator * dz/2)          # 半步线性响应（频域）
    while z < L:
        dz = adaptive_step(phi_max) or L/nsteps
        E = ifft(fft(E) * Hh)                 # 半步线性
        if raman or self-steepening:
            E = RK4_step(E, nonlinear_rhs, dz)  # 数值积分非线性子步
        else:
            E = E * exp(-1j*gamma*8/9*|E|^2*dz) # 解析非线性相位旋转
        E = ifft(fft(E) * Hh)                 # 半步线性
        if pmd: E = random_SOP_rotation(E); E = DGD(E, dz)
        z += dz
    return E
```

**自适应步长**（`phi_max` 参数）：步长由“每步最大非线性相移不超过 `phi_max`”自适应决定（典型 0.005–0.01 rad），信号功率高（靠近入纤端）时步长自动变小，功率衰减后步长变大，兼顾精度与速度（Sinkin et al., 2003）。

### 5.4 受激拉曼散射（SRS）的频域实现

石英光纤的拉曼响应函数（时域）：

$$h_R(t)=\frac{\tau_1^2+\tau_2^2}{\tau_1\tau_2^2}e^{-t/\tau_2}\sin(t/\tau_1)\,u(t)$$

其频域形式 `channel.raman_response`：

$$H_R(\omega)=\frac{\tau_1^2+\tau_2^2}{\tau_1\tau_2^2}\cdot\frac{1}{\tau_1\big[(1/\tau_2+j\omega)^2+1/\tau_1^2\big]}$$

非线性项中用 $(1-f_R)|E|^2+f_R\cdot(h_R\otimes|E|^2)$ 代替纯 Kerr 的 $|E|^2$，$f_R\approx0.18$ 为拉曼贡献比例。物理效果：频率间隔在拉曼增益峰（约 13.2 THz，对应 C 波段约 105 nm）附近的两路信号之间会发生能量从高频（短波长、高光子能量）向低频（长波长）的持续转移——这正是 WDM 系统里"短波长信道被长波长信道抽空"（Raman tilt）现象的物理根源。

### 5.5 偏振模色散（PMD, Polarization Mode Dispersion）

一阶 PMD 用差分群时延（DGD, Differential Group Delay）描述：两个本征偏振态（principal state of polarization）之间存在时延差 $\Delta\tau$。多跨长距离链路中 PMD 是随机累积的，仿真中采用**粗步长模型**：每个 SSFM 步长后插入一次随机偏振态旋转（Jones 矩阵 $\in SU(2)$，在 Poincaré 球上均匀分布）+ 固定 DGD，使累积统计量满足

$$\langle\Delta\tau_{\text{total}}\rangle = D_{\text{PMD}}\sqrt{L}$$

（$D_{\text{PMD}}$ 单位 ps/√km，为 PMD 系数）单步 DGD 取 $\Delta\tau_{\text{step}}=D_{\text{PMD}}\sqrt{\Delta z}\cdot\sqrt{3\pi/8}$，使多步累积（麦克斯韦分布极限）期望值收敛到总 DGD。

### 5.6 多跨链路

`channel.link(E, fs, fib, n_spans)`：每跨 = SSFM 传输（衰减 $\alpha L$ dB）+ 放大器（增益 = 跨段损耗，补偿衰减，同时引入 ASE）。这是标准的**损耗-放大周期性链路**（lumped amplification），每跨引入一次独立的 ASE 噪声，$N$ 跨后噪声功率线性累积 $N$ 倍，而信号经过的非线性作用在每跨都独立发生——这正是长距离传输中 OSNR 随距离下降、同时非线性代价累积的根源。

---

## 6. WDM 系统与 GN 非线性模型

### 6.1 多波段参数文件

`params/fiber_bands.json`（Python 与 MATLAB 共用）存储 O/E/S/C/L/U 六个波段的典型衰减、放大器类型/噪声系数，以及 SSMF/ULL/NZDSF/DSF 四种光纤的色散模型参数（零色散波长 $\lambda_0$、零色散斜率 $S_0$、有效面积、$n_2$、PMD 系数）。色散按 ITU-T G.652 三项 Sellmeier 拟合换算到任意工作波长：

$$D(\lambda)=\frac{S_0}{4}\Big(\lambda-\frac{\lambda_0^4}{\lambda^3}\Big),\qquad S(\lambda)=\frac{S_0}{4}\Big(1+\frac{3\lambda_0^4}{\lambda^4}\Big)$$

`params.fiber_params(band, fiber, L=..., **override)` 一步生成可直接传给 `channel.ssfm`/`dsp.dbp` 的参数字典。

### 6.2 WDM 合波与解复用

`devices.wdm_mux(fields, freqs, fs)`：各信道基带波形乘以 $e^{j2\pi f_k t}$ 搬移到各自频率网格后相加；接收端用超高斯带通滤波器（WSS, Wavelength Selective Switch，`devices.optical_filter`）选出目标信道，本振调谐到相同频率完成下变频。

**关键点**：WDM 合波后对**全场**做 SSFM，XPM（信道间通过 Kerr 效应的交叉调制）与 FWM（四波混频，满足相位匹配条件时在非零色散光纤中较弱、在色散位移光纤 DSF 中很强）都在求解过程中精确包含，无需额外建模。

### 6.3 高斯噪声（GN）非线性模型

GN 模型（Gaussian Noise model, Poggiolini 2012）把 Kerr 非线性在非相干累积、高斯谱近似下等效为一个与 ASE noise 性质相同的加性高斯噪声源，其功率按入纤功率三次方增长：

$$P_{\text{NLI}}=\eta\cdot P_{\text{ch}}^3$$

`channel.gn_eta` 提供两种算法：

- **数值积分**（`method='numeric'`）：直接数值求解 GN 参考积分
  $$G_{\text{NLI}}(0)=\frac{16}{27}\gamma^2\!\iint\! G(f_1)G(f_2)G(f_1+f_2)\left|\frac{1-e^{-\alpha L}e^{j4\pi^2\beta_2 L f_1f_2}}{\alpha-j4\pi^2\beta_2 f_1f_2}\right|^2 df_1df_2$$
- **闭式近似**（`method='closed'`，Poggiolini 2012）：
  $$\eta=\frac{8}{27}\gamma^2 L_{\text{eff}}^2\,\frac{\operatorname{asinh}\!\big(\tfrac{\pi^2}{2}|\beta_2|L_{\text{eff,a}}R_s^2 N_{ch}^{2R_s/\Delta f}\big)}{\pi|\beta_2|L_{\text{eff,a}}R_s^2}$$

全链路 SNR 预测（`channel.gn_model`）：$\mathrm{SNR}=P/(P_{\text{ASE}}+\eta P^3)$，并给出最优入纤功率 $P_{\text{opt}}=(P_{\text{ASE}}/2\eta)^{1/3}$（经典的"非线性香农极限"功率优化结果）。该模型已用仿真做过交叉验证（数值积分与闭式相差 <1 dB），可用于快速预测而无需逐点跑完整 SSFM。局限：对 QAM 信号的 NLI 会有约 0.5–1 dB 的高估（EGN 修正未实现），对近零色散光纤不准确，不含波段间 SRS（ISRS-GN）。

---

## 7. 接收端光电器件模型

### 7.1 偏振分集相干接收机

`devices.coherent_rx` 实现标准的**偏振分集相干接收机**（polarization-diversity coherent receiver）结构：

```
          ┌─ 90° 混频器 ─┐
Es(X) ──► │ I+,I-,Q+,Q- │ ──► BPD(I) ──► I_X
          └──────────────┘ ──► BPD(Q) ──► Q_X     合并 → I_X + jQ_X
Elo/√2 ───┘
          ┌─ 90° 混频器 ─┐
Es(Y) ──► │ I+,I-,Q+,Q- │ ──► BPD(I) ──► I_Y
          └──────────────┘ ──► BPD(Q) ──► Q_Y     合并 → I_Y + jQ_Y
```

- **90° 光混频器**（`devices.hybrid_90`）：4 路输出 $[E_s+E_{LO},\,E_s-E_{LO},\,E_s+jE_{LO},\,E_s-jE_{LO}]/2$；
- **平衡探测器**（BPD, Balanced Photodetector，`devices.bpd`）：两路 PD 电流相减，抑制本振直流项与信号-信号拍频（signal-signal beat interference），保留信号-本振拍频项：
  $$i_I(t)=\mathrm{PD}(E_s+E_{LO})-\mathrm{PD}(E_s-E_{LO})\propto \mathrm{Re}\{E_sE_{LO}^*\}$$

### 7.2 光电探测噪声

`devices.photodiode`：

$$i=R\sum_p|E_p|^2+i_{\text{dark}}+n_{\text{shot}}+n_{\text{thermal}}$$

- **散粒噪声**（shot noise）：方差 $2qIB$（$q$ 电子电荷，$I$ 平均光电流，$B=f_s/2$ 等效噪声带宽）；
- **热噪声**（thermal noise）：方差 $4k_BTB/R_L$（跨阻放大器等效）。

### 7.3 ADC

`devices.adc`：模拟前端带宽限制 → 重采样到 ADC 采样率 → ENOB 量化，与 DAC 对称。

---

## 8. 接收端 DSP 算法链详解

这是整个仓库的核心内容。`system.rx_dsp`（Python）/ `rx_dsp.m`（MATLAB）按下图顺序串联各算法：

```
ADC 输出(2,N) @fs_adc
   │
   ▼
① 前端校正：IQ deskew → GSOP 正交化 → 重采样到 2 sps
   │
   ▼
② 色散/非线性补偿：CDC（频域）或 DBP（逐跨反向 SSFM）
   │
   ▼
③ 匹配滤波（RRC）
   │
   ▼
④ 定时恢复：Godard 前馈 或 Gardner 反馈
   │
   ▼
⑤ 2×2 MIMO 自适应均衡：CMA 预收敛 → RDE 精收敛（偏振解复用）
   │
   ▼
⑥ 频偏估计（FOE）与补偿
   │
   ▼
⑦ 载波相位恢复（CPR）：BPS / VV / 导频辅助
   │
   ▼
⑧ 帧同步 + 偏振交换检测
   │
   ▼
⑨ 相位模糊/周跳消除（训练+导频辅助）
   │
   ▼
⑩（可选）DD-LMS 精均衡
   │
   ▼
⑪ 功率归一化 → 去训练/导频 → 硬判决 → BER/SNR/EVM/GMI
```

下面逐项展开原理、公式与代码。

### 8.1 前端校正

**IQ 偏斜补偿**（`dsp.deskew`）：真实系统中 X/Y 偏振的 I/Q 四路电信号由于走线长度不同存在纳秒级以下的时延差（IQ skew）。补偿方法是对每路施加分数时延的逆：频域线性相位 $e^{-j2\pi f\tau}$（`utils.delay`，循环卷积，支持任意分数采样时延）。

**Gram-Schmidt 正交化**（GSOP，`dsp.gsop`）：补偿接收机前端 I/Q 支路的**幅度不平衡**和**相位不平衡**（quadrature imbalance），源自实际 90° 混频器的非理想正交性：

$$I'=\frac{I-\bar I}{\mathrm{RMS}(I)},\qquad Q'=\frac{(Q-\bar Q)-\langle I'(Q-\bar Q)\rangle I'}{\mathrm{RMS}(\cdot)}$$

即先对 I 归一化，再从 Q 中减去其在 I 方向上的投影（正交化），使处理后 I、Q 严格正交且等功率。该算法思路改写自实验参考代码 `IQimbalance.m`。

### 8.2 色散补偿

**频域色散补偿**（CDC, Chromatic Dispersion Compensation，`dsp.cdc`）：线性色散在频域是纯相位滤波器，可用逆滤波器精确补偿（线性、可逆，无需迭代）：

$$H_{\text{CDC}}(\omega)=\exp\Big[j\Big(\frac{\beta_2}{2}\omega^2+\frac{\beta_3}{6}\omega^3\Big)L\Big]$$

全链路（$N$ 跨 $\times L$ km）只需按总长度 $NL$ 一次性补偿（色散是线性效应，不同跨段可合并处理），计算量远低于逐跨处理。

**盲色散估计**（`dsp.cd_estimate`）：真实系统中链路总长度往往未知或有误差。盲估计思路：扫描一组候选补偿长度 $L$，以**平方律时钟频谱分量**（即 Godard/Oerder-Meyr 时钟音，见 §8.4）强度作为代价函数：

$$J(L)=\Big|\sum_t |x_{\text{CDC}(L)}(t)|^2\cdot e^{-j2\pi R_s t}\Big|$$

残余色散会展宽符号间干扰、抹平平方律检测后的时钟频谱分量；只有补偿长度与真实长度匹配时，$J(L)$ 取最大值。

### 8.3 非线性补偿：数字反向传播（DBP）

**数字反向传播**（DBP, Digital Backpropagation，Ip & Kahn 2008）是补偿确定性 Kerr 非线性（主要是 SPM）的经典方法：把接收信号用**反向符号的光纤参数**做一遍虚拟传播，等效于时间反演地"倒放"Manakov 方程：

$$\text{正向：}\ \alpha,\ \beta_2,\ \gamma \qquad\Longrightarrow\qquad \text{DBP：}\ -\alpha,\ -\beta_2,\ -\gamma$$

`dsp.dbp` 逐跨处理（与正向传播逐跨对称）：

```python
def dbp(x, fs, p_launch_dbm, n_spans, fib, steps, xi=1.0):
    E = set_power(x, p_launch_dbm)             # 还原到物理功率（需知道实际入纤功率）
    b = fib with alpha,D,S,gamma 取反, gamma 乘以补偿系数 xi
    for _ in range(n_spans):
        E = E * 10**(-alpha*L/20)               # 先"反向放大"：撤销该跨的 EDFA 增益
        E = ssfm(E, fs, b, nsteps=steps)         # 再以负参数做 SSFM（即反向 SPM+色散）
    return E
```

- `xi`（非线性补偿系数，0~1）：由于 DBP 无法补偿 ASE-非线性相互作用产生的非线性相位噪声（Gordon-Mollenauer noise），实践中并非系数越大越好，常需要在 0.5~1 之间寻优；
- DBP 的代价是计算复杂度正比于 `steps`（每跨步数），工程上会用更少的步数（每跨 1 步，即"低复杂度 DBP"）配合滤波器优化（ESSFM 等）降低复杂度，本仓库未实现该优化，仅提供教学意义上的标准多步 DBP。
- `nl_comp` 可设为任意可调用对象（`callable`），用于接入自定义或学习型非线性补偿器（见第 11 节学习型 DBP）。

### 8.4 匹配滤波与定时恢复

匹配滤波：接收端 RRC（`tx.matched_filter`），与发射端 RRC 级联构成整体 Nyquist 滤波。

**定时恢复**（Timing Recovery）解决的问题：ADC 采样时刻与符号中心不对齐，需要估计/跟踪分数符号定时偏差 $\tau$。

- **前馈 Godard 方法**（`dsp.timing_ff`）：对整块数据一次性估计，利用平方律非线性在符号率处产生的频谱分量（"时钟音"，clock tone）：

  $$c=\sum_{f>0}X(f)X^*(f-R_s),\qquad \hat\tau=-\frac{\angle c}{2\pi}\cdot\text{sps}$$

  适用于无显著采样时钟频偏（clock frequency offset）的情形，一次 FFT 即可完成，计算高效。

- **Gardner 反馈环**（`dsp.gardner`）：逐符号自适应跟踪，误差信号基于相邻符号与其中间插值点：

  $$e[k]=\mathrm{Re}\{(y[k-1]-y[k])\cdot y^*[k-\tfrac12]\},\qquad \tau\leftarrow\tau+\mu e[k]$$

  配合三次 **Farrow 插值**（三次拉格朗日插值，`dsp._interp`）在任意分数位置重建样本，可以跟踪采样时钟频偏（clock frequency offset），计算量和复杂度高于前馈方法。

### 8.5 自适应均衡 / 偏振解复用（MIMO）

这是接收 DSP 中最关键的一步：同时完成**偏振解复用**（真实光纤中两偏振态会随机旋转、耦合）与**残余线性损伤补偿**（ISI、残余 CD、PMD）。标准结构是 **2×2 蝶形 FIR 滤波器**（butterfly FIR equalizer）：

$$
\begin{bmatrix}y_X[k]\\y_Y[k]\end{bmatrix}=
\begin{bmatrix}w_{XX}&w_{XY}\\w_{YX}&w_{YY}\end{bmatrix}\ast
\begin{bmatrix}x_X[k]\\x_Y[k]\end{bmatrix}
$$

每个 $w_{ij}$ 是长度 `ntaps` 的 FIR 滤波器，逐符号用随机梯度（LMS 族算法）更新 $\mathbf W\leftarrow\mathbf W+\mu\,\mathbf e\,\mathbf x^H$。`dsp.mimo_eq` 支持 5 种误差信号（`mode` 参数）：

| 模式 | 误差信号 $e$ | 原理 | 适用阶段 |
|---|---|---|---|
| `cma` | $y(R_2-|y|^2)$，$R_2=E[|c|^4]/E[|c|^2]$ | **恒模算法**（Constant Modulus Algorithm）：盲算法，驱动输出幅度趋向星座的"等效模值" $R_2$。对高阶 QAM（非恒模星座）是近似收敛，用于**初始粗收敛** | 预收敛 |
| `rde` | $y(R_d^2-|y|^2)$，$R_d$ 为最近星座半径 | **半径判决算法**（Radius-Directed Equalizer）：按多模值判决，比 CMA 更精确地刻画多环 QAM 的星座结构 | 精收敛 |
| `mma` | 实部/虚部分别做 CMA（Modified/Multi-Modulus Algorithm） | 实虚分离，对某些相位模糊更鲁棒 | 备选 |
| `lms` | 训练阶段 $d-y$（$d$=已知训练符号），之后判决引导 $d=\hat c$ | **数据辅助 LMS**：训练阶段收敛快且准确 | 有训练序列时 |
| `dd` | $d-y$，$d$=判决符号（decision-directed） | 载波相位恢复**之后**的精细均衡，要求输入已基本无相位旋转 | 最后一级精调 |

**典型两阶段策略**（`system.rx_dsp` 中采用）：先用前约 10000 符号跑 `cma` 预收敛（避免 RDE 在均衡器未收敛、星座尚是模糊圆环时误判决发散），再用收敛后的抽头作为初值跑全程 `rde` 精收敛：

```python
n0 = min(10000, y.shape[1] // 2)
_, w, _ = dsp.mimo_eq(y[:, :2*n0], ntaps, mu_cma, 2, 'cma', const)
y, w, err = dsp.mimo_eq(y, ntaps, mu_rde, 2, 'rde', const, w=w)
```

**为什么 CMA/RDE 能自动解复用偏振？** 因为两路输出 $y_X,y_Y$ 都独立地被驱动向"恒模/多模"的代价函数最低点收敛，只要加入一个**正交化约束**防止两路收敛到同一个偏振分量（本实现未显式加入 Gram-Schmidt 去相关，依赖两路独立随机初始化+代价函数本身在实践中足以分离，这是 CMA 用于偏振解复用的标准做法，但理论上存在"奇异收敛"小概率风险，工程系统常加额外的去相关项）。

### 8.6 频偏估计（FOE, Frequency Offset Estimation）

相干接收中，发射激光器与本振激光器频率不可能完全一致，残余频偏（典型几百 MHz 量级）会使星座持续旋转，必须先估计并补偿，否则载波相位恢复无法收敛。`dsp.foe` 提供两种基于 **M 次方非线性**（M-th power nonlinearity，消除 QAM 星座的 $2\pi/M$ 相位模糊，暴露出频偏对应的单频分量）方法：

- **FFT 谱峰法**：$s(f)=\sum|\mathrm{FFT}(x^M)|^2$ 的峰值位置 $/M$ 即为频偏估计，补零提高频率分辨率；
- **差分法**（Leven et al., 2007）：$\hat f=\dfrac{\angle\sum_k(x[k]x^*[k-1])^M}{2\pi M}\cdot f_s$，计算量更低，适合逐块跟踪。

补偿：$x\leftarrow x\cdot e^{-j2\pi \hat f t}$。

### 8.7 载波相位恢复（CPR, Carrier Phase Recovery）

补偿完频偏后仍有残余的**激光相位噪声**（线宽展宽导致的随机相位游走），需要逐符号跟踪并补偿。三类方法：

**(a) Viterbi-Viterbi（VV）M 次方算法**（`dsp.vv`）：对无记忆的恒模调制（如 QPSK）最优，对 QAM 是近似：

$$\hat\phi[k]=\frac{1}{M}\Big(\angle\langle x^M\rangle_n - \angle E[c^M]\Big)$$

滑动平均窗口 $n$ 平衡噪声抑制与相位跟踪速度；存在 $2\pi/M$ 的固有相位模糊。

**(b) 盲相位搜索 BPS**（Blind Phase Search，Pfau et al. 2009，`dsp.bps`）：在 $[-\pi/4,\pi/4)$ 范围内测试 `n_test` 个候选相位，对每个候选相位计算窗口内符号到最近星座点的平方距离和，取代价最小的相位：

```python
for each candidate phase φ_i:
    d_i[k] = min_m |x[k]e^{-jφ_i} - c_m|^2       # 逐符号到最近星座点的距离
    D_i[k] = moving_average(d_i, window=n)        # 窗口平滑降噪
φ_hat[k] = argmin_i D_i[k]                        # 取代价最小的候选相位
```

适用于任意 QAM（不要求恒模假设），精度通常优于 VV，代价是计算量正比于 `n_test`；同样存在 $\pi/2$ 固有模糊（因为方形 QAM 旋转 $90°$ 后星座形状不变，无法从星座几何本身分辨）。

**(c) 导频辅助相位恢复**（Pilot-Aided CPE，`dsp.pilot_cpr`，改写自实验参考代码 `PilotBased_CPE.m`）：直接用已知导频符号计算相位旋转 $\angle(x\cdot\text{pilot}^*)$，对导频位置的估计值做滑动平均后线性插值到全部数据符号：

$$\hat\phi[k]=\mathrm{interp}\Big(\{\mathrm{idx}_{\text{pilot}}\},\ \big\{\angle\langle x_{\text{pilot}}\cdot \text{tx}_{\text{pilot}}^*\rangle_n\big\}\Big)$$

**无相位模糊**（因为参考是已知符号），代价是开销（导频比例）与插值精度的折中。

**(d) 判决引导锁相环（DD-PLL）**（`dsp.pll`）：经典二阶数字锁相环，用判决误差驱动相位/频率两个状态：

$$e[k]=\mathrm{Im}\{y[k]\cdot\hat c[k]^*\},\qquad f\leftarrow f+\rho e,\qquad \theta\leftarrow\theta+\mu e+f$$

逐符号迭代，可同时跟踪慢变的残余频偏（积分项 $f$）与相位噪声。

### 8.8 帧同步与偏振交换检测

`dsp.sync`：接收符号流相对发送帧存在未知的**起始位置偏移**（时延）。利用训练序列做**循环互相关**（FFT 实现，计算量 $O(N\log N)$）：

$$c[\tau]=\big|\mathrm{IFFT}\{\mathrm{FFT}(y)\cdot\mathrm{FFT}(\text{ref})^*\}\big|$$

峰值位置即为帧起始偏移 `lag`。同时需要检测**偏振交换**（polarization swap）：经过光纤的随机偏振旋转，接收到的两路 X'/Y' 可能对应发送的 Y/X（而非 X/Y），算法通过比较 4 种互相关组合 $(X\to X, Y\to Y)$ vs $(X\to Y, Y\to X)$ 的相关峰总能量，选出能量更高的配对方式确定 `perm`。

### 8.9 相位模糊与周跳消除

BPS/VV 算法存在固有的 $\pi/2$（方形 QAM）相位模糊——无法区分星座整体旋转 $90°,180°,270°$ 的情形。`dsp.fix_ambiguity` 用已知符号（训练+导频）消除：

$$\hat k=\mathrm{round}\Big(\frac{\angle\sum (y\cdot\text{ref}^*)}{\pi/2}\Big),\qquad y\leftarrow y\cdot e^{-j\hat k\pi/2}$$

- **全局模糊**（`n=None`）：所有已知符号统一估计一个旋转量，适用于相位恢复全程稳定、无周跳（cycle slip）的情形；
- **分段/周跳消除**（给定 `n`）：对已知符号做 `n` 点滑动平均后**逐段**判定旋转量，再插值回全部符号。原因：低 OSNR 下 BPS/VV 可能在跟踪过程中发生**周跳**（相位估计突然跳变 $\pi/2$ 的整数倍，通常由噪声导致代价函数的候选相位错误切换），若只做一次全局估计会被局部周跳污染；分段判定能把误差限制在局部，大幅降低低 OSNR 下的 BER 平台。此改进是本仓库在联调 B2B BER-OSNR 曲线时针对该问题引入的（16 dB OSNR 下 BER 从约 0.2 降到与理论曲线吻合，详见项目 README 第 9 节 FAQ）。

### 8.10 DD-LMS 精均衡

`mimo_eq(..., mode='dd')`：在载波相位恢复完成之后，用判决符号做决策引导 LMS，进行最后一级精细均衡，清除残余的线性损伤（更小步长、更精细跟踪）。

### 8.11 判决与性能指标

去除训练/导频符号后，对数据符号做硬判决、统计误码，并计算 SNR/EVM/GMI（见第 10 节）。

---

## 9. IM-DD 与自相干接收

本仓库同时支持强度调制直接检测（IM-DD）系统，复用部分相同的模块：

- **发射**：PAM-M（`tx.pam`）+ RC/NRZ 成形 + MZM 正交点偏置（`devices.mzm(..., vbias=-vpi/2)`），功率近似线性于驱动电压；
- **信道**：仅需线性色散（`channel.cd`），色散导致的**功率衰落**（dispersion-induced power fading，$\cos$ 型陷波）是 IM-DD 的主要损伤来源；
- **接收**：单路 PD 平方律检测（`devices.photodiode`）+ 实数 **LMS-FFE**（`dsp.ffe`）或**非线性 Volterra 均衡器**（`dsp.volterra`，用岭回归一次性求解核系数，包含二阶、三阶非线性核，补偿平方律检测引入的非线性串扰）；
- **自相干接收**（self-coherent detection）：Kramers-Kronig 接收（`dsp.kk`）利用强载波 + 单边带信号满足最小相位条件（minimum-phase condition），仅用单路光电流通过 Hilbert 变换重建出完整复光场相位信息：
  $$E_{\text{rec}}=\sqrt{I}\cdot\exp\big(j\cdot\mathcal H\{\ln\sqrt I\}\big)$$
  是低成本直接检测系统中少有的能够恢复相位信息、从而补偿色散的方案（Mecozzi et al. 2016）。

---

## 10. 性能指标

对应 `optsim/metrics.py`：

| 指标 | 定义 | 说明 |
|---|---|---|
| BER（误码率） | $N_{\text{err}}/N_{\text{bits}}$ | 最终判据，样本量不足时统计意义有限（经验上需 $\gtrsim 100$ 个误比特） |
| EVM（误差矢量幅度） | $100\sqrt{E|y-x|^2/E|x|^2}$ % | 直观反映星座弥散程度 |
| SNR（有效信噪比） | 先做最小二乘复增益 $a=\langle x,y\rangle/\langle x,x\rangle$ 校正幅相，再算 $10\lg(|a|^2P_x/E|y-ax|^2)$ | 比直接算噪声方差更鲁棒（消除判决前残余的全局增益/相位偏差） |
| Q 因子 | $20\lg\big(\sqrt2\,\mathrm{erfcinv}(2\cdot\mathrm{BER})\big)$ [dB] | BER 的等效高斯信噪比表述，工程上常用 |
| 理论 BER | Gray 映射 M-QAM 最近邻近似：$\frac{4}{k}(1-\tfrac1{\sqrt M})Q\big(\sqrt{3s/(M-1)}\big)$ | 用于与仿真/实验结果比对 |
| **GMI**（广义互信息） | $H(X)-\dfrac1N\sum_k\sum_i\log_2\big(1+e^{-(1-2b_{ki})L_{ki}}\big)$ | 比特交织编码调制（BICM）系统的**可达速率**，比 BER 更能反映软判决 FEC 后的系统容量，是现代光通信系统评估的主流指标 |
| **MI**（互信息） | 符号级，$E\big[\log_2 q(y|x)/\sum_s p(s)q(y|s)\big]$ | 理想无比特位独立假设下的可达速率上界 |
| OSNR↔SNR | $\mathrm{SNR}=\mathrm{OSNR}+10\lg(2B_{\text{ref}}/(N_{\text{pol}}R_s))$ | 光域 OSNR（通常以 0.1 nm = 12.5 GHz 为参考带宽）与电域每偏振符号 SNR 互转 |

---

## 11. 神经网络均衡器与学习型 DBP（简介）

`optsim/nn_eq.py`（需要 `torch`，按需导入）提供：

- **NNEqualizer**：MLP / 1D-CNN / BiLSTM 三种结构的滑窗非线性均衡器，输入为以第 $k$ 个符号为中心的 `ntaps` 个采样窗口（`dsp.sym_windows` 生成，复信号拆成 [Re,Im]×偏振通道），输出中心符号估计，默认采用**残差学习**（网络只学 $y-x_{\text{center}}$ 的修正量）加速收敛；
- **学习型 DBP（LDBP, Learned DBP）**：把 DBP 的分步傅里叶结构展开为多层网络，每层的线性滤波器与非线性步长系数作为可训练参数，用梯度下降代替物理公式直接求解最优参数（Häger & Pfister 2018 思路）。

这两类方法均属于**数据驱动**非线性补偿，训练时需要在**独立实现**（不同比特、不同噪声实现）上评估，否则网络可能"记住"训练序列的具体样本而虚高估计增益。本节仅作方法概览，详细原理建议结合 `Deep_Learning` 与 `Optical_Comm` 技能库中的内容深入学习，不在本文档展开。

---

## 12. 参考文献

- G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press（NLSE、SSFM、拉曼响应、自陡峭、孤子）。
- D. Marcuse, C. R. Menyuk, P. K. A. Wai, "Application of the Manakov-PMD equation to studies of signal propagation in optical fibers with randomly varying birefringence," *J. Lightwave Technol.*, 15(9), 1997.
- O. V. Sinkin, R. Holzlöhner, J. Zweck, C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," *J. Lightwave Technol.*, 21(1), 2003.
- P. Poggiolini, "The GN model of non-linear propagation in uncompensated coherent optical systems," *J. Lightwave Technol.*, 30(24), 2012.
- D. Semrau, R. I. Killey, P. Bayvel, "The Gaussian noise model in the presence of inter-channel stimulated Raman scattering," *J. Lightwave Technol.*, 36(14), 2018.
- S. J. Savory, "Digital coherent optical receivers: Algorithms and subsystems," *IEEE J. Sel. Top. Quantum Electron.*, 16(5), 2010.
- E. Ip, J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," *J. Lightwave Technol.*, 26(20), 2008.
- T. Pfau, S. Hoffmann, R. Noé, "Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations," *J. Lightwave Technol.*, 27(8), 2009.
- A. Leven, N. Kaneda, U.-V. Koc, Y.-K. Chen, "Frequency estimation in intradyne reception," *IEEE Photon. Technol. Lett.*, 19(6), 2007.
- F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled receivers," *IEEE Trans. Commun.*, 34(5), 1986.
- A. Mecozzi, C. Antonelli, M. Shtaif, "Kramers–Kronig coherent receiver," *Optica*, 3(11), 2016.
- G. Böcherer, F. Steiner, P. Schulte, "Bandwidth efficient and rate-matched low-density parity-check coded modulation," *IEEE Trans. Commun.*, 63(12), 2015（概率整形 PAS/PCS）。
- A. Alvarado, T. Fehenberger, B. Chen, F. M. J. Willems, "Achievable information rates for fiber optics: Applications and computations," *J. Lightwave Technol.*, 36(2), 2018（GMI/MI）。
- R. T. Jones, T. A. Eriksson, M. P. Yankov, D. Zibar, "Deep learning of geometric constellation shaping including fiber nonlinearities," *ECOC*, 2018.
- C. Häger, H. D. Pfister, "Deep learning of the nonlinear Schrödinger equation fiber-optic communications," *ISIT*, 2018（学习型 DBP）。
- ITU-T Recommendations G.652 / G.653 / G.654 / G.655（光纤标准）；G.694.1（WDM 频率栅格）。
- 开源参考实现：OptiCommPy（Python）、Robochameleon（MATLAB）。

---

**延伸阅读**：本文档的交互式配套材料见 `Python/notebooks/01_相干DP-QAM全流程教程.ipynb`，其中逐步调用本文档第 3–8 节提到的函数，并在每一步可视化星座图/频谱/收敛曲线的变化，建议对照阅读。
