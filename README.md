# Optical Communication System：光纤通信全流程仿真（Python / MATLAB）

面向科研的光纤通信与光纤传感仿真代码，覆盖 **相干 DP-QAM（单信道与 WDM）**、**IM-DD PAM** 和 **φ-OTDR 分布式声波传感**。

- **两套实现，一一对应**：Python 与 MATLAB 的模块划分、函数功能和默认参数完全对应，结果可交叉验证。两边自检的物理量完全一致。
- **一个参数文件**：波段、光纤、放大器参数集中在 `params/fiber_bands.json`，两种语言共用。
- **完整非线性模型**：SPM / XPM / FWM / SRS / 自陡峭 / PMD / 色散斜率，另附 GN 模型做解析校验。
- **神经网络均衡器**：MLP / 1D-CNN / BiLSTM（替代 FFE、Volterra）与学习型 DBP（替代 DBP），均与传统方法在独立测试数据上对比。
- **代码风格**：简洁、注释直白，不做工程式的参数校验。
- **配套学习材料**：原理文档 [`docs/仿真原理与DSP算法详解.md`](docs/仿真原理与DSP算法详解.md)（公式推导+伪代码+参考文献）与交互式教程 [`Python/notebooks/01_相干DP-QAM全流程教程.ipynb`](Python/notebooks/01_相干DP-QAM全流程教程.ipynb)（逐步调用、逐步可视化，已预先运行并嵌入全部图表输出）。

---

## 目录

1. [快速开始](#1-快速开始)
2. [目录结构](#2-目录结构)
3. [参数文件与多波段](#3-参数文件与多波段)
4. [仿真约定](#4-仿真约定)
5. [物理模型](#5-物理模型)
6. [接收 DSP 算法](#6-接收-dsp-算法)
7. [神经网络均衡器](#7-神经网络均衡器)
8. [API 速查（Python ↔ MATLAB）](#8-api-速查python--matlab)
9. [示例与参考结果](#9-示例与参考结果)
10. [FAQ](#10-faq)
11. [与参考代码的关系](#11-与参考代码的关系)
12. [参考文献](#12-参考文献)

---

## 1. 快速开始

**Python**：使用 conda 环境 `Raymond`，依赖 numpy、scipy、matplotlib。
```bash
cd Python
conda run -n Raymond python examples/selfcheck.py
conda run -n Raymond python examples/coherent_dp_qam.py
conda run -n Raymond python examples/wdm_system.py
```

神经网络均衡器另需 PyTorch（Raymond 环境已有 torch 2.12 + CUDA）；有 GPU 时自动使用，可用环境变量 `OPTSIM_DEVICE=cpu` 强制 CPU。

**MATLAB**：在 R2026a 上测试。需要 Signal Processing Toolbox（`besself`、`hilbert`、`pwelch`、`medfilt1`）；Statistics Toolbox 仅 `pcs_mb` 用到（`randsample`）；神经网络均衡器需要 Deep Learning Toolbox（`trainnet`、`dlarray`；有 GPU 与 Parallel Computing Toolbox 时自动用 GPU）。
```matlab
cd MATLAB
selfcheck
main_coherent_dp_qam
main_wdm_system
```

**最小调用示例**（Python；MATLAB 的写法完全对应）：
```python
from optsim import devices as dev, channel
from optsim.system import Config, tx_signal, rx_dsp

cfg = Config(M=16, rs=32e9, band='L', fiber='ULL', n_spans=20, p_launch=0)   # L 波段 ULL 光纤
fib = cfg.fib                                    # 自动从 JSON 取 α、D、S、γ、NF、f0 ...
ref = tx_signal(cfg)                             # 比特 → QAM → 成帧 → RRC
E   = ref['wave'] * dev.laser(ref['wave'].shape[1], cfg.fs, 0, 100e3)
E   = dev.edfa(E, cfg.fs, p_out_dbm=cfg.p_launch, nf_db=fib['nf'], f0=fib['f0'])
E   = channel.link(E, cfg.fs, fib, cfg.n_spans, phi_max=cfg.phi_max)      # SSFM + 放大器
r   = dev.coherent_rx(E, dev.laser(E.shape[1], cfg.fs, 10, 100e3, 1e9), cfg.fs)
res = rx_dsp(dev.adc(r, cfg.fs, cfg.fs_adc), cfg, ref)                    # → res['ber'], res['snr'], res['gmi']
```

---

## 2. 目录结构

```
Optical Communication System/
├── params/fiber_bands.json        波段 / 光纤 / 放大器 / Raman 参数（两种语言共用）
├── docs/
│   └── 仿真原理与DSP算法详解.md     原理文档：仿真每一级的物理模型 + 接收 DSP 算法链详解，含公式、伪代码、参考文献
├── Python/
│   ├── notebooks/
│   │   └── 01_相干DP-QAM全流程教程.ipynb   配套交互式教程：逐步调用 optsim 函数，分步可视化每个 DSP 模块的效果
│   ├── optsim/
│   │   ├── utils.py      常数、单位、功率、频域滤波、电低通、周期重采样、分数时延、AWGN
│   │   ├── params.py     读取 JSON，fiber_params(band, fiber) → 光纤参数 dict
│   │   ├── tx.py         PRBS、QAM/PAM/PSK、判决/解映射、LLR、PCS、成帧、RRC 成形
│   │   ├── devices.py    DAC/ADC、激光器、MZM/IQ/DP-IQ、放大器、WSS、WDM 合波、偏振器件、PD/BPD、相干接收机
│   │   ├── channel.py    色散、Manakov SSFM（SPM/XPM/FWM/SRS/自陡峭/PMD）、多跨链路、GN 模型
│   │   ├── dsp.py        接收 DSP 算法库
│   │   ├── metrics.py    BER/SER/EVM/SNR/Q/GMI/MI、理论 BER、OSNR↔SNR
│   │   ├── system.py     Config、tx_signal、wdm_tx、wdm_rx、rx_dsp
│   │   ├── nn_eq.py      神经网络均衡器：NNEqualizer (MLP/CNN/BiLSTM)、LDBP（需 torch，不随 optsim 自动导入）
│   │   └── plots.py      星座图、功率谱、眼图
│   └── examples/         selfcheck, coherent_dp_qam, wdm_system, band_gn_analysis,
│                         launch_power_sweep, ber_vs_osnr, imdd_pam4, phi_otdr_das,
│                         nn_imdd_pam4, nn_coherent_nlc
└── MATLAB/
    ├── Utils/  Tx/  Components/  Channel/  DSP/  Metrics/  System/  NN/   （一个函数一个文件）
    ├── setup_paths.m             添加路径（各主脚本开头已调用）
    └── selfcheck.m, main_*.m     与 Python 示例一一对应
```

---

## 3. 参数文件与多波段

`params/fiber_bands.json` 包含以下四部分：

| 部分 | 内容 |
|---|---|
| `bands` | O/E/S/C/L/U 各波段：波长范围、中心波长、典型衰减、放大器类型、NF |
| `fibers` | SSMF (G.652.D)、ULL (G.654.E)、NZDSF (G.655)、DSF (G.653)：零色散波长 λ0、斜率 S0、有效面积、n2、PMD、衰减修正 |
| `raman` | 石英 Raman 响应参数 fR、τ1、τ2 |
| `wdm` | 默认信道间隔 |

`fiber_params(band, fiber)` 按下列模型换算出任意波长处的参数：

- D(λ) = S0/4·(λ − λ0⁴/λ³)，S(λ) = S0/4·(1 + 3λ0⁴/λ⁴)（G.652 三项 Sellmeier）
- A_eff(λ) = A_eff,1550·(λ/1550)，γ = 2π·n2 / (λ·A_eff)
- α = 波段衰减 + 光纤修正；f0 = c/λ（用于 ASE 与自陡峭）

SSMF 各波段中心波长处的参数（`band_table('SSMF')`）：

| 波段 | λ (nm) | α (dB/km) | D (ps/nm/km) | S | γ (1/W/km) | A_eff (µm²) | 放大器 / NF |
|---|---|---|---|---|---|---|---|
| O | 1310 | 0.33 | −0.18 | 0.090 | 1.84 | 67.6 | PDFA/SOA / 6.5 |
| E | 1410 | 0.30 | 7.94 | 0.073 | 1.59 | 72.8 | BDFA / 6.0 |
| S | 1495 | 0.23 | 13.69 | 0.063 | 1.42 | 77.2 | TDFA / 6.0 |
| C | 1550 | 0.20 | 16.97 | 0.057 | 1.32 | 80.0 | C-EDFA / 5.0 |
| L | 1590 | 0.21 | 19.19 | 0.054 | 1.25 | 82.1 | L-EDFA / 5.5 |
| U | 1650 | 0.25 | 22.28 | 0.049 | 1.16 | 85.2 | Raman / 6.5 |

**用法**

- Python：`fiber_params('S', 'ULL', L=100, pmd=0, raman=True)`
- MATLAB：`fiber_params('S', 'ULL', 'L', 100, 'pmd', 0, 'raman', true)`
- 在系统级使用时，`Config(band=..., fiber=..., fib_kw={...})` 或 `sys_config('band', ..., 'fib_kw', {...})` 会自动生成 `cfg.fib`。
- 要添加新光纤或新波段，在 JSON 里加一项即可，不用改代码。

---

## 4. 仿真约定

| 项目 | 约定 |
|---|---|
| 信号形状 | 双偏振 **2×N 复数**（行 = X/Y 偏振）；时间沿最后一维（Python）/ 第 2 维（MATLAB） |
| 光场单位 | √W，\|E\|² 为瞬时功率 [W] |
| 场约定 | **工程约定 E·exp(+jω0t)**：基带正频率 = 更高光频（更短波长）。WDM 信道、β3、SRS 方向都按此约定 |
| 单位 | 长度 km；α dB/km；D ps/(nm·km)；S ps/(nm²·km)；γ 1/(W·km)；λ nm；其余 SI |
| 循环仿真 | 色散、滤波、时延、重采样（FFT 方式）都是循环运算。激光相位噪声取 Brownian bridge，频偏取整到频率网格，保证首尾连续。<br>例外是相位跟踪（BPS / VV / 导频 / 周跳消除）。它们的滑动平均**不循环**：FO 补偿后残余频偏使相位不再周期，若首尾相接，记录末尾几个符号会出错 |
| 采样率 | 单信道 4 sps；WDM 的仿真带宽需覆盖全部信道（如 32 GBd × 16 sps = 512 GHz，可容纳 7×50 GHz） |
| 相干输出 | 相干接收输出 ∝ R·E_s·E_LO*：LO 频偏 +Δf 时，信号呈现 −Δf 频偏 |
| 帧结构 | [训练 2048 | 每 32 符号 1 个导频]，`kind` 数组标记 0 数据 / 1 训练 / 2 导频 |

---

## 5. 物理模型

### 5.1 收发器件

| 器件 | 模型要点 | Python / MATLAB |
|---|---|---|
| DAC / ADC | ENOB 量化（按 RMS 削波）+ 高斯/贝塞尔/巴特沃斯带宽；ADC 含周期重采样 | `dac`/`adc` · `dac_model`/`adc_model` |
| 激光器 | Wiener 相位噪声（线宽 Δν）、RIN、频偏 | `laser` · `laser_cw` |
| MZM | 推挽余弦传输，零点/正交点偏置 | `mzm` |
| IQ / DP-IQ 调制器 | 嵌套 MZM，正交误差与 IQ 增益不平衡；PBS/PBC | `iq_mod`, `dp_iq_mod` |
| 光放大器 | 增益或 APC + ASE：S_ASE = n_sp·hν·(G−1)，n_sp = NF/2，ν 随波段变化 | `edfa` |
| OSNR 加载 | 按 0.1 nm 参考带宽直接加 ASE | `set_osnr` |
| WSS / 光滤波器 | 超高斯带通，可设中心频率（WDM 选信道） | `optical_filter` |
| WDM 合波 | 各信道频移后求和 | `wdm_mux`, `freq_shift` |
| 偏振器件 | 旋转器、随机 SU(2)、PDL、DGD | `rotator`/`jones_rotator`, `random_unitary`, `pdl`/`pdl_elem`, `dgd`/`dgd_elem` |
| 90° 混频器 | [Es±Elo, Es±jElo]/2 | `hybrid_90` · `hybrid90` |
| PD / BPD | 平方律 + 散粒噪声 2qIB + 热噪声 4kTB/R + 带宽 | `photodiode`, `bpd` |
| 相干接收机 | PBS + 2 个混频器 + 4 个 BPD + IQ skew | `coherent_rx` |

### 5.2 光纤传播（`channel.ssfm` / `ssfm_manakov`）

采用 Manakov 方程（快速随机双折射下平均后的耦合 NLSE），工程约定下为：

```
∂E/∂z = −α/2·E − j(β2ω²/2 + β3ω³/6)·E
        − j·γ·(8/9)·(1 + ω/ω0)·F{ E·[(1−fR)|E|² + fR·(h_R ⊗ |E|²)] }
```

单行输入时退化为标量 NLSE（去掉 8/9 因子）。求解用对称分步傅里叶法：线性半步 → 非线性整步 → 线性半步。

| 效应 | 如何包含 | 开关 / 参数 |
|---|---|---|
| 色散 β2、色散斜率 β3 | 线性算子 | `fib.D`、`fib.S`（随波段自动变化） |
| 损耗 | 线性算子 | `fib.alpha` |
| **SPM** | Kerr 项 | 总是开启 |
| **XPM、FWM** | 对 WDM **全场**求解时自动产生（信道间相互作用不做近似） | 用 `wdm_tx` 生成多信道 |
| 信道内 IXPM / IFWM | 单信道全场 SSFM 自动包含 | — |
| **SRS**（受激拉曼散射） | 延迟响应 fR·h_R，频域 H_R(ω) | `fib.raman = True` |
| **自陡峭** | (1 + ω/ω0) 因子 | `fib.steep = True` |
| PMD | 每步随机 SOP + DGD（粗步长模型），平均 DGD = PMD·√L | `fib.pmd` |
| 非线性相位噪声（Gordon–Mollenauer） | 每跨注入的 ASE 随信号一起非线性传播 | 自动 |
| PDL | 集总元件 | `pdl` |

另有两点说明：

- 开启 SRS 或自陡峭时，非线性子步用 RK4 求解；否则用精确的相位旋转。
- **步长**有两种设法：`nsteps` 为均匀步长；`phi_max` 为自适应步长，每步最大非线性相移，WDM 推荐 0.005 rad。

**未建模**：受激布里渊散射 (SBS) 只对窄线宽 CW 或低速信号重要。高速调制信号的谱宽远大于约 20 MHz 的布里渊增益带宽，阈值很高，这里不考虑。

**物理正确性自检**（`selfcheck` 第 8–12 项）：

| 检查 | 对比对象 | 结果 |
|---|---|---|
| CW 非线性相移 | 解析值 γPL_eff | 误差 < 1e‑3 |
| 基阶孤子传输 5 个色散长度 | 形状不变 | 误差 7.8e‑6（能同时验证色散与 Kerr 符号） |
| 双音 SRS | 解析解 P = 1/(1+e^{−gPz}) | 0.665 W vs 0.665 W |
| FWM | DSF 与 SSMF 的产物强度 | 相差约 37 dB |
| GN 模型 | 数值积分 vs 闭式近似 | 相差 0.19 dB |

### 5.3 GN 模型（`gn_eta` / `gn_model`）

非相干 GN 模型，矩形谱，用于估计中心信道的非线性干扰：P_NLI = η·P³，SNR = P / (P_ASE + η·P³)。

- `method='numeric'`：直接数值积分 GN 参考公式（适合信道数较少的情况）。
- `method='closed'`：Poggiolini 闭式近似，η = 8/27·γ²·L_eff²·asinh(π²/2·|β2|·L_eff,a·Rs²·N^{2Rs/Δf}) / (π·|β2|·L_eff,a·Rs²)，适合满载波段。
- 输出包括最优入纤功率 P_opt = (P_ASE/2η)^{1/3} 和最优 SNR。
- 局限：GN 对 QAM 会高估 NLI 约 0.5–1 dB（EGN 修正未实现），对近零色散光纤不准确，也不含波段间 SRS（ISRS-GN）。

---

## 6. 接收 DSP 算法

`rx_dsp` 依次执行下表各步骤，所有步骤也可以单独调用。

| 步骤 | 算法 | 函数（Python / MATLAB） | 关键参数 |
|---|---|---|---|
| 前端校正 | IQ deskew、GSOP 正交化（IQ 不平衡、去直流） | `deskew`/`iq_deskew`, `gsop` | skew [s] |
| 色散补偿 | 频域 CDC（含 β3）；盲 CD 估计（时钟音扫描） | `cdc`, `cd_estimate` | L, D, S |
| 非线性补偿 | DBP（逐跨反向 Manakov SSFM） | `dbp` | steps/跨，ξ |
| 匹配滤波 | RRC | `matched_filter` | β |
| 定时恢复 | Godard 前馈；Gardner 反馈 + Farrow 插值 | `timing_ff`, `gardner`/`gardner_tr` | μ |
| 偏振解复用 / 均衡 | 2×2 蝶形 FIR：CMA / RDE / MMA / LMS（训练）/ DD-LMS | `mimo_eq` | ntaps, μ |
| 频偏估计 | M 次方 FFT 谱峰；差分（Leven） | `foe`, `fo_comp` | M |
| 载波相位恢复 | BPS；Viterbi-Viterbi；导频辅助；DD-PLL | `bps`/`cpr_bps`, `vv`/`cpr_vv`, `pilot_cpr`/`cpr_pilot`, `pll`/`cpr_pll` | 测试相位数、窗长 |
| 同步 | FFT 循环互相关 + 偏振交换检测 | `sync`/`frame_sync` | — |
| 模糊 / 周跳消除 | 训练 + 导频逐段判定 π/2 旋转 | `fix_ambiguity` | 平均长度 n |
| IM-DD | 实数 LMS-FFE | `ffe`/`ffe_lms` | ntaps, μ |
| 自相干 | Kramers-Kronig 场重建 | `kk`/`kk_rx` | — |

性能指标：`ber`、`ser`、`evm`、`snr`（最小二乘复增益后的有效 SNR）、`q_factor`、`ber_theory`、`gmi`、`mi`、`osnr2snr`、`snr2osnr`。

---

## 7. 神经网络均衡器

### 7.1 什么时候 NN 能替代传统均衡器

线性均衡器（FFE、MIMO、CDC）只能补偿线性 ISI。失真本质上是**非线性且记忆较短**时，神经网络才有明显优势；若失真记忆很长，或者有现成的物理模型可用，物理引导结构更合适。

| 场景 | 失真来源 | 传统方法 | 推荐 NN | 本项目结论 |
|---|---|---|---|---|
| IM-DD 短距 | 平方律检测 × 色散功率衰落、MZM 余弦非线性、DAC/ADC | FFE、Volterra | **MLP / CNN / BiLSTM**（黑盒） | NN 比 Volterra 低 1–2 个数量级 BER |
| 相干长距 | Kerr 非线性（SPM/IXPM/IFWM），记忆约等于色散记忆（800 km 约 100+ 符号） | DBP（多步 SSFM） | **学习型 DBP（LDBP，物理引导）** | 用约一半的乘法量逼近全 DBP |
| 相干长距 | 同上 | — | 黑盒 CNN 后均衡（CDC+CPR 之后） | **几乎无增益**：41 符号窗看不到 100+ 符号的非线性记忆，且失真以相位噪声形式出现 |

### 7.2 实现（Python `optsim/nn_eq.py` ↔ MATLAB `NN/`）

**黑盒均衡器** `NNEqualizer(arch, ntaps, sps)` ↔ `nn_eq_train` / `nn_eq_apply` / `nn_eq_mults`

- **输入**：以第 k 个符号为中心、长 ntaps 采样的滑窗（`sym_windows`）。复信号拆成 [Re, Im] × 偏振个通道。
- **输出**：中心符号，默认**残差学习**：网络只学 y − x_center 的修正量，收敛更快，小网络也够用。
- **结构**：
  - `mlp`：展平 → 全连接（LeakyReLU）
  - `cnn`：1D 卷积（'same' 填充）→ 展平 → 全连接
  - `bilstm`：双向 LSTM，取窗口中心时刻的输出
- **训练**：MSE + Adam；以末尾 10% 样本为验证集，验证集早停并恢复最佳权重。
- **复杂度**：`mults_per_symbol()` / `nn_eq_mults` 给出每个输出符号的实数乘法次数，便于与 FFE / Volterra / DBP 公平比较。

**传统基准**：`volterra(x, ref, n_train, taps=(L1, L2, L3))` ↔ `volterra_eq` 为 1/2/3 阶 Volterra，用岭回归最小二乘求核。`taps=(L, 0, 0)` 时退化为最小二乘 FFE（`ffe_ls`）。

**学习型 DBP** `LDBP(fs, fib, n_spans, p_launch, steps, ntaps)` ↔ `ldbp_init` / `ldbp_train` / `ldbp_apply`

- **结构**：与 DBP 相同，逐跨逆序、每跨 `steps` 步。每步为可学习短复 FIR（色散逆）→ 非线性相位旋转 E·exp(jθ_k(|Ex|²+|Ey|²))，θ_k 可学习。
- **初始化**：
  - FIR 用**加权最小二乘**设计：通带逼近段色散逆，过渡带与阻带也加约束。直接截断会在多步级联中发散，带外增益会被非线性混入带内。
  - θ 初始化为 (8/9)γ·L_eff,段。
  - 因此训练前它就是一个低步数 DBP。
- **训练**：端到端，以匹配滤波后的有效 SNR 为目标（各偏振最小二乘复增益后求 MSE）。训练实现不含激光相位噪声。
- **使用**：训练后作为 `rx_dsp` 的非线性补偿级：`Config(nl_comp=ldbp.apply)` ↔ `cfg.nl_comp = @(y) ldbp_apply(p, y)`，后续 MIMO/CPR 不变。
- **约束**：`ntaps` 必须覆盖单段色散记忆。80 km SSMF 在 2 sps 下约 22 个采样，因此每跨 1 步时 21 抽头无法收敛，41 抽头可以。

### 7.3 结果（测试数据均为独立实现：不同比特、不同噪声）

**IM-DD**（`nn_imdd_pam4`）：56 GBd PAM4，4 km SSMF @1550 nm，MZM 摆幅 0.5 Vπ，41 抽头窗（2 sps）。

| 均衡器 | BER @ −10 dBm | BER @ −6 dBm | BER @ −2 dBm | 乘法/符号 |
|---|---|---|---|---|
| FFE（LS） | 3.1e‑2 | 2.7e‑2 | 2.6e‑2 | 42 |
| Volterra (41, 15, 5) | 2.8e‑3 | 8.7e‑4 | 5.9e‑4 | 197 |
| MLP (128, 64) | 1.4e‑3 | 1.1e‑4 | 5.3e‑5 | 13 504 |
| CNN (16, 16, k=7) | 7.6e‑4 | 5.3e‑5 | 6.1e‑5 | 78 720 |
| BiLSTM (h=32) | 5.0e‑4 | 2.3e‑5 | 7.6e‑6 | 346 432 |

NN 训练（尤其 GPU）不是逐位确定的，BER 很低时误码数也很少，因此 NN 各行的 BER 在重复运行间约有 ±2 倍波动；MATLAB 版（`main_nn_imdd_pam4`）的排序相同，数值在同一量级（例如 −6 dBm：Volterra 8.2e‑4，MLP 3.2e‑4，CNN 5.3e‑5，BiLSTM 1.2e‑4）。

**相干**（`nn_coherent_nlc`）：DP-16QAM 32 GBd，10×80 km SSMF，单信道。测试实现另加 100 kHz 激光线宽。

| 补偿方式 | SNR @0 dBm | @2 dBm | @4 dBm | @6 dBm | 乘法/符号 |
|---|---|---|---|---|---|
| CDC | 20.9 | 20.1 | 17.5 | 13.8 | — |
| CNN 后均衡（黑盒） | 20.9 | 20.1 | 17.5 | 13.8 | 120 704 |
| DBP 1 步/跨 | 21.5 | 21.4 | 19.5 | 16.2 | 2 320 |
| **LDBP 1 步/跨**（41 抽头） | 22.1 | 23.3 | **23.4** | 21.7 | 6 800 |
| **LDBP 4 步/跨**（15 抽头） | 22.0 | 23.5 | 24.7 | **25.6** | 10 560 |
| DBP 10 步/跨 | 22.4 | 24.1 | 25.6 | 26.9 | 23 200 |

DBP 的乘法数按分块 FFT（N = 4096）估算，与 LDBP 同口径。MATLAB 版（`main_nn_coherent_nlc`）与上表相差 ≤ 0.2 dB（例如 6 dBm：LDBP 4 步/跨 25.50 dB，DBP 10 步/跨 26.75 dB）。

### 7.4 使用要点与常见陷阱

- **训练与测试必须用独立实现**（不同随机比特、不同噪声）。用同一序列（尤其是 PRBS）测试，网络会“记住”序列，严重高估增益。
- **数据量**：黑盒 NN 每次训练用约 6.5 万符号；数据太少会欠拟合或过拟合，看验证集 MSE 曲线判断。
- **每个工作点单独训练**（ROP、入纤功率）是常见做法。跨工作点泛化需要混合训练数据。
- **复杂度比 BER 更重要**：NN 的增益要与乘法数一起报告。IM-DD 中 MLP 约为 Volterra 的 70 倍。
- **GPU**：几十秒内即可完成训练；CPU 也能跑，只是较慢。本机实测：**不要让 Python（torch）与 MATLAB 的 GPU 任务同时运行**，否则 torch 初始化 CUDA 可能崩溃（access violation）。若 CUDA 初始化异常，可设 `OPTSIM_DEVICE=cpu`。

---

## 8. API 速查（Python ↔ MATLAB）

Python 函数位于 `optsim.<模块>`，MATLAB 为同名或近名函数文件。每个函数都有说明：Python 用 `help(optsim.dsp.mimo_eq)` 查看，MATLAB 用 `help mimo_eq` 查看。

| 功能 | Python | MATLAB |
|---|---|---|
| 光纤参数 | `params.fiber_params(band, fiber, L=, **kw)` | `fiber_params(band, fiber, 'L',..)` |
| 参数表 | `params.band_table(fiber)` | `band_table(fiber)` |
| 系统配置 | `system.Config(...)`, `cfg.fib`, `cfg.fs` | `sys_config(...)`, `cfg.fib`, `cfg.fs` |
| 单信道发射 | `system.tx_signal(cfg)` | `tx_signal(cfg)` |
| WDM 发射 / 接收 | `system.wdm_tx(cfg)`, `system.wdm_rx(E, cfg, f_ch)` | `wdm_tx(cfg)`, `wdm_rx(E, cfg, f_ch)` |
| 接收 DSP 链 | `system.rx_dsp(r, cfg, ref)` | `rx_dsp(r, cfg, ref)` |
| 星座 | `tx.qam(M)`, `tx.pam(M)`, `tx.psk(M)` | `qam_const`, `pam_const`, `psk_const` |
| 映射 / 解映射 | `tx.modulate`, `tx.demodulate`, `tx.decide` | `bits2sym`, `sym2bits`, `sym_decide` |
| LLR / PCS | `tx.llr`, `tx.pcs`, `tx.pcs_symbols` | `calc_llr`, `pcs_mb` |
| 成帧 / 成形 | `tx.frame`, `tx.pulse_shape`, `tx.matched_filter` | `framing`, `pulse_shape`, `matched_filter` |
| 单跨传输 | `channel.ssfm(E, fs, fib, nsteps=/phi_max=)` | `ssfm_manakov(E, fs, fib, 'phi_max', ..)` |
| 多跨链路 | `channel.link(E, fs, fib, n_spans, phi_max=)` | `fiber_link(E, fs, fib, n_spans, ase, ..)` |
| 线性色散 | `channel.cd(E, fs, L, D, lam, S)` | `fiber_cd(E, fs, L, D, lam, S)` |
| GN 模型 | `channel.gn_model(P, fib, rs, n_spans, n_ch, spacing, method)` | `gn_model(...)` |
| Raman 响应 | `channel.raman_response(w)` | `raman_response(w, tau1, tau2)` |
| 滑窗 / Volterra / LS-FFE | `dsp.sym_windows`, `dsp.volterra`, `dsp.volterra_features`, `dsp.ffe_ls` | `sym_windows`, `volterra_eq`, `volterra_features` |
| NN 均衡器 | `nn_eq.NNEqualizer(arch, ntaps, sps).fit(x, y)`, `eq(x)`, `eq.mults_per_symbol()` | `nn_eq_train`, `nn_eq_apply`, `nn_eq_mults` |
| 学习型 DBP | `nn_eq.LDBP(fs, fib, n_spans, P, steps, ntaps).fit(r, frame, beta)`, `.apply(r)` | `ldbp_init`, `ldbp_train`, `ldbp_apply` |
| 绘图 | `plots.constellation`, `plots.spectrum`, `plots.eye` | `plot_const`, `plot_spectrum`, `plot_eye` |

---

## 9. 示例与参考结果

所有结果均在本机实际运行得到，Python 与 MATLAB 一致（差异来自随机数）。

| 脚本（Python / MATLAB） | 内容 | 参考结果 |
|---|---|---|
| `selfcheck` | 8 项 DSP 恢复 + 5 项光纤物理 | 13/13 OK |
| `coherent_dp_qam` / `main_coherent_dp_qam` | 40 GBd DP-16QAM，单信道，10×80 km C-SSMF，含 DAC/ADC/激光/IQ/PMD/PDL/WSS 全部损伤 | SNR ≈ 19.0 dB，BER ≈ 3–5e‑5，GMI ≈ 4.0 |
| `wdm_system` / `main_wdm_system` | 7×32 GBd，50 GHz 间隔，10×80 km，理想收发机；对比 ASE-only / 单信道 / WDM / WDM+SRS | 见下表 |
| `band_gn_analysis` / `main_band_gn_analysis` | O–U 各波段 × SSMF/ULL/NZDSF，满载 GN 最优 SNR | C-SSMF 19.1 dB；C-ULL 23.1 dB；O-SSMF 4.2 dB |
| `launch_power_sweep` / `main_launch_power_sweep` | 单信道 CDC vs DBP vs GN | CDC 峰值 20.6 dB@0 dBm；GN 预测 20.9 dB@0.7 dBm；6 dBm 时 DBP 比 CDC 高 11.5 dB |
| `ber_vs_osnr` / `main_ber_vs_osnr` | 背靠背 BER–OSNR，与理论对比 | 代价约 0.2 dB |
| `imdd_pam4` / `main_imdd_pam4` | 56 GBd PAM4，2 km，ROP 扫描 + FFE | CD 功率衰落与 RIN 造成约 4e‑4 的误码平台 |
| `phi_otdr_das` / `main_phi_otdr_das` | φ-OTDR 相干 DAS | 定位误差 < 2 m，波形相关系数 1.000 |
| `nn_imdd_pam4` / `main_nn_imdd_pam4` | PAM4 IM-DD：FFE / Volterra / MLP / CNN / BiLSTM | 见第 7.3 节；NN 比 Volterra 低 1–2 个数量级 BER |
| `nn_coherent_nlc` / `main_nn_coherent_nlc` | 相干长距：CDC / DBP / LDBP / 黑盒 CNN | 见第 7.3 节；LDBP 4 步/跨约 25.6 dB，乘法量约为全 DBP 的一半 |

**WDM 结果**（中心信道，每信道 1 dBm，ideal TRx）：

| 情形 | 包含的效应 | SNR（Python / MATLAB） | GN 模型 |
|---|---|---|---|
| ASE only（γ = 0） | 线性传输 + ASE | 22.8 / 23.0 dB | 24.0 dB（仅 ASE） |
| 单信道 | + SPM（含 IXPM/IFWM） | 20.4 / 20.7 dB | 21.1 dB |
| 7 信道 WDM | + XPM + FWM | **19.2 / 19.4 dB** | 18.7 dB |
| 7 信道 WDM + SRS + 自陡峭 | + 拉曼、自陡峭 | 19.2 / 19.4 dB（无可见变化） | — |
| 满载 C 波段（88 信道，GN 闭式） | — | — | 1 dBm 时 16.7 dB；最优 −2.2 dBm 时 19.1 dB |

结果解读：

- **仿真比 GN 的 ASE 值低约 1 dB**：这是 DSP 的实现代价，来自自适应均衡噪声增强、CPR 和激光相位噪声。
- **WDM 仿真比 GN 高约 0.6 dB**：GN 的高斯假设会高估 QAM 信号的 NLI，EGN 理论预言的偏差正是这个量级。
- **邻道带来约 1.2 dB 代价**：XPM/FWM 使中心信道 SNR 比单信道再降约 1.2 dB。边缘信道的邻道较少，SNR 高约 0.3 dB（见 `results/wdm_system.png`）。
- **SRS 无可见影响**：7×50 GHz（约 350 GHz 带宽）内的拉曼功率转移可以忽略。要看到明显的 SRS 倾斜，需要 THz 量级带宽，例如 C+L 波段。
- **样本量提示**：每次仿真约 5 万比特。SNR 约 19–20 dB 时只有零星误码，此时 BER 统计意义有限，应以 SNR/GMI 为准。

---

## 10. FAQ

**Q1：单信道 DP-16QAM 接收 SNR 约 18.5–19 dB，是否偏高？非线性考虑了吗？**

考虑了：单信道仿真中 SPM（以及信道内 IXPM/IFWM）一直存在。逐项分解如下（10×80 km，NF 5 dB，40 GBd，1 dBm）：

| 噪声 / 损伤来源 | SNR |
|---|---|
| 只有 ASE | 58 + P − NF − 跨损耗 − 10lgN − 10lg(Rs/12.5G) = **22.9 dB** |
| + 单信道 SPM | **≈ 20.6 dB**（SSFM 仿真；GN 预测 20.9 dB） |
| + 收发机损伤（6-bit DAC/ADC、带宽、IQ 误差、PMD、PDL） | **≈ 19 dB** ← 原示例的结果 |
| 换成 7 信道 WDM（+XPM、FWM） | 再降约 1.2 dB（第 9 节：20.4 → 19.2 dB，理想收发机） |
| 满载 C 波段（87 信道，GN） | 最优功率下约 19 dB（仅光纤）；1 dBm 时更低 |
| 实际收发机 SNR 上限（约 20–22 dB） | 与上一项合并后约 16–17 dB |

因此 18.5 dB 对“单信道 + 800 km”本身是合理的。之所以显得偏高，是因为原示例**没有 WDM 邻道（XPM/FWM）**，也没有计入实际收发机的 SNR 上限。新增的 `wdm_system` 和 `band_gn_analysis` 补上了前者。如果要和实验（例如 1800 km）对比，可以修改 `n_spans`，并在 Tx 端用 `awgn` 加入收发机噪声。

**Q2：如何换波段、换光纤？**
设置 `Config(band='S', fiber='ULL')` 或 `sys_config('band','S','fiber','ULL')`。α、D、S、γ、NF、f0 会全部随之变化。注意 O 波段 SSMF 的 D≈0，在 WDM 下 FWM 很强。

**Q3：WDM 仿真太慢怎么办？**
可以从以下几方面入手：
- 减少 `n_blocks`、`n_ch` 或 `sps`（仿真带宽只需覆盖全部信道）；
- 增大 `phi_max`（例如 0.01，需先确认结果已收敛）；
- 关闭 `raman`/`steep`（50 GHz 栅格、几百 GHz 带宽内 SRS 几乎可以忽略）；
- 大信道数场景用 GN 模型估计。

**Q4：为什么激光相位噪声要做首尾连接、频偏要取整到网格？相位跟踪又为什么不循环？**
仿真是循环的（FFT）。如果相位首尾不连续，跳变会经 CDC 和滤波扩散到附近上百个符号，形成边缘误码。修正后，单信道 BER 从 4.7e‑4 降到约 3e‑5，而 SNR 几乎不变；后者与 18.9 dB 下的 AWGN 理论值一致。

反过来，CPR 是逐符号的时序过程。FO 补偿后的残余频偏使相位轨迹不再周期，所以 CPR 的滑动窗必须用非循环边界。否则记录末尾的几个符号会和开头的符号混在一起平均，产生零星误码。

**Q5：如何扩展？**
- **新器件**：照 `devices.py` / `Components/` 写一个函数，输入输出都是光场（2×N），即可插入链路。
- **EGN、ISRS-GN**：在 `gn_eta` 中加修正项。
- **BOTDA / OFDR 等传感**：可复用 `laser`、`hybrid_90`、`bpd` 与光纤模块。

---

## 11. 与参考代码的关系

`Long Haul Transmission Experiment Data/Process_Codes` **未作任何修改**。本项目参考其思路重写并推广了以下函数：

| 参考代码 | 本项目 | 主要变化 |
|---|---|---|
| `IQimbalance` | `gsop` | — |
| `FiberDispComp` | `cdc` | 增加 β3 |
| `DBP_Raymond` | `dbp` | 同为 Manakov 8/9、对称 SSFM、逐跨撤销增益；改用参数文件 |
| `Train_LMS_RDE_2by2` / `LMS_DD` | `mimo_eq` | 统一为多模式 |
| `FreOffsetEst` | `foe` | — |
| `BPS_DualPol` | `cpr_bps` | 改为滑动窗 + 解卷绕 |
| `PilotBased_CPE` | `cpr_pilot` | 增加插值 |
| `Synchronization_*_abs` | `frame_sync` | — |
| `SNR_eval` / `EVM_eval` | `calc_snr` / `calc_evm` | — |
| `SymbolMappingAndFraming` / `RemovePilotSymbols` | `framing` + `kind` 标记 | — |

---

## 12. 参考文献

- G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (NLSE、SSFM、Raman 响应、自陡峭、孤子)
- D. Marcuse, C. R. Menyuk, P. K. A. Wai, “Application of the Manakov-PMD equation…,” JLT 15(9), 1997
- O. V. Sinkin et al., “Optimization of the split-step Fourier method…,” JLT 21(1), 2003（步长选择）
- R.-J. Essiambre et al., “Capacity limits of optical fiber networks,” JLT 28(4), 2010
- P. Poggiolini, “The GN model of non-linear propagation in uncompensated coherent optical systems,” JLT 30(24), 2012
- D. Semrau, R. Killey, P. Bayvel, “The Gaussian noise model in the presence of inter-channel stimulated Raman scattering,” JLT 36(14), 2018
- S. J. Savory, “Digital coherent optical receivers: algorithms and subsystems,” IEEE JSTQE 16(5), 2010
- E. Ip, J. M. Kahn, “Compensation of dispersion and nonlinear impairments using digital backpropagation,” JLT 26(20), 2008
- T. Pfau et al., “Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM,” JLT 27(8), 2009
- A. Leven et al., “Frequency estimation in intradyne reception,” PTL 19(6), 2007
- F. M. Gardner, “A BPSK/QPSK timing-error detector for sampled receivers,” IEEE TCOM 34(5), 1986
- A. Mecozzi, C. Antonelli, M. Shtaif, “Kramers–Kronig coherent receiver,” Optica 3(11), 2016
- G. Böcherer et al., “Bandwidth efficient and rate-matched LDPC coded modulation,” IEEE TCOM 63(12), 2015（PAS/PCS）
- A. Alvarado et al., “Achievable information rates for fiber optics,” JLT 36(2), 2018（GMI/MI）
- ITU-T G.652 / G.653 / G.654 / G.655；G.694.1（WDM 栅格）
- Y. Lu et al., “Distributed vibration sensor based on coherent detection of phase-OTDR,” JLT 28(22), 2010
- C. Häger, H. D. Pfister, “Physics-based deep learning for fiber-optic communication systems,” IEEE JSAC 39(1), 2021（LDBP）
- P. J. Freire et al., “Performance versus complexity study of neural network equalizers in coherent optical systems,” JLT 39(19), 2021
- S. Zhang et al., “Field and lab experimental demonstration of nonlinear impairment compensation using neural networks,” Nat. Commun. 10, 3033, 2019
- B. Karanov et al., “End-to-end deep learning of optical fiber communications,” JLT 36(20), 2018
- N. Stojanovic et al., “Volterra and Wiener equalizers for short-reach 100G PAM-4 applications,” JLT 35(21), 2017
- T. A. Eriksson et al., “Applying neural networks in optical communication systems: possible pitfalls,” PTL 29(23), 2017（训练/测试数据陷阱）
- 开源参考：OptiCommPy（Python）、Robochameleon（MATLAB）
