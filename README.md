# Optical Communication System：光纤通信全流程仿真（Python / MATLAB）

本项目由香港理工大学光子研究院开发，包含面向科研的光纤通信与光纤传感仿真代码，覆盖 **相干 DP-QAM / DP-M-PSK（单信道与 WDM）**、**差分 DPSK/DQPSK（DLI 直接检测）**、**IM-DD PAM2/4/8（MZM / EML / DML，PIN / APD）**、**CO-OFDM**，以及 **光纤传感（OTDR / φ-OTDR-DAS / OFDR / BOTDA / 拉曼 DTS / FBG）** 和 **通信-传感一体化（ISAC）**。

- **双版本代码**：Python 与 MATLAB 的模块划分、函数功能和默认参数完全对应，结果可交叉验证，自检的物理量完全一致。
- **统一参数文件**：波段、光纤、放大器参数集中在 `params/fiber_bands.json`，两种编程语言共用。
- **完整非线性模型**：SPM / XPM / FWM / SRS / 自陡峭 / PMD / 色散斜率，另附 GN 模型做解析校验。
- **新型光纤**：空芯光纤（HCF，低时延 / 低非线性 / 模间干扰）、少模光纤（FMF，随机模式耦合 + 多模 Manakov 非线性 + 2M×2M MIMO 相干 SDM 收发）、多模光纤（MMF，850 nm 模式群时延 / EMB / 激励条件 + VCSEL IM-DD）。见 5.5 节。
- **神经网络均衡器**：MLP / 1D-CNN / BiLSTM（替代 FFE、Volterra）与学习型 DBP（LDBP, 替代 DBP），均与传统方法在独立测试数据上对比。
- **常用算法库**：除相干 DSP 链外，还有 PB-NLC、DFE / MLSE、OFDM（Schmidl-Cox、LS 信道估计、CPE）、差分 QAM、规则 LDPC（最小和译码）、NGMI。
- **光纤传感与通感一体**：六种分布式/点式传感技术的前向模型与解调；两种 ISAC 方案（CP-OFDM 背向散射 DAS、相干收发机前向 SOP/相位传感）。见第 8 节。
- **配套学习材料**：原理文档 [`docs/仿真原理与DSP算法详解.md`](docs/仿真原理与DSP算法详解.md)（公式推导+伪代码+参考文献）、传感/通感调研 [`docs/光纤传感与通感一体调研.md`](docs/光纤传感与通感一体调研.md)（技术路线、代表工作对比、研究切入点）与交互式教程 [`Python/notebooks/01_相干DP-QAM全流程教程.ipynb`](Python/notebooks/01_相干DP-QAM全流程教程.ipynb)（逐步调用、逐步可视化，已预先运行并嵌入全部图表输出）。

---

## 目录

1. [快速开始](#1-快速开始)
2. [目录结构](#2-目录结构)
3. [参数文件与多波段](#3-参数文件与多波段)
4. [仿真约定](#4-仿真约定)
5. [物理模型](#5-物理模型)
6. [接收 DSP 算法](#6-接收-dsp-算法)
7. [神经网络均衡器](#7-神经网络均衡器)
8. [光纤传感与通感一体](#8-光纤传感与通感一体)
9. [API 速查（Python ↔ MATLAB）](#9-api-速查python--matlab)
10. [示例与参考结果](#10-示例与参考结果)
11. [FAQ](#11-faq)
12. [与参考代码的关系](#12-与参考代码的关系)
13. [参考文献](#13-参考文献)

---

## 1. 快速开始

**Python**（≥ 3.9，任意平台；系统 Python、venv、conda 环境均可）：依赖见 `Python/requirements.txt`（numpy、scipy、matplotlib，教程另需 pandas、pillow、Jupyter）。
```bash
cd Python
pip install -r requirements.txt        # 一键安装依赖（建议先建虚拟环境：python -m venv .venv）
python examples/selfcheck.py
python examples/coherent_dp_qam.py
python examples/wdm_system.py
```

神经网络均衡器另需 PyTorch：`pip install -r requirements-nn.txt`（GPU 版按 pytorch.org 选择对应 CUDA 的安装命令）；可用环境变量 `OPTSIM_DEVICE=cpu` 强制 CPU。

**MATLAB**：在 R2026a 上测试。需要 Signal Processing Toolbox（`besself`、`hilbert`、`pwelch`、`medfilt1`）；Statistics Toolbox 仅 `pcs_mb` 用到（`randsample`）；神经网络均衡器需要 Deep Learning Toolbox（`trainnet`、`dlarray`；有 GPU 与 Parallel Computing Toolbox 时自动用 GPU）。
```matlab
cd MATLAB
selfcheck
main_coherent_dp_qam
main_wdm_system
```

**最小调用示例：**
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
│   ├── 仿真原理与DSP算法详解.md     原理文档：仿真每一级的物理模型 + 接收 DSP 算法链详解，含公式、伪代码、参考文献
│   └── 光纤传感与通感一体调研.md     传感 / ISAC 调研：技术路线、代表工作对比、现存问题、研究切入点
├── Python/
│   ├── requirements.txt          依赖清单（pip install -r requirements.txt）；requirements-nn.txt 另加 PyTorch
│   ├── notebooks/
│   │   └── 01_相干DP-QAM全流程教程.ipynb   配套交互式教程：逐步调用 optsim 函数，分步可视化每个 DSP 模块的效果
│   ├── optsim/
│   │   ├── utils.py      常数、单位、功率、频域滤波、电低通、周期重采样、分数时延、AWGN
│   │   ├── params.py     读取 JSON，fiber_params(band, fiber) → 光纤参数 dict
│   │   ├── tx.py         PRBS、QAM/PAM/PSK、判决/解映射、差分 PSK/QAM、LLR、PCS、成帧、RRC 成形、OFDM 调制
│   │   ├── devices.py    DAC/ADC、激光器、MZM/IQ/DP-IQ、放大器、WSS、WDM 合波、偏振器件、PD/BPD、相干接收机
│   │   ├── channel.py    色散、Manakov SSFM（SPM/XPM/FWM/SRS/自陡峭/PMD）、多跨链路、GN 模型、空芯光纤模间干扰 imi
│   │   ├── multimode.py  少模/多模光纤：耦合段模型、多模 Manakov 非线性系数、MDL/群时延、SDM 收发与 MIMO DSP、MMF 模式群/EMB/激励
│   │   ├── dsp.py        接收 DSP 算法库（相干链、PB-NLC、FFE/DFE/Volterra/MLSE、OFDM、KK）
│   │   ├── metrics.py    BER/SER/EVM/SNR/Q/GMI/NGMI/MI、理论 BER、OSNR↔SNR
│   │   ├── fec.py        规则 LDPC：构造、系统编码、归一化最小和译码
│   │   ├── filecodec.py  文件编解码：任意文件 → 封装/LDPC/加扰/成帧（tx_signal 格式）→ 接收端还原文件并校验 CRC32
│   │   ├── sensing.py    光纤传感：OTDR、φ-OTDR/DAS、LFM 脉冲压缩、OFDR、BOTDA、拉曼 DTS、FBG
│   │   ├── isac.py       通感一体：CP-OFDM 背向散射距离剖面、Jones/SOP/相位估计
│   │   ├── system.py     Config、tx_signal、wdm_tx、wdm_rx、rx_dsp
│   │   ├── nn_eq.py      神经网络均衡器：NNEqualizer (MLP/CNN/BiLSTM)、LDBP（需 torch，不随 optsim 自动导入）
│   │   └── plots.py      星座图、功率谱、眼图
│   └── examples/         selfcheck, coherent_dp_qam, wdm_system, band_gn_analysis,
│                         launch_power_sweep, ber_vs_osnr, imdd_pam4, imdd_system, psk_coherent,
│                         dpsk_dli, phi_otdr_das, fiber_sensing, isac_ofdm_das, isac_forward_sop,
│                         nn_imdd_pam4, nn_coherent_nlc, hcf_link, fmf_sdm, mmf_vcsel
└── MATLAB/
    ├── Utils/  Tx/  Components/  Channel/  DSP/  Metrics/  System/  NN/   （一个函数一个文件）
    ├── FEC/  Sensing/  ISAC/      LDPC；光纤传感；通感一体
    ├── Multimode/                少模/多模光纤（与 Python optsim/multimode.py 一一对应）
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
- 要添加新光纤或新波段，在 JSON 里加一项即可。

**空芯光纤 `HCF`**（反谐振 DNANF 型典型值，具体数值随光纤设计变化，待核实）：光在空气中传播，光纤条目使用以下可选字段，`fiber_params` 会自动识别：

| 字段 | 含义 | HCF 取值 |
|---|---|---|
| `alpha_db_km` | 绝对衰减（不叠加石英波段衰减） | 0.11 dB/km |
| `D1550` / `S1550` | 线性色散 D(λ) = D1550 + S1550·(λ − 1550) | 2.5 ps/(nm·km) / 0.007 |
| `n2`、`aeff_um2` | 含少量玻璃交叠的等效 n2、有效面积 | 3×10⁻²³ m²/W、500 µm² → γ ≈ 2.4×10⁻⁴ /(W·km)（约为 SSMF 的 1/5000） |
| `ng` | 群折射率（所有光纤都输出 `ng` 与 `latency` [µs/km]，未给出时取 1.468） | 1.0003 → 3.34 µs/km（SSMF 4.90 µs/km，低约 32%） |
| `medium` | `'air'` 时 fR = 0（空气中无石英拉曼响应） | air |

**少模 / 多模光纤**参数在 JSON 的 `multimode` 部分，用 `mm_params(name)` 读取：

| 名称 | 类型 | 主要参数 |
|---|---|---|
| `FMF3` | 3 模 FMF（LP01 + LP11a/b，6 个空间-偏振通道） | 基模模场半径 6 µm，DMGD 50 ps/km，MDL 0.01 dB/km，强耦合（lc = 1 km） |
| `FMF6` | 6 模 FMF（3 个模式群，12 个通道） | 模场半径 7 µm，DMGD 0 / 60 / 120 ps/km，弱耦合（群间 −30 dB/km） |
| `OM3` / `OM4` | 50 µm 渐变折射率 MMF，850 nm | NA 0.2，V ≈ 37，18 个模式群；剖面指数偏差按满注入 EMB 2000 / 4700 MHz·km 标定 |

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
| 帧结构 | [训练 2048 每 32 符号 1 个导频]，`kind` 数组标记 0 数据 / 1 训练 / 2 导频 |

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
| PD / BPD | 平方律 + 散粒噪声 2qIB + 热噪声 4kTB/R 或 TIA 输入噪声电流密度 irnd + 带宽 | `photodiode`, `bpd` |
| EML（电吸收调制） | 指数吸收 T(u) = 10^(−ER(1−u)/20)，瞬态啁啾 E·T^((1+jα)/2) | `eml` |
| DML（直调激光） | P = P̄(1+m·u)，啁啾 Δν = α/4π·(d ln P/dt + κP)（瞬态 + 绝热），线宽、RIN | `dml` |
| APD | 增益 M，倍增噪声 2qM²F(M)(RP̄+i_d)B，F = k_A·M + (1−k_A)(2−1/M) | `apd` |
| DLI 差分接收机 | 两个 1 符号延迟 MZI（φ = 0、π/2）+ 2 个 BPD → z ∝ y_k·y*_{k−1}，偏振不敏感 | `dli`, `dli_rx` |
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

### 5.4 IM-DD / PAM 与 PSK / DPSK 链路

**IM-DD PAM-M**（`ImddConfig` → `imdd_tx` → `imdd_link` → `imdd_rx`；MATLAB 为 `imdd_config` 等同名函数）：

```
比特 → Gray PAM-M（电平 ±1/(M−1)…）→ RC 成形 → DAC → MZM（正交偏置，零啁啾）/ EML（α）/ DML（α, κ）
     → SSMF（色散 + 斜率 + 损耗，参数随波段）→ VOA（设定 ROP）→ PIN 或 APD + TIA 噪声 → ADC 2 sps
     → FFE / FFE+DFE / Volterra / FFE+后滤波+MLSE → PAM 判决 → BER、KP4 灵敏度
```

平方律检测把色散变成功率衰落，小信号响应为 |H(f)| = |cos θ − α·sin θ|，其中 θ = πλ²DLf²/c（`power_fading`）。这是 C 波段 IM-DD 的主要距离限制；O 波段 D≈0，几乎不受影响。

`imdd_system` 的主要结果（Python，MATLAB 一致）：

| 场景 | 结果 | 解读 |
|---|---|---|
| 112 Gb/s，O 波段 2 km，MZM + PIN + FFE | KP4 灵敏度：PAM2 −15.3 dBm，PAM4 −11.5 dBm，PAM8 −7.2 dBm | 同比特率下每升一阶约差 4 dB（电平间距缩小）；PAM8 对 ENOB 敏感（ENOB 6 时 SNR 封顶约 21 dB，无法过 KP4，示例用 ENOB 7） |
| PAM4 + APD（M = 10，k_A = 0.5） | −14.7 dBm，比 PIN 好约 3.1 dB | 增益压过 TIA 热噪声 |
| C 波段 10 km 小信号响应 | MZM/EML 仿真与理论陷波位置、深度吻合（误差 < 0.01） | α>0 的 EML 陷波前移（加剧衰落），α<0 后移；DML 绝热啁啾额外改变低频响应 |
| C 波段 PAM4 56 GBd，ROP −4 dBm，KP4 最大距离 | FFE 3.0 km，FFE+DFE 3.5 km，Volterra 4.0 km，FFE+PF+MLSE 3.0 km | 约 5 km 处第一个衰落陷波（27 GHz）进入信号带宽，线性均衡无法恢复谱零点 → BER 陡升 |

**相干 M-PSK**：`Config(mod='psk', M=...)` 直接复用相干 DSP 链。`rx_dsp` 用 `rot_order` 求星座的旋转对称阶数 rot（BPSK 2，QPSK 4，8PSK 8，16PSK 16），并据此自动设定：
- FOE 的 M 次方阶数；
- BPS 的搜索范围（±π/rot）和解卷绕周期；
- 模糊消除的步长（2π/rot）。

背靠背 HD-FEC 门限处的 OSNR 与 AWGN 理论相差 0.04–0.36 dB。同为 4 bit/符号，16QAM 比 16PSK 少需 3.7 dB OSNR，因为最小欧氏距离更大。

**差分 M-DPSK + DLI**（`dpsk_dli`）：
- **流程**：差分编码（`diff_psk`）→ RRC → 激光 → OSNR → 光匹配滤波 → `dli_rx` → 符号中心采样 → `diff_psk_demod`。
- **优点**：无需本振和载波恢复。
- **代价**：相对相干检测，DBPSK、DQPSK、D8PSK 在 HD-FEC 处分别多需 1.4、2.4、2.9 dB OSNR；DQPSK、D8PSK 仿真与差分检测理论吻合到 0.2 dB 以内。
- **DBPSK 偏差**：仿真比理论差约 0.9 dB。原因是偏振不敏感的 DLI 还收到正交偏振的 ASE-ASE 拍频，而理论只计同偏振噪声。已验证：只保留同偏振 ASE 时，仿真与理论完全重合。
- **相位噪声容限**：BER 在 Δν·T ≈ 10⁻³ 以下基本不受影响，超过后 D8PSK 最先劣化。

### 5.5 空芯、少模与多模光纤

**空芯光纤（HCF）**：直接作为新光纤类型进入现有链路（`Config(fiber='HCF')` / `sys_config('fiber', 'HCF')`），SSFM、GN 模型、DBP 都不用改。与 SSMF 的主要区别：
- 非线性系数低约 3.7 个数量级 → 最优入纤功率高 20 dB 以上（实际受放大器输出功率限制）；
- 色散约 2.5 ps/(nm·km)，色散补偿负担小；
- 群折射率 ≈ 1 → 时延比石英光纤低约 32%；
- 特有损伤**模间干扰（IMI）**：基模分布式耦合到高阶模、以不同群速度传播后再耦合回来，形成一串弱的随机相位延迟回波（`channel.imi` / `fiber_imi`）。回波时延远大于均衡器窗口时 IMI 表现为加性干扰，1/SNR ≈ 1/SNR₀ + N·η。

**少模光纤（FMF）——耦合段模型**（`multimode.ssfm_mm` / `mm_link`，MATLAB `Multimode/`）。光场为 2M×N（M 个空间模式 × 2 偏振）。光纤按耦合长度 lc 分段：
- 段内各通道独立传播：模式群各自的衰减（MDL）、群时延（DMGD，去均值）、色散 D/S；
- 段间乘随机酉矩阵：`strong` 为全部 2M 通道 Haar 随机强耦合；`weak` 只在简并模式群内强耦合、群间按 `xt_db_km` 弱耦合；
- 非线性取模式群平均的**多模 Manakov 方程** ∂E_g/∂z = −j·Σ_h Γ_gh |E_h|² E_g（Mecozzi 2012；Mumtaz 2013）。系数 Γ = (n2ω0/c)·κ，κ 由 Kerr 张量对群内 Haar 随机酉变换求平均得到：用复高斯矩缩并，κ_gg = Σ_{i,j∈g} S_ij/(n_g(n_g+1))，κ_gh = Σ S_ij/(n_g n_h)，S_ij = f_ab·(2/3 + 4/3·δ_pq)，f_ab = ∬ψa²ψb² dA 为模场交叠积分（抛物线剖面 Hermite-Gauss 模）。单模时 κ = 8/9·f，即标准 Manakov 系数（selfcheck 验证到 10⁻⁶）；
- 少模 EDFA：补偿平均跨段损耗，叠加模式相关增益 MDG 与各通道 ASE。

`mm_transfer` 给出与同一种子的 `mm_link` 逐点一致的线性传递矩阵 H(f)，由此计算 MDL（奇异值比 `mdl_db`）和群时延（群时延算符 j·H⁻¹·dH/dω 的本征值 `group_delays`）。强耦合时群时延扩展 ∝ √L，弱耦合时 ∝ L（Ho & Kahn 2011）。

**SDM 收发**：`sdm_tx`（M 个 DP-QAM 发射机共用一个激光器）→ `mm_link` → `sdm_rx`（M 个相干接收机共用本振）→ `sdm_rx_dsp`：
- 模式混合与群时延扩展未补偿时，4 次方频偏估计失效（实测频偏为 0 时也估出 1.1 GHz），因此 MIMO 之前用**训练序列辅助粗频偏估计**（`foe_training`）：所有通道频偏相同，y_i(t)·s_j*(t−k) 的频谱在 Δf 处有峰，对全部 (i, j, k) 非相干累加；
- 2M×2M MIMO：训练序列 DA-LMS 预收敛，再以其抽头为初值全帧 RDE；步长按 LMS 稳定条件 μ ∝ 1/(P·ntaps) 归一化（固定 1e‑3 在 12×12、161 抽头时发散）；
- 抽头数须覆盖群时延扩展，经验上 ntaps ≈ 4×扩展符号数 + 1；12×12 MIMO 需要更长的训练序列（示例用 8192）。

**多模光纤（MMF，850 nm）**：α 剖面渐变折射率光纤，G = floor(V/2) 个主模式群，第 g 群含 g 个空间模式：
- 群时延用 Gloge-Marcatili / Olshansky-Keck 公式（忽略剖面色散 ε）：τ_g = L·ng/c·[1 + Δ(α−2)/(α+2)·x^{2α/(α+2)} + Δ²(3α−2)/(2(α+2))·x^{4α/(α+2)}]，x ≈ g²/(G(G+1))；最优剖面 α_opt ≈ 2 − 12Δ/5；
- 激励条件 `mmf_launch`：满注入（各模式等功率）或高斯光斑对 HG 模的交叠积分（模场失配时只激发偶数阶群，P_N = (1−r²)·r^N；横向偏移时为泊松分布，均与解析式一致到 10⁻⁶）；
- 全面积探测器上各模式正交，探测功率 = Σ_g |E_g|²，所以 `mmf_link` 输出 G×N 群场，可直接送入 `imdd_rx`；
- `mmf_emb` 由模式群冲激响应计算 EMB。离散冲激响应在 −3 dB 带宽之外有深陷波，比实际光纤（DMD 连续、含模式耦合）更悲观；未建模模式噪声、模式分配噪声和多横模 VCSEL 的光谱宽度（VCSEL 用 `dml` 近似）。

---

## 6. 接收 DSP 算法

`rx_dsp` 依次执行下表各步骤，所有步骤也可以单独调用。

| 步骤 | 算法 | 函数（Python / MATLAB） | 关键参数 |
|---|---|---|---|
| 前端校正 | IQ deskew、GSOP 正交化（IQ 不平衡、去直流） | `deskew`/`iq_deskew`, `gsop` | skew [s] |
| 色散补偿 | 频域 CDC（含 β3）；盲 CD 估计（时钟音扫描） | `cdc`, `cd_estimate` | L, D, S |
| 非线性补偿 | DBP（逐跨反向 Manakov SSFM）；一阶微扰 PB-NLC（三元组 \|mn\| ≤ L，训练符号 LS 拟合系数，CPR 之后使用） | `dbp`, `pb_nlc` | steps/跨，ξ；L |
| 匹配滤波 | RRC | `matched_filter` | β |
| 定时恢复 | Godard 前馈；Gardner 反馈 + Farrow 插值 | `timing_ff`, `gardner`/`gardner_tr` | μ |
| 偏振解复用 / 均衡 | 2×2 蝶形 FIR：CMA / RDE / MMA / LMS（训练）/ DD-LMS | `mimo_eq` | ntaps, μ |
| 频偏估计 | M 次方 FFT 谱峰；差分（Leven） | `foe`, `fo_comp` | M |
| 载波相位恢复 | BPS（搜索范围 ±π/rot）；Viterbi-Viterbi（M = rot）；导频辅助；DD-PLL | `bps`/`cpr_bps`, `vv`/`cpr_vv`, `pilot_cpr`/`cpr_pilot`, `pll`/`cpr_pll` | 测试相位数、窗长、rot |
| 同步 | FFT 循环互相关 + 偏振交换检测 | `sync`/`frame_sync` | — |
| 模糊 / 周跳消除 | 训练 + 导频逐段判定 2π/rot 旋转（rot 为星座旋转对称阶数，`rot_order`：QAM/QPSK 4，BPSK 2，8PSK 8） | `fix_ambiguity` | 平均长度 n, rot |
| IM-DD 均衡 | 实数 LMS-FFE；FFE+DFE；Volterra（2/3 阶）；FFE + 噪声白化后滤波 + MLSE | `ffe`/`ffe_lms`, `dfe`/`dfe_lms`, `volterra`/`volterra_eq`, `pf_mlse` | ntaps, μ, nb |
| 差分检测 | M-DPSK 差分判决（DLI 输出或数字差分积） | `diff_psk_demod` | — |
| 自相干 | Kramers-Kronig 场重建 | `kk`/`kk_rx` | — |
| IM-DD 序列检测 | LS 信道估计 + Viterbi MLSE（M^L 状态） | `ch_est`, `mlse`/`mlse_viterbi` | 信道抽头数 |
| OFDM | 加/去 CP + (I)FFT；Schmidl-Cox 定时与频偏；多训练符号 LS 信道估计（先对齐公共相位再平均）+ 导频 CPE | `tx.ofdm_mod`/`ofdm_mod`, `ofdm_demod`, `schmidl_cox`, `ofdm_eq` | nfft, ncp, 训练符号数 |
| 差分 QAM | 象限差分 + 象限内 Gray，对 π/2 模糊与周跳免疫（M=4 即 DQPSK） | `tx.diff_qam`/`diff_qam`, `diff_qam_demod` | — |
| 信道编码 | 规则 (dv, dc) LDPC：Gallager 构造 + GF(2) 消元系统编码 + 归一化最小和译码 | `fec.ldpc_make/encode/decode`, `ldpc_make/encode/decode` | n, dv, dc, 迭代次数 |
| 文件还原 | 训练/导频估计噪声方差 → 逐比特 LLR → 解扰 → LDPC 译码 → CRC32 校验（仅 Python） | `filecodec.decode_file` | iters |

性能指标：`ber`、`ser`、`evm`、`snr`（最小二乘复增益后的有效 SNR）、`q_factor`、`ber_theory`、`gmi`、`ngmi`/`calc_ngmi`、`mi`、`osnr2snr`、`snr2osnr`。

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
- **GPU**：几十秒内即可完成训练；CPU 也能跑，只是较慢。**不要让 Python（torch）与 MATLAB 的 GPU 任务同时运行**，否则 torch 初始化 CUDA 可能崩溃（access violation）。若 CUDA 初始化异常，可设 `OPTSIM_DEVICE=cpu`。

---

## 8. 光纤传感与通感一体

背景、技术路线和文献对比见 [`docs/光纤传感与通感一体调研.md`](docs/光纤传感与通感一体调研.md)。代码在 Python `optsim/sensing.py`、`optsim/isac.py` ↔ MATLAB `Sensing/`、`ISAC/`。

### 8.1 建模约定

- 光纤离散为间距 dz 的瑞利散射单元，r_i ~ CN(0, αs·S·dz)，并乘往返损耗 e^{−αz}（αs = 4.6e‑5 /m，S = 1.5e‑3）。
- 取 **dz = c/(2·n_g·fs)**（`cell_len`）时，第 i 个单元恰好延迟 i 个采样，背向散射就是探测光与 h = r·e^{jφ} 的卷积（`backscatter`）。扰动只改变 φ（往返相位 = 2 × 单程）。
- 传感系数取石英单模光纤 @1550 nm 的典型值：BFS ≈ 10.85 GHz、1 MHz/K、0.05 MHz/με；拉曼频移 13.2 THz；FBG 约 1.2 pm/με、11 pm/K；OFDR 应变系数 K ≈ 0.78。具体光纤需要标定。

### 8.2 传感技术与函数

| 技术 | 前向模型 | 解调 | Python / MATLAB |
|---|---|---|---|
| OTDR | P(z) ∝ P0·(cW/2n_g)·αs·S·e^{−2αz} × 事件损耗，按脉宽平滑 | 平滑斜率中位数 → 衰减；去斜后左右窗均值差 → 事件 | `otdr_trace`, `otdr_events` |
| φ-OTDR / DAS | 脉冲 ⊛ h，相干 IQ 接收 | 标距 G 差分相位、距离平均、慢时间解卷绕；中值滤波定位 | `rayleigh_fiber`, `backscatter`, `das_phase`, `vib_locate`, `ddc`（外差） |
| 啁啾脉冲 φ-OTDR | LFM 探测 | 脉冲压缩（匹配滤波），分辨率 c/(2n_gB) | `lfm`/`lfm_pulse`, `pulse_compress` |
| OFDR | 扫频拍频 = h 的 FFT；应变 → 局部附加时延 | 距离域分段 → 局部瑞利谱互相关 → 谱移 Δν = −f0·K·ε | `ofdr_scan`, `ofdr_shift` |
| BOTDA | 洛伦兹增益谱 g(f − BFS(z))，按泵浦脉宽平均 | 逐位置二次拟合峰值 → BFS → ΔT / ε | `bfs`/`bfs_shift`, `brillouin_gain`, `botda`/`botda_bgs`, `bfs_fit` |
| 拉曼 DTS | R(T) = (λs/λas)⁴·e^{−hΔν/kT} | 参考段标定 + 差分损耗修正 → T | `raman_ratio`, `dts_temp` |
| FBG | 均匀光栅耦合模解析解（Erdogan 1997） | 半高以上质心 → Δλ → ε / ΔT | `fbg_reflect`, `fbg_shift`, `fbg_peak` |

### 8.3 通感一体（ISAC）

| 方案 | 原理 | 函数 | 关键约束 |
|---|---|---|---|
| **后向：CP-OFDM 数据信号作 DAS 探测光**（可定位） | 发射端接收自身信号的瑞利背向散射；逐子载波 Y/X（OFDM 雷达式处理）再 IFFT → 每个 OFDM 符号得到一次 h(z)，之后同 φ-OTDR | `isac.ofdm_profile` / `ofdm_profile` | CP ≥ 光纤往返时延（否则空间 ISI）；距离分辨率 ≈ c/(2n_gB)；慢时间采样率 = 1/OFDM 符号时长；子载波间隔小，需要窄线宽激光 |
| **前向：相干收发机 DSP 传感**（积分式） | 用判决符号做数据辅助块 Jones 估计 → 输出 SOP（Stokes）与公共相位 ∠det(J)/2 的时间序列 → 频谱 / 带通 | `isac.jones_est`, `jones_sop`, `jones_phase`, `stokes` / `stokes_vec` | 无空间分辨率；相位传感要求激光线宽远小于扰动频率 |

### 8.4 结果（Python / MATLAB 均实测，二者一致）

**`fiber_sensing` / `main_fiber_sensing`**（六种技术，各自给出解调误差）

| 技术 | 场景 | 结果 |
|---|---|---|
| OTDR | 20 km，100 ns；5 km 处 0.5 dB、12 km 处 1.0 dB | 衰减 0.200 dB/km；事件 5.00 km/0.49 dB、12.00 km/0.99 dB |
| φ-OTDR 分辨率 | 矩形 100 ns vs LFM 1 µs / 100 MHz | −3 dB 宽度 10.2 m vs 1.23 m（理论 1.02 m） |
| BOTDA | 2 km，2 m 泵浦；+30 K 热点、300 με 应变段 | BFS rms 误差 0.33 MHz；热点 30.0 K；应变 299 με |
| 拉曼 DTS | 4 km，0.5% 强度比噪声；70 °C 热区 | 背景 rms 0.67 K；热区 70.1 °C（MATLAB 69.8 °C） |
| FBG | 500 / 1000 με | 峰值漂移 604.5 / 1209 pm → 500 / 1000 με |
| OFDR | 1 THz 扫频（0.1 mm 单元），26 mm 窗；200 με 段 | 201 με；谱移量化步长约 3.2 με |

**`phi_otdr_das` / `main_phi_otdr_das`**：2 km，500 Hz 振动于 1200–1210 m，定位 1208.2 m（MATLAB 1210.8 m），波形相关系数 1.000。

**`isac_ofdm_das` / `main_isac_ofdm_das`**：1 km 光纤，500 MS/s 16QAM-OFDM（nfft 16384，CP 5000，导频每 64 个子载波一个，8 个训练符号），100 Hz 线宽激光。

| | Python | MATLAB |
|---|---|---|
| 传感：800 Hz 振动（真值 600–605 m） | 定位 601.4 m，相关系数 1.000 | 602.5 m，0.998 |
| 距离分辨率 / 慢时间采样率 | ≈ 0.23 m / 23.4 kHz | 同左 |
| 通信（AWGN 20 dB） | SNR 17.0 dB，BER 9.0e‑4 | SNR 17.2 dB，BER 7.8e‑4 |
| 净速率 / CP 开销 | 1.36 Gb/s / 23% | 同左 |

通信 SNR 比 AWGN 低约 3 dB，主要来自激光相噪引起的载波间干扰：子载波间隔仅 30.5 kHz，两个 100 Hz 激光的 ICI 约为 π·Δν·T ≈ −17 dB。这是"CP ≥ 往返时延"迫使 nfft 很大所带来的代价。

**`isac_forward_sop` / `main_isac_forward_sop`**：DP-16QAM，10 MBd，50 ms；60 Hz 双折射摆动 + 120 Hz 振动相位；发/收合计线宽 10 Hz（Δν/Rs = 1e‑6，相当于 32 GBd 下 32 kHz）。

| | Python | MATLAB |
|---|---|---|
| 通信（2×2 LMS → DD，AWGN 18 dB） | SNR 17.6 dB，BER 2.9e‑4 | SNR 17.6 dB，BER 2.8e‑4 |
| SOP 峰 / 相位峰 | 60 Hz / 120 Hz | 60 Hz / 120 Hz |
| 120 Hz 相位：相关系数 / 幅度（真值 1 rad） | 1.000 / 0.96 rad | 0.989 / 1.02 rad |

### 8.5 使用要点

- **缩小规模仿真**：慢时间（kHz 级振动）需要 ms 级记录，因此示例把采样率降到 MS/s–百 MS/s 量级。**激光相噪按 Δν·Ts 起作用**，降速时线宽必须同比例缩小，否则 DD-LMS 会周跳（曾经实测 200 Hz @ 10 MBd 时 BER ≈ 0.36）。
- **频率分辨率 = 1/记录时长**：扰动频率不在网格上时谱峰会偏到相邻格（50 ms 记录下分辨率为 20 Hz）。
- **相干衰落**：瑞利幅度服从瑞利分布，弱点处相位不可靠。`das_phase` 的 `n_avg` 做距离平均，`vib_locate` 的中值滤波去掉伪峰。
- **OFDR 统计**：窗口跨越应变边界时会得到中间值，这是空间分辨率效应。评估噪声时只用整窗都不受应变的位置。

---

## 9. API 速查（Python ↔ MATLAB）

Python 函数位于 `optsim.<模块>`，MATLAB 为同名或近名函数文件。每个函数都有说明：Python 用 `help(optsim.dsp.mimo_eq)` 查看，MATLAB 用 `help mimo_eq` 查看。

| 功能 | Python | MATLAB |
|---|---|---|
| 光纤参数 | `params.fiber_params(band, fiber, L=, **kw)` | `fiber_params(band, fiber, 'L',..)` |
| 参数表 | `params.band_table(fiber)` | `band_table(fiber)` |
| 系统配置 | `system.Config(...)`, `cfg.fib`, `cfg.fs` | `sys_config(...)`, `cfg.fib`, `cfg.fs` |
| 单信道发射 | `system.tx_signal(cfg)` | `tx_signal(cfg)` |
| WDM 发射 / 接收 | `system.wdm_tx(cfg)`, `system.wdm_rx(E, cfg, f_ch)` | `wdm_tx(cfg)`, `wdm_rx(E, cfg, f_ch)` |
| 接收 DSP 链 | `system.rx_dsp(r, cfg, ref)` | `rx_dsp(r, cfg, ref)` |
| 星座 | `tx.qam(M)`, `tx.pam(M)`, `tx.psk(M)`, `tx.constellation(fmt, M)`, `tx.rot_order(c)` | `qam_const`, `pam_const`, `psk_const`, `constellation`, `rot_order` |
| 差分 PSK | `tx.diff_psk(bits, M)`, `tx.diff_psk_demod(z, M)` | `diff_psk`, `diff_psk_demod` |
| IM-DD 系统 | `system.ImddConfig(...)`, `imdd_tx(cfg)`, `imdd_link(E, cfg)`, `imdd_rx(E, cfg, ref, rop)` | `imdd_config(...)`, `imdd_tx`, `imdd_link`, `imdd_rx` |
| IM-DD 器件 | `devices.eml`, `devices.dml`, `devices.apd`, `photodiode(irnd=)` | `eml`, `dml`, `apd`, `photodiode(.., struct('irnd',..))` |
| DPSK 接收 | `devices.dli(E, fs, tau, phi)`, `devices.dli_rx(E, fs, rs)` | `dli`, `dli_rx` |
| 功率衰落 / 灵敏度 | `channel.power_fading(f, L, D, lam, alpha)`, `metrics.sensitivity(x, ber, thr)`, `metrics.KP4_FEC` | `power_fading`, `sensitivity` |
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
| 差分 QAM | `tx.diff_qam(bits, M)`, `tx.diff_qam_demod(y, M)` | `diff_qam`, `diff_qam_demod`, `qam_quadrant` |
| OFDM | `tx.ofdm_mod(X, ncp)`, `dsp.ofdm_demod(y, nfft, ncp)`, `dsp.ofdm_eq(Y, X, n_train, pilots)`, `dsp.schmidl_cox(y, L, fs)` | `ofdm_mod`, `ofdm_demod`, `ofdm_eq`, `schmidl_cox` |
| PB-NLC | `dsp.pb_nlc(y, ref, n_train, L)` | `pb_nlc` |
| DFE / MLSE | `dsp.dfe(x, ref, nf, nb, mu, sps, n_train, levels)`, `dsp.ch_est(x, ref, ntaps)`, `dsp.mlse(x, h, levels)` | `dfe_lms`, `ch_est`, `mlse_viterbi` |
| LDPC | `fec.ldpc_make(n, dv, dc)`, `fec.ldpc_encode(u, code)`, `fec.ldpc_decode(llr, code)` | `ldpc_make`, `ldpc_encode`, `ldpc_decode` |
| 文件编解码 | `filecodec.file_tx_signal(path, cfg, fec_kind)`, `filecodec.decode_file(y, ref, out_dir)`, `pack_file`, `unpack_file` | —（仅 Python） |
| 空芯光纤 | `fiber_params('C', 'HCF')`, `fib['latency']`, `channel.imi(E, fs, imi_db, tau_max)` | `fiber_params('C', 'HCF')`, `fib.latency`, `fiber_imi` |
| 少模/多模参数 | `multimode.mm_params(name, L=, **kw)` | `mm_params(name, 'L', ..)` |
| 模场 / 非线性系数 | `multimode.hg1d`, `overlap(hg, w)`, `kerr_coeffs(f, labels)`, `nl_matrix(mmf)` | `hg1d`, `mm_overlap`, `kerr_coeffs`, `mm_nl_matrix` |
| FMF 传输 | `multimode.ssfm_mm(E, fs, mmf, rng=)`, `mm_link(E, fs, mmf, n_spans, rng=, noise_rng=)`, `coupling`, `haar_unitary` | `ssfm_mm(E, fs, mmf, stream)`, `mm_link(E, fs, mmf, n_spans, ase, seed)`, `mm_coupling`, `haar_unitary` |
| 传递矩阵 / MDL / 群时延 | `multimode.mm_transfer(fs, n, mmf, n_spans, rng)`, `mdl_db(H)`, `group_delays(H, fs)` | `mm_transfer`（2M×2M×n）, `mm_mdl`, `mm_group_delays` |
| SDM 收发 | `multimode.sdm_tx(cfg, M)`, `sdm_rx(E, cfg)`, `foe_training`, `sdm_rx_dsp(r, cfg, refs, mmf, n_spans, ntaps)` | `sdm_tx`, `sdm_rx`, `foe_training`, `sdm_rx_dsp` |
| MMF | `multimode.mmf_groups(mmf)`, `mmf_launch(mmf, kind, w_um, offset_um)`, `mmf_link(E, fs, mmf, P, L)`, `mmf_emb(mmf, P)` | `mmf_groups`, `mmf_launch`, `mmf_link`, `mmf_emb` |
| NGMI | `metrics.ngmi(rx, tx, const)` | `calc_ngmi` |
| 瑞利传感 | `sensing.cell_len`, `rayleigh_fiber`, `backscatter`, `lfm`, `pulse_compress`, `ddc`, `das_phase`, `vib_locate`, `otdr_trace`, `otdr_events`, `ofdr_scan`, `ofdr_shift` | 同名（`lfm` → `lfm_pulse`；常数见 `sens_const`） |
| 布里渊 / 拉曼 / FBG | `sensing.bfs`, `brillouin_gain`, `botda`, `bfs_fit`, `raman_ratio`, `dts_temp`, `fbg_reflect`, `fbg_shift`, `fbg_peak` | `bfs_shift`, `brillouin_gain`, `botda_bgs`, `bfs_fit`, 其余同名 |
| 通感一体 | `isac.ofdm_profile`, `isac.jones_est`, `isac.jones_sop`, `isac.jones_phase`, `isac.stokes` | `ofdm_profile`, `jones_est`, `jones_sop`, `jones_phase`, `stokes_vec` |

---

## 10. 示例与参考结果

注意：数值差异仅来自随机数。

| 脚本（Python / MATLAB） | 内容 | 参考结果 |
|---|---|---|
| `selfcheck` | 8 项 DSP 恢复 + 5 项光纤物理 + 5 项扩展算法（OFDM/Schmidl-Cox、差分 QAM 抗周跳、PAM4 FFE/DFE/MLSE、PB-NLC、LDPC）+ 6 项空芯/少模/多模光纤（HCF SPM/时延/IMI、多模 Kerr 系数、模式耦合酉性与传递矩阵、群时延 √L / L 规律、MMF 激励解析式、EMB 标定） | 24/24 OK |
| `coherent_dp_qam` / `main_coherent_dp_qam` | 40 GBd DP-16QAM，单信道，10×80 km C-SSMF，含 DAC/ADC/激光/IQ/PMD/PDL/WSS 全部损伤 | SNR ≈ 19.0 dB，BER ≈ 3–5e‑5，GMI ≈ 4.0 |
| `wdm_system` / `main_wdm_system` | 7×32 GBd，50 GHz 间隔，10×80 km，理想收发机；对比 ASE-only / 单信道 / WDM / WDM+SRS | 见下表 |
| `band_gn_analysis` / `main_band_gn_analysis` | O–U 各波段 × SSMF/ULL/NZDSF，满载 GN 最优 SNR | C-SSMF 19.1 dB；C-ULL 23.1 dB；O-SSMF 4.2 dB |
| `launch_power_sweep` / `main_launch_power_sweep` | 单信道 CDC vs PB-NLC vs DBP vs GN | CDC 峰值 20.7 dB@0 dBm；GN 预测 20.9 dB@0.7 dBm；PB-NLC（L=10）峰值 21.2 dB@2 dBm，4/6 dBm 时比 CDC 高 1.1/1.3 dB；6 dBm 时 DBP 比 CDC 高 11.4 dB。PB-NLC 增益有限，因为 ±10 个符号的三元组远小于 800 km 的色散记忆（上百个符号） |
| `ber_vs_osnr` / `main_ber_vs_osnr` | 背靠背 BER–OSNR，与理论对比 | 代价约 0.2 dB |
| `imdd_pam4` / `main_imdd_pam4` | 56 GBd PAM4，2 km，ROP 扫描 + FFE | CD 功率衰落与 RIN 造成约 4e‑4 的误码平台 |
| `imdd_system` / `main_imdd_system` | 112 Gb/s PAM2/4/8 灵敏度；C 波段功率衰落（MZM/EML/DML）；FFE/DFE/MLSE/Volterra 距离对比；眼图 | 见 5.4 节 |
| `psk_coherent` / `main_psk_coherent` | DP-BPSK/QPSK/8PSK/16PSK 与 DP-16QAM 背靠背 BER–OSNR | 与理论相差 0.04–0.36 dB；16QAM 比 16PSK 省 3.7 dB |
| `dpsk_dli` / `main_dpsk_dli` | DBPSK/DQPSK/D8PSK + DLI：BER–OSNR、激光线宽容限 | 差分代价 1.4 / 2.4 / 2.9 dB，与理论一致 |
| `phi_otdr_das` / `main_phi_otdr_das` | φ-OTDR 相干 DAS | 定位误差 < 2 m，波形相关系数 1.000 |
| `fiber_sensing` / `main_fiber_sensing` | OTDR / LFM 脉冲压缩 / BOTDA / 拉曼 DTS / FBG / OFDR | 见 8.4 节；各项解调误差均在 1% 量级 |
| `isac_ofdm_das` / `main_isac_ofdm_das` | CP-OFDM 数据信号同时作 DAS 探测光 | 定位 601.4 m（真值 600–605），相关系数 1.000；通信 BER 9e‑4 |
| `isac_forward_sop` / `main_isac_forward_sop` | 相干收发机前向 SOP / 相位传感 | 扰动频率 60 / 120 Hz 正确识别；通信 BER 2.9e‑4 |
| `nn_imdd_pam4` / `main_nn_imdd_pam4` | PAM4 IM-DD：FFE / Volterra / MLP / CNN / BiLSTM | 见第 7.3 节；NN 比 Volterra 低 1–2 个数量级 BER |
| `nn_coherent_nlc` / `main_nn_coherent_nlc` | 相干长距：CDC / DBP / LDBP / 黑盒 CNN | 见第 7.3 节；LDBP 4 步/跨约 25.6 dB，乘法量约为全 DBP 的一半 |
| `hcf_link` / `main_hcf_link` | 空芯光纤 vs SSMF：参数与时延、单信道 10×80 km 功率扫描（SSFM vs GN）、IMI 容限 | 见下表 |
| `fmf_sdm` / `main_fmf_sdm` | FMF3/FMF6：群时延扩展与 MDL 随长度、多模非线性系数、FMF3 强耦合 4×50 km 6×6 MIMO 功率扫描、FMF6 弱耦合 10 km 12×12 MIMO | 见下表；约 4 分钟 |
| `mmf_vcsel` / `main_mmf_vcsel` | OM3/OM4 850 nm：模式群时延、EMB 随剖面偏差、激励条件、53.125 GBd PAM4 BER 随长度 | 见下表 |

**空芯 / 少模 / 多模光纤结果**（Python / MATLAB；两边随机数实现不同，统计结果一致，确定性物理量逐位一致）：

| 场景 | 结果 | 解读 |
|---|---|---|
| HCF vs SSMF 参数（C 波段，10×80 km） | γ 2.4×10⁻⁴ vs 1.32 /(W·km)；时延 3.34 vs 4.90 µs/km（800 km：2.67 vs 3.92 ms）；GN 最优功率 +20.9 vs +0.7 dBm，最优 SNR 48.8 vs 20.9 dB | 低非线性把最优功率推高约 20 dB；低损耗与低色散使 ASE 极限 SNR 也更高 |
| SSFM + CDC 功率扫描 | SSMF 峰值 20.6 / 20.6 dB @ 0 dBm（GN 20.9 dB）；HCF 峰值 36.1 / 35.4 dB @ +20 dBm，+32 dBm 时 27.6 / 27.7 dB | HCF 高功率时约 36 dB 封顶，来自接收 DSP 自身的 SNR 上限（自适应均衡失调、BPS 窗口噪声），不是光纤效应；+32 dBm 时 SSFM 比 GN 低约 2–3 dB，推测与 HCF 低色散下 GN 的高斯假设偏差有关（未单独验证） |
| HCF IMI 容限（+10 dBm，每跨一次） | −50 → −30 dB/跨：SNR 33.5 → 19.8 dB，与 1/SNR₀ + N·η 相差 ≤ 0.05 dB；−20 dB/跨时仿真低于理论约 1 dB | 回波时延（10 ps/m × 80 km）远大于均衡器窗口，IMI 等效为加性干扰；强 IMI 时还影响自适应均衡与 CPR |
| 多模非线性系数 Γ | FMF3 强耦合 0.592；弱耦合 [[0.828, 0.621], [0.621, 0.497]]；FMF6 强耦合 0.328 /(W·km) | 模式越多、模场越大，平均非线性越弱；单模退化为 8/9·γ |
| FMF3 群时延标准差（5 → 200 km） | 强耦合 53 → 333 ps（Python）/ 59 → 332 ps（MATLAB），增大 6.2 倍（√40 = 6.3）；弱耦合 118 → 4371 ps ∝ L，50 km 时 1157 ps（不耦合上限 1179 ps） | 强耦合把群时延扩展从 ∝ L 降为 ∝ √L，MIMO 抽头数随之大幅减少 |
| FMF3 MDL（5 → 200 km） | 强耦合 0.32 → 0.42 dB；弱耦合 0.25 → 1.66 dB | 强耦合使 MDL 在模式间平均，累积更慢；弱耦合在短距离先降后升，因为少模 EDFA 给 LP11 多 0.3 dB 增益（MDG），与 LP11 多出的 0.01 dB/km 损耗在约 30 km 处抵消 |
| FMF3 强耦合 4×50 km，6×6 MIMO（121 抽头） | 每模式 −4 / 0 / 4 / 8 / 12 dBm：平均 SNR 19.9 / 20.5 / 20.3 / 17.6 / 10.5 dB（Python），MATLAB 18.0–21.3 dB 区间同趋势 | 低功率端被 6-bit DAC/ADC 等收发机损伤限制在约 21 dB；+8 dBm 起多模非线性代价明显。MATLAB 该次实现中 LP01 比其他模式低约 2.5 dB，来自随机 MDL/MDG 分配 |
| FMF6 弱耦合 1×10 km，12×12 MIMO（157 抽头，训练 8192） | 各模式 SNR 24.2–26.2 dB（Python）/ 25.8–27.4 dB（MATLAB），BER ≈ 1e‑4 | 误码集中在帧末尾十几个符号：最后一个导频之后的一次周跳，属帧边界效应 |
| OM3 / OM4 EMB（满注入） | 2000 / 4700 MHz·km（标定）；α_opt 时 10770 MHz·km；居中高斯光斑（w = 4 µm）> 10⁶，偏移 8 µm 时 1.27×10⁵ MHz·km | 受限激励只激发低阶模式群，有效带宽远高于满注入 |
| 53.125 GBd PAM4，ROP −4 dBm，满注入 | 低于 KP4 门限的最大长度：OM3 约 60 m，OM4 约 150 m（两种语言一致）；只有色度色散时 300 m 仍无明显代价 | 850 nm 短距链路主要受模式带宽限制；离散模式群模型比实际光纤更悲观，结果只说明趋势 |

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

## 11. FAQ

**Q1：单信道 DP-16QAM 接收 SNR 约 18.5–19 dB，是否偏高？非线性考虑了吗？**

考虑了：单信道仿真中 SPM（以及信道内 IXPM/IFWM）一直存在。逐项分解如下（10×80 km，NF 5 dB，40 GBd，1 dBm）：

| 噪声 / 损伤来源 | SNR |
|---|---|
| 只有 ASE | 58 + P − NF − 跨损耗 − 10lgN − 10lg(Rs/12.5G) = **22.9 dB** |
| + 单信道 SPM | **≈ 20.6 dB**（SSFM 仿真；GN 预测 20.9 dB） |
| + 收发机损伤（6-bit DAC/ADC、带宽、IQ 误差、PMD、PDL） | **≈ 19 dB** ← 原示例的结果 |
| 换成 7 信道 WDM（+XPM、FWM） | 再降约 1.2 dB（第 10 节：20.4 → 19.2 dB，理想收发机） |
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
- **新的传感 / 通感方案**：前向模型用 `rayleigh_fiber` + `backscatter`（瑞利类）或 `botda`（布里渊），接收用 `laser`、`hybrid_90`、`bpd`，解调可复用 `das_phase`、`pulse_compress`、`ofdm_profile`、`jones_est`。

**Q6：如何把一个真实文件（文本、图片等）调制到光信号上，传输后再还原？**
用 `optsim/filecodec.py`。`file_tx_signal` 的返回值与 `tx_signal` 格式相同，所以只需在任一相干示例（如 `coherent_dp_qam.py`）里替换发射这一行，器件、链路和 `rx_dsp` 都不用改：
```python
from optsim import filecodec
ref = filecodec.file_tx_signal('photo.png', cfg, fec_kind='ldpc34')   # 代替 ref = tx_signal(cfg)
# ... DAC / 激光器 / DP-IQ 调制器 / 链路 / 相干接收机 / ADC，与原示例相同 ...
res = rx_dsp(r, cfg, ref)
out = filecodec.decode_file(res['y'], ref, out_dir='results')         # 写出 results/rx_photo.png
print(out['crc_ok'], out['ber_pre'], out['n_fail'])                   # CRC 是否通过、纠错前 BER、未收敛码字数
```
处理流程：
- 发射端：文件封装为“文件头（魔数、版本、文件名、长度、CRC32）+ 内容”→ 比特 → LDPC 编码（`ldpc34` 码率 3/4、`ldpc12` 码率 1/2、`none`，码长 1944）→ 固定种子加扰（使星座点等概率）→ Gray QAM（比特流前半送 X 偏振、后半送 Y 偏振）→ 成帧 → RRC 成形。
- 接收端：用训练序列和导频估计每偏振噪声方差，计算软信息 LLR，解扰后做 LDPC 译码，再按 CRC32 判断文件是否完好。
- 小文件会用伪随机比特填充到每偏振至少 2¹⁴ 个数据符号，保证 MIMO 均衡有足够的收敛长度。

实测（`seed=1`，40 GBd，链路同 `coherent_dp_qam.py`）：49.5 kB 的 README 经 10×80 km、1 dBm 传输，纠错前 BER 2.7e‑5，还原文件与原文件逐字节一致（约 1.5 min）；44.5 kB 的 PNG 以 DP-64QAM 背靠背、OSNR 26 dB 发送，纠错前 BER 2.3e‑2，LDPC 3/4 后逐字节一致；OSNR 12 dB 时 LDPC 无法收敛，`crc_ok=False`。

注意：
- `rx_dsp` 的帧同步（`dsp.sync`）用整帧发送符号做相关，含数据段，因此不是严格的盲接收；LLR、解扰和译码本身只用训练序列、导频和协议参数（种子、码字数）。
- 耗时与内存随文件大小线性增长，主要耗在 SSFM；大文件建议先用背靠背（`n_spans=0`、`dev.set_osnr`）验证。

---

## 13. 参考文献

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
- A. H. Gnauck, P. J. Winzer, “Optical phase-shift-keyed transmission,” JLT 23(1), 2005（DPSK/DQPSK、DLI 接收）
- T. L. Koch, J. E. Bowers, “Nature of wavelength chirping in directly modulated semiconductor lasers,” Electron. Lett. 20(25), 1984（DML 啁啾）
- R. J. McIntyre, “Multiplication noise in uniform avalanche diodes,” IEEE Trans. Electron Devices 13(1), 1966（APD 过剩噪声）
- J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., 2008（M-PSK / DPSK 误码率）
- A. Mecozzi, C. Antonelli, M. Shtaif, “Kramers–Kronig coherent receiver,” Optica 3(11), 2016
- G. Böcherer et al., “Bandwidth efficient and rate-matched LDPC coded modulation,” IEEE TCOM 63(12), 2015（PAS/PCS）
- A. Alvarado et al., “Achievable information rates for fiber optics,” JLT 36(2), 2018（GMI/MI）
- ITU-T G.652 / G.653 / G.654 / G.655；G.694.1（WDM 栅格）
- Y. Lu et al., “Distributed vibration sensor based on coherent detection of phase-OTDR,” JLT 28(22), 2010
- A. H. Hartog, *An Introduction to Distributed Optical Fibre Sensors*, CRC Press, 2017（分布式传感综述）
- X. Bao, L. Chen, “Recent progress in distributed fiber optic sensors,” Sensors 12(7), 8601–8639, 2012
- J. Pastor-Graells et al., “Single-shot distributed temperature and strain tracking using direct detection phase-sensitive OTDR with chirped pulses,” Opt. Express 24(12), 2016（啁啾脉冲 φ-OTDR）
- M. Froggatt, J. Moore, “High-spatial-resolution distributed strain measurement in optical fiber with Rayleigh scatter,” Appl. Opt. 37(10), 1998（OFDR 局部瑞利谱互相关）
- T. Horiguchi, M. Tateda, “BOTDA—nondestructive measurement of single-mode optical fiber attenuation characteristics using Brillouin interaction: theory,” JLT 7(8), 1989
- J. P. Dakin et al., “Distributed optical fibre Raman temperature sensor using a semiconductor light source and detector,” Electron. Lett. 21(13), 1985
- T. Erdogan, “Fiber grating spectra,” JLT 15(8), 1997；A. D. Kersey et al., “Fiber grating sensors,” JLT 15(8), 1997
- G. Marra et al., “Ultrastable laser interferometry for earthquake detection with terrestrial and submarine cables,” Science 361(6401), 2018
- Z. Zhan et al., “Optical polarization–based seismic and water wave sensing on transoceanic cables,” Science 371(6532), 2021；A. Mecozzi et al., “Polarization sensing using submarine optical cables,” Optica 8(6), 2021
- H. He et al., “Integrated sensing and communication in an optical fibre,” Light Sci. Appl. 12, 25, 2023
- C. Sturm, W. Wiesbeck, “Waveform design and signal processing aspects for fusion of wireless communications and radar sensing,” Proc. IEEE 99(7), 2011（OFDM 雷达处理）
- T. M. Schmidl, D. C. Cox, “Robust frequency and timing synchronization for OFDM,” IEEE TCOM 45(12), 1997
- R. G. Gallager, “Low-density parity-check codes,” IRE Trans. Inf. Theory 8(1), 1962（LDPC）
- Z. Tao et al., “Multiplier-free intrachannel nonlinearity compensating algorithm operating at symbol rate,” JLT 29(17), 2011（PB-NLC）
- C. Häger, H. D. Pfister, “Physics-based deep learning for fiber-optic communication systems,” IEEE JSAC 39(1), 2021（LDBP）
- P. J. Freire et al., “Performance versus complexity study of neural network equalizers in coherent optical systems,” JLT 39(19), 2021
- S. Zhang et al., “Field and lab experimental demonstration of nonlinear impairment compensation using neural networks,” Nat. Commun. 10, 3033, 2019
- B. Karanov et al., “End-to-end deep learning of optical fiber communications,” JLT 36(20), 2018
- N. Stojanovic et al., “Volterra and Wiener equalizers for short-reach 100G PAM-4 applications,” JLT 35(21), 2017
- T. A. Eriksson et al., “Applying neural networks in optical communication systems: possible pitfalls,” PTL 29(23), 2017（训练/测试数据陷阱）
- F. Poletti, “Nested antiresonant nodeless hollow core fiber,” Opt. Express 22(20), 23807–23828, 2014（反谐振空芯光纤）
- G. T. Jasion et al., “0.174 dB/km hollow core double nested antiresonant nodeless fiber (DNANF),” OFC 2022, Th4C.7（待核实）；M. Petrovich et al., “Broadband optical fibre with an attenuation lower than 0.1 decibel per kilometre,” Nat. Photonics, 2025（待核实卷期）
- A. Mecozzi, C. Antonelli, M. Shtaif, “Nonlinear propagation in multi-mode fibers in the strong coupling regime,” Opt. Express 20(11), 11673–11678, 2012
- S. Mumtaz, R.-J. Essiambre, G. P. Agrawal, “Nonlinear propagation in multimode and multicore fibers: generalization of the Manakov equations,” JLT 31(3), 398–406, 2013
- K.-P. Ho, J. M. Kahn, “Statistics of group delays in multimode fiber with strong mode coupling,” JLT 29(21), 3119–3128, 2011
- R. Ryf et al., “Mode-division multiplexing over 96 km of few-mode fiber using coherent 6×6 MIMO processing,” JLT 30(4), 521–531, 2012
- D. Gloge, E. A. J. Marcatili, “Multimode theory of graded-core fibers,” Bell Syst. Tech. J. 52(9), 1563–1578, 1973
- R. Olshansky, D. B. Keck, “Pulse broadening in graded-index optical fibers,” Appl. Opt. 15(2), 483–491, 1976（α 剖面群时延与 α_opt）
- 开源参考：OptiCommPy（Python）、Robochameleon（MATLAB）
