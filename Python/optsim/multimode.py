"""多模 / 少模光纤：少模光纤（FMF）相干空分复用（SDM）与多模光纤（MMF）850 nm 短距 IM-DD。

参数来自 params/fiber_bands.json 的 "multimode" 部分（mm_params）。场约定同 channel.py：E·exp(+jω0t)，
FMF 光场为 (2M, N)，行顺序 [模式1 X, 模式1 Y, 模式2 X, ...]（M 个空间模式 × 2 偏振）。

少模光纤（FMF）—— 耦合段模型（coupled-section model）
-------------------------------------------------------
每段长 lc 的光纤内各通道独立传播（模式群各自的衰减 MDL、群时延 DMGD、色散 D/S），段与段之间乘随机酉矩阵：
    regime='strong'：全部 2M 个通道 Haar 随机强耦合（群时延扩展 ∝ √(L·lc)，MDL 也按 √段数 累积）
    regime='weak'  ：只在简并模式群内（含偏振）Haar 随机强耦合，群间按 xt_db_km 弱耦合（群时延扩展 ∝ L）
Kerr 非线性取模式群平均的多模 Manakov 形式（Mecozzi et al. 2012；Mumtaz et al. 2013）：
    ∂E_g/∂z = ... - j·Σ_h Γ_gh·|E_h|²·E_g，Γ_gh = (n2·ω0/c)·κ_gh
κ_gh 由 Kerr 张量对各群内 Haar 随机酉矩阵求平均得到（高斯矩缩并，见 kerr_coeffs），只需模场交叠积分
f_ab = ∬ψa²ψb² dA；单模单群时 κ = 8/9·f_11 = 8/9/Aeff，即标准 Manakov 方程（selfcheck 验证）。
模场用抛物线折射率近似的 Hermite-Gauss 模（同一模式群内简并，基的选取不影响群平均结果）。

多模光纤（MMF）—— 模式群模型
-----------------------------
α 剖面渐变折射率光纤，G = floor(V/2) 个主模式群，第 g 群（HG 阶数 N=g-1）含 g 个空间模式。
群时延（Gloge-Marcatili / Olshansky-Keck，忽略剖面色散 ε）：
    τ_g = L·ng/c·[1 + Δ(α-2)/(α+2)·x^{2α/(α+2)} + Δ²(3α-2)/(2(α+2))·x^{4α/(α+2)}]，x = m/M ≈ g²/(G(G+1))
最优剖面 α_opt ≈ 2 - 12Δ/5（ε=0）。全面积探测器上各模式正交，探测功率 = Σ_g |E_g|²，
故 MMF 输出用 (G, N) 的各模式群光场表示，可直接送入 devices.photodiode / system.imdd_rx。
未建模：模式噪声、模式分配噪声、多横模 VCSEL 的光谱宽度（VCSEL 用 devices.dml 近似）。

函数
----
mm_params                   读取 FMF/MMF 参数
hg1d, overlap, kerr_coeffs, nl_matrix   模场、交叠积分、非线性系数
haar_unitary, coupling      随机模式耦合矩阵
ssfm_mm, mm_link            FMF 单跨 / 多跨传输（少模 EDFA：增益 + MDG + ASE）
mm_transfer, mdl_db, group_delays       线性传递矩阵 H(f)、模式相关损耗、群时延算符本征值
foe_training                训练序列辅助粗频偏估计（MIMO 前）
sdm_tx, sdm_rx, sdm_rx_dsp  SDM 发射机（M 个 DP-IQ 共用一个激光器）、相干接收机阵列、2M×2M MIMO 接收 DSP
mmf_groups, mmf_launch, mmf_link, mmf_emb   MMF 模式群时延、激励功率分布、链路、有效模式带宽
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import expm
from .utils import C, freq, fft_filter, delay, normalize, resample, set_power
from .params import load, dispersion
from .channel import beta2, beta3, cd
from . import tx, dsp, metrics, devices as dev


# ============================== 参数 ==============================
def mm_params(name='FMF3', L=None, lam=None, **override):
    """生成少模/多模光纤参数 dict（JSON "multimode" 部分）。

    参数: name 'FMF3' | 'FMF6' | 'OM3' | 'OM4'（可在 JSON 中添加）; L 长度 [km]（默认 FMF 50、MMF 0.1）;
          lam 波长 [nm]（默认 JSON 的 lam_nm）; override 覆盖任意字段（先覆盖原始字段，再计算派生量）
    返回 dict（除 JSON 字段外）：name, lam [nm], f0 [Hz], L [km]；
      FMF: n_modes 空间模式数 M, groups/hg/mdl_db_km/D/S/dmgd_ps_km/mdg_db 为 numpy 数组
      MMF: a [µm] 纤芯半径, delta 相对折射率差, V 归一化频率, n_groups 模式群数 G, w0_um 基模模场半径,
           alpha_opt 最优剖面指数, alpha_profile 实际剖面指数, D/S [ps/(nm km)], [ps/(nm² km)]
    """
    raw = {k: v for k, v in load()['multimode'][name].items() if not k.startswith('_')}
    raw.update({k: v for k, v in override.items() if k in raw})
    p = dict(raw, name=name)
    p['lam'] = raw['lam_nm'] if lam is None else lam
    p['f0'] = C / (p['lam'] * 1e-9)
    if raw['type'] == 'FMF':
        p['L'] = 50.0 if L is None else L
        for k in ('groups', 'hg', 'mdl_db_km', 'D', 'S', 'dmgd_ps_km', 'mdg_db'):
            p[k] = np.asarray(raw[k], dtype=int if k in ('groups', 'hg') else float)
        p['n_modes'] = len(p['groups'])
    else:
        p['L'] = 0.1 if L is None else L
        a, na, n1, lam_m = raw['core_um'] / 2, raw['na'], raw['n1'], p['lam'] * 1e-9
        k0 = 2 * np.pi / lam_m
        p.update(a=a, delta=na ** 2 / (2 * n1 ** 2), V=k0 * a * 1e-6 * na)
        p['n_groups'] = int(np.floor(p['V'] / 2))
        p['w0_um'] = np.sqrt(2 * a * 1e-6 / (k0 * na)) * 1e6              # 抛物线剖面基模模场半径
        p['alpha_opt'] = 2 - 12 * p['delta'] / 5
        p['alpha_profile'] = p['alpha_opt'] + raw['alpha_profile_offset']
        p['D'], p['S'] = dispersion(p['lam'], raw['lambda0_nm'], raw['S0'])
    p.update({k: v for k, v in override.items() if k not in raw})
    return p


# ============================== 模场与非线性系数 ==============================
def hg1d(nmax, x, w):
    """归一化一维 Hermite-Gauss 模 u_n(x)，n = 0..nmax，返回 (nmax+1, len(x))，∫u_n² dx = 1。
    u_n(x) = (2/w²)^{1/4}·h_n(√2·x/w)，h_n 为 Hermite 函数（三项递推，数值稳定）。"""
    s = np.sqrt(2) * np.asarray(x) / w
    h = np.zeros((nmax + 1, s.size))
    h[0] = np.pi ** -0.25 * np.exp(-s ** 2 / 2)
    if nmax > 0:
        h[1] = np.sqrt(2) * s * h[0]
    for k in range(1, nmax):
        h[k + 1] = np.sqrt(2 / (k + 1)) * s * h[k] - np.sqrt(k / (k + 1)) * h[k - 1]
    return h * (2 / w ** 2) ** 0.25


def overlap(hg, w, npts=4001):
    """模场交叠积分 f_ab = ∬ψa²ψb² dA [1/m²]，ψ_(m,n)(x,y) = u_m(x)·u_n(y)（可分离，逐维积分）。

    参数: hg (M, 2) 各空间模式的 HG 阶数; w 模场半径 [m]。f_aa = 1/Aeff,a（基模 Aeff = π·w²）
    """
    hg = np.asarray(hg)
    nmax = int(hg.max())
    x = np.linspace(-1, 1, npts) * w * (6 + nmax)
    u2 = hg1d(nmax, x, w) ** 2
    I = u2 @ u2.T * (x[1] - x[0])                                    # I[i, j] = ∫u_i² u_j² dx
    return I[hg[:, 0][:, None], hg[:, 0][None]] * I[hg[:, 1][:, None], hg[:, 1][None]]


def kerr_coeffs(f, labels):
    """模式群平均的非线性系数 κ_gh [1/m²]（labels：每个空间-偏振通道所属的群，长度 2M）。

    推导：通道 i=(a,p)（模式 a，偏振 p）的 Kerr 项 NL_i = Σ T_ijkl E_j* E_k E_l，各向同性石英
    P_NL ∝ (1/3)[2(E*·E)E + (E·E)E*]。对各群内独立的 Haar 随机酉变换求平均后，只能剩下
    Σ_h κ_gh |E_h|² E_g；用复高斯矩 <x_i*x_j*x_k x_l> = δ_ik δ_jl + δ_il δ_jk 缩并两边得
        κ_gg = Σ_{i,j∈g} S_ij / (n_g(n_g+1))，κ_gh = Σ_{i∈g, j∈h} S_ij / (n_g n_h)，
        S_ij = T_ijij + T_ijji = f_ab·(2/3 + 4/3·δ_pq)
    单模（一个群、2 个偏振通道）：κ = (16/3·f)/6 = 8/9·f，即 Manakov 系数。
    返回: (κ (G, G), 群标签数组 (G,))
    """
    labels = np.asarray(labels)
    M = f.shape[0]
    a, p = np.repeat(np.arange(M), 2), np.tile([0, 1], M)
    S = f[a[:, None], a[None]] * (2 / 3 + 4 / 3 * (p[:, None] == p[None]))
    g = np.unique(labels)
    K = np.zeros((len(g), len(g)))
    for i, gi in enumerate(g):
        for j, gj in enumerate(g):
            mi, mj = labels == gi, labels == gj
            ni, nj = mi.sum(), mj.sum()
            K[i, j] = S[np.ix_(mi, mj)].sum() / (ni * (ni + 1) if i == j else ni * nj)
    return K, g


def nl_matrix(mmf):
    """FMF 非线性系数矩阵。返回 (lab (2M,) 每个通道所属非线性群的下标, Γ (G', G') [1/(W m)])。
    regime='strong' 时全部通道为一个群（G' = 1）；'weak' 时为各模式群。"""
    comp = np.repeat(mmf['groups'], 2)
    labels = np.zeros_like(comp) if mmf['regime'] == 'strong' else comp
    K, g = kerr_coeffs(overlap(mmf['hg'], mmf['w0_um'] * 1e-6), labels)
    lab = np.searchsorted(g, labels)
    return lab, mmf['n2'] * 2 * np.pi * mmf['f0'] / C * K


# ============================== 随机模式耦合 ==============================
def haar_unitary(n, rng=None):
    """n×n Haar 随机酉矩阵（复高斯矩阵 QR 分解并校正 R 对角相位）。"""
    rng = np.random.default_rng(rng)
    Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def coupling(mmf, dz_km, rng=None):
    """一个耦合段末端的随机耦合矩阵 (2M, 2M)。
    strong：Haar(2M)；weak：各模式群内 Haar（块对角）后乘 expm(j√xt·H)，H 为只含群间元素的随机厄米矩阵
    （按行归一化，使每段从一个通道耦合到其他群的功率 ≈ xt = 10^(xt_db_km/10)·dz_km）。"""
    rng = np.random.default_rng(rng)
    comp = np.repeat(mmf['groups'], 2)
    n = comp.size
    if mmf['regime'] == 'strong':
        return haar_unitary(n, rng)
    U = np.zeros((n, n), complex)
    for g in np.unique(comp):
        idx = np.flatnonzero(comp == g)
        U[np.ix_(idx, idx)] = haar_unitary(idx.size, rng)
    xt = 10 ** (mmf['xt_db_km'] / 10) * dz_km
    if xt > 0 and np.unique(comp).size > 1:
        Hm = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Hm[comp[:, None] == comp[None]] = 0
        Hm = (Hm + Hm.conj().T) / 2
        Hm /= np.sqrt(np.mean(np.sum(np.abs(Hm) ** 2, 1)))
        U = expm(1j * np.sqrt(xt) * Hm) @ U
    return U


def _lin_op(mmf, w):
    """单位长度线性算子 [1/m]（每个通道一行）：-α/2 - j(τω + β2ω²/2 + β3ω³/6)，τ 为去均值后的 DMGD。"""
    gi = np.repeat(mmf['groups'], 2) - 1
    a = (mmf['alpha_db_km'] + mmf['mdl_db_km'][gi]) * np.log(10) / 10 / 1e3
    tau = mmf['dmgd_ps_km'][gi] * 1e-15
    tau = tau - tau.mean()                                        # 以通道平均群时延为参考帧
    D, S = mmf['D'][gi], mmf['S'][gi]
    ph = tau[:, None] * w + beta2(D, mmf['lam'])[:, None] / 2 * w ** 2 + beta3(D, S, mmf['lam'])[:, None] / 6 * w ** 3
    return -a[:, None] / 2 - 1j * ph


# ============================== FMF 传输 ==============================
def ssfm_mm(E, fs, mmf, phi_max=0.005, rng=None, nonlinear=True):
    """FMF 单跨传输：耦合段模型 + 对称分步傅里叶（线性半步 - 群平均 Kerr 相移 - 线性半步）。

    参数: E (2M, N) [√W]; mmf mm_params 输出（L [km]）; phi_max 每步最大非线性相移 [rad];
          rng 只用于随机耦合矩阵（与 mm_transfer 用同一种子即得到同一条光纤实现）; nonlinear False 为纯线性
    """
    rng = np.random.default_rng(rng)
    E = np.asarray(E, complex)
    lin = _lin_op(mmf, 2 * np.pi * freq(E.shape[-1], fs))
    lab, Gam = nl_matrix(mmf)
    Lm, lc = mmf['L'] * 1e3, mmf['lc_km'] * 1e3
    z, z_next = 0.0, lc
    while z < Lm - 1e-9:
        dz = min(z_next - z, Lm - z)
        if nonlinear:
            pmax = np.max(np.sum(np.abs(E) ** 2, 0))
            dz = min(dz, phi_max / max(Gam.max() * pmax, 1e-30))
        Hh = np.exp(lin * dz / 2)
        E = fft_filter(E, Hh)
        if nonlinear:                                              # 精确相位旋转（各群功率不变）
            P = np.stack([np.sum(np.abs(E[lab == g]) ** 2, 0) for g in range(len(Gam))])
            E = E * np.exp(-1j * (Gam @ P)[lab] * dz)
        E = fft_filter(E, Hh)
        z += dz
        if z >= z_next - 1e-9:
            E = coupling(mmf, mmf['lc_km'], rng) @ E
            z_next += lc
    return E


def _amp_gain(mmf):
    """少模 EDFA 各通道增益 [dB]：补偿平均跨段损耗 + 模式相关增益 MDG（去均值）。"""
    gi = np.repeat(mmf['groups'], 2) - 1
    loss = np.mean(mmf['alpha_db_km'] + mmf['mdl_db_km'][gi]) * mmf['L']
    mdg = mmf['mdg_db'][gi]
    return loss + mdg - mdg.mean()


def mm_link(E, fs, mmf, n_spans, ase=True, rng=None, noise_rng=None, **kw):
    """FMF 多跨链路：每跨 ssfm_mm + 少模 EDFA（各通道增益 _amp_gain，ASE 按各自增益与 nf_db）。
    rng 用于模式耦合，noise_rng 用于 ASE（分开，使 mm_transfer 可复现同一光纤实现）。kw 透传 ssfm_mm。"""
    rng, noise_rng = np.random.default_rng(rng), np.random.default_rng(noise_rng)
    g = _amp_gain(mmf)
    for _ in range(n_spans):
        E = ssfm_mm(E, fs, mmf, rng=rng, **kw) * 10 ** (g[:, None] / 20)
        if ase:
            psd = np.array([dev.ase_psd(10 ** (gk / 10), mmf['nf_db'], mmf['f0']) for gk in g])[:, None]
            E = dev.add_noise(E, psd, fs, noise_rng)
    return E


def mm_transfer(fs, n, mmf, n_spans=1, rng=None):
    """线性传递矩阵 H(f) (n, 2M, 2M)（FFT 频率顺序），与 mm_link(..., rng=同一种子, ase=False, nonlinear=False)
    逐点一致（含跨段放大器增益与 MDG，不含非线性与 ASE）。"""
    rng = np.random.default_rng(rng)
    lin = _lin_op(mmf, 2 * np.pi * freq(n, fs)).T                  # (n, 2M)
    g = 10 ** (_amp_gain(mmf) / 20)
    H = np.broadcast_to(np.eye(lin.shape[1], dtype=complex), (n,) + (lin.shape[1],) * 2).copy()
    Lm, lc = mmf['L'] * 1e3, mmf['lc_km'] * 1e3
    for _ in range(n_spans):
        z, z_next = 0.0, lc
        while z < Lm - 1e-9:
            dz = min(z_next - z, Lm - z)
            H = np.exp(lin * dz)[:, :, None] * H
            z += dz
            if z >= z_next - 1e-9:
                H = coupling(mmf, mmf['lc_km'], rng) @ H
                z_next += lc
        H = g[None, :, None] * H
    return H


def mdl_db(H):
    """模式相关损耗 MDL [dB] = 20lg(σmax/σmin)（H(f) 的奇异值，每个频点一个值）。"""
    s = np.linalg.svd(H, compute_uv=False)
    return 20 * np.log10(s[:, 0] / s[:, -1])


def group_delays(H, fs):
    """群时延算符 j·H⁻¹·dH/dω 的本征值 [s]（每个频点 2M 个，按实部排序；H = e^{-jωτ} 时得 τ）。
    dH/dω 用中心差分，要求频点间隔 fs/n 远小于 1/(2·最大群时延差)，否则相位卷绕使结果失真。"""
    n = H.shape[0]
    dw = 2 * np.pi * fs / n
    dH = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) / (2 * dw)
    ev = np.linalg.eigvals(1j * np.linalg.solve(H, dH))
    return np.sort(ev.real, -1)


# ============================== SDM 收发机 ==============================
def sdm_tx(cfg, n_modes, rng=None, swing=0.2, enob=6):
    """SDM 发射机：M 个独立 DP-QAM 信号（各自数据/训练序列，seed = cfg.seed + 100·m），
    共用一个激光器（分束），各经 DAC + DP-IQ 调制器，每个空间模式入纤功率 cfg.p_launch [dBm]。
    模式复用器按理想一一映射（任何酉变换都可并入随机模式耦合）。
    返回: (E (2M, N), refs 各模式的 system.tx_signal 输出列表)
    """
    from .system import tx_signal
    rng = np.random.default_rng(rng)
    refs = [tx_signal(replace(cfg, seed=cfg.seed + 100 * m)) for m in range(n_modes)]
    N, fs = refs[0]['wave'].shape[1], cfg.fs
    cw = dev.laser(N, fs, 16, cfg.linewidth, rng=rng)
    E = [dev.dp_iq_mod(cw, swing * dev.dac(r['wave'], fs, enob=enob, bw=0.7 * cfg.rs), phase_err=np.deg2rad(1))
         for r in refs]
    return np.vstack([set_power(e, cfg.p_launch) for e in E]), refs


def sdm_rx(E, cfg, rng=None, f_offset=1.5e9, lo_dbm=13, enob=6):
    """相干接收机阵列：模式解复用（理想）后 M 个偏振分集相干接收机，共用一个本振（分束），再经 ADC。
    返回: (2M, N') ADC 输出 @cfg.fs_adc"""
    rng = np.random.default_rng(rng)
    lo = dev.laser(E.shape[1], cfg.fs, lo_dbm, cfg.linewidth, f_offset=f_offset, rng=rng)
    r = np.vstack([dev.coherent_rx(E[2 * m:2 * m + 2], lo, cfg.fs, R=0.8, bw=0.75 * cfg.rs, T=300, rng=rng)
                   for m in range(E.shape[0] // 2)])
    return dev.adc(r, cfg.fs, cfg.fs_adc, enob=enob, bw=0.75 * cfg.rs)


def foe_training(y, train, max_lag, sps=2, fs=1.0, nfft=None):
    """训练序列辅助的粗频偏估计（MIMO 之前、模式混合与群时延扩展未补偿时使用）。

    各通道共用同一对收发激光器，频偏 Δf 相同：y_i(t) = e^{j2πΔf t}·Σ_j h_ij ⊗ s_j。对每个时延 k ∈ [-max_lag, max_lag]
    （符号）计算 z_ij,k(t) = y_i(t)·s_j*(t-k)，其频谱在 Δf 处有峰；对全部 (i, j, k) 非相干累加 |FFT|² 后取峰值。
    参数: y (P, N) sps 倍过采样的接收信号（帧起始在 0 附近）; train (P, T) 各通道训练序列（1 sps）;
          max_lag 搜索时延范围 [符号]，应覆盖群时延扩展; fs 符号率（返回值单位随之）
    返回: 频偏估计（范围 ±fs/2，分辨率 fs/nfft）
    """
    P, T = train.shape
    nfft = nfft or 4 * 2 ** int(np.ceil(np.log2(T)))
    x = y[:, ::sps]
    acc = np.zeros(nfft)
    for k in range(-max_lag, max_lag + 1):
        z = np.roll(x, -k, axis=-1)[:, None, :T] * np.conj(train)[None]          # (P, P, T)
        acc += np.sum(np.abs(np.fft.fft(z, nfft, axis=-1)) ** 2, (0, 1))
    return freq(nfft, fs)[np.argmax(acc)]


def sdm_rx_dsp(r, cfg, refs, mmf, n_spans, ntaps=31, mu_lms=None, mu_rde=None, train_epochs=4):
    """SDM 接收 DSP：GSOP → 2 sps → CDC（通道平均色散）→ 匹配滤波 → 定时
    → 训练序列辅助粗频偏估计（foe_training）→ 2M×2M MIMO（训练序列 DA-LMS 预收敛 train_epochs 遍，
    再以其抽头为初值全帧 RDE）→ 残余频偏（4 次方谱）→ BPS → 帧同步 → 导频辅助模糊/周跳消除 → 判决。

    抽头数 ntaps（2 sps）须覆盖模式群时延扩展（见 group_delays；经验上 ntaps ≈ 4×扩展符号数 + 1）。
    步长默认按 LMS 稳定条件 μ < 2/tr(R) ∝ 1/(P·ntaps) 归一化：mu_lms = 0.5/(P·ntaps)，mu_rde = 0.15/(P·ntaps)
    （P = 2M；对 2×2、15 抽头即约 1.7e-2 / 5e-3，对 6×6、81 抽头即 1e-3 / 3e-4）。仿真中帧起始在 0 附近（循环仿真、
    CDC 已去除平均时延）；实际系统需先用前导码做粗帧同步。
    返回 dict: ber, ber_mode (M,), snr_mode [dB], snr（平均）, fo [Hz], lag, sym (2M, nd), W
    """
    c, kind = refs[0]['const'], refs[0]['kind']
    f = np.vstack([q['frame'] for q in refs])
    rot = tx.rot_order(c)
    fs2 = 2 * cfg.rs
    gi = np.repeat(mmf['groups'], 2) - 1
    y = resample(dsp.gsop(r), cfg.fs_adc, fs2)
    y = dsp.cdc(y, fs2, n_spans * mmf['L'], np.mean(mmf['D'][gi]), mmf['lam'], np.mean(mmf['S'][gi]))
    y = normalize(tx.matched_filter(y, 2, cfg.beta))
    y, _ = dsp.timing_ff(y, 2)
    nt, P = cfg.train_len, r.shape[0]
    mu_lms = 0.5 / (P * ntaps) if mu_lms is None else mu_lms
    mu_rde = 0.15 / (P * ntaps) if mu_rde is None else mu_rde
    fo = foe_training(y, f[:, :nt], ntaps // 4 + 1, 2, cfg.rs)
    y = dsp.fo_comp(y, fo, fs2)
    _, W, _ = dsp.mimo_eq(y[:, :2 * nt], ntaps, mu_lms, 2, 'lms', c, ref=f, n_train=nt, epochs=train_epochs)
    y, W, _ = dsp.mimo_eq(y, ntaps, mu_rde, 2, 'rde', c, w=W)
    y = normalize(y)
    fo2 = dsp.foe(y, cfg.rs, m=rot)
    y = dsp.fo_comp(y, fo2, cfg.rs)
    fo += fo2
    y, _ = dsp.bps(y, c, rot=rot)
    _, lag, _ = dsp.sync(y[:2], refs[0]['frame'])
    y = np.roll(y, -lag, axis=-1)[:, :f.shape[1]]
    y = dsp.fix_ambiguity(y, f, kind > 0, n=5, rot=rot)
    y = y / np.sqrt(np.mean(np.abs(y[:, kind > 0]) ** 2, -1, keepdims=True))
    d, s = y[:, kind == 0], f[:, kind == 0]
    rb = tx.demodulate(d, c)
    bits = np.vstack([q['bits'] for q in refs])
    M = len(refs)
    ber_mode = np.array([metrics.ber(bits[2 * m:2 * m + 2], rb[2 * m:2 * m + 2])[0] for m in range(M)])
    snr_mode = np.array([metrics.snr(d[2 * m:2 * m + 2], s[2 * m:2 * m + 2]) for m in range(M)])
    return dict(ber=float(ber_mode.mean()), ber_mode=ber_mode, snr_mode=snr_mode,
                snr=float(10 * np.log10(np.mean(10 ** (snr_mode / 10)))), fo=fo, lag=lag, sym=d, W=W)


# ============================== MMF（850 nm IM-DD） ==============================
def mmf_groups(mmf):
    """MMF 主模式群：返回 dict
    n_groups G, n_modes (G,) 每群空间模式数 g, tau [s/km] 各群单位长度群时延（绝对值，Olshansky-Keck ε=0），
    x (G,) 归一化模式序号 m/M ≈ g²/(G(G+1))"""
    G, a, D = mmf['n_groups'], mmf['alpha_profile'], mmf['delta']
    g = np.arange(1, G + 1)
    x = g ** 2 / (G * (G + 1))
    tau = mmf['ng'] / C * 1e3 * (1 + D * (a - 2) / (a + 2) * x ** (2 * a / (a + 2))
                                 + D ** 2 * (3 * a - 2) / (2 * (a + 2)) * x ** (4 * a / (a + 2)))
    return dict(n_groups=G, n_modes=g, tau=tau, x=x)


def mmf_launch(mmf, kind='ofl', w_um=None, offset_um=0.0, npts=4001):
    """激励条件 -> 各模式群功率 P_g（G,）。

    kind='ofl'  ：满注入（overfilled launch），各模式等功率，P_g ∝ g，ΣP_g = 1
    kind='gauss'：高斯光斑（模场半径 w_um，横向偏移 offset_um，如 VCSEL / 单模尾纤对接），
                  对 HG 模做交叠积分 |∫ψ_launch·ψ_mn dA|²，按 N = m+n 归入第 N+1 群；ΣP_g ≤ 1（其余耦合到非导模）
    """
    G = mmf['n_groups']
    if kind == 'ofl':
        P = np.arange(1, G + 1, dtype=float)
        return P / P.sum()
    w0 = mmf['w0_um'] * 1e-6
    ws, d = (w_um or mmf['w0_um']) * 1e-6, offset_um * 1e-6
    x = np.linspace(-1, 1, npts) * w0 * (6 + G) + d / 2
    dx = x[1] - x[0]
    u = hg1d(G - 1, x, w0)
    gx = np.exp(-(x - d) ** 2 / ws ** 2)
    gy = np.exp(-x ** 2 / ws ** 2)
    gx, gy = gx / np.sqrt(np.sum(gx ** 2) * dx), gy / np.sqrt(np.sum(gy ** 2) * dx)
    cx, cy = (u @ gx * dx) ** 2, (u @ gy * dx) ** 2
    P = np.array([sum(cx[n] * cy[N - n] for n in range(N + 1)) for N in range(G)])
    return P


def mmf_link(E, fs, mmf, P, L=None):
    """MMF 链路：单模输入光场 E (N,) -> 各模式群光场 (G, N)。
    第 g 群 = √P_g · 衰减 · 色散(D, S @ lam) · 时延 (τ_g - τ̄)·L（τ̄ 为按 P_g 加权的平均群时延）。
    探测时 photodiode 对各行 |E|² 求和（全面积探测器，模式正交）。"""
    L = mmf['L'] if L is None else L
    q = mmf_groups(mmf)
    P = np.asarray(P, float)
    if P.size != q['n_groups']:
        raise ValueError(f"P 长度应为模式群数 {q['n_groups']}（单个模式群用 np.eye(G)[0]）")
    tau = (q['tau'] - np.sum(P * q['tau']) / P.sum()) * L
    Ec = cd(E, fs, L, mmf['D'], mmf['lam'], mmf['S']) * 10 ** (-mmf['alpha_db_km'] * L / 20)
    return np.stack([np.sqrt(P[g]) * delay(Ec, tau[g], fs) for g in range(len(P))])


def mmf_emb(mmf, P, f_max=1e12, n=100001):
    """有效模式带宽 EMB [MHz·km]：模式群冲激响应 h(t) = Σ P_g δ(t - τ_g·1 km) 的 -3 dB 光带宽 × 1 km
    （|H(f)| 首次降到 0.5，H(0) = 1；只含模式色散，不含色度色散；频率分辨率 f_max/(n-1)·1 km）。
    返回 (EMB, f [Hz], |H(f)|)；f_max 内未降到 0.5 时 EMB = inf。"""
    q = mmf_groups(mmf)
    P = np.asarray(P, float) / np.sum(P)
    f = np.linspace(0, f_max, n)
    H = np.abs(np.exp(-2j * np.pi * f[:, None] * q['tau'][None]) @ P)
    i = np.flatnonzero(H < 0.5)
    return (f[i[0]] / 1e6 if i.size else np.inf), f, H
