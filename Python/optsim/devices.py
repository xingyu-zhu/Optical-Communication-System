"""光/电器件模型。

电域   : quantize, dac, adc
光源   : laser
调制器 : mzm, iq_mod, dp_iq_mod, phase_mod, eml（电吸收调制，带啁啾）, dml（直调激光器，瞬态+绝热啁啾）
放大/滤波 : ase_psd, add_noise, edfa, set_osnr, voa, optical_filter
WDM    : freq_shift, wdm_mux
偏振   : rotator, random_unitary, pdl, dgd
接收   : hybrid_90, photodiode（PIN + TIA 噪声）, apd, bpd, coherent_rx, dli, dli_rx（差分直接检测）

约定：光场单位 √W，|E|² 为瞬时功率 [W]；双偏振为 (2, N)；工程约定 E·exp(+jω0t)。
"""
import numpy as np
from .utils import (H, Q, KB, F0, db2lin, dbm2w, total_power, freq, fft_filter, lpf, resample, iq_delay, delay)


# ================= 电域器件 =================
def quantize(x, enob, clip=3.0):
    """按有效位数量化（复数时 I/Q 分别量化）。

    参数: enob 有效位数; clip 满量程 = clip × 每路 RMS（决定削波程度）
    """
    def q(v):
        fsr = clip * np.sqrt(np.mean(v ** 2, axis=-1, keepdims=True))
        step = 2 * fsr / 2 ** enob
        return np.clip(np.round(v / step - 0.5) + 0.5, -2 ** enob / 2 + 0.5, 2 ** enob / 2 - 0.5) * step
    return q(x.real) + 1j * q(x.imag) if np.iscomplexobj(x) else q(x)


def dac(x, fs, enob=6, bw=None, clip=3.0, kind='gaussian'):
    """DAC：ENOB 量化 + 模拟带宽（含驱动放大器响应）。

    参数: fs 采样率 [Hz]; enob=None 不量化; bw 3 dB 带宽 [Hz]（None 不限带）; kind 滤波器类型（见 utils.lpf）
    """
    y = quantize(x, enob, clip) if enob else x
    return lpf(y, fs, bw, kind) if bw else y


def adc(x, fs_in, fs_out, enob=6, bw=None, clip=3.0, kind='gaussian'):
    """ADC：前端带宽 -> 重采样到 fs_out（有理数比）-> ENOB 量化。"""
    y = lpf(x, fs_in, bw, kind) if bw else x
    y = resample(y, fs_in, fs_out)
    return quantize(y, enob, clip) if enob else y


# ================= 激光器 =================
def laser(n, fs, p_dbm=0.0, linewidth=0.0, f_offset=0.0, rin_db=None, rng=None):
    """连续光激光器。

    参数: n 采样点数; fs 采样率; p_dbm 输出功率; linewidth 线宽 Δν [Hz]（Wiener 相位噪声，
          相位增量方差 2πΔν/fs）; f_offset 相对参考载波的频偏 [Hz]（WDM 信道频率也用它设置）;
          rin_db 相对强度噪声 [dB/Hz]
    返回: (n,) 复光场

    为与循环（FFT）仿真相容：相位噪声取 Brownian bridge（首尾相接），f_offset 取整到频率网格 fs/n，
    保证整段信号周期连续，否则首尾相位跳变会经 CD/滤波扩散成误码。
    """
    rng = np.random.default_rng(rng)
    t = np.arange(n) / fs
    f_offset = np.round(f_offset * n / fs) * fs / n
    phi = np.cumsum(np.sqrt(2 * np.pi * linewidth / fs) * rng.standard_normal(n))
    phi -= np.arange(1, n + 1) / n * phi[-1]
    p = dbm2w(p_dbm) * np.ones(n)
    if rin_db is not None:
        p = p * (1 + np.sqrt(db2lin(rin_db) * fs / 2) * rng.standard_normal(n))
    return np.sqrt(np.abs(p)) * np.exp(1j * (phi + 2 * np.pi * f_offset * t))


# ================= 调制器 =================
def mzm(E, v, vpi=1.0, vbias=0.0):
    """推挽 MZM 场传输：E·cos(π(v+vbias)/(2Vπ))。

    vbias = -Vπ：零点偏置（相干 IQ 子调制器，场 ∝ sin，近线性）
    vbias = -Vπ/2：正交点偏置（IM-DD，功率 ∝ (1+sin)/2）
    """
    return E * np.cos(np.pi * (v + vbias) / (2 * vpi))


def iq_mod(E, u, vpi=1.0, vbias=None, phase_err=0.0, gain_imb_db=0.0):
    """IQ 调制器（两个零点偏置子 MZM + 90° 移相）。

    参数: u 复驱动电压（实部→I，虚部→Q）; phase_err 正交误差 [rad]; gain_imb_db I/Q 幅度不平衡 [dB]
    返回: E/2·[MZM(uI) + e^{j(π/2+φe)}·MZM(uQ)]
    """
    vbias = -vpi if vbias is None else vbias
    g = db2lin(gain_imb_db / 2) ** 0.5
    return 0.5 * (g * mzm(E, u.real, vpi, vbias) + np.exp(1j * (np.pi / 2 + phase_err)) / g * mzm(E, u.imag, vpi, vbias))


def dp_iq_mod(E, u, vpi=1.0, **kw):
    """双偏振 IQ 调制器：PBS 均分 -> 两路 iq_mod -> PBC 合束。u 为 (2, N) 驱动，kw 透传 iq_mod。"""
    return np.stack([iq_mod(E / np.sqrt(2), u[i], vpi, **kw) for i in range(2)])


def phase_mod(E, v, vpi=1.0):
    """相位调制器：E·exp(jπv/Vπ)。"""
    return E * np.exp(1j * np.pi * v / vpi)


def eml(E, u, er_db=8.0, alpha=0.5):
    """电吸收调制器 EAM（EML 中与激光集成）：吸收随电压近似指数变化，功率透过率
    T(u) = 10^{-ER·(1-u)/20}（u ∈ [-1, 1]，u=1 最亮、u=-1 最暗，消光比 ER [dB]），
    并带瞬态啁啾（Henry 因子 α）：E_out = E_in·T^{(1+jα)/2}，即 Δφ = (α/2)·ln T。
    α>0 时在 C 波段 SSMF 中加剧色散功率衰落，α<0 可部分抵消。"""
    T = 10 ** (-er_db * (1 - np.clip(u, -1, 1)) / 20)
    return E * T ** ((1 + 1j * alpha) / 2)


def dml(u, fs, p_dbm=5.0, er_db=4.0, alpha=3.0, k_ad=1e12, bw=None, linewidth=1e6, rin_db=-145, rng=None):
    """直调激光器 DML（大信号啁啾模型，Koch & Bowers）：P(t) = P̄·(1 + m·u)，m = (ER-1)/(ER+1)，
    频率啁啾 Δν = α/(4π)·[d ln P/dt + κP]，其中绝热啁啾系数 k_ad = ακ/(4π) [Hz/W]（典型 0.5~2 GHz/mW = 0.5~2e12 Hz/W）。
    参数: u 归一化驱动 [-1, 1]（bw 为驱动/弛豫振荡带宽限制）; linewidth、rin_db 见 laser
    返回: (N,) 复光场。绝热啁啾的平均频移（与 P̄ 成正比）只是载波偏移，已去除以保持循环连续。"""
    rng = np.random.default_rng(rng)
    u = lpf(u, fs, bw).real if bw else u
    m = (db2lin(er_db) - 1) / (db2lin(er_db) + 1)
    P = dbm2w(p_dbm) * np.maximum(1 + m * u, 1e-3)
    phi = alpha / 2 * np.log(P) + 2 * np.pi * k_ad * np.cumsum(P - P.mean()) / fs
    return np.sqrt(P) * np.exp(1j * phi) * laser(len(P), fs, 0, linewidth, rin_db=rin_db, rng=rng) / np.sqrt(1e-3)


# ================= 放大器、衰减、滤波 =================
def ase_psd(gain_lin, nf_db, f0=F0):
    """单偏振 ASE 功率谱密度 [W/Hz] = n_sp·hν·(G-1)，n_sp = NF/2。"""
    return db2lin(nf_db) / 2 * H * f0 * max(gain_lin - 1, 0)


def add_noise(E, psd, fs, rng=None):
    """按单偏振 PSD [W/Hz] 在仿真带宽 fs 内给每个偏振加复高斯白噪声。"""
    rng = np.random.default_rng(rng)
    n = rng.standard_normal(E.shape) + 1j * rng.standard_normal(E.shape)
    return E + np.sqrt(psd * fs / 2) * n


def edfa(E, fs, gain_db=None, p_out_dbm=None, nf_db=5.0, f0=F0, rng=None):
    """光放大器（EDFA/TDFA/...通用集总模型）：增益 + ASE。

    参数: gain_db 固定增益；或 p_out_dbm 自动功率控制（总输出功率）；nf_db 噪声系数；f0 中心频率（波段相关）
    """
    G = db2lin(gain_db) if p_out_dbm is None else dbm2w(p_out_dbm) / total_power(E)
    return add_noise(np.sqrt(G) * E, ase_psd(G, nf_db, f0), fs, rng)


def set_osnr(E, osnr_db, fs, bref=12.5e9, rng=None):
    """直接按 OSNR 加 ASE（参考带宽 bref，默认 0.1 nm@1550 = 12.5 GHz，双偏振噪声）。"""
    psd = total_power(E) / (2 * db2lin(osnr_db) * bref)
    return add_noise(E, psd, fs, rng)


def voa(E, att_db):
    """可调光衰减器。"""
    return E * db2lin(-att_db / 2)


def optical_filter(E, fs, bw, order=2, fc=0.0):
    """超高斯光带通（WSS / 解复用器）：|H|² = exp(-ln2·(2(f-fc)/bw)^{2·order})。

    参数: bw 3 dB 带宽 [Hz]; order 阶数（1 为高斯，越大越接近矩形）; fc 中心频率偏移 [Hz]
    """
    f = freq(E.shape[-1], fs) - fc
    return fft_filter(E, np.exp(-np.log(2) / 2 * (2 * f / bw) ** (2 * order)))


# ================= WDM =================
def freq_shift(E, f, fs):
    """频移 f [Hz]（乘 exp(j2πft)），用于把基带信道搬到 WDM 栅格或搬回基带。"""
    return E * np.exp(2j * np.pi * f * np.arange(E.shape[-1]) / fs)


def wdm_mux(fields, freqs, fs):
    """WDM 合波器（理想耦合，无插损）：sum_k freq_shift(fields[k], freqs[k])。"""
    return sum(freq_shift(E, f, fs) for E, f in zip(fields, freqs))


# ================= 偏振器件 =================
def rotator(theta, phi=0.0):
    """偏振旋转 Jones 矩阵（方位角 theta、相位差 phi）。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s * np.exp(-1j * phi)], [s * np.exp(1j * phi), c]])


def random_unitary(rng=None):
    """随机 SU(2) Jones 矩阵（在 Poincaré 球上均匀分布的 SOP 旋转）。"""
    a = np.random.default_rng(rng).standard_normal(4)
    a /= np.linalg.norm(a)
    return np.array([[a[0] + 1j * a[1], a[2] + 1j * a[3]], [-a[2] + 1j * a[3], a[0] - 1j * a[1]]])


def pdl(E, pdl_db, theta=0.0):
    """偏振相关损耗：方位角 theta 方向的偏振衰减 pdl_db。"""
    R = rotator(theta)
    return R @ np.diag([1.0, db2lin(-pdl_db / 2)]) @ R.conj().T @ E


def dgd(E, fs, tau):
    """差分群时延（一阶 PMD）：两本征偏振相对时延 tau [s]。"""
    w = 2 * np.pi * freq(E.shape[-1], fs)
    return np.stack([fft_filter(E[0], np.exp(1j * w * tau / 2)), fft_filter(E[1], np.exp(-1j * w * tau / 2))])


# ================= 接收端光电器件 =================
def hybrid_90(Es, Elo):
    """2×4 90° 光混频器，输出 [I+, I-, Q+, Q-] = [Es+Elo, Es-Elo, Es+jElo, Es-jElo]/2。"""
    return np.stack([Es + Elo, Es - Elo, Es + 1j * Elo, Es - 1j * Elo]) / 2


def photodiode(E, fs, R=1.0, bw=None, i_dark=0.0, T=None, RL=50.0, shot=True, irnd=None, rng=None):
    """PIN 光电二极管：i = R·Σ|E|² + i_dark + 散粒噪声 + 热噪声，再经带宽 bw 低通。

    参数: R 响应度 [A/W]; i_dark 暗电流 [A]; T 温度 [K]（None 不计热噪声）; RL 负载/跨阻等效电阻 [Ω];
          shot 是否计散粒噪声（方差 2q·<i>·fs/2）;
          irnd TIA 输入等效噪声电流密度 [A/√Hz]（典型 10~20 pA/√Hz，即 1e-11~2e-11），给出时方差 irnd²·fs/2
    """
    rng = np.random.default_rng(rng)
    i = R * np.sum(np.abs(np.atleast_2d(E)) ** 2, axis=0) + i_dark
    B = fs / 2
    var = (2 * Q * np.mean(i) * B if shot else 0) + (4 * KB * T * B / RL if T else 0) + (irnd ** 2 * B if irnd else 0)
    i = i + np.sqrt(var) * rng.standard_normal(i.shape)
    return lpf(i, fs, bw).real if bw else i


def apd(E, fs, M=10.0, R=0.8, ka=0.5, bw=None, i_dark=10e-9, irnd=None, T=None, RL=50.0, rng=None):
    """雪崩光电二极管 APD：i = M·R·P；倍增散粒噪声方差 2q·M²·F(M)·(R·P̄ + i_dark)·B，
    过剩噪声因子 F = ka·M + (1-ka)(2 - 1/M)（McIntyre，ka 为电离系数比；InGaAs/InP ≈ 0.4~0.5，Si ≈ 0.02）；
    再加 TIA 噪声（irnd 或 T/RL）。增益提升信号、压过热噪声，但 M 过大时倍增噪声主导 → 存在最优 M。"""
    rng = np.random.default_rng(rng)
    i = M * R * np.sum(np.abs(np.atleast_2d(E)) ** 2, axis=0)
    B = fs / 2
    F = ka * M + (1 - ka) * (2 - 1 / M)
    var = 2 * Q * M ** 2 * F * (np.mean(i) / M + i_dark) * B + (4 * KB * T * B / RL if T else 0) + (irnd ** 2 * B if irnd else 0)
    i = i + np.sqrt(var) * rng.standard_normal(i.shape)
    return lpf(i, fs, bw).real if bw else i


def bpd(E1, E2, fs, **kw):
    """平衡探测器：photodiode(E1) - photodiode(E2)（抑制 LO 直流与信号-信号拍频）。kw 透传 photodiode。"""
    rng = np.random.default_rng(kw.pop('rng', None))
    return photodiode(E1, fs, rng=rng, **kw) - photodiode(E2, fs, rng=rng, **kw)


def coherent_rx(E, Elo, fs, skew=(0, 0, 0, 0), rng=None, **pd_kw):
    """偏振分集相干接收机：PBS + 2 个 90° 混频器 + 4 个 BPD。

    参数: E (2, N) 信号光; Elo (N,) 本振（45° 线偏振，功率均分到两偏振）;
          skew [XI, XQ, YI, YQ] 各支路时延 [s]; pd_kw 透传 photodiode（R, bw, T, shot...）
    返回: (2, N) 复电流 I + jQ ∝ R·Es·conj(Elo)
    """
    rng = np.random.default_rng(rng)
    out = []
    for p in range(2):
        o = hybrid_90(E[p], Elo / np.sqrt(2))
        i = bpd(o[0], o[1], fs, rng=rng, **pd_kw) + 1j * bpd(o[2], o[3], fs, rng=rng, **pd_kw)
        out.append(iq_delay(i, skew[2 * p], skew[2 * p + 1], fs) if any(skew) else i)
    return np.stack(out)


def dli(E, fs, tau, phi=0.0):
    """延迟线干涉仪（DLI，非对称 MZI）：两臂时延差 tau、相位偏置 phi。
    输出 (相长口, 相消口) = (E(t) + e^{jφ}E(t-τ))/2, (E(t) - e^{jφ}E(t-τ))/2。"""
    Ed = np.exp(1j * phi) * delay(E, tau, fs)
    return (E + Ed) / 2, (E - Ed) / 2


def dli_rx(E, fs, rs, rng=None, **pd_kw):
    """差分相位直接检测接收机（DPSK/DQPSK/D8PSK）：3 dB 分光 -> 两个 1 符号延迟 DLI（φ = 0、π/2）-> 两个 BPD。
    I = Re{E(t)E*(t-T)}/2，Q = Im{E(t)E*(t-T)}/2（φ=π/2 臂），返回复电流 z = I + jQ ∝ y_k·y*_{k-1}。
    无需本振、对激光相位噪声（线宽 << 符号率）与小频偏不敏感；多行输入（双偏振）时功率相加，偏振不敏感。"""
    rng = np.random.default_rng(rng)
    out = []
    for phi in (0.0, np.pi / 2):
        a, b = dli(E / np.sqrt(2), fs, 1 / rs, phi)
        out.append(bpd(a, b, fs, rng=rng, **pd_kw))
    return out[0] + 1j * out[1]
