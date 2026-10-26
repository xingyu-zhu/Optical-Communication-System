"""光纤信道：色散、损耗、Kerr 非线性（SPM/XPM/FWM）、受激拉曼散射 (SRS)、自陡峭、PMD、多跨链路、GN 模型。

场约定（工程约定）：E(t)·exp(+jω0 t)，基带正频率 = 更高光频（更短波长）。
光纤参数统一用 dict `fib`（由 params.fiber_params 生成），字段与单位：
    L [km], alpha [dB/km], D [ps/(nm km)], S [ps/(nm^2 km)], gamma [1/(W km)], lam [nm], f0 [Hz],
    pmd [ps/√km], nf [dB], raman (bool), steep (bool), fR, tau1, tau2 [s]

传播方程（Manakov，双偏振平均后的耦合 NLSE；单行输入时退化为标量 NLSE）：
    ∂E/∂z = -α/2·E - j·β2/2·∂²E/∂t² ... (频域: -jβ2ω²/2 - jβ3ω³/6)
            - j·γ·(8/9)·(1 + ω/ω0)·F{ E·[(1-fR)|E|² + fR·(h_R ⊗ |E|²)] }
    - SPM / XPM / FWM：对全场（单信道或 WDM 合波）求解即自动包含（含信道内 IXPM/IFWM）
    - SRS：fR·h_R 的延迟响应项，WDM 中表现为高频信道向低频信道转移功率
    - 自陡峭：(1 + ω/ω0) 因子
    - PMD：每步随机 SOP 旋转 + DGD（粗步长模型）；ASE-信号非线性相互作用（非线性相位噪声）在多跨中自然包含
"""
import numpy as np
from .utils import C, freq, fft_filter
from .devices import edfa, random_unitary


def beta2(D=17.0, lam=1550.0):
    """色散参数换算：D [ps/(nm km)], lam [nm] -> β2 [s^2/m]。"""
    return -D * 1e-6 * (lam * 1e-9) ** 2 / (2 * np.pi * C)


def beta3(D=17.0, S=0.057, lam=1550.0):
    """色散斜率换算：S [ps/(nm^2 km)] -> β3 [s^3/m]。β3 = (λ/2πc)²·(λ²S + 2λD)。"""
    l = lam * 1e-9
    return (l / (2 * np.pi * C)) ** 2 * (l ** 2 * S * 1e3 + 2 * l * D * 1e-6)


def _disp_phase(w, D, S, lam):
    """单位长度 [1/m] 的色散相位 β2ω²/2 + β3ω³/6。"""
    return beta2(D, lam) / 2 * w ** 2 + (beta3(D, S, lam) / 6 * w ** 3 if S else 0)


def cd(E, fs, L, D=17.0, lam=1550.0, S=0.0):
    """线性色散传输（无损耗、无非线性）。

    参数: E (P,N) 或 (N,) 光场; fs 采样率 [Hz]; L [km]; D [ps/(nm km)]; lam [nm]; S 色散斜率（0 则忽略 β3）
    """
    return fft_filter(E, np.exp(-1j * _disp_phase(2 * np.pi * freq(E.shape[-1], fs), D, S, lam) * L * 1e3))


def power_fading(f, L, D=17.0, lam=1550.0, alpha=0.0):
    """IM-DD 色散功率衰落的小信号频率响应 |H(f)| = |cos(θ) - α·sin(θ)|，θ = π·λ²·D·L·f²/c（Henry 因子 α 的啁啾）。
    α=0 时第 k 个陷波位于 f_k = sqrt((2k-1)·c / (2·λ²·D·L))。f [Hz]，L [km]，D [ps/(nm km)]，lam [nm]。"""
    th = np.pi * (lam * 1e-9) ** 2 * D * 1e-6 * L * 1e3 * np.asarray(f) ** 2 / C
    return np.abs(np.cos(th) - alpha * np.sin(th))


def raman_response(w, tau1=12.2e-15, tau2=32e-15):
    """石英 Raman 响应的频域形式 H_R(ω)（h_R(t) 的 FFT，H_R(0)=1）。w 为角频率 [rad/s]。"""
    return (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2) / tau1 / ((1 / tau2 + 1j * w) ** 2 + 1 / tau1 ** 2)


def ssfm(E, fs, fib, nsteps=None, phi_max=None, dz_max=2.0, rng=None):
    """对称分步傅里叶法单跨传输（Manakov / 标量 NLSE），见模块说明的方程。

    参数
    ----
    E       : (2, N) 双偏振或 (N,) 单偏振光场 [√W]
    fs      : 采样率 [Hz]（WDM 时须覆盖全部信道带宽）
    fib     : 光纤参数 dict（params.fiber_params），fib['raman'] / fib['steep'] 控制 SRS / 自陡峭
    nsteps  : 均匀步数；与 phi_max 二选一
    phi_max : 自适应步长：每步最大非线性相移 [rad]（推荐 WDM 用 0.005~0.01）
    dz_max  : 自适应步长上限 [km]
    返回: 传输后光场（形状同输入）
    """
    rng = np.random.default_rng(rng)
    E = np.atleast_2d(E).astype(complex)
    Lm = fib['L'] * 1e3
    a = fib['alpha'] * np.log(10) / 10 / 1e3                  # 功率衰减 [1/m]
    w = 2 * np.pi * freq(E.shape[-1], fs)
    lin = -a / 2 - 1j * _disp_phase(w, fib['D'], fib.get('S', 0), fib['lam'])   # 线性算子 [1/m]
    g = fib['gamma'] / 1e3 * (8 / 9 if E.shape[0] == 2 else 1)
    full = fib.get('raman') or fib.get('steep')
    if full:
        HR = raman_response(w, fib['tau1'], fib['tau2']) if fib.get('raman') else None
        fR = fib['fR'] if fib.get('raman') else 0.0
        sh = 1 + w / (2 * np.pi * fib['f0']) if fib.get('steep') else 1.0

    def nl(E):                                               # 完整非线性算子 dE/dz|NL
        P = np.sum(np.abs(E) ** 2, 0)
        if fR:
            P = (1 - fR) * P + fR * np.real(np.fft.ifft(np.fft.fft(P) * HR))
        return -1j * g * np.fft.ifft(np.fft.fft(E * P, axis=-1) * sh, axis=-1)

    z = 0.0
    while z < Lm - 1e-9:
        if phi_max:
            dz = min(dz_max * 1e3, phi_max / max(abs(g) * np.max(np.sum(np.abs(E) ** 2, 0)), 1e-30), Lm - z)
        else:
            dz = Lm / nsteps
        Hh = np.exp(lin * dz / 2)
        E = fft_filter(E, Hh)
        if full:                                             # RK4 求解非线性子步
            k1 = dz * nl(E); k2 = dz * nl(E + k1 / 2); k3 = dz * nl(E + k2 / 2); k4 = dz * nl(E + k3)
            E = E + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:                                                # 纯 Kerr：精确相位旋转
            E = E * np.exp(-1j * g * np.sum(np.abs(E) ** 2, 0) * dz)
        E = fft_filter(E, Hh)
        if fib.get('pmd') and E.shape[0] == 2:               # 粗步长 PMD，使平均 DGD = pmd·√L
            tau = fib['pmd'] * 1e-12 * np.sqrt(dz / 1e3) * np.sqrt(3 * np.pi / 8)
            E = random_unitary(rng) @ E
            E = fft_filter(E, np.stack([np.exp(1j * w * tau / 2), np.exp(-1j * w * tau / 2)]))
        z += dz
    return E


def link(E, fs, fib, n_spans, ase=True, rng=None, **ssfm_kw):
    """多跨段链路：每跨 ssfm + 放大器（增益 = 跨段损耗，NF = fib['nf']，ASE 中心频率 fib['f0']）。

    参数: n_spans 跨数; ase=False 时理想无噪放大; ssfm_kw 透传给 ssfm（nsteps / phi_max / dz_max）
    """
    rng = np.random.default_rng(rng)
    G = fib['alpha'] * fib['L']
    for _ in range(n_spans):
        E = ssfm(E, fs, fib, rng=rng, **ssfm_kw)
        E = edfa(E, fs, gain_db=G, nf_db=fib['nf'], f0=fib['f0'], rng=rng) if ase else E * 10 ** (G / 20)
    return E


# ======================= GN 模型（非线性干扰解析估计） =======================
def gn_eta(fib, rs, n_ch=1, spacing=None, method='numeric', ppc=40):
    """单跨、中心信道的非线性干扰系数 η [1/W²]：P_NLI = η·P_ch³（非相干 GN 模型，矩形谱，双偏振）。

    method = 'numeric'：数值积分 GN 参考公式
        G_NLI(0) = 16/27·γ²·∬ G(f1)G(f2)G(f1+f2)·|(1-e^{-αL}e^{j4π²β2L f1f2}) / (α - j4π²β2 f1f2)|² df1 df2
    method = 'closed'：Poggiolini 闭式近似
        η = 8/27·γ²·Leff²·asinh(π²/2·|β2|·Leff,a·Rs²·Nch^{2Rs/Δf}) / (π·|β2|·Leff,a·Rs²)
    参数: rs 符号率（信道带宽）[Hz]; n_ch 信道数; spacing 信道间隔 [Hz]; ppc 每信道积分点数
    """
    spacing = spacing or rs
    a = fib['alpha'] * np.log(10) / 10 / 1e3
    Lm, g, b2 = fib['L'] * 1e3, fib['gamma'] / 1e3, abs(beta2(fib['D'], fib['lam']))
    Leff, Lea = (1 - np.exp(-a * Lm)) / a, 1 / a
    if method == 'closed':
        return 8 / 27 * g ** 2 * Leff ** 2 * np.arcsinh(np.pi ** 2 / 2 * b2 * Lea * rs ** 2 * n_ch ** (2 * rs / spacing)) \
            / (np.pi * b2 * Lea * rs ** 2)
    fc = (np.arange(n_ch) - (n_ch - 1) / 2) * spacing
    B = fc[-1] - fc[0] + rs
    df = rs / ppc
    f = np.arange(-B / 2 + df / 2, B / 2, df)
    inband = lambda x: np.any(np.abs(x[..., None] - fc) <= rs / 2, -1)
    m2 = inband(f)
    tot = 0.0
    for i in np.flatnonzero(m2):                             # 逐行积分，节省内存
        f2 = f[m2]
        x = 4 * np.pi ** 2 * b2 * Lm * f[i] * f2
        mu = (1 - np.exp(-a * Lm) * np.exp(1j * x)) / (a - 1j * x / Lm)
        tot += np.sum(inband(f[i] + f2) * np.abs(mu) ** 2)
    return 16 / 27 * g ** 2 * tot * df ** 2 / rs ** 3 * rs


def gn_model(p_ch_dbm, fib, rs, n_spans, n_ch=1, spacing=None, method='numeric'):
    """GN 模型链路 SNR 预测（理想收发机，非相干累积 ε=0）。

    参数: p_ch_dbm 每信道入纤功率 [dBm]（可为数组）; 其余同 gn_eta
    返回 dict: snr, snr_ase, snr_nli [dB]; eta [1/W²]（全链路）; p_opt_dbm, snr_opt [dB]
    """
    from .utils import H, dbm2w
    P = dbm2w(np.asarray(p_ch_dbm, float))
    G = 10 ** (fib['alpha'] * fib['L'] / 10)
    p_ase = n_spans * 10 ** (fib['nf'] / 10) * H * fib['f0'] * (G - 1) * rs   # 双偏振、信号带宽内
    eta = n_spans * gn_eta(fib, rs, n_ch, spacing, method)
    p_nli = eta * P ** 3
    p_opt = (p_ase / (2 * eta)) ** (1 / 3)
    db = lambda x: 10 * np.log10(x)
    return dict(snr=db(P / (p_ase + p_nli)), snr_ase=db(P / p_ase), snr_nli=db(P / p_nli), eta=eta,
                p_opt_dbm=db(p_opt / 1e-3), snr_opt=db(p_opt / (1.5 * p_ase)))
