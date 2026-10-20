"""接收端 DSP 算法库（相干为主，兼顾 IM-DD / 自相干）。

典型相干流程（system.rx_dsp 已串好）：
    前端校正 (deskew, gsop) -> CDC / DBP -> 匹配滤波 -> 定时恢复 (timing_ff / gardner)
    -> 2x2 MIMO (mimo_eq) -> 频偏估计 (foe) -> 载波相位恢复 (bps / vv / pilot_cpr / pll)
    -> 帧同步 (sync) -> 模糊/周跳消除 (fix_ambiguity) -> 判决
另有：微扰非线性补偿 pb_nlc；IM-DD 的 ffe / dfe / volterra / mlse；OFDM 的 ofdm_demod / ofdm_eq / schmidl_cox；KK 接收。
输入输出均为 (P, N) 复数组（P 个偏振，沿最后一维为时间）。
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert
from .utils import freq, fft_filter, delay, iq_delay, set_power, total_power
from .channel import beta2, beta3, ssfm
from .tx import decide


def _movmean(x, n):
    """沿最后一维的 n 点滑动平均（非循环，边缘取最近值；相位跟踪是时序过程，不能首尾相接）。"""
    f = lambda v: uniform_filter1d(v, n, axis=-1, mode='nearest')
    return f(x.real) + 1j * f(x.imag) if np.iscomplexobj(x) else f(x)


# =============== 前端校正 ===============
def dc_block(x):
    """去直流。"""
    return x - np.mean(x, axis=-1, keepdims=True)


def deskew(x, fs, skew):
    """补偿 IQ skew：对各支路施加相反时延。skew = [XI, XQ, YI, YQ] [s]（标定/估计值）。"""
    return np.stack([iq_delay(x[p], -skew[2 * p], -skew[2 * p + 1], fs) for p in range(x.shape[0])])


def gsop(x):
    """Gram-Schmidt 正交化：去直流并补偿每路 IQ 幅度/相位不平衡，输出 I、Q 单位功率且正交。"""
    I, Qd = dc_block(x.real), dc_block(x.imag)
    I = I / np.sqrt(np.mean(I ** 2, -1, keepdims=True))
    Qd = Qd - np.mean(I * Qd, -1, keepdims=True) * I
    Qd = Qd / np.sqrt(np.mean(Qd ** 2, -1, keepdims=True))
    return I + 1j * Qd


# =============== 色散补偿 / 非线性补偿 ===============
def cdc(x, fs, L, D=17.0, lam=1550.0, S=0.0):
    """频域色散补偿（channel.cd 的逆）。L 总长度 [km]；S≠0 时同时补偿色散斜率。"""
    w = 2 * np.pi * freq(x.shape[-1], fs)
    ph = beta2(D, lam) / 2 * w ** 2 + (beta3(D, S, lam) / 6 * w ** 3 if S else 0)
    return fft_filter(x, np.exp(1j * ph * L * 1e3))


def cd_estimate(x, fs, rs, L_grid, D=17.0, lam=1550.0):
    """盲色散估计：扫描补偿长度，代价函数为平方律时钟音 |Σ|x|²·e^{-j2πRs t}|（正确补偿时最大）。

    返回: (最佳长度 [km], 代价函数数组)
    """
    tone = np.exp(-2j * np.pi * rs * np.arange(x.shape[-1]) / fs)
    m = np.array([np.abs(np.sum(np.sum(np.abs(cdc(x, fs, L, D, lam)) ** 2, 0) * tone)) for L in L_grid])
    return L_grid[np.argmax(m)], m


def dbp(x, fs, p_launch_dbm, n_spans, fib, steps=10, xi=1.0):
    """数字反向传播（逐跨：撤销放大器增益 -> 以 -α, -D, -S, -γ 做 SSFM）。

    参数: p_launch_dbm 每跨入纤总功率（用于把接收信号还原为物理功率）; fib 光纤参数 dict;
          steps 每跨步数; xi 非线性补偿系数（0 退化为 CDC，常取 0.5~1 优化）
    """
    s = np.sqrt(total_power(x))
    E = set_power(x, p_launch_dbm)
    b = dict(fib, alpha=-fib['alpha'], D=-fib['D'], S=-fib.get('S', 0), gamma=-fib['gamma'] * xi,
             pmd=0, raman=False, steep=False)
    for _ in range(n_spans):
        E = E * 10 ** (-fib['alpha'] * fib['L'] / 20)
        E = ssfm(E, fs, b, nsteps=steps)
    return E / np.sqrt(total_power(E)) * s


def pb_nlc(y, ref, n_train, L=10, lam=1e-3):
    """一阶微扰非线性补偿（PB-NLC），微扰系数由训练符号最小二乘拟合：
    Δy_p[k] = Σ_{m,n} C_mn·(y_p[k+m]·y_p*[k+m+n] + y_q[k+m]·y_q*[k+m+n])·y_p[k+n]，
    取 |m|,|n| ≤ L 且 |m·n| ≤ L 的三元组（双曲剪枝），两偏振共享 C；输出 y − Δy。
    参数: y、ref (P, N) CDC + CPR 后、已同步的 1 sps 符号; n_train 训练符号数; lam 岭回归系数
    返回: (补偿后符号, C)
    """
    y = np.atleast_2d(y)
    mn = [(m, n) for m in range(-L, L + 1) for n in range(-L, L + 1) if abs(m * n) <= L]
    sh = lambda v, d: np.roll(v, -d, axis=-1)                # sh(v, d)[k] = v[k+d]
    P = lambda m, n: np.sum(sh(y, m) * np.conj(sh(y, m + n)), 0)
    T = np.stack([P(m, n) * sh(y, n) for m, n in mn], -1)    # (P, N, F)
    A = T[:, :n_train].reshape(-1, len(mn))
    c = np.linalg.solve(A.conj().T @ A + lam * len(A) * np.eye(len(mn)),
                        A.conj().T @ (y[:, :n_train] - ref[:, :n_train]).ravel())
    return y - T @ c, c


# =============== 定时恢复 ===============
def _interp(x, t):
    """三次 Lagrange (Farrow) 插值，t 为分数采样位置（循环索引）。"""
    i = int(np.floor(t))
    m = t - i
    c = np.array([-m * (m - 1) * (m - 2) / 6, (m + 1) * (m - 1) * (m - 2) / 2,
                  -(m + 1) * m * (m - 2) / 2, (m + 1) * m * (m - 1) / 6])
    return x[:, (i + np.arange(-1, 3)) % x.shape[-1]] @ c


def timing_ff(x, sps=2):
    """前馈 Godard 时钟音定时估计：c = Σ_{f>0} X(f)·X*(f-Rs)，τ = -∠c/(2π)·sps。

    适用：块处理、无时钟频偏；2 sps 即可。返回 (校正后信号, τ [采样])
    """
    X = np.fft.fft(x, axis=-1)
    N = X.shape[-1]
    k = np.arange(N // 2)
    c = np.sum(X[:, k] * np.conj(X[:, (k - N // sps) % N]))
    tau = -np.angle(c) / (2 * np.pi) * sps
    return delay(x, -tau, 1.0), tau


def gardner(x, mu=2e-3, sps=2):
    """Gardner 反馈定时恢复：e = Re{(y[k-1] - y[k])·y*[k-1/2]}，τ ← τ + μe，三次插值。

    可跟踪时钟频偏。输入/输出均为 2 sps。返回 (y, 每符号 τ 轨迹)
    """
    P, N = x.shape
    nsym = N // sps
    y = np.zeros((P, 2 * nsym), complex)
    tau, taus = 0.0, np.zeros(nsym)
    for k in range(1, nsym):
        t = k * sps + tau
        yk, ym, yp = _interp(x, t), _interp(x, t - sps / 2), _interp(x, t - sps)
        tau += mu * np.sum(np.real((yp - yk) * np.conj(ym)))
        y[:, 2 * k], y[:, 2 * k - 1], taus[k] = yk, ym, tau
    return y, taus


# =============== 自适应均衡 / 偏振解复用 ===============
def mimo_eq(x, ntaps=15, mu=1e-3, sps=2, mode='cma', const=None, ref=None, n_train=0, w=None, epochs=1):
    """2x2 蝶形 FIR 自适应均衡（逐符号随机梯度 w ← w + μ·e·x*），完成偏振解复用与残余 CD/PMD/ISI 补偿。

    mode : 'cma' 恒模  e = y(R2 - |y|²)
           'rde' 半径判决 e = y(R_d² - |y|²)，R_d 为最近星座半径（多模 QAM）
           'mma' 实虚分离多模
           'lms' 前 n_train 符号用 ref 训练（数据辅助），之后判决引导
           'dd'  判决引导 LMS（需先完成载波恢复）
    参数: ntaps 抽头数; mu 步长; sps 输入过采样（输出 1 sps）; const 星座（cma 可省略，rde/mma/lms/dd 必须给出）;
          w 初始抽头 (P,P,ntaps)，None 为中心单抽头
    返回: (y (P, nsym), w (P, P, ntaps), |e| (P, nsym))
    """
    P, N = x.shape
    nsym, h = N // sps, ntaps // 2
    xp = np.concatenate([x[:, N - h:], x, x[:, :h]], 1)
    W = np.zeros((P, P * ntaps), complex) if w is None else w.reshape(P, -1).copy()
    if w is None:
        for i in range(P):
            W[i, i * ntaps + h] = 1
    if const is None:                                        # 未给星座：按单位功率恒模（QPSK）处理，仅适用 cma/mma
        const = np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7]))
    R2 = np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2)
    Rr = np.mean(const.real ** 4) / np.mean(const.real ** 2)
    radii = np.unique(np.round(np.abs(const), 6))
    y, err = np.zeros((P, nsym), complex), np.zeros((P, nsym))
    for _ in range(epochs):
        for k in range(nsym):
            X = xp[:, k * sps:k * sps + ntaps].ravel()
            yk = W @ X
            if mode == 'cma':
                e = yk * (R2 - np.abs(yk) ** 2)
            elif mode == 'rde':
                r = radii[np.argmin(np.abs(np.abs(yk)[:, None] - radii), 1)]
                e = yk * (r ** 2 - np.abs(yk) ** 2)
            elif mode == 'mma':
                e = yk.real * (Rr - yk.real ** 2) + 1j * yk.imag * (Rr - yk.imag ** 2)
            else:
                d = ref[:, k] if (mode == 'lms' and k < n_train) else const[decide(yk, const)]
                e = d - yk
            W += mu * np.outer(e, X.conj())
            y[:, k], err[:, k] = yk, np.abs(e)
    return y, W.reshape(P, P, ntaps), err


def ffe(x, ref, ntaps=21, mu=1e-3, sps=1, n_train=None, levels=None):
    """实数 LMS 前馈均衡器（IM-DD / PAM）。

    参数: x 实信号; ref 训练符号; n_train 训练长度（None 全程训练）; levels 判决电平（DD 阶段）
    返回: (均衡输出 1 sps, 抽头)
    """
    N, h = len(x) // sps, ntaps // 2
    xp = np.r_[x[-h:], x, x[:h]]
    w = np.zeros(ntaps)
    w[h] = 1
    n_train = N if n_train is None else n_train
    y = np.zeros(N)
    for k in range(N):
        X = xp[k * sps:k * sps + ntaps]
        y[k] = w @ X
        d = ref[k] if k < n_train else levels[np.argmin(np.abs(levels - y[k]))]
        w += mu * (d - y[k]) * X
    return y, w


def sym_windows(x, ntaps, sps=1):
    """滑窗取样：以第 k 个符号（采样点 k·sps）为中心取 ntaps 个采样（循环边界）。

    参数: x (N,) 实信号，或 (P, N) 复信号
    返回: (Nsym, ntaps, C) 实数组；实信号 C=1，复信号 C=2P，顺序 [Re p1, Im p1, Re p2, Im p2, ...]
    NN 均衡器与 Volterra 均衡器共用此函数。
    """
    x = np.atleast_2d(x)
    if np.iscomplexobj(x):
        x = np.stack([v for p in x for v in (p.real, p.imag)])
    N = x.shape[-1]
    idx = (np.arange(N // sps)[:, None] * sps + np.arange(ntaps) - ntaps // 2) % N
    return np.moveaxis(x[:, idx], 0, -1)


def volterra_features(x, taps, sps):
    """实数 Volterra 特征矩阵 (Nsym, F)：[1, 一阶 L1 项, 二阶 L2(L2+1)/2 项, 三阶 C(L3+2,3) 项]（各阶为居中子窗）。
    训练得到核 w 后，对新数据的均衡输出 = volterra_features(x_new, taps, sps) @ w。"""
    from itertools import combinations_with_replacement as cwr
    W = sym_windows(x, taps[0], sps)[..., 0]                 # (Nsym, L1)
    c = taps[0] // 2
    feats = [np.ones((W.shape[0], 1)), W]
    for order, L in ((2, taps[1]), (3, taps[2])):
        if L:
            sub = W[:, c - L // 2:c - L // 2 + L]
            feats.append(np.stack([np.prod(sub[:, list(i)], 1) for i in cwr(range(L), order)], 1))
    return np.hstack(feats)


def volterra(x, ref, n_train, taps=(31, 9, 5), sps=2, lam=1e-6):
    """实数 Volterra 均衡器（IM-DD 传统非线性均衡）：用前 n_train 个符号做岭回归最小二乘求核，再作用于全序列。

    参数: x 实信号（sps 倍过采样）; ref 训练符号（至少 n_train 个）; taps = (一阶, 二阶, 三阶) 记忆长度，
          二/三阶置 0 即退化为最小二乘 FFE; lam 岭回归正则化系数
    返回: (均衡输出 1 sps, 核系数 w)；乘法复杂度 ≈ len(w) 次/符号
    """
    F = volterra_features(x, taps, sps)
    A = F[:n_train]
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ ref[:n_train])
    return F @ w, w


def ffe_ls(x, ref, n_train, ntaps=31, sps=2):
    """最小二乘 FFE（线性均衡基准，等价于 volterra(taps=(ntaps, 0, 0))）。返回 (输出 1 sps, 抽头)。"""
    return volterra(x, ref, n_train, (ntaps, 0, 0), sps)


def dfe(x, ref, nf=21, nb=3, mu=1e-3, sps=1, n_train=None, levels=None):
    """实数 LMS 判决反馈均衡（IM-DD / PAM）：y_k = w_f·x_k − w_b·[d_{k-1}, ..., d_{k-nb}]，
    训练段 d 取 ref，之后取判决（levels 为判决电平）。FFE 抑制前导 ISI，DFE 无噪声增强地消除后续 ISI。
    返回: (均衡输出 1 sps, (w_f, w_b))
    """
    N, h = len(x) // sps, nf // 2
    xp = np.r_[x[-h:], x, x[:h]]
    wf, wb, d = np.zeros(nf), np.zeros(nb), np.zeros(nb)
    wf[h] = 1
    n_train = N if n_train is None else n_train
    y = np.zeros(N)
    for k in range(N):
        X = xp[k * sps:k * sps + nf]
        y[k] = wf @ X - wb @ d
        dk = ref[k] if k < n_train else levels[np.argmin(np.abs(levels - y[k]))]
        e = dk - y[k]
        wf += mu * e * X
        wb -= mu * e * d
        d = np.r_[dk, d[:-1]]
    return y, (wf, wb)


def ch_est(x, ref, ntaps):
    """最小二乘估计因果 FIR 信道 x_k ≈ Σ_{i<ntaps} h_i·ref_{k-i}（1 sps，循环）。用于 MLSE 的信道模型。"""
    A = np.stack([np.roll(ref, i) for i in range(ntaps)], 1)
    return np.linalg.lstsq(A, x[:len(ref)], rcond=None)[0]


def mlse(x, h, levels):
    """MLSE（Viterbi）序列检测：x_k = Σ_{i=0}^{L} h_i·a_{k-i} + 噪声，a ∈ levels，状态为最近 L 个符号（M^L 个）。
    通常接在短 FFE 之后，此时剩余 ISI 只有 2~3 个抽头（h 由 ch_est 得到）。返回判决电平序列。"""
    M, L = len(levels), len(h) - 1
    S, D = M ** L, M ** (L - 1)
    sp = np.arange(S)
    pred = (sp % D)[:, None] * M + np.arange(M)              # 状态 s' 的 M 个前驱 s
    digits = (pred[..., None] // M ** np.arange(L - 1, -1, -1)) % M   # 前驱 s 的符号 a_{k-1..k-L}
    out = h[0] * levels[sp // D][:, None] + levels[digits] @ h[1:]   # 分支期望输出 (S, M)
    metric, tb = np.zeros(S), np.zeros((len(x), S), np.int8)
    for k, xk in enumerate(x):
        c = metric[pred] + (xk - out) ** 2
        tb[k] = np.argmin(c, 1)
        metric = c[sp, tb[k]]
        metric -= metric.min()
    s, a = int(np.argmin(metric)), np.zeros(len(x))
    for k in range(len(x) - 1, -1, -1):
        a[k] = levels[s // D]
        s = pred[s, tb[k, s]]
    return a


def pf_mlse(ye, ref, n_train, levels, mem=1):
    """FFE + 噪声白化后滤波 + MLSE（IM-DD 带宽受限 / 色散功率衰落的常用方案）：
    FFE 输出的噪声因高频增强而有色。在训练段用最小二乘拟合 AR(mem) 预测器 e_k ≈ -Σ p_i·e_{k-i}，
    后滤波 z = (1 + Σ p_i·D^i)·ye 使噪声变白，同时引入已知 ISI h = [1, p_1..p_mem]，再用 mlse 做序列检测。
    返回判决电平序列（1 sps）。"""
    e = ye[:n_train] - ref[:n_train]
    A = np.stack([np.roll(e, i) for i in range(1, mem + 1)], 1)[mem:]
    p = -np.linalg.lstsq(A, e[mem:], rcond=None)[0]
    h = np.r_[1.0, p]
    z = sum(h[i] * np.roll(ye, i) for i in range(mem + 1))
    return mlse(z, h, levels)


# =============== 频偏估计 ===============
def foe(x, fs, method='fft', m=4, nfft=None):
    """频偏估计 [Hz]。

    'fft' : M 次方谱峰法，峰值频率/M（补零 4 倍提高分辨率），范围 ±fs/(2M)
    'diff': 差分 M 次方（Leven）f = ∠Σ(x[k]x*[k-1])^M / (2πM)·fs
    x 一般为 1 sps 符号（fs = 符号率）。
    """
    x = np.atleast_2d(x)
    if method == 'fft':
        nfft = nfft or 4 * 2 ** int(np.ceil(np.log2(x.shape[-1])))
        s = np.sum(np.abs(np.fft.fft(x ** m, nfft, axis=-1)) ** 2, 0)
        return freq(nfft, fs)[np.argmax(s)] / m
    d = x[:, 1:] * np.conj(x[:, :-1])
    return np.angle(np.sum(d ** m)) / (2 * np.pi * m) * fs


def fo_comp(x, f, fs):
    """频偏补偿：x·exp(-j2πft)。"""
    return x * np.exp(-2j * np.pi * f * np.arange(x.shape[-1]) / fs)


# =============== 载波相位恢复 ===============
def vv(x, n=64, m=4, const=None):
    """Viterbi-Viterbi M 次方相位估计：φ = (unwrap∠<x^M>_n - ∠E[c^M]) / M。返回 (校正后, φ)。含 2π/M 模糊。"""
    ref = np.angle(np.sum(const ** m)) if const is not None else np.pi
    phi = (np.unwrap(np.angle(_movmean(x ** m, n)), axis=-1) - ref) / m
    return x * np.exp(-1j * phi), phi


def bps(x, const, n_test=32, n=64, rot=4):
    """盲相位搜索 BPS（Pfau 2009）：在 [-π/rot, π/rot) 内测试 n_test 个相位，取 n 点窗内判决距离最小者，
    再做 2π/rot 周期解卷绕。rot 为星座旋转对称阶数（tx.rot_order：QAM/QPSK 4，BPSK 2，8PSK 8）。
    返回 (校正后, φ)，含 2π/rot 模糊。"""
    phis = (np.arange(n_test) / n_test - 0.5) * 2 * np.pi / rot
    x = np.atleast_2d(x)
    d = np.stack([np.min(np.abs((x * np.exp(-1j * p))[..., None] - const) ** 2, -1) for p in phis], -1)
    phi = phis[np.argmin(uniform_filter1d(d, n, axis=1, mode='nearest'), -1)]
    phi = np.unwrap(phi * rot, axis=-1) / rot
    return x * np.exp(-1j * phi), phi


def pilot_cpr(x, tx, mask, n=3):
    """导频辅助相位恢复：导频处 ∠(x·tx*)，n 个导频滑动平均，线性插值到全部符号。无相位模糊。

    参数: x、tx 已帧同步的 (P, N)；mask 导频位置布尔数组
    """
    idx = np.flatnonzero(mask)
    r = _movmean(x[:, idx] * np.conj(tx[:, idx]), n)
    ph = np.unwrap(np.angle(r), axis=-1)
    phi = np.stack([np.interp(np.arange(x.shape[-1]), idx, p) for p in ph])
    return x * np.exp(-1j * phi), phi


def pll(x, const, mu=0.05, rho=1e-3):
    """二阶判决引导锁相环：e = Im{y·d*}，f ← f + ρe，θ ← θ + μe + f。返回 (校正后, θ)。"""
    x = np.atleast_2d(x)
    y, phi = np.zeros_like(x), np.zeros(x.shape)
    for p in range(x.shape[0]):
        th, fr = 0.0, 0.0
        for k in range(x.shape[1]):
            y[p, k] = x[p, k] * np.exp(-1j * th)
            e = np.imag(y[p, k] * np.conj(const[decide(y[p, k], const)]))
            fr += rho * e
            th += mu * e + fr
            phi[p, k] = th
    return y, phi


# =============== 同步与模糊消除 ===============
def sync(y, ref, use_abs=False):
    """帧同步 + 偏振交换检测（FFT 循环互相关，取 |相关| 峰）。

    参数: y (2, N) 接收符号; ref (2, Nr) 发送帧; use_abs 用 |y| 相关（相位未恢复时）
    返回: (对齐后 y[:, :Nr], lag, perm)
    """
    N, Nr = y.shape[-1], ref.shape[-1]
    f = (lambda v: np.abs(v) - np.mean(np.abs(v), -1, keepdims=True)) if use_abs else (lambda v: v)
    Y, R = np.fft.fft(f(y), axis=-1), np.fft.fft(f(ref), N, axis=-1)
    c = np.abs(np.fft.ifft(Y[:, None] * np.conj(R[None]), axis=-1))     # c[i, j]: y_i 与 ref_j
    perm = [0, 1] if c[0, 0].max() + c[1, 1].max() >= c[0, 1].max() + c[1, 0].max() else [1, 0]
    lag = int(np.argmax(c[perm[0], 0] + c[perm[1], 1]))
    return np.roll(y[perm], -lag, axis=-1)[:, :Nr], lag, perm


def fix_ambiguity(y, ref, mask=None, n=None, rot=4):
    """用已知符号（mask 标记的训练/导频）消除每偏振 2π/rot 整数倍相位模糊（QAM/QPSK rot=4，8PSK rot=8）。

    n=None：全局一个旋转量；给 n：对已知符号 n 点滑动平均逐段判定（导频辅助周跳消除）。
    """
    N = y.shape[-1]
    step = 2 * np.pi / rot
    idx = np.arange(N) if mask is None else np.flatnonzero(mask)
    r = y[:, idx] * np.conj(ref[:, idx])
    if n is None:
        k = np.round(np.angle(np.sum(r, -1)) / step)[:, None]
    else:
        k = np.round(np.angle(_movmean(r, n)) / step)
        k = k[:, np.maximum(np.searchsorted(idx, np.arange(N), 'right') - 1, 0)]
    return y * np.exp(-1j * k * step)


# =============== OFDM ===============
def ofdm_demod(y, nfft, ncp):
    """OFDM 解调：串并转换 -> 去循环前缀 -> FFT（与 tx.ofdm_mod 互逆）。返回 (nsym, nfft)。"""
    n = nfft + ncp
    return np.fft.fft(y[:len(y) // n * n].reshape(-1, n)[:, ncp:], axis=-1) / np.sqrt(nfft)


def ofdm_eq(Y, X, n_train=1, pilots=None):
    """OFDM 单抽头频域均衡：前 n_train 个训练符号做 LS 信道估计 H = <Y/X>（各训练符号先对齐公共相位再平均，
    避免激光相噪使平均失效），Y/H；给出 pilots（导频子载波下标）时再逐符号去除公共相位误差 CPE = ∠Σ_p Y·X*。
    X 为已知发送符号（只用到训练行与导频列）。返回 (均衡后 Y, H)。"""
    Hk = Y[:n_train] / np.where(X[:n_train] == 0, 1, X[:n_train])
    Hc = np.mean(Hk * np.exp(-1j * np.angle(np.sum(Hk * np.conj(Hk[:1]), 1)))[:, None], 0)
    Z = Y / np.where(Hc == 0, 1, Hc)
    if pilots is not None:
        Z = Z * np.exp(-1j * np.angle(np.sum(Z[:, pilots] * np.conj(X[:, pilots]), 1)))[:, None]
    return Z, Hc


def schmidl_cox(y, L, fs=1.0):
    """Schmidl-Cox 定时与频偏估计：前导为两个相同的半符号（各 L 点）。
    M(d) = |P(d)|²/R(d)²，P(d) = Σ_m y*[d+m]·y[d+m+L]，R(d) = Σ_m |y[d+m+L]|²；
    返回 (d̂ = argmax M, 频偏 ∠P(d̂)·fs/(2πL)（范围 ±fs/(2L)）, M)。"""
    c = lambda v: np.convolve(v, np.ones(L))[L - 1:len(y) - L]
    P = c(np.conj(y[:-L]) * y[L:])
    R = c(np.abs(y[L:]) ** 2)
    M = np.abs(P) ** 2 / np.maximum(R, 1e-30) ** 2
    d = int(np.argmax(M))
    return d, np.angle(P[d]) * fs / (2 * np.pi * L), M


# =============== 自相干 ===============
def kk(i_pd):
    """Kramers-Kronig 接收：由单 PD 光电流（强载波 + 单边带信号，满足最小相位条件）重建复光场（含载波）。
    E = √I·exp(j·H{ln√I})，H 为 Hilbert 变换。"""
    a = np.log(np.sqrt(np.maximum(i_pd, 1e-12 * np.max(i_pd))))
    return np.exp(a + 1j * np.imag(hilbert(a)))
