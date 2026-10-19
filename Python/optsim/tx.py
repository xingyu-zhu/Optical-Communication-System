"""发射端数字部分：比特、星座映射/解映射、LLR、概率整形、成帧、脉冲成形与匹配滤波。

星座数组约定：const[label] 为比特标签 label（MSB 在前）对应的符号；QAM 归一化到单位平均功率。
"""
import numpy as np
from .utils import freq, fft_filter, upsample


# ---------------- 比特 ----------------
def prbs(order, n):
    """PRBS 序列（Fibonacci LFSR，多项式 x^order + x^tap + 1），order ∈ {7, 9, 11, 15, 23, 31}。"""
    taps = {7: 6, 9: 5, 11: 9, 15: 14, 23: 18, 31: 28}[order]
    reg, out = (1 << order) - 1, np.empty(n, dtype=np.uint8)
    for i in range(n):
        b = ((reg >> (order - 1)) ^ (reg >> (taps - 1))) & 1
        reg = ((reg << 1) | b) & ((1 << order) - 1)
        out[i] = b
    return out


def rand_bits(n, rng=None):
    """均匀随机比特，n 可为整数或形状元组，如 (2, 4096)。"""
    return np.random.default_rng(rng).integers(0, 2, n, dtype=np.uint8)


def bits2int(bits, k):
    """每 k 比特（MSB 在前）组成一个整数标签。"""
    return bits.reshape(-1, k) @ (1 << np.arange(k - 1, -1, -1))


def int2bits(idx, k):
    """整数标签 -> 比特（MSB 在前），返回一维数组。"""
    return ((np.asarray(idx)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


# ---------------- 星座 ----------------
def _gray(n):
    return np.arange(n) ^ (np.arange(n) >> 1)


def pam(M):
    """Gray 映射 PAM-M 电平（±1, ±3, ...，未归一化）。"""
    lv = np.empty(M)
    lv[_gray(M)] = 2 * np.arange(M) - M + 1
    return lv


def qam(M):
    """Gray 映射方形 M-QAM（M = 4, 16, 64, 256, ...），单位平均功率。"""
    k = int(np.log2(M)) // 2
    L = 2 ** k
    p = pam(L)
    lab = np.arange(M)
    c = p[lab >> k] + 1j * p[lab & (L - 1)]
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def psk(M):
    """Gray 映射 M-PSK（QPSK 旋转 π/4）。"""
    c = np.empty(M, complex)
    c[_gray(M)] = np.exp(1j * (2 * np.pi * np.arange(M) / M + np.pi / M * (M == 4)))
    return c


def modulate(bits, const):
    """比特 -> 符号。bits 形状 (n·k,) 或 (P, n·k)，返回 (n,) 或 (P, n)。"""
    k = int(np.log2(len(const)))
    return np.array([const[bits2int(b, k)] for b in np.atleast_2d(bits)]).reshape(
        np.shape(bits)[:-1] + (-1,))


def decide(x, const):
    """最小欧氏距离硬判决，返回符号标签（与 x 同形状）。判决后符号为 const[decide(x, const)]。"""
    return np.argmin(np.abs(np.asarray(x)[..., None] - const), axis=-1)


def demodulate(x, const):
    """硬判决解映射：符号 -> 比特，形状 (P, n) -> (P, n·k)。"""
    k = int(np.log2(len(const)))
    idx = decide(x, const)
    return np.array([int2bits(i, k) for i in np.atleast_2d(idx)]).reshape(np.shape(x)[:-1] + (-1,))


def llr(x, const, sigma2, prior=None, maxlog=False):
    """逐比特对数似然比 LLR = log P(b=0|y) / P(b=1|y)（高斯信道）。

    参数: sigma2 复噪声方差; prior 符号先验（PCS）; maxlog True 用 max-log 近似
    返回: (N·k,) 数组，顺序为 [符号1的k比特, 符号2的k比特, ...]
    """
    k = int(np.log2(len(const)))
    m = -np.abs(np.ravel(x)[:, None] - const) ** 2 / sigma2
    if prior is not None:
        m = m + np.log(prior)
    lab = int2bits(np.arange(len(const)), k).reshape(-1, k).astype(bool)
    red = (lambda a: a.max(1)) if maxlog else (lambda a: np.logaddexp.reduce(a, axis=1))
    return np.stack([red(np.where(~lab[:, i], m, -np.inf)) - red(np.where(lab[:, i], m, -np.inf))
                     for i in range(k)], axis=1).ravel()


def pcs(const, nu):
    """Maxwell-Boltzmann 分布 p ∝ exp(-ν|c|²)。返回 (概率, 熵 [bit/sym])；ν 越大整形越强。"""
    p = np.exp(-nu * np.abs(const) ** 2)
    p /= p.sum()
    return p, -np.sum(p * np.log2(p))


def pcs_symbols(const, nu, n, rng=None):
    """按 MB 分布直接抽样 n 个 PCS 符号（理想分布匹配器近似，未实现 CCDM）。

    返回: (symbols 重新归一化为单位功率, labels, p)
    """
    p, _ = pcs(const, nu)
    lab = np.random.default_rng(rng).choice(len(const), n, p=p)
    c = const / np.sqrt(np.sum(p * np.abs(const) ** 2))
    return c[lab], lab, p


# ---------------- 成帧 ----------------
def frame(data, train, pilot, period):
    """帧结构：[训练序列 | 数据块...]，每个数据块 period 个符号、第 1 个为导频。

    参数: data (P, n_d)、train (P, n_t)、pilot (P, ≥n_blk)；n_d 最好为 (period-1) 的整数倍
    返回: (frame (P, N_f), kind (N_f,))，kind: 0 数据 / 1 训练 / 2 导频
    """
    P, nd = data.shape
    nblk = int(np.ceil(nd / (period - 1)))
    body = np.zeros((P, nblk * period), complex)
    kind = np.zeros(nblk * period, np.int8)
    kind[::period] = 2
    body[:, ::period] = pilot[:, :nblk]
    body[:, kind == 0] = np.pad(data, ((0, 0), (0, nblk * (period - 1) - nd)))
    return np.hstack([train, body]), np.r_[np.ones(train.shape[1], np.int8), kind]


# ---------------- 脉冲成形 ----------------
def nyquist_response(f, rs, beta, root=True):
    """升余弦 (root=False) / 根升余弦 (root=True) 频率响应，f 与 rs 单位一致。"""
    f = np.abs(f)
    f1, f2 = (1 - beta) * rs / 2, (1 + beta) * rs / 2
    h = np.where(f <= f1, 1.0, 0.0)
    tr = (f > f1) & (f <= f2)
    h[tr] = 0.5 * (1 + np.cos(np.pi / (beta * rs) * (f[tr] - f1)))
    return np.sqrt(h) if root else h


def pulse_shape(sym, sps, beta, kind='rrc'):
    """脉冲成形：插零上采样 + 频域滤波。

    kind: 'rrc' 根升余弦（相干默认）| 'rc' 升余弦 | 'rect' NRZ 矩形 | 'gaussian'（beta 为 3 dB 带宽-符号周期积 BT）
    """
    if kind == 'rect':
        return np.repeat(sym, sps, axis=-1)
    x = upsample(sym, sps)
    f = freq(x.shape[-1], sps)              # 归一化到符号率
    if kind in ('rrc', 'rc'):
        h = nyquist_response(f, 1.0, beta, kind == 'rrc')
    elif kind == 'gaussian':
        h = np.exp(-(np.pi * f) ** 2 / (2 * np.log(2) * beta ** 2))
    return fft_filter(x, h * sps)


def matched_filter(x, sps, beta):
    """接收端 RRC 匹配滤波（输入 sps 倍过采样，输出同采样率）。"""
    return fft_filter(x, nyquist_response(freq(x.shape[-1], sps), 1.0, beta))
