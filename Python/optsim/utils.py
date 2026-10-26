"""通用工具：物理常数、单位换算、功率、频率轴、频域滤波、电低通、重采样、分数时延、AWGN。

约定：双偏振信号为 (2, N) 复数组，单偏振为 (N,)；所有运算沿最后一维（时间）。光场单位 √W。
"""
import numpy as np
from scipy import signal as ss

C = 299792458.0          # 光速 [m/s]
H = 6.62607015e-34       # 普朗克常数 [J·s]
Q = 1.602176634e-19      # 电子电荷 [C]
KB = 1.380649e-23        # 玻尔兹曼常数 [J/K]
F0 = 193.414e12          # 默认载波频率 [Hz]（1550 nm）


def db2lin(x):
    """dB -> 线性（功率比）。"""
    return 10 ** (np.asarray(x) / 10)


def lin2db(x):
    """线性 -> dB。"""
    return 10 * np.log10(x)


def dbm2w(p):
    """dBm -> W。"""
    return 1e-3 * 10 ** (np.asarray(p) / 10)


def w2dbm(p):
    """W -> dBm。"""
    return 10 * np.log10(p / 1e-3)


def power(x):
    """每路平均功率 E|x|²（沿最后一维）；双偏振返回长度 2 的数组。"""
    return np.mean(np.abs(x) ** 2, axis=-1)


def total_power(x):
    """所有偏振的总平均功率 [W]。"""
    return float(np.sum(power(np.atleast_2d(x))))


def normalize(x):
    """每路归一化到单位平均功率。"""
    return x / np.sqrt(power(x))[..., None]


def set_power(x, p_dbm):
    """缩放信号使总功率（所有偏振之和）为 p_dbm。"""
    return x * np.sqrt(dbm2w(p_dbm) / total_power(x))


def freq(n, fs):
    """FFT 顺序的频率轴 [Hz]（与 np.fft.fft 输出一一对应）。"""
    return np.fft.fftfreq(n, 1 / fs)


def fft_filter(x, h):
    """频域滤波（循环卷积）：ifft(fft(x)·h)。h 为 FFT 顺序的频率响应，可按行广播。"""
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def lpf(x, fs, bw, kind='gaussian', order=4):
    """电低通滤波（DAC/ADC/驱动/PD 带宽）。

    参数: bw 3 dB 带宽 [Hz]; kind 'gaussian'(零相位) | 'bessel'(线性相位, 有群时延) | 'butter' | 'brickwall';
          order 贝塞尔/巴特沃斯阶数
    """
    f = freq(x.shape[-1], fs)
    if kind == 'gaussian':
        h = np.exp(-np.log(2) / 2 * (f / bw) ** 2)
    elif kind == 'brickwall':
        h = (np.abs(f) <= bw).astype(float)
    elif kind in ('bessel', 'butter'):
        b, a = (ss.bessel(order, 2 * np.pi * bw, analog=True, norm='mag') if kind == 'bessel'
                else ss.butter(order, 2 * np.pi * bw, analog=True))
        h = ss.freqs(b, a, 2 * np.pi * f)[1]
    return fft_filter(x, h)


def resample(x, fs_in, fs_out):
    """周期（FFT）重采样，沿最后一维；等效理想砖墙抗混叠，与本仿真的循环信号假设一致（无边缘瞬态）。
    输出点数 = round(N·fs_out/fs_in)。"""
    return ss.resample(x, int(round(x.shape[-1] * fs_out / fs_in)), axis=-1)


def upsample(x, sps):
    """插零上采样 sps 倍。"""
    y = np.zeros(x.shape[:-1] + (x.shape[-1] * sps,), dtype=x.dtype)
    y[..., ::sps] = x
    return y


def delay(x, tau, fs):
    """分数时延 tau [s]（频域线性相位，循环）。tau 可为标量，或每行一个值的数组。"""
    tau = np.asarray(tau, dtype=float)[..., None] if np.ndim(tau) else tau
    return fft_filter(x, np.exp(-2j * np.pi * freq(x.shape[-1], fs) * tau))


def iq_delay(x, tau_i, tau_q, fs):
    """I、Q 分别施加时延（模拟或补偿 IQ skew）。"""
    return delay(x.real, tau_i, fs).real + 1j * delay(x.imag, tau_q, fs).real


def awgn(x, snr_db, sps=1, rng=None):
    """按符号 SNR [dB] 加复高斯白噪声；sps>1 时噪声铺满整个仿真带宽（匹配滤波后 SNR = snr_db）。"""
    rng = np.random.default_rng(rng)
    sigma2 = power(x) * sps / db2lin(snr_db)
    n = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    return x + np.sqrt(sigma2 / 2)[..., None] * n
