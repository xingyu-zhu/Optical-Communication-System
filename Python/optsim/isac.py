"""光纤通信-传感一体化（ISAC）：同一个通信信号同时完成数据传输与分布式/积分式振动传感。

后向（可定位，类 φ-OTDR）：发射端用自身的 CP-OFDM 数据信号作探测光，接收其瑞利背向散射。
    逐子载波除以已知发送符号（OFDM 雷达式处理，Sturm & Wiesbeck 2011），再 IFFT，
    得到每个 OFDM 符号时刻的散射冲激响应 h(z)，后续与 φ-OTDR 相同（sensing.das_phase）。
    条件：CP ≥ 散射记忆（被测光纤往返时延对应的采样数），否则会出现空间 ISI。
前向（积分式，类 Marra 2018 / Zhan 2021）：远端相干接收机 DSP 中的信道估计随外界扰动变化。
    数据辅助块估计 Jones 矩阵，得到偏振态（Stokes）与光相位的时间序列，用于振动/地震检测。
"""
import numpy as np
from .dsp import ofdm_demod


def ofdm_profile(y, X, nfft, ncp):
    """CP-OFDM 背向散射 -> 距离剖面：H = ofdm_demod(y)/X（空载波置 0），h = IFFT(H)。
    y: 与发送对齐的背向散射基带信号; X (nsym, nfft) 已知发送子载波符号。
    返回 (nsym, nfft) 复剖面，第 i 列对应距离单元 i（单元长 c/(2·ng·fs)）。"""
    Y = ofdm_demod(y, nfft, ncp)[:len(X)]
    return np.fft.ifft(np.where(X == 0, 0, Y / np.where(X == 0, 1, X)), axis=-1)


def jones_est(y, x, n):
    """数据辅助块 Jones 矩阵估计：每 n 个符号最小二乘 J = Y·X^H·(X·X^H)^{-1}。
    y、x: (2, N) 接收 / 已知（或判决）发送符号（1 sps，已同步）。返回 (N//n, 2, 2)。"""
    nb = y.shape[1] // n
    Y = y[:, :nb * n].reshape(2, nb, n).transpose(1, 0, 2)
    X = x[:, :nb * n].reshape(2, nb, n).transpose(1, 0, 2)
    Xh = X.conj().transpose(0, 2, 1)
    return Y @ Xh @ np.linalg.inv(X @ Xh)


def stokes(E):
    """Jones 向量 E (2, ...) -> 归一化 Stokes 向量 (..., 3)：[|Ex|²−|Ey|², 2Re(Ex·Ey*), −2Im(Ex·Ey*)]/S0。"""
    s0 = np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2
    return np.stack([np.abs(E[0]) ** 2 - np.abs(E[1]) ** 2, 2 * np.real(E[0] * np.conj(E[1])),
                     -2 * np.imag(E[0] * np.conj(E[1]))], -1) / s0[..., None]


def jones_sop(J, e_in=(1, 0)):
    """输入偏振 e_in 经 Jones 矩阵 J (..., 2, 2) 后的输出 Stokes 向量 (..., 3)。"""
    return stokes(np.moveaxis(J @ np.asarray(e_in, complex), -1, 0))


def jones_phase(J):
    """Jones 矩阵的公共相位 ∠det(J)/2（沿时间解卷绕）：前向传输光相位（含激光相噪）。"""
    return np.unwrap(np.angle(np.linalg.det(J))) / 2
