"""性能指标：BER/SER/EVM/SNR/Q、GMI/MI（可达信息速率）、AWGN 理论 BER、OSNR <-> SNR。"""
import numpy as np
from scipy.special import erfc, erfcinv
from .tx import decide, int2bits, llr

HD_FEC = 3.8e-3    # 7% 开销硬判决 FEC 门限
SD_FEC = 2.0e-2    # ~20% 开销软判决 FEC 门限


def ber(tx_bits, rx_bits):
    """误码率。返回 (BER, 误比特数)。"""
    e = int(np.sum(np.asarray(tx_bits) != np.asarray(rx_bits)))
    return e / np.size(tx_bits), e


def ser(tx_sym, rx_sym, const):
    """误符号率（两者都先硬判决再比较）。"""
    return np.mean(decide(tx_sym, const) != decide(rx_sym, const))


def evm(rx, tx):
    """rms EVM [%] = sqrt(E|rx-tx|² / E|tx|²)。"""
    return 100 * np.sqrt(np.mean(np.abs(rx - tx) ** 2) / np.mean(np.abs(tx) ** 2))


def snr(rx, tx):
    """数据辅助有效 SNR [dB]：先求最小二乘复增益 a = <tx, rx>/<tx, tx>，SNR = |a|²P_tx / E|rx - a·tx|²。"""
    a = np.vdot(tx, rx) / np.vdot(tx, tx)
    return 10 * np.log10(np.abs(a) ** 2 * np.mean(np.abs(tx) ** 2) / np.mean(np.abs(rx - a * tx) ** 2))


def q_factor(b):
    """由 BER 换算 Q 因子 [dB]：20·log10(√2·erfcinv(2·BER))。"""
    return 20 * np.log10(np.sqrt(2) * erfcinv(2 * b))


def ber_theory(M, snr_db, kind='qam'):
    """AWGN 下 Gray 映射理论 BER（最近邻近似）。kind: 'qam' | 'psk' | 'pam'；snr_db 为每符号 SNR。"""
    s, k = 10 ** (np.asarray(snr_db) / 10), np.log2(M)
    Qf = lambda v: 0.5 * erfc(v / np.sqrt(2))
    if kind == 'qam':
        return 4 / k * (1 - 1 / np.sqrt(M)) * Qf(np.sqrt(3 * s / (M - 1)))
    if kind == 'psk':
        return 2 / k * Qf(np.sqrt(2 * s) * np.sin(np.pi / M)) if M > 4 else Qf(np.sqrt(s))
    return 2 * (M - 1) / (M * k) * Qf(np.sqrt(3 * s / (M ** 2 - 1)))


def gmi(rx, tx, const, prior=None):
    """广义互信息 GMI [bit/sym/pol]（比特交织编码调制 BICM 的可达速率，高斯辅助信道）：
    GMI = H(X) - (1/N)·Σ_k Σ_i log2(1 + exp(-(1-2b_ki)·LLR_ki))。prior 为 PCS 先验。"""
    rx, tx = np.ravel(rx), np.ravel(tx)
    k = int(np.log2(len(const)))
    s2 = np.mean(np.abs(rx - tx) ** 2)
    L = llr(rx, const, s2, prior)
    b = int2bits(decide(tx, const), k)
    Hx = k if prior is None else -np.sum(prior * np.log2(prior))
    return Hx - np.mean(np.logaddexp(0, -(1 - 2.0 * b) * L)) / np.log(2) * k


def mi(rx, tx, const, prior=None):
    """符号级互信息 MI [bit/sym/pol]（高斯辅助信道）：E[log2 q(y|x) / Σ_s p(s)q(y|s)]。"""
    rx, tx = np.ravel(rx), np.ravel(tx)
    p = np.ones(len(const)) / len(const) if prior is None else prior
    s2 = np.mean(np.abs(rx - tx) ** 2)
    num = -np.abs(rx - tx) ** 2 / s2
    den = np.logaddexp.reduce(-np.abs(rx[:, None] - const) ** 2 / s2 + np.log(p), axis=1)
    return np.mean(num - den) / np.log(2)


def osnr2snr(osnr_db, rs, bref=12.5e9, npol=2):
    """OSNR（参考带宽 bref）-> 每偏振符号 SNR [dB]：SNR = OSNR + 10·log10(2·bref / (npol·rs))。"""
    return osnr_db + 10 * np.log10(2 * bref / (npol * rs))


def snr2osnr(snr_db, rs, bref=12.5e9, npol=2):
    """每偏振符号 SNR -> OSNR [dB]（osnr2snr 的逆）。"""
    return snr_db - 10 * np.log10(2 * bref / (npol * rs))
