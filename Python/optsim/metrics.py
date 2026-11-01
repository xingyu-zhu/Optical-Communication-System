"""性能指标：BER/SER/EVM/SNR/Q、GMI/MI（可达信息速率）、AWGN 理论 BER、OSNR <-> SNR。"""
import numpy as np
from scipy.special import erfc, erfcinv
from .tx import decide, int2bits, llr

HD_FEC = 3.8e-3    # 7% 开销硬判决 FEC 门限
SD_FEC = 2.0e-2    # ~20% 开销软判决 FEC 门限
KP4_FEC = 2.4e-4   # IEEE 802.3 KP4 RS(544,514) 门限（数据中心 IM-DD PAM4 常用）


def ber(tx_bits, rx_bits):
    """误码率。返回 (BER, 误比特数)。"""
    e = int(np.sum(np.asarray(tx_bits) != np.asarray(rx_bits)))
    return e / np.size(tx_bits), e


def ser(tx_sym, rx_sym, const):
    """误符号率（两者都先硬判决再比较）。"""
    return np.mean(decide(tx_sym, const) != decide(rx_sym, const))


def moments(const, prior=None):
    """星座归一化高阶矩 (μ4, μ6) = (E|x|⁴/(E|x|²)², E|x|⁶/(E|x|²)³)。
    决定非线性干扰对调制格式的依赖（EGN 模型的修正项含 μ4−2 与 μ6）：QPSK (1, 1)，16QAM (1.32, 1.96)，
    64QAM (1.381, 2.226)，复高斯 (2, 6)；μ4 越接近 2（高阶 QAM、概率整形），NLI 越接近 GN 模型的预测（越大）。"""
    p = np.ones(len(const)) / len(const) if prior is None else np.asarray(prior)
    m2 = np.sum(p * np.abs(const) ** 2)
    return np.sum(p * np.abs(const) ** 4) / m2 ** 2, np.sum(p * np.abs(const) ** 6) / m2 ** 3


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
    """AWGN 下 Gray 映射理论 BER。snr_db 为每符号 SNR (Es/N0)。
    kind: 'qam'（最近邻近似）| 'psk'（相干，BPSK/QPSK 精确）| 'pam' |
          'dpsk'（差分检测：DBPSK 精确 ½e^{-SNR}，DQPSK 精确 Marcum-Q 式，M≥8 近似）"""
    s, k = 10 ** (np.asarray(snr_db) / 10), np.log2(M)
    Qf = lambda v: 0.5 * erfc(v / np.sqrt(2))
    if kind == 'qam':
        return 4 / k * (1 - 1 / np.sqrt(M)) * Qf(np.sqrt(3 * s / (M - 1)))
    if kind == 'psk':
        if M == 2:
            return Qf(np.sqrt(2 * s))
        return Qf(np.sqrt(s)) if M == 4 else 2 / k * Qf(np.sqrt(2 * s) * np.sin(np.pi / M))
    if kind == 'dpsk':
        if M == 2:
            return 0.5 * np.exp(-s)
        if M == 4:                                           # Proakis 式 (8.6-62)，γb = SNR/2
            from scipy.stats import ncx2
            from scipy.special import i0e
            a2, b2 = s * (1 - 1 / np.sqrt(2)), s * (1 + 1 / np.sqrt(2))
            return ncx2.sf(b2, 2, a2) - 0.5 * i0e(np.sqrt(a2 * b2)) * np.exp(np.sqrt(a2 * b2) - (a2 + b2) / 2)
        return 2 / k * Qf(np.sqrt(2 * s) * np.sin(np.pi / (np.sqrt(2) * M)))
    return 2 * (M - 1) / (M * k) * Qf(np.sqrt(3 * s / (M ** 2 - 1)))


def sensitivity(x, ber_values, thr=KP4_FEC):
    """接收灵敏度：在 log10(BER)–x 曲线上线性插值，求 BER 恰为 thr 时的 x（如 ROP [dBm] 或 OSNR [dB]）。
    BER 须随 x 单调下降；未跨过门限时返回 nan。"""
    lb = np.log10(np.maximum(np.asarray(ber_values, float), 1e-12))
    i = np.flatnonzero((lb[:-1] >= np.log10(thr)) & (lb[1:] < np.log10(thr)))
    if len(i) == 0:
        return np.nan
    i = i[0]
    return x[i] + (np.log10(thr) - lb[i]) / (lb[i + 1] - lb[i]) * (x[i + 1] - x[i])


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


def ngmi(rx, tx, const, prior=None):
    """归一化 GMI：NGMI = 1 − (H(X) − GMI)/log2(M)（SD-FEC 门限常用 NGMI 判定，如 0.8~0.9）。"""
    Hx = np.log2(len(const)) if prior is None else -np.sum(prior * np.log2(prior))
    return 1 - (Hx - gmi(rx, tx, const, prior)) / np.log2(len(const))


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
