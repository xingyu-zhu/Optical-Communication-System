"""差分相位调制 DBPSK / DQPSK / D8PSK + 延迟线干涉仪（DLI）直接检测（无本振、无载波恢复）。
发射：差分编码 tx.diff_psk -> RRC 成形 -> 激光（相位噪声）；信道：OSNR 加载（ASE 在两个偏振上）；
接收：光匹配滤波（RRC 光带通）-> devices.dli_rx（两个 1 符号延迟 DLI + BPD，偏振不敏感）-> 符号中心采样 -> 差分判决。
(a) BER–OSNR 与差分检测理论（metrics.ber_theory(..., 'dpsk')）及同阶相干 PSK 理论对比。
    注：理论只计信号同偏振 ASE；偏振不敏感 DLI 还会收到正交偏振的 ASE-ASE 拍频，低 OSNR 下（DBPSK 约 7 dB）
    带来约 0.9 dB 额外代价（DLI 前加偏振片即可消除，仿真已验证此时与理论完全重合）；DQPSK/D8PSK 工作 OSNR 高，影响 ≤ 0.2 dB。
(b) 相位噪声容限：固定 OSNR，扫描激光线宽 Δν，BER 随 Δν·T 上升（相干接收由 CPR 跟踪，DLI 只受相邻符号相位差影响）
运行：python examples/dpsk_dli.py（约 1 分钟）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import tx, devices as dev, metrics

rs, sps, beta, nsym = 28e9, 8, 0.2, 2 ** 15
fs = rs * sps


def run(M, osnr, linewidth=100e3, seed=0):
    """单偏振 M-DPSK 一次传输，返回 BER（跳过首个符号）。"""
    rng = np.random.default_rng(seed)
    k = int(np.log2(M))
    bits = tx.rand_bits(nsym * k, rng)
    E = tx.pulse_shape(tx.diff_psk(bits, M), sps, beta) * dev.laser(nsym * sps, fs, 0, linewidth, rng=rng)
    E = np.stack([E, np.zeros_like(E)])                      # 信号在 X 偏振，ASE 落在两个偏振
    E = dev.set_osnr(E, osnr, fs, rng=rng)
    E = tx.matched_filter(E, sps, beta)                       # 光匹配滤波（光带通）
    z = dev.dli_rx(E, fs, rs, shot=False, rng=rng)[::sps]
    return metrics.ber(bits[k:], tx.diff_psk_demod(z[1:], M))[0]


os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ranges = {2: np.arange(5, 14, 1.0), 4: np.arange(9, 18, 1.0), 8: np.arange(14, 24, 1.0)}
names = {2: 'DBPSK', 4: 'DQPSK', 8: 'D8PSK'}
for (M, osnr), col in zip(ranges.items(), plt.rcParams['axes.prop_cycle'].by_key()['color']):
    ber = [run(M, o, seed=int(o)) for o in osnr]
    o2 = np.linspace(osnr[0], osnr[-1], 100)
    snr = metrics.osnr2snr(o2, rs, npol=1)
    ax[0].semilogy(osnr, np.maximum(ber, 1e-6), 'o', color=col, label=f'{names[M]} + DLI (sim)')
    ax[0].semilogy(o2, metrics.ber_theory(M, snr, 'dpsk'), '-', color=col, lw=1)
    ax[0].semilogy(o2, metrics.ber_theory(M, snr, 'psk'), ':', color=col, lw=1)
    s_sim = metrics.sensitivity(osnr, ber, metrics.HD_FEC)
    og = np.linspace(0, 30, 3001)                             # 理论灵敏度在宽范围内求
    sg = metrics.osnr2snr(og, rs, npol=1)
    s_dp = metrics.sensitivity(og, metrics.ber_theory(M, sg, 'dpsk'), metrics.HD_FEC)
    s_co = metrics.sensitivity(og, metrics.ber_theory(M, sg, 'psk'), metrics.HD_FEC)
    print(f'{names[M]}: OSNR @ HD-FEC sim {s_sim:.2f} dB, DPSK theory {s_dp:.2f} dB, '
          f'coherent PSK theory {s_co:.2f} dB (differential penalty {s_dp - s_co:.2f} dB)')
ax[0].axhline(metrics.HD_FEC, color='k', ls='--', lw=0.8, label='HD-FEC')
ax[0].set(xlabel='OSNR (dB / 0.1 nm)', ylabel='BER', ylim=(1e-6, 0.3),
          title='Single-pol M-DPSK, 28 GBd (solid: DPSK theory, dotted: coherent PSK)')
ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(fontsize=8)

lws = np.logspace(5, 9, 9)                                    # 100 kHz ~ 1 GHz
for M, o, col in ((2, 10.0, 'C0'), (4, 14.0, 'C1'), (8, 20.0, 'C2')):
    ax[1].loglog(lws * 1 / rs, [max(run(M, o, lw, seed=3), 1e-6) for lw in lws], 'o-', color=col,
                 label=f'{names[M]} @ OSNR {o:.0f} dB')
ax[1].axhline(metrics.HD_FEC, color='k', ls='--', lw=0.8)
ax[1].set(xlabel='Linewidth × symbol period  Δν·T', ylabel='BER', title='(b) Laser phase-noise tolerance of DLI detection')
ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('results/dpsk_dli.png', dpi=130)
