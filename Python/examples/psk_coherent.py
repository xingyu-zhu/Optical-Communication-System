"""相干 DP-M-PSK（BPSK / QPSK / 8PSK / 16PSK）背靠背 BER–OSNR，与 AWGN 理论对比。
复用 system.rx_dsp（Config(mod='psk')）：FOE 的 M 次方、BPS 搜索范围、模糊消除步长均由星座旋转对称阶数
tx.rot_order 自动确定（BPSK 2，QPSK 4，8PSK 8，16PSK 16）。含 100 kHz 收发激光线宽与 0.5 GHz 频偏。
另加 DP-16QAM 作对比：同为 4 bit/符号，16QAM 最小欧氏距离更大，HD-FEC 门限处比 16PSK 约省 3.7 dB OSNR。
运行：python examples/psk_coherent.py（约 2 分钟）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, metrics, plots
from optsim.system import Config, tx_signal, rx_dsp


def b2b(cfg, osnr, seed=0):
    """理想调制 + 发射/本振激光相噪 + 频偏 + OSNR 加载 -> 相干接收 -> rx_dsp。返回 rx_dsp 结果。"""
    ref = tx_signal(cfg)
    N = ref['wave'].shape[1]
    E = ref['wave'] * dev.laser(N, cfg.fs, 0, 100e3, rng=seed)
    E = dev.set_osnr(E, osnr, cfg.fs, rng=seed + 1)
    r = dev.coherent_rx(E, dev.laser(N, cfg.fs, 10, 100e3, 0.5e9, rng=seed + 2), cfg.fs, shot=False, rng=seed + 3)
    return rx_dsp(r, cfg, ref), ref


fmts = {('psk', 2): np.arange(4, 13, 1.0), ('psk', 4): np.arange(8, 17, 1.0),
        ('psk', 8): np.arange(13, 22, 1.0), ('psk', 16): np.arange(18, 27, 1.0),
        ('qam', 16): np.arange(14, 23, 1.0)}
os.makedirs('results', exist_ok=True)
fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(1, 3, (1, 2))
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
syms = {}
for (mod, M), col in zip(fmts, cols):
    cfg = Config(M=M, mod=mod, rs=32e9, n_blocks=300, n_spans=0, fs_adc=128e9)
    osnr, ber = fmts[(mod, M)], []
    for o in osnr:
        res, ref = b2b(cfg, o)
        ber.append(res['ber'])
    syms[(mod, M)] = res['sym'][0]
    name = {2: 'BPSK', 4: 'QPSK'}.get(M, f'{M}{mod.upper()}')
    o2 = np.linspace(osnr[0], osnr[-1], 100)
    ax.semilogy(osnr, np.maximum(ber, 1e-6), 'o', color=col, label=f'DP-{name} (sim)')
    ax.semilogy(o2, metrics.ber_theory(M, metrics.osnr2snr(o2, cfg.rs), mod), '-', color=col, lw=1)
    print(f'DP-{name:5s}: OSNR @ HD-FEC {metrics.sensitivity(osnr, ber, metrics.HD_FEC):5.2f} dB '
          f'(theory {metrics.sensitivity(o2, metrics.ber_theory(M, metrics.osnr2snr(o2, cfg.rs), mod), metrics.HD_FEC):5.2f} dB)')
ax.axhline(metrics.HD_FEC, color='k', ls=':', label='HD-FEC')
ax.set(xlabel='OSNR (dB / 0.1 nm)', ylabel='BER', ylim=(1e-6, 0.3), title='DP-M-PSK, 32 GBd B2B (lines: AWGN theory)')
ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8, ncol=2)
plots.constellation(syms[('psk', 8)], fig.add_subplot(1, 3, 3), 'DP-8PSK @ 21 dB OSNR, X pol')
plt.tight_layout(); plt.savefig('results/psk_coherent.png', dpi=130)
