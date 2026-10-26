"""背靠背 BER-OSNR 曲线：仿真（含激光相噪、频偏、完整 Rx DSP）vs AWGN 理论。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, metrics
from optsim.system import Config, tx_signal, rx_dsp

cfg = Config(M=16, n_blocks=400, n_spans=0, fs_adc=160e9)
rng = np.random.default_rng(0)
ref = tx_signal(cfg)
N = ref['wave'].shape[1]
osnr = np.arange(16, 25, 1.0)
ber = []
for o in osnr:
    E = ref['wave'] * dev.laser(N, cfg.fs, 0, 100e3, rng=rng)          # 理想调制 + 发射激光相噪
    E = dev.set_osnr(E, o, cfg.fs, rng=rng)
    lo = dev.laser(N, cfg.fs, 10, 100e3, f_offset=0.8e9, rng=rng)
    r = dev.coherent_rx(E, lo, cfg.fs, shot=False, rng=rng)
    res = rx_dsp(r, cfg, ref)
    ber.append(res['ber'])
    print(f'OSNR {o:.0f} dB: BER {res["ber"]:.2e}, SNR {res["snr"]:.2f} dB')

snr_th = metrics.osnr2snr(osnr, cfg.rs)
os.makedirs('results', exist_ok=True)
plt.semilogy(osnr, ber, 'o-', label='simulation')
plt.semilogy(osnr, metrics.ber_theory(cfg.M, snr_th), 'k--', label='theory (AWGN)')
plt.axhline(metrics.HD_FEC, color='r', ls=':', label='HD-FEC')
plt.xlabel('OSNR (dB/0.1nm)'); plt.ylabel('BER'); plt.grid(True, which='both'); plt.legend()
plt.savefig('results/ber_vs_osnr.png', dpi=120)
