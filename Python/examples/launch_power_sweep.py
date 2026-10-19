"""单信道入纤功率扫描：CDC vs DBP，并与 GN 模型预测对比（Kerr 非线性与最优入纤功率）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, channel
from optsim.system import Config, tx_signal, rx_dsp

cfg = Config(M=16, n_blocks=300, n_spans=10, fs_adc=160e9, fib_kw={'pmd': 0})
fib = cfg.fib
ref = tx_signal(cfg)
N = ref['wave'].shape[1]
powers = np.arange(-2, 7, 2.0)
snr = {'cdc': [], 'dbp': []}
for p in powers:
    rng = np.random.default_rng(1)
    E = dev.edfa(ref['wave'] * dev.laser(N, cfg.fs, 0, rng=rng), cfg.fs, p_out_dbm=p, nf_db=fib['nf'], f0=fib['f0'], rng=rng)
    E = channel.link(E, cfg.fs, fib, cfg.n_spans, phi_max=cfg.phi_max, rng=rng)
    r = dev.coherent_rx(E, dev.laser(N, cfg.fs, 10), cfg.fs, shot=False)
    for m in snr:
        snr[m].append(rx_dsp(r, replace(cfg, p_launch=p, nl_comp=m, dbp_steps=10), ref)['snr'])
    print(f"P {p:+.0f} dBm: SNR CDC {snr['cdc'][-1]:.2f} dB, DBP {snr['dbp'][-1]:.2f} dB")

pg = np.linspace(-4, 8, 100)
gn = channel.gn_model(pg, fib, cfg.rs, cfg.n_spans)
print(f"GN model: P_opt {gn['p_opt_dbm']:.2f} dBm, SNR_opt {gn['snr_opt']:.2f} dB")

os.makedirs('results', exist_ok=True)
for m in snr:
    plt.plot(powers, snr[m], 'o-', label=f'SSFM + {m.upper()}')
plt.plot(pg, gn['snr'], 'k--', label='GN model (1 ch)')
plt.plot(pg, gn['snr_ase'], 'k:', label='ASE only')
plt.ylim(10, 30)
plt.xlabel('Launch power (dBm)'); plt.ylabel('SNR (dB)'); plt.grid(); plt.legend()
plt.savefig('results/launch_power_sweep.png', dpi=120)
