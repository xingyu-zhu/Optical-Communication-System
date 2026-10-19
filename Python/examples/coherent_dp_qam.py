"""单信道相干 DP-16QAM 全流程仿真（含全部收发机器件损伤）：
Tx DSP -> DAC -> 激光器 -> DP-IQ 调制器 -> 功放 -> N×(SSMF + 放大器) -> PDL -> WSS
-> 本振 + 相干接收机(90°混频器 + BPD) -> ADC -> Rx DSP -> BER/SNR/EVM/GMI

注意：这是单信道（只有 SPM 等信道内非线性），实际 WDM 系统还有 XPM/FWM，SNR 会更低，见 wdm_system.py。
运行：python examples/coherent_dp_qam.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, channel, dsp, plots
from optsim.system import Config, tx_signal, rx_dsp

# 波段/光纤参数自动取自 params/fiber_bands.json；fib_kw 覆盖个别字段
cfg = Config(M=16, rs=40e9, band='C', fiber='SSMF', n_spans=10, L=80, p_launch=1.0,
             fib_kw={'pmd': 0.1}, nl_comp='cdc', cpr='bps')
fib, fs = cfg.fib, cfg.fs
rng = np.random.default_rng(cfg.seed)

# ---------------- 发射端 ----------------
ref = tx_signal(cfg)
u = 0.2 * dev.dac(ref['wave'], fs, enob=6, bw=0.7 * cfg.rs)               # DAC + 驱动带宽，摆幅 0.2Vπ
E = dev.dp_iq_mod(dev.laser(u.shape[1], fs, 16, 100e3, rng=rng), u, phase_err=np.deg2rad(1))  # 100 kHz 激光 + DP-IQ MZM
E = dev.edfa(E, fs, p_out_dbm=cfg.p_launch, nf_db=fib['nf'], f0=fib['f0'], rng=rng)          # 功放
E_tx = E.copy()

# ---------------- 光纤链路（SPM + 色散/斜率 + PMD + ASE；可在 fib_kw 打开 raman/steep）----------------
E = channel.link(E, fs, fib, cfg.n_spans, phi_max=cfg.phi_max, rng=rng)
E = dev.pdl(E, 0.5, theta=0.3)                                               # 0.5 dB PDL
E = dev.optical_filter(E, fs, bw=1.2 * cfg.rs * (1 + cfg.beta), order=3)     # WSS

# ---------------- 相干接收机 ----------------
lo = dev.laser(E.shape[1], fs, p_dbm=13, linewidth=100e3, f_offset=1.5e9, rng=rng)
skew = [0, 1.5e-12, 0, -1e-12]                                               # Rx IQ skew [s]
r = dev.coherent_rx(E, lo, fs, skew=skew, R=0.8, bw=0.75 * cfg.rs, T=300, rng=rng)
r = dev.adc(r, fs, cfg.fs_adc, enob=6, bw=0.75 * cfg.rs)                     # 160 -> 100 GSa/s

# ---------------- 接收端 DSP ----------------
r = dsp.deskew(r, cfg.fs_adc, skew)
res = rx_dsp(r, cfg, ref)
print(f"FO est {res['fo'] / 1e9:.3f} GHz | BER {res['ber']:.2e} (X {res['ber_xy'][0]:.2e}, "
      f"Y {res['ber_xy'][1]:.2e}) | SNR {res['snr']:.2f} dB | EVM {res['evm']:.1f}% | GMI {res['gmi']:.3f} b/sym")

# ---------------- 绘图 ----------------
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
plots.spectrum(E_tx, fs, ax[0], 'Tx'); plots.spectrum(E, fs, ax[0], 'Rx'); ax[0].legend()
plots.constellation(res['sym'][0], ax[1], 'X pol')
plots.constellation(res['sym'][1], ax[2], 'Y pol')
plt.tight_layout(); plt.savefig('results/coherent_dp_qam.png', dpi=120)
