"""多波段 (O/E/S/C/L/U) × 多光纤类型的满载 WDM 性能估计（GN 模型闭式，快速）。
展示参数文件 params/fiber_bands.json 的用法：每个波段自动取对应的 α、D、S、γ、放大器 NF。

注意：GN 模型假设高色散（近零色散的 O 波段 SSMF / C 波段 DSF 下 FWM 相位匹配，GN 结果仅供参考），
且未计入波段间 SRS（ISRS-GN）与收发机噪声。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import channel
from optsim.params import load, fiber_params
from optsim.utils import C

rs, spacing, n_spans, L = 32e9, 50e9, 10, 80
bands = list(load()['bands'])
fibers = ['SSMF', 'ULL', 'NZDSF']
res = {f: [] for f in fibers}
print(f"{'fiber':6s} {'band':4s} {'Nch':>4s} {'D':>6s} {'α':>5s} {'γ':>5s} {'NF':>4s} {'P_opt(dBm)':>10s} {'SNR_opt(dB)':>11s}")
for fb in fibers:
    for bd in bands:
        fib = fiber_params(bd, fb, L=L)
        lo, hi = load()['bands'][bd]['range_nm']
        n_ch = int((C / (lo * 1e-9) - C / (hi * 1e-9)) // spacing)
        g = channel.gn_model(0, fib, rs, n_spans, n_ch, spacing, method='closed')
        res[fb].append(g['snr_opt'])
        print(f"{fb:6s} {bd:4s} {n_ch:4d} {fib['D']:6.2f} {fib['alpha']:5.2f} {fib['gamma']:5.2f} {fib['nf']:4.1f} "
              f"{g['p_opt_dbm']:10.2f} {g['snr_opt']:11.2f}")

# C 波段 SSMF：信道数对 SNR-功率曲线的影响
fib = fiber_params('C', 'SSMF', L=L)
p = np.linspace(-6, 8, 200)
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for n in (1, 7, 21, 87):
    ax[0].plot(p, channel.gn_model(p, fib, rs, n_spans, n, spacing, 'closed')['snr'], label=f'{n} ch')
ax[0].set_xlabel('Launch power per channel (dBm)'); ax[0].set_ylabel('SNR (dB)')
ax[0].set_title('C-band SSMF, 10×80 km (GN)'); ax[0].grid(); ax[0].legend(); ax[0].set_ylim(5, 30)
xb = np.arange(len(bands))
for i, fb in enumerate(fibers):
    ax[1].bar(xb + (i - 1) * 0.27, res[fb], 0.27, label=fb)
ax[1].set_xticks(xb, bands); ax[1].set_ylabel('Optimum SNR (dB)'); ax[1].set_title('Fully-loaded band, 10×80 km (GN)')
ax[1].legend(); ax[1].grid(axis='y')
plt.tight_layout(); plt.savefig('results/band_gn_analysis.png', dpi=120)
