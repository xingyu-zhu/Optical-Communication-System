"""IM-DD PAM-M 系统仿真（system.ImddConfig / imdd_tx / imdd_link / imdd_rx）：
(a) 112 Gb/s 下 PAM2 / PAM4 / PAM8 的 BER–ROP 与 KP4 灵敏度，O 波段 2 km，MZM + PIN；另加 PAM4 + APD
    （DAC/ADC 取 ENOB 7：ENOB 6 时量化+削波使 PAM8 有效 SNR 封顶约 21 dB，低于 KP4 所需约 23.7 dB）
(b) C 波段 10 km 色散功率衰落：MZM（α=0）、EML（α=±1）、DML（瞬态+绝热啁啾）的小信号响应 vs 理论
(c) C 波段 PAM4 56 GBd 色散受限传输距离：FFE / DFE / FFE+后滤波+MLSE / Volterra 的 BER–光纤长度
    （第一个功率衰落陷波 f1 = sqrt(c/(2λ²DL)) 进入信号带宽后，线性均衡无法恢复 → BER 陡升）
(d) O 波段 PAM4 接收眼图
运行：python examples/imdd_system.py（约 2~3 分钟）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, channel, metrics, plots
from optsim.system import ImddConfig, imdd_tx, imdd_link, imdd_rx
from optsim.utils import freq, lpf

os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(12, 9))


def ber_curve(cfg, rops):
    """一次发射与传输，扫描 ROP 返回 BER 数组。"""
    ref = imdd_tx(cfg)
    E = imdd_link(ref['E'], cfg)
    return np.array([imdd_rx(E, cfg, ref, p, rng=k)['ber'] for k, p in enumerate(rops)])


# ---------------- (a) 112 Gb/s：PAM2 / PAM4 / PAM8 ----------------
rop = np.arange(-18, -1, 1.0)
q = dict(dac_enob=7, adc_enob=7)
cases = {'PAM2 112 GBd': ImddConfig(M=2, rs=112e9, n_sym=2 ** 16, **q),
         'PAM4 56 GBd': ImddConfig(M=4, rs=56e9, n_sym=2 ** 15, **q),
         'PAM8 37.3 GBd': ImddConfig(M=8, rs=112e9 / 3, n_sym=21846, **q),
         'PAM4 56 GBd + APD': ImddConfig(M=4, rs=56e9, n_sym=2 ** 15, rx='apd', **q)}
print('(a) O-band 2 km, 112 Gb/s, KP4 sensitivity:')
for name, cfg in cases.items():
    b = ber_curve(cfg, rop)
    ax[0, 0].semilogy(rop, np.maximum(b, 1e-6), 'o-', ms=4, label=name)
    print(f'    {name:18s}: {metrics.sensitivity(rop, b):6.2f} dBm')
ax[0, 0].axhline(metrics.KP4_FEC, color='k', ls=':', label='KP4 FEC')
ax[0, 0].set(xlabel='ROP (dBm)', ylabel='BER', title='(a) 112 Gb/s, O-band 2 km, FFE', ylim=(1e-6, 0.5))
ax[0, 0].grid(True, which='both', alpha=0.3); ax[0, 0].legend(fontsize=8)

# ---------------- (b) C 波段 10 km 色散功率衰落 ----------------
n, fs, L = 2 ** 16, 400e9, 10.0
rng = np.random.default_rng(0)
u = lpf(rng.standard_normal(n), fs, 60e9).real
u = 0.1 * u / u.std()                                         # 小信号驱动
cw = np.sqrt(1e-3) * np.ones(n)
f = freq(n, fs)
fp = np.linspace(0.5e9, 40e9, 400)
txs = {'MZM, α=0': (dev.mzm(cw, 0.3 * u, 1, -0.5), 0.0),
       'EML, α=+1': (dev.eml(cw, 0.5 + u, 8, 1.0), 1.0),
       'EML, α=-1': (dev.eml(cw, 0.5 + u, 8, -1.0), -1.0),
       'DML, α=3, κ=1 GHz/mW': (dev.dml(u, fs, 5, 4, 3.0, 1e12, linewidth=0, rin_db=None), None)}
for (name, (E0, a)), col in zip(txs.items(), plt.rcParams['axes.prop_cycle'].by_key()['color']):
    P0, P1 = np.abs(E0) ** 2, np.abs(channel.cd(E0, fs, L, 17, 1550)) ** 2
    Hs = np.abs(np.fft.fft(P1 - P1.mean()) / np.fft.fft(P0 - P0.mean()))
    k = (f > 0) & (f < 40e9)
    sm = np.convolve(Hs[k], np.ones(15) / 15, 'same')
    ax[0, 1].plot(f[k] / 1e9, 20 * np.log10(sm), color=col, alpha=0.8, label=name + ' (sim)')
    if a is not None:
        ax[0, 1].plot(fp / 1e9, 20 * np.log10(channel.power_fading(fp, L, 17, 1550, a) + 1e-6), '--', color=col, lw=1)
ax[0, 1].set(xlabel='Frequency (GHz)', ylabel='|H(f)| (dB)', title='(b) C-band 10 km power fading (dashed: theory)',
             ylim=(-30, 10))
ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=8)
f1 = np.sqrt(3e8 / (2 * (1550e-9) ** 2 * 17e-6 * L * 1e3))
print(f'(b) first fading notch (α=0) at {f1 / 1e9:.1f} GHz')

# ---------------- (c) C 波段 PAM4：色散受限距离与均衡器对比 ----------------
Ls = np.arange(0, 6.01, 0.5)
base = ImddConfig(M=4, rs=56e9, band='C', n_sym=2 ** 14, ntaps=31)
ROP_C = -4.0
res = {k: [] for k in ('ffe', 'dfe', 'mlse', 'volterra')}
for Lk in Ls:
    c = replace(base, L=Lk)
    ref = imdd_tx(c)
    E = imdd_link(ref['E'], c)
    for eq in res:
        res[eq].append(imdd_rx(E, replace(c, eq=eq), ref, ROP_C, rng=1)['ber'])
print(f'(c) C-band PAM4 56 GBd, ROP {ROP_C:.0f} dBm: max reach at KP4 [km]')
for eq, lab in (('ffe', 'FFE (LMS)'), ('dfe', 'FFE+DFE'), ('mlse', 'FFE+PF+MLSE'), ('volterra', 'Volterra')):
    b = np.array(res[eq])
    ok = Ls[b <= metrics.KP4_FEC]
    ax[1, 0].semilogy(Ls, np.maximum(b, 1e-6), 'o-', ms=4, label=lab)
    print(f'    {lab:12s}: {ok.max() if len(ok) else 0:.1f} km')
ax[1, 0].axhline(metrics.KP4_FEC, color='k', ls=':', label='KP4 FEC')
ax[1, 0].set(xlabel='Fiber length (km)', ylabel='BER', title=f'(c) PAM4 56 GBd, C-band SSMF, ROP {ROP_C:.0f} dBm',
             ylim=(1e-6, 0.5))
ax[1, 0].grid(True, which='both', alpha=0.3); ax[1, 0].legend(fontsize=8)

# ---------------- (d) 眼图 ----------------
cfg = replace(cases['PAM4 56 GBd'], n_sym=4096)
ref = imdd_tx(cfg)
i = dev.photodiode(dev.voa(imdd_link(ref['E'], cfg), 10), cfg.fs, 0.8, bw=0.7 * cfg.rs, irnd=15e-12, rng=0)
plots.eye(np.roll(i, cfg.sps // 2) * 1e3, cfg.sps, ax[1, 1], n=600)       # 移半个符号使眼图居中
ax[1, 1].set(ylabel='Photocurrent (mA)', title='(d) PAM4 eye, O-band 2 km (before EQ)')
plt.tight_layout(); plt.savefig('results/imdd_system.png', dpi=130)
