"""IM-DD PAM4 短距仿真：DAC -> MZM(正交偏置) -> SSMF(CD 功率衰落) -> VOA -> PIN(热/散粒噪声) -> ADC -> LMS-FFE。
扫描接收光功率 (ROP) 得 BER 曲线。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import tx, devices as dev, channel, dsp, metrics
from optsim.utils import w2dbm, total_power

rs, sps, M, L = 56e9, 4, 4, 2.0           # 56 GBd PAM4, 2 km SSMF @1550 nm
fs = rs * sps
rng = np.random.default_rng(0)
lv = tx.pam(M) / 3                         # Gray PAM4 电平，归一化到 [-1, 1]
bits = tx.rand_bits(2 * 2 ** 15, rng)
sym = lv[tx.bits2int(bits, 2)]
u = dev.dac(tx.pulse_shape(sym, sps, 0.2, 'rc').real, fs, enob=6, bw=0.6 * rs)
E = dev.laser(len(u), fs, p_dbm=10, rin_db=-150, rng=rng)
E = dev.mzm(E, 0.4 * u, vpi=1.0, vbias=-0.5)                 # 正交点偏置，±0.4Vπ 摆幅
E = channel.cd(E, fs, L) * 10 ** (-0.2 * L / 20)

n_train = 4096
rop = np.arange(-14, -3, 1.0)
ber = []
for p in rop:
    Er = dev.voa(E, w2dbm(total_power(E)) - p)                 # VOA 调到目标 ROP
    i = dev.photodiode(Er, fs, R=0.8, bw=0.7 * rs, T=300, RL=50, rng=rng)
    y = dev.adc(i, fs, 2 * rs, enob=6)
    y = (y - y.mean()) / y.std() * lv.std()
    ye, _ = dsp.ffe(y, sym, ntaps=21, mu=1e-3, sps=2, n_train=n_train, levels=lv)
    rb = tx.int2bits(tx.decide(ye[n_train:], lv), 2)
    ber.append(metrics.ber(bits[2 * n_train:], rb)[0])
    print(f'ROP {p:+.0f} dBm: BER {ber[-1]:.2e}')

os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogy(rop, np.maximum(ber, 1e-6), 'o-'); ax[0].axhline(metrics.HD_FEC, color='r', ls=':')
ax[0].set_xlabel('ROP (dBm)'); ax[0].set_ylabel('BER'); ax[0].grid(True, which='both')
ax[1].hist(ye[n_train:], 200); ax[1].set_title('FFE output histogram')
plt.tight_layout(); plt.savefig('results/imdd_pam4.png', dpi=120)
