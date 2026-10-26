"""光纤传感综合示例（6 种常用技术，各自给出解调误差）：
(a) OTDR：衰减与损耗事件识别          (b) φ-OTDR：矩形脉冲 vs LFM 脉冲压缩的空间分辨率
(c) BOTDA：布里渊增益谱 -> BFS -> 温度/应变   (d) 拉曼 DTS：反斯托克斯/斯托克斯比 -> 温度
(e) FBG：反射谱峰值解调应变                (f) OFDR：局部瑞利谱互相关 -> 毫米级应变分布"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import sensing as sen
from optsim.utils import F0

rng = np.random.default_rng(0)
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
ax = ax.ravel()

# ---- (a) OTDR：20 km，100 ns 脉冲，5 km 处 0.5 dB、12 km 处 1.0 dB 损耗 ----
z = np.arange(0, 20e3, 1.0)
p = sen.otdr_trace(z, 20, 100e-9, 0.2, [(5e3, 0.5), (12e3, 1.0)])
p_db = 10 * np.log10(np.abs(p + 2e-3 * p[0] * 10 ** -0.8 * rng.standard_normal(len(z))))   # 平均后的残余噪声
a_est, idx, loss = sen.otdr_events(p_db, 1.0, 200, 0.1)
print(f'(a) OTDR: attenuation {a_est:.3f} dB/km (true 0.200); events ' +
      ', '.join(f'{z[i] / 1e3:.2f} km/{l:.2f} dB' for i, l in zip(idx, loss)) + ' (true 5.00/0.50, 12.00/1.00)')
ax[0].plot(z / 1e3, p_db - p_db[0]); [ax[0].axvline(z[i] / 1e3, color='r', ls=':') for i in idx]
ax[0].set(xlabel='Distance (km)', ylabel='Backscatter (dB, round trip)', title='(a) OTDR events')

# ---- (b) φ-OTDR 空间分辨率：矩形 100 ns vs LFM 1 µs / 100 MHz（同峰值功率）----
fs = 250e6
dz = sen.cell_len(fs)
zb, r = sen.rayleigh_fiber(300, dz, 0.2, rng)
i1, i2 = int(150 / dz), int(153 / dz)
rect, chirp = np.ones(int(100e-9 * fs)), sen.lfm(1e-6, 100e6, fs)
width = lambda y: np.sum(np.abs(y) ** 2 >= 0.5 * np.max(np.abs(y) ** 2)) * dz   # 单反射点响应 -3 dB 宽度
one = np.zeros(len(zb), complex); one[i1] = 1
w_r = width(sen.backscatter(rect, one, len(zb)))
w_c = width(sen.pulse_compress(sen.backscatter(chirp, one, len(zb)), chirp))
h = r.copy(); h[[i1, i2]] += 20 * np.std(r)                       # 两个相距 3 m 的强反射点
yr = sen.backscatter(rect, h, len(zb))
yc = sen.pulse_compress(sen.backscatter(chirp, h, len(zb)), chirp)
print(f'(b) phi-OTDR resolution: rect {w_r:.1f} m, LFM compressed {w_c:.2f} m (theory c/2nB = {sen.cell_len(100e6):.2f} m)')
m = (zb > 140) & (zb < 165)
ax[1].plot(zb[m], 20 * np.log10(np.abs(yr[m]) / np.abs(yr[m]).max()), label='rect 100 ns')
ax[1].plot(zb[m], 20 * np.log10(np.abs(yc[m]) / np.abs(yc[m]).max()), label='LFM 1 µs, 100 MHz')
ax[1].set(xlabel='Distance (m)', ylabel='|trace| (dB)', title='(b) Pulse compression', ylim=(-40, 2)); ax[1].legend()

# ---- (c) BOTDA：2 km，20 ns 泵浦（2 m），600 m 处 +30 K、1200 m 处 300 με ----
zc = np.arange(0, 2000, 0.5)
dT = np.where((zc >= 600) & (zc < 620), 30.0, 0.0)
eps = np.where((zc >= 1200) & (zc < 1230), 300.0, 0.0)
f_scan = np.arange(10.75e9, 11.0e9, 2e6)
g = sen.botda(sen.bfs(dT, eps), f_scan, int(2 / 0.5))
vb = sen.bfs_fit(f_scan, g + 0.02 * rng.standard_normal(g.shape))
hot, st, bg = (zc >= 605) & (zc < 615), (zc >= 1205) & (zc < 1225), (dT == 0) & (eps == 0)
print(f'(c) BOTDA: BFS rms error {np.std(vb[bg] - 10.85e9) / 1e6:.2f} MHz; hot spot {np.mean(vb[hot] - 10.85e9) / 1e6:.1f} MHz -> '
      f'{np.mean(vb[hot] - 10.85e9) / 1e6:.1f} K (true 30); strain {np.mean(vb[st] - 10.85e9) / 0.05e6:.0f} µε (true 300)')
ax[2].imshow(g, aspect='auto', origin='lower', extent=[zc[0], zc[-1], f_scan[0] / 1e9, f_scan[-1] / 1e9], cmap='viridis')
ax[2].plot(zc, vb / 1e9, 'w', lw=0.6)
ax[2].set(xlabel='Distance (m)', ylabel='Pump-probe offset (GHz)', title='(c) BOTDA gain spectrum + BFS')

# ---- (d) 拉曼 DTS：4 km，基线 20 °C，2000~2050 m 处 70 °C，前 100 m 为已知温度的参考段 ----
zd = np.arange(0, 4000, 1.0)
T = 293.15 + np.where((zd >= 2000) & (zd < 2050), 50.0, 0.0)
dalpha = 0.05 * np.log(10) / 10 / 1e3 * 2                  # 两波长往返损耗差 0.05 dB/km
R = uniform_filter1d(sen.raman_ratio(T), 3) * np.exp(-dalpha * zd) * (1 + 5e-3 * rng.standard_normal(len(zd)))
ref = zd < 100
T_est = sen.dts_temp(R, np.mean(R[ref] * np.exp(dalpha * zd[ref])), 293.15, zd, dalpha)
inr = (zd >= 2005) & (zd < 2045)
print(f'(d) Raman DTS: background rms error {np.std(T_est[zd > 2100] - 293.15):.2f} K; hot section {np.mean(T_est[inr]) - 273.15:.1f} °C (true 70.0)')
ax[3].plot(zd, T_est - 273.15, lw=0.6, label='DTS'); ax[3].plot(zd, T - 273.15, 'k--', label='true')
ax[3].set(xlabel='Distance (m)', ylabel='Temperature (°C)', title='(d) Raman DTS'); ax[3].legend()

# ---- (e) FBG：λB = 1550 nm，10 mm，Δn = 1e-4；应变 0 / 500 / 1000 με ----
lam = np.linspace(1549, 1552, 3001)
for e in (0, 500, 1000):
    Rf = sen.fbg_reflect(lam, 1550 + sen.fbg_shift(1550, eps=e)) * 10 ** (rng.standard_normal(len(lam)) * 0.002)
    d_pk = sen.fbg_peak(lam, Rf) - 1550
    print(f'(e) FBG: strain {e:4d} µε -> peak shift {d_pk * 1e3:7.1f} pm -> {d_pk / sen.fbg_shift(1550, eps=1):.0f} µε')
    ax[4].plot(lam, Rf, label=f'{e} µε')
ax[4].set(xlabel='Wavelength (nm)', ylabel='Reflectivity', title='(e) FBG spectrum'); ax[4].legend()

# ---- (f) OFDR：1 THz 扫频（分辨率 0.1 mm），3.3 m 光纤，1.0~1.2 m 处 200 με、2.0~2.05 m 处 100 με ----
B, nk = 1e12, 2 ** 15
dzf = sen.cell_len(B)
zf, rf = sen.rayleigh_fiber(nk * dzf, dzf, 0.2, rng)
ef = np.where((zf >= 1.0) & (zf < 1.2), 200.0, 0.0) + np.where((zf >= 2.0) & (zf < 2.05), 100.0, 0.0)
noise = lambda s: s + 0.03 * np.std(s) * (rng.standard_normal(nk) + 1j * rng.standard_normal(nk))
iw, dnu = sen.ofdr_shift(noise(sen.ofdr_scan(rf, dzf)), noise(sen.ofdr_scan(rf, dzf, ef)), 256, B)
e_est = -dnu / (0.78 * F0) * 1e6
ins = (zf[iw] > 1.03) & (zf[iw] < 1.17)
print(f'(f) OFDR: window {256 * dzf * 1e3:.1f} mm; strain in 1.0-1.2 m {np.mean(e_est[ins]):.0f} µε (true 200), '
      f'unstrained rms {np.std(e_est[uniform_filter1d(ef, 256)[iw] == 0]):.1f} µε (windows fully outside strain)')
ax[5].plot(zf[iw], e_est, '.-', ms=3, label='OFDR'); ax[5].plot(zf, ef, 'k--', label='true')
ax[5].set(xlabel='Distance (m)', ylabel='Strain (µε)', title='(f) OFDR strain'); ax[5].legend()

os.makedirs('results', exist_ok=True)
plt.tight_layout(); plt.savefig('results/fiber_sensing.png', dpi=120)
