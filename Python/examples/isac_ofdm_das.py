"""通感一体示例 1：CP-OFDM 数据信号同时作 DAS 探测光（后向，可定位）。

发射机：16QAM-OFDM（nfft 16384，CP 5000 ≥ 1 km 光纤往返时延 4900 采样）-> 窄线宽激光 -> 环形器 -> 1 km 光纤
  通信：远端相干接收 -> 去 CP/FFT -> LS 信道估计 + 导频 CPE -> 16QAM 判决
  传感：近端用同一激光作本振接收瑞利背向散射 -> 每个 OFDM 符号逐子载波除以已知发送符号 -> IFFT
        -> 散射冲激响应 h(z)（一个 OFDM 符号 = 一次 "脉冲"）-> 差分相位解调（同 φ-OTDR）
仿真为缩小规模（500 MS/s），原理与高速系统相同：距离分辨率 ≈ c/(2·ng·B_OFDM)，慢时间采样率 = 1/OFDM 符号时长。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import tx, dsp, metrics, devices as dev, sensing as sen, isac
from optsim.utils import dbm2w, awgn

rng = np.random.default_rng(0)
# ---- 参数 ----
fs, nfft, ncp, K = 500e6, 16384, 5000, 300       # 采样率 = OFDM 带宽；K 个 OFDM 符号
dz = sen.cell_len(fs)                             # 距离单元 0.20 m
nz = int(1000 / dz)                               # 1 km 光纤（散射记忆 4900 采样 ≤ CP）
f_vib, A_vib, z1, z2 = 800.0, 1.0, 600.0, 605.0  # 振动：800 Hz，单程 1 rad，600~605 m
G, n_avg = int(10 / dz), 5                        # 标距 10 m，距离平均 1 m
Ts = (nfft + ncp) / fs                            # OFDM 符号时长 = 慢时间采样间隔

# ---- 发射：16QAM-OFDM，DC 与两侧 10% 子载波置空，每 64 个子载波 1 个导频 ----
c = tx.qam(16)
used = np.abs(np.fft.fftfreq(nfft)) < 0.45
used[0] = False
X = np.where(used, c[rng.integers(0, 16, (K, nfft))], 0)
pil = np.flatnonzero(used)[::64]
x = tx.ofdm_mod(X, ncp) / np.sqrt(np.mean(used))  # 单位平均功率
n_s = nfft + ncp
lo = dev.laser(K * n_s, fs, 0, 100, rng=rng) / np.sqrt(1e-3)   # 100 Hz 超窄线宽（DAS 常用），发射与本振同源（单位功率）
Etx = np.sqrt(dbm2w(10)) * x * lo                 # 入纤 10 dBm

# ---- 光纤：瑞利散射 + 振动（往返相位 2φ）。每个 OFDM 符号内扰动视为不变 ----
z, r = sen.rayleigh_fiber(nz * dz, dz, 0.2, rng)
t_slow = np.arange(K) * Ts
vib = A_vib * np.sin(2 * np.pi * f_vib * t_slow)
h = r * np.exp(2j * vib[:, None] * np.clip((z - z1) / (z2 - z1), 0, 1))
Ebs = sen.backscatter(Etx.reshape(K, n_s), h, n_s).ravel()   # 各符号自身 CP 吸收散射记忆

# ---- 传感接收：同源本振 0 dBm，90° 混频 + BPD ----
o = dev.hybrid_90(Ebs, np.sqrt(dbm2w(0)) * lo)
kw = dict(R=0.8, T=300, RL=50, rng=rng)
i_bs = dev.bpd(o[0], o[1], fs, **kw) + 1j * dev.bpd(o[2], o[3], fs, **kw)
prof = isac.ofdm_profile(i_bs, X, nfft, ncp)[:, :nz]          # (K, nz) 散射冲激响应
dphi = sen.das_phase(prof, G, n_avg)
zc = z[:-G] + G * dz / 2
iz, e = sen.vib_locate(dphi, 25)
true = 2 * vib
corr = abs(np.corrcoef(dphi[:, iz], true)[0, 1])
print(f'Sensing: located {zc[iz]:.1f} m (true {z1:.0f}-{z2:.0f} m), waveform corr {corr:.3f}, '
      f'range resolution ~{sen.cell_len(0.9 * fs):.2f} m, slow-time rate {1 / Ts / 1e3:.1f} kHz')

# ---- 通信接收：远端，前向单程振动相位 + 独立本振（100 Hz）+ AWGN 20 dB ----
fwd = np.repeat(np.exp(1j * vib), n_s)            # 前向扰动相位（单程，不可定位）
Erx = Etx * fwd * dev.laser(K * n_s, fs, 0, 100, rng=rng).conj() / np.sqrt(1e-3)
y = awgn(Erx / np.sqrt(dbm2w(10)), 20, rng=rng)
nt = 8                                            # 训练符号数（LS 信道估计）
Z, _ = dsp.ofdm_eq(dsp.ofdm_demod(y, nfft, ncp), X, nt, pil)
dat = used.copy(); dat[pil] = False
rb = tx.demodulate(Z[nt:, dat].ravel(), c)
tb = tx.demodulate(X[nt:, dat].ravel(), c)
rate = dat.sum() * 4 / Ts
print(f'Comm: BER {metrics.ber(tb, rb)[0]:.2e}, SNR {metrics.snr(Z[nt:, dat], X[nt:, dat]):.2f} dB (AWGN 20 dB), '
      f'net rate {rate / 1e9:.2f} Gb/s, CP overhead {ncp / n_s * 100:.0f} %')

os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(19, 4))
ax[0].imshow(dphi.T, aspect='auto', origin='lower', extent=[0, t_slow[-1] * 1e3, zc[0], zc[-1]], cmap='RdBu', vmin=-2.5, vmax=2.5)
ax[0].set(xlabel='Time (ms)', ylabel='Distance (m)', title='ISAC: differential phase')
ax[1].plot(zc, e); ax[1].set(xlabel='Distance (m)', ylabel='RMS (rad)', title='Vibration energy')
ax[2].plot(t_slow * 1e3, dphi[:, iz], label='recovered'); ax[2].plot(t_slow * 1e3, true, '--', label='true')
ax[2].set(xlabel='Time (ms)', ylabel='Phase (rad)', title='Waveform'); ax[2].legend()
zz = Z[nt:, dat].ravel()[:20000]
ax[3].plot(zz.real, zz.imag, '.', ms=1); ax[3].set(aspect='equal', title='Comm: 16QAM-OFDM')
plt.tight_layout(); plt.savefig('results/isac_ofdm_das.png', dpi=120)
