"""光纤传感示例：相位敏感 OTDR (φ-OTDR) 分布式声波传感 (DAS)，相干 IQ 解调。
窄线宽激光 -> AOM 脉冲 -> 光纤（随机瑞利散射点 + 局部振动调制相位）-> 背向散射与本振在 90° 混频器拍频 -> BPD
DSP：IQ -> 相位 -> 标距 G 的差分相位 -> 慢时间解卷绕 -> 振动定位与波形恢复。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, sensing as sen
from optsim.utils import dbm2w

rng = np.random.default_rng(0)
# ---- 参数 ----
fs = 250e6
dz = sen.cell_len(fs)                  # 每采样对应光纤长度 (~0.41 m)
W, prr, K = 100e-9, 10e3, 1000         # 脉宽 100 ns (~10 m 分辨率)、重复频率、脉冲数
f_vib, A_vib, z1, z2 = 500.0, 1.5, 1200.0, 1210.0   # 振动：500 Hz、1.5 rad（单程）、位置 1200~1210 m
G = int(20 / dz)                       # 差分相位标距 20 m

# ---- 光纤与扰动：振动区内相位线性累积，其后保持（往返相位 = 2×单程）----
z, r = sen.rayleigh_fiber(2000, dz, 0.2, rng)
nz = len(z)
t_slow = np.arange(K) / prr
prof = np.clip((z - z1) / (z2 - z1), 0, 1)
h = r * np.exp(2j * A_vib * np.sin(2 * np.pi * f_vib * t_slow)[:, None] * prof)   # (K, nz)
pulse = np.ones(int(W * fs))

# ---- 前向：背向散射 + 激光相噪（相对本振）+ 相干接收 ----
Es = np.sqrt(dbm2w(20)) * sen.backscatter(pulse, h, nz)
Es *= np.stack([dev.laser(nz, fs, 0, 1e3, rng=rng) for _ in range(K)]) / np.sqrt(1e-3)   # 1 kHz 线宽
kw = dict(R=0.8, T=300, RL=50, rng=rng)
rx = lambda o: dev.bpd(o[0], o[1], fs, **kw) + 1j * dev.bpd(o[2], o[3], fs, **kw)
traces = np.stack([rx(dev.hybrid_90(e, np.sqrt(dbm2w(0)) * np.ones(nz))) for e in Es])

# ---- DSP：差分相位解调 + 定位 ----
dphi = sen.das_phase(traces, G, len(pulse))
zc = z[:-G] + G * dz / 2
iz, rms = sen.vib_locate(dphi, 2 * len(pulse))   # 中值滤波抑制相干衰落点的伪峰
true = 2 * A_vib * np.sin(2 * np.pi * f_vib * t_slow)
print(f'Vibration located at {zc[iz]:.1f} m (true {z1:.0f}-{z2:.0f} m), waveform corr {abs(np.corrcoef(dphi[:, iz], true)[0, 1]):.3f}')

os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(dphi.T, aspect='auto', origin='lower', extent=[0, t_slow[-1] * 1e3, zc[0], zc[-1]], cmap='RdBu', vmin=-2.5 * A_vib, vmax=2.5 * A_vib)
ax[0].set_xlabel('Time (ms)'); ax[0].set_ylabel('Distance (m)'); ax[0].set_title('Differential phase')
ax[1].plot(zc, rms); ax[1].set_xlabel('Distance (m)'); ax[1].set_ylabel('RMS (rad)')
ax[2].plot(t_slow * 1e3, dphi[:, iz], label='recovered'); ax[2].plot(t_slow * 1e3, true, '--', label='true')
ax[2].set_xlabel('Time (ms)'); ax[2].set_ylabel('Phase (rad)'); ax[2].legend()
plt.tight_layout(); plt.savefig('results/phi_otdr_das.png', dpi=120)
