"""通感一体示例 2：相干收发机前向传感（积分式，类 Zhan 2021 / Mecozzi 2021 / Mazur 2022）。

链路：DP-16QAM -> 光纤（某处扰动使双折射轴以 60 Hz 摆动 -> SOP 振荡；另一处振动产生 120 Hz 相位调制）
      -> 相干接收 -> 2×2 LMS（训练后判决引导，同时跟踪 SOP 与相位）-> 16QAM 判决（通信）
传感：用判决符号做数据辅助块 Jones 矩阵估计（每 1000 符号一次 = 10 kHz 采样）
      -> 输出 SOP（Stokes）与公共相位的时间序列 -> 频谱识别扰动频率、带通恢复振动波形。
缩小规模：10 MBd、50 ms。激光相噪按 Δν·Ts 起作用，故发/收合计线宽取 10 Hz（Δν/Rs = 1e-6，相当于 32 GBd 下 32 kHz；
相位传感本身也需要超稳激光，见 Marra 2018）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.signal import detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import tx, dsp, metrics, devices as dev, isac
from optsim.utils import awgn, freq

rng = np.random.default_rng(0)
rs, N, nb = 10e6, 500_000, 1000                   # 符号率、符号数（50 ms）、Jones 估计块长
t = np.arange(N) / rs
c = tx.qam(16)
bits = tx.rand_bits((2, 4 * N), rng)
s = tx.modulate(bits, c)

# ---- 信道：R2·R(θ(t))·R1 的 SOP 扰动 + 振动相位 + 激光相噪（发/收合计 10 Hz）+ AWGN ----
f_sop, f_ph = 60.0, 120.0                        # 取在频率网格（1/50 ms = 20 Hz）上
theta = 0.4 * np.sin(2 * np.pi * f_sop * t)
phi = 1.0 * np.sin(2 * np.pi * f_ph * t)
R1, R2 = dev.random_unitary(rng), dev.random_unitary(rng)
J = R2 @ np.moveaxis(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), -1, 0) @ R1   # (N, 2, 2)
lw = np.angle(dev.laser(N, rs, 0, 10, rng=rng))
y = np.einsum('kij,jk->ik', J, s) * np.exp(1j * (phi + lw))
y = awgn(y, 18, rng=rng)

# ---- 通信：LMS 训练 5000 符号后判决引导（ntaps=3，1 sps）----
ye, _, _ = dsp.mimo_eq(y, 3, 3e-3, 1, 'lms', c, s, 5000)
d = c[tx.decide(ye, c)]
d[:, :5000] = s[:, :5000]                         # 训练段用已知训练符号
ber = metrics.ber(bits[:, 4 * 10000:], tx.demodulate(ye[:, 10000:], c))[0]
print(f'Comm: BER {ber:.2e}, SNR {metrics.snr(ye[:, 10000:], s[:, 10000:]):.2f} dB (AWGN 18 dB)')

# ---- 传感：判决符号辅助的块 Jones 估计 -> SOP 与相位 ----
Jh = isac.jones_est(y, d, nb)
S = isac.jones_sop(Jh)                              # (nblk, 3)
ph = isac.jones_phase(Jh)
fsj = rs / nb
tb = (np.arange(len(ph)) + 0.5) / fsj
f = freq(len(ph), fsj)
psd = lambda v: np.abs(np.fft.fft(detrend(v, axis=0), axis=0)) ** 2
P_sop, P_ph = psd(S).sum(1), psd(ph)
band = (f > 5) & (f < 1000)
fs_hat, fp_hat = f[band][np.argmax(P_sop[band])], f[band][np.argmax(P_ph[band])]
bp = lambda v, f0: np.real(np.fft.ifft(np.fft.fft(v) * (np.abs(np.abs(f) - f0) < 20)))   # ±20 Hz 带通
ph_rec, ph_true = bp(ph, fp_hat), np.sin(2 * np.pi * f_ph * tb)
print(f'Sensing: SOP peak {fs_hat:.0f} Hz (true {f_sop:.0f}), phase peak {fp_hat:.0f} Hz (true {f_ph:.0f}); '
      f'phase waveform corr {np.corrcoef(ph_rec, ph_true)[0, 1]:.3f}, amplitude {np.std(ph_rec) * np.sqrt(2):.2f} rad (true 1.00)')

os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(19, 4))
ax[0].plot(tb * 1e3, S); ax[0].set(xlabel='Time (ms)', ylabel='Normalized Stokes', title='SOP from receiver DSP')
ax[0].legend(['S1', 'S2', 'S3'])
m = (f > 0) & (f < 1000)
ax[1].semilogy(f[m], P_sop[m] / P_sop[m].max(), label='SOP'); ax[1].semilogy(f[m], P_ph[m] / P_ph[m].max(), label='phase')
ax[1].set(xlabel='Frequency (Hz)', ylabel='Normalized PSD', title='Spectra'); ax[1].legend()
ax[2].plot(tb * 1e3, ph_rec, label='recovered (band-pass)'); ax[2].plot(tb * 1e3, ph_true, '--', label='true')
ax[2].set(xlabel='Time (ms)', ylabel='Phase (rad)', title=f'{f_ph:.0f} Hz vibration'); ax[2].legend()
ax[3].plot(ye[0, -20000:].real, ye[0, -20000:].imag, '.', ms=1); ax[3].set(aspect='equal', title='Comm: X-pol after LMS')
plt.tight_layout(); plt.savefig('results/isac_forward_sop.png', dpi=120)
