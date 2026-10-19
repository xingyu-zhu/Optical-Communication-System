"""自检：DSP 算法恢复能力 + 光纤物理模型（SPM/孤子/SRS/FWM/GN）正确性。运行：python examples/selfcheck.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from optsim import tx, dsp, metrics, channel, devices
from optsim.params import fiber_params
from optsim.utils import awgn, delay, normalize, freq

rng = np.random.default_rng(1)
M, rs, sps = 16, 32e9, 2
fs = rs * sps
c = tx.qam(M)
bits = tx.rand_bits((2, 4 * 2 ** 14), rng)
s = tx.modulate(bits, c)
x = normalize(tx.pulse_shape(s, sps, 0.1))
mf = lambda v: tx.matched_filter(v, sps, 0.1)


def report(name, ok, info):
    print(f"[{'OK ' if ok else 'BAD'}] {name:28s} {info}")


print('---- DSP ----')
# 1) 映射/解映射 + AWGN 理论 BER
r = awgn(s, 14, rng=rng)
b, _ = metrics.ber(bits, tx.demodulate(r, c))
bt = metrics.ber_theory(M, 14)
report('QAM map / AWGN BER', abs(b / bt - 1) < 0.15, f'sim {b:.2e}  theory {bt:.2e}  GMI {metrics.gmi(r, s, c):.2f}')

# 2) CD(含斜率) + CDC + 盲 CD 估计
y = channel.cd(x, fs, 500, S=0.057)
Lh, _ = dsp.cd_estimate(y, fs, rs, np.arange(300, 701, 20))
z = mf(dsp.cdc(y, fs, 500, S=0.057))[:, ::sps]
report('CD / CDC / CD estimate', metrics.snr(z, s) > 25 and Lh == 500, f'SNR {metrics.snr(z, s):.1f} dB, L_est {Lh} km')

# 3) FOE
xf = dsp.fo_comp(s, -1.2e9, rs)
fa, fd = dsp.foe(xf, rs), dsp.foe(xf, rs, 'diff')
report('FOE fft / diff', abs(fa - 1.2e9) < 5e6 and abs(fd - 1.2e9) < 5e7, f'{fa / 1e9:.4f} / {fd / 1e9:.4f} GHz (true 1.2)')

# 4) CPR：合计 200 kHz 线宽
r = awgn(s * devices.laser(s.shape[1], rs, linewidth=200e3, rng=rng) / np.sqrt(1e-3), 18, rng=rng)
sb = metrics.snr(dsp.fix_ambiguity(dsp.bps(r, c)[0], s), s)
sv = metrics.snr(dsp.fix_ambiguity(dsp.vv(r, 64, 4, c)[0], s), s)
report('CPR BPS / VV', min(sb, sv) > 16, f'SNR BPS {sb:.1f}  VV {sv:.1f} dB (AWGN 18)')
mask = np.zeros(s.shape[1], bool); mask[::32] = True
sp = metrics.snr(dsp.pilot_cpr(r, s, mask, 5)[0], s)
sl = metrics.snr(dsp.fix_ambiguity(dsp.pll(r * np.exp(0.3j), c)[0], s), s)
report('CPR pilot / DD-PLL', min(sp, sl) > 16, f'SNR pilot {sp:.1f}  PLL {sl:.1f} dB')

# 5) 定时恢复：0.37 采样时延
xd = delay(x, 0.37 / fs, fs)
_, tau = dsp.timing_ff(xd, sps)
_, taus = dsp.gardner(xd, 2e-3, sps)
report('Timing FF / Gardner', abs(tau - 0.37) < 0.05 and abs(taus[-1] - 0.37) < 0.1, f'FF {tau:.3f}, Gardner {taus[-1]:.3f} sa (true 0.37)')

# 6) MIMO：随机 SOP + DGD 10 ps
xm = mf(awgn(devices.dgd(devices.random_unitary(rng) @ x, fs, 10e-12), 20, sps, rng))
_, w, _ = dsp.mimo_eq(xm, 15, 1e-3, sps, 'cma', c)
ym, _, _ = dsp.mimo_eq(xm, 15, 5e-4, sps, 'rde', c, w=w)
ys, lag, perm = dsp.sync(dsp.bps(ym, c)[0], s)
sm = metrics.snr(dsp.fix_ambiguity(ys, s)[:, 5000:], s[:, 5000:])
report('MIMO CMA->RDE + sync', sm > 17, f'SNR {sm:.1f} dB (AWGN 20), lag {lag}, perm {perm}')

# 7) Kramers-Kronig：载波 + 单边带（偏移 0.6 Rs），CSPR 10 dB
xs = normalize(tx.pulse_shape(s[0], 4, 0.1)); t = np.arange(len(xs)); fsh = 0.6 / 4
Ek = dsp.kk(np.abs(np.sqrt(10) + xs * np.exp(2j * np.pi * fsh * t)) ** 2)
rec = tx.matched_filter((Ek - np.mean(Ek)) * np.exp(-2j * np.pi * fsh * t), 4, 0.1)[::4]
report('Kramers-Kronig', metrics.snr(rec, s[0]) > 20, f'SNR {metrics.snr(rec, s[0]):.1f} dB')

print('---- Fiber physics ----')
fib = fiber_params('C', 'SSMF', pmd=0)
# 8) SPM：CW 非线性相移 = γ·P·Leff
P0, n = 0.1, 1024
Eo = channel.ssfm(np.sqrt(P0) * np.ones(n), 100e9, fib, nsteps=200)
a = fib['alpha'] * np.log(10) / 10
phi_th = fib['gamma'] * P0 * (1 - np.exp(-a * fib['L'])) / a
report('SPM phase (CW)', abs(-np.angle(Eo[0, 0]) / phi_th - 1) < 1e-3, f'{-np.angle(Eo[0, 0]):.4f} rad vs γPLeff {phi_th:.4f}')

# 9) 基阶孤子：P0 = |β2|/(γT0²)，无损传输 5 个色散长度后形状不变
T0, fsol = 5e-12, 2e12
tt = (np.arange(4096) - 2048) / fsol
b2 = abs(channel.beta2(fib['D'], fib['lam']))
Ps = b2 / (fib['gamma'] / 1e3 * T0 ** 2)
LD = T0 ** 2 / b2 / 1e3
u0 = np.sqrt(Ps) / np.cosh(tt / T0)
f0 = dict(fib, alpha=0, S=0, L=5 * LD)
u1 = channel.ssfm(u0, fsol, f0, nsteps=500)[0]
u2 = channel.ssfm(u0, fsol, dict(f0, gamma=0), nsteps=10)[0]
err = np.max(np.abs(np.abs(u1) - np.abs(u0))) / np.sqrt(Ps)
report('Fundamental soliton', err < 0.01, f'shape error {err:.1e} (linear-only peak drops to {np.max(np.abs(u2)) ** 2 / Ps:.2f})')

# 10) SRS：相距 13.2 THz 的两 CW（各 0.5 W），能量从高频流向低频；
#     与双波解析解 P_low(z) = P/(1 + e^{-g·P·z}) 比较，g = 2γ·fR·|Im H_R(Ω)|
fsr, nr, fr = 40e12, 4000, 6.6e12                 # 频率落在 FFT 网格上，避免泄漏
tr = np.arange(nr) / fsr
E2 = np.sqrt(0.5) * (np.exp(2j * np.pi * fr * tr) + np.exp(-2j * np.pi * fr * tr))
Er = channel.ssfm(E2, fsr, dict(fib, alpha=0, L=1, raman=True), nsteps=50)[0]
Sp = np.abs(np.fft.fft(Er) / nr) ** 2
lo = Sp[np.argmin(abs(freq(nr, fsr) + fr))]
gR = 2 * fib['gamma'] / 1e3 * fib['fR'] * abs(np.imag(channel.raman_response(2 * np.pi * 2 * fr, fib['tau1'], fib['tau2'])))
lo_th = 1 / (1 + np.exp(-gR * 1.0 * 1e3))
report('SRS (2-tone, 1 km)', abs(lo / lo_th - 1) < 0.01, f'P_low {lo:.3f} W vs analytic {lo_th:.3f} W (Raman gain {gR * 1e3:.2f} /W/km)')

# 11) FWM：两 CW（±25 GHz）在 DSF 与 SSMF 中产生的 ±75 GHz 分量
fw, nw = 400e9, 4096
tw = np.arange(nw) / fw
Ew = np.sqrt(0.01) * (np.exp(2j * np.pi * 25e9 * tw) + np.exp(-2j * np.pi * 25e9 * tw))
fwm = {}
for fbname in ('DSF', 'SSMF'):
    fp = fiber_params('C', fbname, pmd=0)
    Sw = np.abs(np.fft.fft(channel.ssfm(Ew, fw, fp, nsteps=400)[0])) ** 2
    fwm[fbname] = 10 * np.log10(Sw[np.argmin(abs(freq(nw, fw) - 75e9))] / Sw[np.argmin(abs(freq(nw, fw) - 25e9))])
report('FWM (DSF vs SSMF)', fwm['DSF'] > fwm['SSMF'] + 15, f"product/pump: DSF {fwm['DSF']:.1f} dB, SSMF {fwm['SSMF']:.1f} dB")

# 12) GN 模型：数值积分 vs 闭式（5 信道 Nyquist WDM）
en = channel.gn_eta(fib, 32e9, 5, 32e9, 'numeric')
ec = channel.gn_eta(fib, 32e9, 5, 32e9, 'closed')
report('GN numeric vs closed', abs(10 * np.log10(en / ec)) < 1, f'η {en:.1f} vs {ec:.1f} 1/W² per span ({10 * np.log10(en / ec):+.2f} dB)')
