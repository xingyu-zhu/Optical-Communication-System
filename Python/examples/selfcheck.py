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

print('---- More algorithms ----')
from optsim import fec
# 13) OFDM：Schmidl-Cox 定时/频偏 + 3 径信道 + LS 信道估计 + 导频 CPE
nfft, ncp, Lsc = 256, 16, 64
Xo = c[rng.integers(0, M, (40, nfft))]
Xo[:, 0] = 0
half = np.exp(2j * np.pi * rng.random(Lsc))
sig = np.r_[np.zeros(100), half, half, tx.ofdm_mod(Xo, ncp), np.zeros(10)]
yo = awgn(np.convolve(sig, [1, 0.3 + 0.2j, 0.1])[:len(sig)] * np.exp(2j * np.pi * 1e-3 * np.arange(len(sig))), 25, rng=rng)
d, fo, _ = dsp.schmidl_cox(yo, Lsc)
st = d + 2 * Lsc - ncp // 2                          # 定时回退半个 CP，避免晚采引入符号间干扰
Yo = dsp.ofdm_demod(dsp.fo_comp(yo, fo, 1.0)[st:], nfft, ncp)[:len(Xo)]
Zo, _ = dsp.ofdm_eq(Yo, Xo, 2, np.arange(1, nfft, 16))
so = metrics.snr(Zo[2:, 1:], Xo[2:, 1:])
# 理论 SNR = 1/mean(σ²·(1 + E[1/|X|²]/n_train)/|H|²)：频率选择性信道 + 2 个训练符号的 LS 估计噪声
Hf = np.abs(np.fft.fft([1, 0.3 + 0.2j, 0.1], nfft)[1:]) ** 2
sth = -10 * np.log10(np.mean((1 + np.mean(1 / np.abs(c) ** 2) / 2) * 10 ** -2.5 * np.mean(np.abs(sig) ** 2) * 1.14 / Hf))
report('OFDM + Schmidl-Cox', abs(fo - 1e-3) < 1e-4 and so > sth - 1, f'd {d} (true 100), CFO {fo:.2e} (true 1e-3), SNR {so:.1f} dB (theory {sth:.1f})')

# 14) 差分 QAM：中途 π/2 周跳不扩散
bd = tx.rand_bits((2, 4 * 8000), rng)
rd = awgn(tx.diff_qam(bd, 16), 16, rng=rng)
rd[:, 4000:] *= 1j
bdd, _ = metrics.ber(bd, tx.diff_qam_demod(rd, 16))
report('Differential 16QAM + slip', bdd < 3 * metrics.ber_theory(16, 16), f'BER {bdd:.2e} (Gray, no slip {metrics.ber_theory(16, 16):.2e})')

# 15) IM-DD：FFE / DFE / MLSE（PAM4，强 ISI 信道 [1, 0.9, 0.5]）
lv = tx.pam(4)
a = lv[rng.integers(0, 4, 40000)]
xr = np.convolve(a, [1, 0.9, 0.5])[:len(a)] + 0.25 * rng.standard_normal(len(a))
dec = lambda v: lv[np.argmin(np.abs(v[:, None] - lv), 1)]
ser_f = np.mean(dec(dsp.ffe(xr, a, 21, 2e-4, 1, 8000, lv)[0])[10000:] != a[10000:])
ser_d = np.mean(dec(dsp.dfe(xr, a, 21, 3, 2e-4, 1, 8000, lv)[0])[10000:] != a[10000:])
ser_m = np.mean(dsp.mlse(xr, dsp.ch_est(xr[:8000], a[:8000], 3), lv)[10000:] != a[10000:])
report('PAM4 FFE / DFE / MLSE', ser_m < ser_d < ser_f, f'SER {ser_f:.2e} / {ser_d:.2e} / {ser_m:.2e}')

# 16) PB-NLC：合成的一阶 SPM/IXPM 型失真 y = s·exp(jε(|s_k|² + 0.5|s_k±1|²)) + 噪声
pw = np.sum(np.abs(s) ** 2, 0)
yn = awgn(s * np.exp(0.25j * (pw + 0.5 * np.roll(pw, 1) + 0.5 * np.roll(pw, -1))), 24, rng=rng)
yp, _ = dsp.pb_nlc(yn, s, 8000, L=3)
s0_, s1_ = metrics.snr(yn[:, 8000:], s[:, 8000:]), metrics.snr(yp[:, 8000:], s[:, 8000:])
report('PB-NLC (synthetic)', s1_ > s0_ + 2, f'SNR {s0_:.1f} -> {s1_:.1f} dB')

# 17) LDPC (3,6) n=1944 + QPSK @ 3 dB：软判决译码
code = fec.ldpc_make(1944, 3, 6, rng)
u = rng.integers(0, 2, (20, code['k']))
cw = fec.ldpc_encode(u, code)
q4 = tx.qam(4)
sq = tx.modulate(cw.ravel(), q4)
rq = awgn(sq, 3, rng=rng)
bh, ok = fec.ldpc_decode(tx.llr(rq, q4, 10 ** -0.3).reshape(len(u), -1), code)
pre, post = np.mean(tx.demodulate(rq, q4) != cw.ravel()), np.mean(bh[:, code['free']] != u)
report('LDPC NMS decoding', post < 1e-4 < pre, f'pre-FEC {pre:.2e} -> post-FEC {post:.1e}, NGMI {metrics.ngmi(rq, sq, q4):.3f}, rate {code["k"] / 1944:.3f}')

print('---- Hollow-core / few-mode / multimode fiber ----')
from optsim import multimode as mm
# 18) 空芯光纤：SPM 相移 = γPLeff（线性色散模型 + medium=air 时 fR=0 的 RK4 路径）、时延比与 IMI 功率
hcf = fiber_params('C', 'HCF', pmd=0, raman=True)
P0h = 1.0
Eh = channel.ssfm(np.sqrt(P0h) * np.ones(1024), 100e9, hcf, nsteps=50)
ah = hcf['alpha'] * np.log(10) / 10
ph_h = hcf['gamma'] * P0h * (1 - np.exp(-ah * hcf['L'])) / ah
Ei = normalize(rng.standard_normal(2 ** 14) + 1j * rng.standard_normal(2 ** 14))
eta = np.mean(np.abs(channel.imi(Ei, 64e9, -30, 50e-9, rng=rng) - np.sqrt(1 - 1e-3) * Ei) ** 2)
report('HCF SPM / latency / IMI', abs(-np.angle(Eh[0, 0]) / ph_h - 1) < 1e-3 and abs(eta / 1e-3 - 1) < 0.1,
       f"φ {-np.angle(Eh[0, 0]):.2e} vs γPLeff {ph_h:.2e} rad, latency {hcf['latency']:.3f} vs SSMF {fib['latency']:.3f} µs/km, IMI {10 * np.log10(eta):.1f} dB (set -30)")

# 19) 多模 Kerr 系数：单模退化为 Manakov 8/9；强耦合 FMF 中 CW 的非线性相移 = Γ·P_tot·L（与随机耦合对易）
w = 6e-6
k1 = mm.kerr_coeffs(mm.overlap(np.array([[0, 0]]), w), [0, 0])[0][0, 0] * np.pi * w ** 2
q = mm.mm_params('FMF3', L=20, alpha_db_km=0, mdl_db_km=[0, 0], mdg_db=[0, 0], dmgd_ps_km=[0, 0], D=[0, 0], S=[0, 0])
Ec = np.ones((6, 256)) * np.sqrt(0.05 / 6) * np.exp(2j * np.pi * rng.random((6, 1)))
Hq = mm.mm_transfer(64e9, 256, q, 1, rng=3)[0]
Gm = mm.nl_matrix(q)[1][0, 0]
dev_nl = np.max(np.abs(mm.ssfm_mm(Ec, 64e9, q, rng=3) - Hq @ Ec * np.exp(-1j * Gm * 0.05 * 20e3))) / np.sqrt(0.05 / 6)
report('Multimode Kerr coefficient', abs(k1 - 8 / 9) < 1e-6 and dev_nl < 1e-3, f'κ·Aeff {k1:.6f} (8/9), FMF3 CW phase error {dev_nl:.1e}')

# 20) 随机模式耦合：无损时 H 酉；mm_link（线性、无 ASE）与 mm_transfer 逐点一致
q = mm.mm_params('FMF3', L=30)
Hu = mm.mm_transfer(64e9, 512, mm.mm_params('FMF3', L=30, alpha_db_km=0, mdl_db_km=[0, 0], mdg_db=[0, 0]), 1, rng=1)
uerr = np.max(np.abs(Hu @ np.conj(np.swapaxes(Hu, 1, 2)) - np.eye(6)))
Ex = rng.standard_normal((6, 2048)) + 1j * rng.standard_normal((6, 2048))
E1 = mm.mm_link(Ex, 64e9, q, 2, ase=False, rng=5, nonlinear=False)
E2 = np.fft.ifft(np.einsum('nij,jn->in', mm.mm_transfer(64e9, 2048, q, 2, rng=5), np.fft.fft(Ex, axis=1)), axis=1)
lerr = np.linalg.norm(E1 - E2) / np.linalg.norm(E2)
report('Mode coupling / transfer', uerr < 1e-12 and lerr < 1e-10, f'|HH^H - I| {uerr:.1e}, link vs transfer {lerr:.1e}')

# 21) 群时延扩展：强耦合 ∝ √L（10 → 40 km 约 2 倍），弱耦合 ≈ 不耦合值 DMGD·L·√(2/9)
gstd = lambda reg, L: np.mean([np.mean(np.std(mm.group_delays(mm.mm_transfer(
    64e9, 4096, mm.mm_params('FMF3', L=L, regime=reg), 1, rng=s), 64e9), 1)) for s in range(6)])
r_s = gstd('strong', 40) / gstd('strong', 10)
r_w = gstd('weak', 10) / (50e-12 * 10 * np.sqrt(2 / 9))
report('GD spread strong / weak', abs(r_s - 2) < 0.3 and abs(r_w - 1) < 0.1, f'strong 40/10 km ratio {r_s:.2f} (√4 = 2), weak / uncoupled {r_w:.3f}')

# 22) MMF 激励：模场失配高斯光斑 P_2k = (1-r²)·C(2k,k)/4^k·r^{2k}（2D 合并后 P_N = (1-r²)·r^N，N 偶），
#     横向偏移 d 的匹配光斑 P_1 = e^{-(d/w0)²}（泊松分布，均值 (d/w0)²）
om = mm.mm_params('OM4')
w0 = om['w0_um']
rr = (3 ** 2 - w0 ** 2) / (3 ** 2 + w0 ** 2)
Pm = mm.mmf_launch(om, 'gauss', w_um=3.0)
Po = mm.mmf_launch(om, 'gauss', offset_um=5.0)
th_m = (1 - rr ** 2) * rr ** np.array([0, 2, 4])
report('MMF launch (mismatch / offset)', np.max(np.abs(Pm[[0, 2, 4]] - th_m)) < 1e-6 and abs(Po[0] / np.exp(-(5 / w0) ** 2) - 1) < 1e-6,
       f'P_0,2,4 {np.round(Pm[[0, 2, 4]], 4)} vs {np.round(th_m, 4)}, offset P_1 {Po[0]:.4f} vs {np.exp(-(5 / w0) ** 2):.4f}')

# 23) MMF 有效模式带宽：OM3 / OM4 标定值，α_opt 附近带宽最大
e3 = mm.mmf_emb(mm.mm_params('OM3'), mm.mmf_launch(om, 'ofl'))[0]
e4 = mm.mmf_emb(om, mm.mmf_launch(om, 'ofl'))[0]
e0 = mm.mmf_emb(mm.mm_params('OM4', alpha_profile_offset=0.0), mm.mmf_launch(om, 'ofl'))[0]
report('MMF EMB (OFL)', abs(e3 / 2000 - 1) < 0.02 and abs(e4 / 4700 - 1) < 0.02 and e0 > e4, f'OM3 {e3:.0f}, OM4 {e4:.0f}, α_opt {e0:.0f} MHz·km')
