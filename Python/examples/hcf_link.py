"""空芯光纤（HCF，反谐振 DNANF 型）与 SSMF 对比：参数与时延、入纤功率扫描（SSFM vs GN 模型）、模间干扰（IMI）容限。

1) 参数表：衰减、色散、非线性系数、群折射率与单程时延（params/fiber_bands.json 中的 'HCF' 典型值）
2) 单信道 DP-16QAM 40 GBd、10×80 km：SSFM + CDC 的接收 SNR 随入纤功率变化，与 GN 模型对比
   （HCF 的 γ 约为 SSMF 的 1/5000，最优入纤功率高出 20 dB 以上，实际受放大器输出功率限制；
   HCF 高功率时 SSFM 结果约 36 dB 封顶，来自接收 DSP 自身的 SNR 上限（自适应均衡失调、BPS 窗口噪声），不是光纤效应）
3) HCF 链路每跨加一次 IMI（channel.imi），时延远大于均衡器窗口时 IMI 表现为加性干扰：
   1/SNR ≈ 1/SNR_0 + N·η（η = 10^(IMI/10)，N 跨数）
运行：python examples/hcf_link.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, channel, metrics
from optsim.system import Config, tx_signal, rx_dsp

base = Config(M=16, rs=40e9, n_blocks=300, n_spans=10, L=80, fs_adc=160e9, fib_kw={'pmd': 0})
ref = tx_signal(base)
N, f = ref['wave'].shape[1], ref['frame']
data = ref['kind'] == 0


def run(cfg, p, imi_db=None, seed=1):
    """理想收发机：激光器 + 功放 -> 链路（可选每跨 IMI）-> 相干接收 -> rx_dsp，返回数据符号 SNR [dB]。"""
    fib, rng = cfg.fib, np.random.default_rng(seed)
    E = dev.edfa(ref['wave'] * dev.laser(N, cfg.fs, 0, rng=rng), cfg.fs, p_out_dbm=p, nf_db=fib['nf'], f0=fib['f0'], rng=rng)
    for _ in range(cfg.n_spans):
        E = channel.link(E, cfg.fs, fib, 1, phi_max=cfg.phi_max, rng=rng)
        if imi_db is not None:                      # LP11 与基模 DMGD 取 10 ps/m（典型量级，待核实）
            E = channel.imi(E, cfg.fs, imi_db, 10e-12 * fib['L'] * 1e3, rng=rng)
    r = dev.coherent_rx(E, dev.laser(N, cfg.fs, 10), cfg.fs, shot=False)
    y = rx_dsp(r, replace(cfg, p_launch=p), ref)['y']
    return metrics.snr(y[:, data], f[:, data])


# ---------------- 1) 参数与时延 ----------------
cfgs = {'SSMF': base, 'HCF': replace(base, fiber='HCF')}
print(f"{'fiber':5s} {'α dB/km':>8s} {'D':>6s} {'γ 1/W/km':>9s} {'ng':>7s} {'µs/km':>6s} {'800 km ms':>9s} {'GN P_opt dBm':>12s} {'SNR_opt dB':>10s}")
gn = {}
for name, c in cfgs.items():
    fib = c.fib
    gn[name] = channel.gn_model(np.linspace(-6, 36, 400), fib, c.rs, c.n_spans)
    print(f"{name:5s} {fib['alpha']:8.3f} {fib['D']:6.2f} {fib['gamma']:9.2e} {fib['ng']:7.4f} {fib['latency']:6.3f} "
          f"{fib['latency'] * c.n_spans * c.L / 1e3:9.3f} {gn[name]['p_opt_dbm']:12.2f} {gn[name]['snr_opt']:10.2f}")

# ---------------- 2) 入纤功率扫描 ----------------
powers = {'SSMF': np.arange(-2, 7, 2.0), 'HCF': np.array([0, 10, 20, 26, 32.0])}
snr = {}
for name, c in cfgs.items():
    snr[name] = np.array([run(c, p) for p in powers[name]])
    print(name, 'SSFM SNR [dB] @', powers[name], 'dBm:', np.round(snr[name], 2))

# ---------------- 3) IMI 容限（HCF，+10 dBm） ----------------
imis = np.arange(-50, -19, 5.0)
p_imi = 10.0
s0 = run(cfgs['HCF'], p_imi)
s_imi = np.array([run(cfgs['HCF'], p_imi, q) for q in imis])
th = -10 * np.log10(10 ** (-s0 / 10) + base.n_spans * 10 ** (imis / 10))
print(f'HCF IMI @ {p_imi:+.0f} dBm: 无 IMI SNR {s0:.2f} dB')
for q, s, t in zip(imis, s_imi, th):
    print(f'  IMI {q:+.0f} dB/跨: SNR {s:.2f} dB（理论 {t:.2f} dB）')

# ---------------- 绘图 ----------------
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
pg = np.linspace(-6, 36, 400)
for name, col in (('SSMF', 'C0'), ('HCF', 'C3')):
    ax[0].plot(powers[name], snr[name], 'o', color=col, label=f'{name} SSFM + CDC')
    ax[0].plot(pg, gn[name]['snr'], '-', color=col, lw=1, label=f'{name} GN model')
ax[0].set_ylim(10, 52); ax[0].set_xlabel('Launch power per channel (dBm)'); ax[0].set_ylabel('SNR (dB)')
ax[0].set_title('10×80 km, DP-16QAM 40 GBd, 1 ch'); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
Lk = np.linspace(0, 2000, 50)
for name, col in (('SSMF', 'C0'), ('HCF', 'C3')):
    ax[1].plot(Lk, cfgs[name].fib['latency'] * Lk / 1e3, color=col, label=f"{name} (ng = {cfgs[name].fib['ng']:.4f})")
ax[1].set_xlabel('Distance (km)'); ax[1].set_ylabel('One-way latency (ms)'); ax[1].set_title('Propagation latency')
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
ax[2].plot(imis, s_imi, 'o', color='C3', label='SSFM + IMI per span')
ax[2].plot(imis, th, 'k--', lw=1, label=r'$1/\mathrm{SNR}_0 + N\eta$')
ax[2].set_xlabel('IMI per span (dB)'); ax[2].set_ylabel('SNR (dB)'); ax[2].set_title(f'HCF IMI tolerance, {p_imi:+.0f} dBm')
ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('results/hcf_link.png', dpi=120)
