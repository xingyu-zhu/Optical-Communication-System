"""少模光纤（FMF）相干空分复用（SDM）：模式耦合、群时延扩展、MDL、多模非线性与 2M×2M MIMO 接收。

模型（optsim.multimode）：耦合段模型（每段 lc 内各模式群独立的 DMGD/MDL/色散，段间随机酉耦合）+ 模式群平均的
多模 Manakov 非线性 + 少模 EDFA（MDG + ASE）；M 个 DP-16QAM 发射机共用一个激光器，接收端 M 个相干接收机共用本振，
DSP 为训练序列辅助粗频偏估计 + 2M×2M DA-LMS→RDE MIMO + BPS（multimode.sdm_rx_dsp）。
1) FMF3（LP01 + LP11a/b）群时延扩展随长度：强耦合 ∝ √L，弱耦合 ∝ L（不耦合上限 DMGD·L·√(2/9)）；MDL 随长度累积
2) FMF3 强耦合 4×50 km、6×6 MIMO：入纤功率扫描（每模式功率），非线性系数 Γ = (n2ω0/c)·κ
   （低功率端 SNR 约 21 dB 封顶，来自 6-bit DAC/ADC 与发射机带宽等收发机损伤）
3) FMF6（3 个模式群）弱耦合 1×10 km、12×12 MIMO：各模式 BER / SNR
   （seed=7 时 LP21/LP02 群的少量误码集中在帧末尾最后十几个符号：最后一个导频之后的一次周跳，属于帧边界效应）
运行：python examples/fmf_sdm.py（约 3 分钟）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import multimode as mm, plots
from optsim.system import Config


def spread_sym(mmf, n_spans, cfg, seed):
    """该光纤实现的群时延扩展（最大-最小，频率平均）[符号]。"""
    gd = mm.group_delays(mm.mm_transfer(cfg.fs, 2 ** 14, mmf, n_spans, rng=seed), cfg.fs)
    return np.mean(gd[:, -1] - gd[:, 0]) * cfg.rs


def sdm_run(cfg, mmf, n_spans, seed=7):
    """SDM 全链路一次：发射 -> 链路 -> 相干接收 -> MIMO DSP。抽头数按群时延扩展自动选取。"""
    ntaps = 4 * int(np.ceil(spread_sym(mmf, n_spans, cfg, seed))) + 1
    E, refs = mm.sdm_tx(cfg, mmf['n_modes'], rng=1)
    E = mm.mm_link(E, cfg.fs, mmf, n_spans, rng=seed, noise_rng=seed + 1)
    r = mm.sdm_rx(E, cfg, rng=2, f_offset=1.5e9)
    return mm.sdm_rx_dsp(r, cfg, refs, mmf, n_spans, ntaps=ntaps), ntaps


# ---------------- 1) 群时延扩展与 MDL ----------------
cfg = Config(M=16, rs=32e9, n_blocks=480, fs_adc=80e9)
lengths = np.array([5, 10, 20, 50, 100, 200.0])
gd_std, mdl = {}, {}
for reg in ('strong', 'weak'):
    g, d = [], []
    for L in lengths:
        q = mm.mm_params('FMF3', L=L, regime=reg)
        s, m = [], []
        for seed in range(4):
            H = mm.mm_transfer(64e9, 2 ** 14, q, 1, rng=seed)
            s.append(np.mean(np.std(mm.group_delays(H, 64e9), 1)))
            m.append(np.mean(mm.mdl_db(H)))
        g.append(np.mean(s)); d.append(np.mean(m))
    gd_std[reg], mdl[reg] = np.array(g) * 1e12, np.array(d)
    print(f"FMF3 {reg:6s}：群时延标准差 [ps] @ {lengths.astype(int)} km：{np.round(gd_std[reg], 1)}；MDL [dB]：{np.round(mdl[reg], 2)}")
q = mm.mm_params('FMF3')
uncoupled = q['dmgd_ps_km'][1] * lengths * np.sqrt(2 / 9)
for nm in ('FMF3', 'FMF6'):
    for reg in ('strong', 'weak'):
        print(f"{nm} {reg:6s} 非线性系数 Γ [1/(W·km)]：", np.round(mm.nl_matrix(mm.mm_params(nm, regime=reg))[1] * 1e3, 3).tolist())

# ---------------- 2) FMF3 强耦合 4×50 km 功率扫描 ----------------
fmf3 = mm.mm_params('FMF3', L=50, regime='strong')
powers = np.array([-4, 0, 4, 8, 12.0])
snr_p = []
for p in powers:
    res, ntaps = sdm_run(Config(M=16, rs=32e9, n_blocks=480, fs_adc=80e9, p_launch=p), fmf3, 4)
    snr_p.append(res['snr_mode'])
    print(f"FMF3 强耦合 4×50 km，{p:+.0f} dBm/模式，{ntaps} 抽头：BER {res['ber']:.2e}，各模式 SNR {np.round(res['snr_mode'], 2)} dB")
snr_p = np.array(snr_p)

# ---------------- 3) FMF6 弱耦合 1×10 km，12×12 MIMO ----------------
fmf6 = mm.mm_params('FMF6', L=10)
res6, nt6 = sdm_run(Config(M=16, rs=32e9, n_blocks=960, train_len=8192, fs_adc=80e9, p_launch=0.0), fmf6, 1)
print(f"FMF6 弱耦合 1×10 km，12×12 MIMO，{nt6} 抽头：BER {res6['ber']:.2e}")
for m, name in enumerate(fmf6['modes']):
    print(f"  {name:5s} BER {res6['ber_mode'][m]:.2e}  SNR {res6['snr_mode'][m]:.2f} dB")

# ---------------- 绘图 ----------------
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
ax[0].loglog(lengths, gd_std['strong'], 'o-', label='strong coupling')
ax[0].loglog(lengths, gd_std['weak'], 's-', label='weak coupling')
ax[0].loglog(lengths, gd_std['strong'][3] * np.sqrt(lengths / lengths[3]), 'k:', lw=1, label=r'$\propto\sqrt{L}$')
ax[0].loglog(lengths, uncoupled, 'k--', lw=1, label=r'uncoupled $\mathrm{DMGD}\cdot L\sqrt{2/9}$')
ax[0].set_xlabel('Length (km)'); ax[0].set_ylabel('Group-delay std (ps)'); ax[0].set_title('FMF3 group-delay spread')
ax[0].grid(alpha=0.3, which='both'); ax[0].legend(fontsize=8)
ax[1].plot(lengths, mdl['strong'], 'o-', label='strong coupling')
ax[1].plot(lengths, mdl['weak'], 's-', label='weak coupling')
ax[1].set_xlabel('Length (km)'); ax[1].set_ylabel('Mean MDL (dB)'); ax[1].set_title('FMF3 mode-dependent loss')
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
for m, name in enumerate(fmf3['modes']):
    ax[2].plot(powers, snr_p[:, m], 'o-', label=name)
ax[2].set_xlabel('Launch power per mode (dBm)'); ax[2].set_ylabel('SNR (dB)')
ax[2].set_title('FMF3 strong, 4×50 km, 6×6 MIMO'); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
plots.constellation(res6['sym'][10], ax[3], f"FMF6 weak 10 km, {fmf6['modes'][5]} X")
plt.tight_layout(); plt.savefig('results/fmf_sdm.png', dpi=120)
