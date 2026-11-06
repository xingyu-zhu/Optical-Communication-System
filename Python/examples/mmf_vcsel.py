"""多模光纤（OM3 / OM4，850 nm）短距 IM-DD：模式群时延、有效模式带宽（EMB）、激励条件、PAM4 BER 与传输距离。

模型（optsim.multimode）：α 剖面渐变折射率光纤，G = floor(V/2) 个主模式群，群时延按 Olshansky-Keck（ε=0）计算；
激励功率分布由满注入（OFL）或高斯光斑对 HG 模的交叠积分给出；探测器对各模式群功率求和。
发射机用 devices.dml 近似直调 VCSEL（啁啾 + RIN），接收复用 system.imdd_rx（PIN + TIA + FFE）。
1) 各模式群相对群时延（类似 DMD 测量）：α_opt 与 OM3/OM4 标定剖面
2) 满注入 EMB 随剖面指数偏差的变化（OM3/OM4 的 alpha_profile_offset 按标准 EMB 2000 / 4700 MHz·km 标定）
3) 激励条件：满注入、居中高斯光斑、偏移光斑的模式群功率分布与对应 EMB
4) 53.125 GBd PAM4（≈100 Gb/s）BER 随光纤长度变化（OM3 / OM4，满注入，ROP 固定）
注意：离散模式群冲激响应 Σ P_g δ(t - τ_g) 在 -3 dB 带宽之外有深陷波，比实际光纤（DMD 连续、含模式间耦合）更悲观；
     传输距离结果只用于说明 EMB 对距离的限制趋势，不对应具体标准的链路预算。
运行：python examples/mmf_vcsel.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import multimode as mm
from optsim.system import ImddConfig, imdd_tx, imdd_rx

fibers = {k: mm.mm_params(k) for k in ('OM3', 'OM4')}
m4 = fibers['OM4']
print(f"OM3/OM4 @ 850 nm: V = {m4['V']:.2f}，模式群数 G = {m4['n_groups']}，基模模场半径 {m4['w0_um']:.2f} µm，"
      f"Δ = {m4['delta']:.4f}，α_opt = {m4['alpha_opt']:.4f}，D = {m4['D']:.1f} ps/(nm·km)")

# ---------------- 1) 模式群时延 ----------------
dmd = {}
for name, m in {'α_opt': mm.mm_params('OM4', alpha_profile_offset=0.0), **fibers}.items():
    t = mm.mmf_groups(m)['tau'] * 1e12                       # ps/km
    dmd[name] = t - t[0]
    print(f"{name:5s} α = {m['alpha_profile']:.4f}：群时延相对 LP01 群最大 {np.ptp(dmd[name]):.1f} ps/km，"
          f"满注入 EMB {mm.mmf_emb(m, mm.mmf_launch(m, 'ofl'))[0]:.0f} MHz·km（标准 {m['emb_nominal_mhz_km']} MHz·km）")

# ---------------- 2) EMB 随剖面偏差 ----------------
offs = np.linspace(-0.04, 0.05, 91)
emb = np.array([mm.mmf_emb(mm.mm_params('OM4', alpha_profile_offset=o), mm.mmf_launch(m4, 'ofl'))[0] for o in offs])

# ---------------- 3) 激励条件 ----------------
launch = {'OFL': mm.mmf_launch(m4, 'ofl'),
          'Gaussian w = 4 µm': mm.mmf_launch(m4, 'gauss', w_um=4.0),
          'Gaussian w = 4 µm, offset 8 µm': mm.mmf_launch(m4, 'gauss', w_um=4.0, offset_um=8.0)}
for k, P in launch.items():
    e = mm.mmf_emb(m4, P)[0]
    print(f"OM4 {k:32s}: 耦合效率 {P.sum():.3f}，EMB {'> 10⁶' if np.isinf(e) else f'{e:.0f}'} MHz·km")

# ---------------- 4) PAM4 BER vs 长度 ----------------
cfg = ImddConfig(M=4, rs=53.125e9, n_sym=2 ** 15, tx='dml', p_tx=0.0, er_db=6.0, alpha=3.0, k_ad=1e12,
                 linewidth=50e6, rin_db=-145, dac_bw=0.6, rx_bw=0.7, ntaps=31)
ref = imdd_tx(cfg)
rop = -4.0
lengths = np.array([20, 40, 60, 80, 100, 150, 200, 300.0])
ber = {}
for name, m in fibers.items():
    P = mm.mmf_launch(m, 'ofl')
    ber[name] = np.array([imdd_rx(mm.mmf_link(ref['E'], cfg.fs, m, P, L / 1e3), cfg, ref, rop, rng=1)['ber']
                          for L in lengths])
    print(f"{name} PAM4 53.125 GBd，ROP {rop} dBm，BER @ {lengths.astype(int)} m：", np.array2string(ber[name], precision=2))
one = np.array([imdd_rx(mm.mmf_link(ref['E'], cfg.fs, m4, np.eye(m4['n_groups'])[0], L / 1e3), cfg, ref, rop, rng=1)['ber']
                for L in lengths])
print('仅色度色散（单个模式群）参考：', np.array2string(one, precision=2))

# ---------------- 绘图 ----------------
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
for k, v in dmd.items():
    ax[0].plot(np.arange(1, len(v) + 1), v, 'o-', ms=4, label=k)
ax[0].set_xlabel('Principal mode group'); ax[0].set_ylabel('Relative group delay (ps/km)')
ax[0].set_title('Mode-group delay (Olshansky-Keck, ε = 0)'); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[1].semilogy(offs, emb, 'k-')
for name, m in fibers.items():
    ax[1].semilogy(m['alpha_profile_offset'], m['emb_nominal_mhz_km'], 'o', label=f"{name} ({m['emb_nominal_mhz_km']} MHz·km)")
ax[1].set_xlabel(r'$\alpha - \alpha_{opt}$'); ax[1].set_ylabel('EMB, OFL (MHz·km)'); ax[1].set_title('EMB vs index profile')
ax[1].grid(alpha=0.3, which='both'); ax[1].legend(fontsize=8)
w = 0.27
for i, (k, P) in enumerate(launch.items()):
    ax[2].bar(np.arange(1, len(P) + 1) + (i - 1) * w, P / P.sum(), w, label=k)
ax[2].set_xlabel('Principal mode group'); ax[2].set_ylabel('Normalized power'); ax[2].set_title('Launch conditions (OM4)')
ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
for name in fibers:
    ax[3].semilogy(lengths, np.maximum(ber[name], 1e-6), 'o-', label=f'{name}, OFL')
ax[3].semilogy(lengths, np.maximum(one, 1e-6), 'k--', label='CD only')
ax[3].axhline(2.4e-4, color='gray', ls=':', label='KP4 FEC threshold')
ax[3].set_xlabel('Fiber length (m)'); ax[3].set_ylabel('BER'); ax[3].set_title(f'PAM4 53.125 GBd, ROP {rop:.0f} dBm')
ax[3].grid(alpha=0.3, which='both'); ax[3].legend(fontsize=8)
plt.tight_layout(); plt.savefig('results/mmf_vcsel.png', dpi=120)
