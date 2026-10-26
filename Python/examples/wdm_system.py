"""WDM 相干系统仿真：全场 SSFM 同时包含 SPM / XPM / FWM（可选 SRS、自陡峭），
对比 ASE-only、单信道、WDM 三种情形下中心信道的 SNR，并给出 GN 模型预测与各信道 SNR 分布。

    N 个 DP-16QAM 信道（独立数据与激光相噪）-> WDM 合波 -> 10×80 km SSMF + 放大器
    -> WSS 选信道 -> 本振调谐 -> 相干接收 -> rx_dsp

收发机设为理想（无 DAC/ADC 量化），差异纯粹来自光纤传输。改 band/fiber 即可换波段与光纤类型。
运行：python examples/wdm_system.py   （约数分钟）
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import channel, plots
from optsim.system import Config, wdm_tx, wdm_rx, rx_dsp

N_CH = 7                       # WDM 信道数（奇数）
RUN_SRS = True                 # 是否额外跑一次含 SRS + 自陡峭的 WDM（较慢）
base = Config(M=16, rs=32e9, sps=16, n_blocks=200, band='C', fiber='SSMF', n_spans=10, L=80,
              p_launch=1.0, spacing=50e9, fs_adc=80e9, fib_kw={'pmd': 0})


def run(cfg, all_channels=False):
    """发射 -> 链路 -> 接收，返回 (各被测信道结果列表, 发射谱, 接收谱)。"""
    rng = np.random.default_rng(7)
    E, refs, freqs = wdm_tx(cfg, rng)
    Et = E.copy()
    t0 = time.time()
    E = channel.link(E, cfg.fs, cfg.fib, cfg.n_spans, phi_max=cfg.phi_max, rng=rng)
    chs = range(cfg.n_ch) if all_channels else [cfg.n_ch // 2]
    res = [rx_dsp(wdm_rx(E, cfg, freqs[k], rng=rng), cfg, refs[k]) for k in chs]
    print(f'    (link {time.time() - t0:.0f} s)')
    return res, Et, E


cases = [('ASE only (γ=0)', replace(base, n_ch=1, fib_kw={'pmd': 0, 'gamma': 0})),
         ('1 ch: SPM', replace(base, n_ch=1)),
         (f'{N_CH} ch: SPM+XPM+FWM', replace(base, n_ch=N_CH))]
if RUN_SRS:
    cases.append((f'{N_CH} ch: +SRS+self-steepening', replace(base, n_ch=N_CH, fib_kw={'pmd': 0, 'raman': True, 'steep': True})))

out = {}
for name, cfg in cases:
    print(f'{name} ...')
    out[name] = run(cfg, all_channels=cfg.n_ch > 1)
    r = out[name][0][len(out[name][0]) // 2]
    print(f'    center channel: SNR {r["snr"]:.2f} dB, BER {r["ber"]:.2e}, GMI {r["gmi"]:.3f}')

print('\nGN model (center channel, ideal transceiver):')
fib = base.fib
for n in (1, N_CH):
    g = channel.gn_model(base.p_launch, fib, base.rs, base.n_spans, n, base.spacing)
    print(f'    {n} ch: SNR {g["snr"]:.2f} dB (ASE {g["snr_ase"]:.2f}, NLI {g["snr_nli"]:.2f}); '
          f'P_opt {g["p_opt_dbm"]:.2f} dBm -> SNR_opt {g["snr_opt"]:.2f} dB')
full = int(4.4e12 // base.spacing)
g = channel.gn_model(base.p_launch, fib, base.rs, base.n_spans, full, base.spacing, method='closed')
print(f'    full C band ({full} ch): SNR {g["snr"]:.2f} dB; P_opt {g["p_opt_dbm"]:.2f} dBm -> SNR_opt {g["snr_opt"]:.2f} dB')

# ---------------- 绘图 ----------------
os.makedirs('results', exist_ok=True)
wdm_name = cases[2][0]
res_w, Et, Er = out[wdm_name]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
plots.spectrum(Et, base.fs, ax[0], 'Tx'); plots.spectrum(Er, base.fs, ax[0], 'Rx'); ax[0].legend()
ax[0].set_title(f'{N_CH}-ch WDM spectrum (X pol)')
idx = np.arange(N_CH) - N_CH // 2
for name, _ in cases[2:]:
    ax[1].plot(idx, [r['snr'] for r in out[name][0]], 'o-', label=name)
for name, _ in cases[:2]:
    ax[1].axhline(out[name][0][0]['snr'], ls='--', color='gray')
    ax[1].text(idx[0], out[name][0][0]['snr'] + 0.1, name, fontsize=8)
ax[1].set_xlabel('Channel index'); ax[1].set_ylabel('SNR (dB)'); ax[1].legend(fontsize=8); ax[1].grid()
plots.constellation(res_w[N_CH // 2]['sym'][0], ax[2], 'Center channel, X pol')
plt.tight_layout(); plt.savefig('results/wdm_system.png', dpi=120)
