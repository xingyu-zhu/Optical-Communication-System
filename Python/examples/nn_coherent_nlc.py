"""神经网络非线性补偿 vs DBP：相干 DP-16QAM 长距单信道（入纤功率扫描）。

对比（均接在同一套 rx_dsp 链中，只替换色散/非线性补偿级）：
    CDC                      仅色散补偿（线性）
    DBP 1 / 10 步每跨        传统数字反向传播（FFT 实现）
    LDBP 1 / 4 步每跨        学习型 DBP：DBP 结构 + 可学习短 FIR 与非线性系数（物理引导神经网络）
    CNN 后均衡               黑盒神经网络作用于 CDC+CPR 后的符号（对照：说明为何相干场景需要物理引导结构）
训练实现：无激光相位噪声；测试实现：独立比特/噪声 + 100 kHz 激光线宽（检验泛化）。
运行（需 torch，约 10~15 分钟）：python examples/nn_coherent_nlc.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import devices as dev, channel, dsp, metrics
from optsim.utils import resample
from optsim.system import Config, tx_signal, rx_dsp
from optsim.nn_eq import LDBP, NNEqualizer

base = Config(M=16, rs=32e9, n_blocks=1000, n_spans=10, fs_adc=128e9, fib_kw={'pmd': 0})
fib = base.fib
powers = [0.0, 2.0, 4.0, 6.0]
LDBP_CFG = {'LDBP 1 st/span': (1, 41), 'LDBP 4 st/span': (4, 15)}


def realization(cfg, seed, lw):
    """一次独立的传输，返回 (cfg, ref, ADC 输出)。"""
    c = replace(cfg, seed=seed)
    ref = tx_signal(c)
    rng = np.random.default_rng(seed)
    N = ref['wave'].shape[1]
    E = dev.edfa(ref['wave'] * dev.laser(N, c.fs, 0, lw, rng=rng), c.fs, p_out_dbm=c.p_launch, nf_db=fib['nf'], f0=fib['f0'], rng=rng)
    E = channel.link(E, c.fs, fib, c.n_spans, phi_max=c.phi_max, rng=rng)
    return c, ref, dev.coherent_rx(E, dev.laser(N, c.fs, 10, lw, rng=rng), c.fs, shot=False, rng=rng)


names = ['CDC', 'DBP 1 st/span', 'DBP 10 st/span', *LDBP_CFG, 'CNN post-eq']
snr = {k: [] for k in names}
cplx = {}
for P in powers:
    t0 = time.time()
    cfg = replace(base, p_launch=P)
    c1, ref1, r1 = realization(cfg, 11, 0)            # 训练实现
    c2, ref2, r2 = realization(cfg, 22, 100e3)        # 测试实现
    res = rx_dsp(r2, c2, ref2)
    snr['CDC'].append(res['snr'])
    for k, st in (('DBP 1 st/span', 1), ('DBP 10 st/span', 10)):
        snr[k].append(rx_dsp(r2, replace(c2, nl_comp='dbp', dbp_steps=st), ref2)['snr'])
    x1 = resample(dsp.gsop(r1), cfg.fs_adc, 2 * cfg.rs)
    for k, (st, nt) in LDBP_CFG.items():
        ld = LDBP(2 * cfg.rs, fib, cfg.n_spans, P, steps=st, ntaps=nt).fit(x1, ref1['frame'], cfg.beta, verbose=False)
        snr[k].append(rx_dsp(r2, replace(c2, nl_comp=ld.apply), ref2)['snr'])
        cplx[k] = ld.mults_per_symbol()
    eq = NNEqualizer('cnn', ntaps=41, sps=1, channels=(16, 16), kernel=9).fit(rx_dsp(r1, c1, ref1)['y'], ref1['frame'], verbose=False)
    m = ref2['kind'] == 0
    snr['CNN post-eq'].append(metrics.snr(eq(res['y'])[:, m], ref2['frame'][:, m]))
    cplx['CNN post-eq'] = eq.mults_per_symbol()
    print(f'P {P:+.0f} dBm: ' + ' | '.join(f'{k} {snr[k][-1]:.2f}' for k in names) + f'  ({time.time() - t0:.0f} s)', flush=True)

# DBP 复杂度（FFT 实现，每符号实数乘法近似，与 LDBP.mults_per_symbol 同口径）：
#   每采样每步 = 2 偏振 × [FFT+IFFT 4·log2(N) + 频域相乘 4] + 非线性 12；每符号 × sps(=2)；N 为分块 FFT 长度
nfft = 2 ** 12
for k, st in (('DBP 1 st/span', 1), ('DBP 10 st/span', 10)):
    cplx[k] = int(base.n_spans * st * 2 * (2 * (4 * np.log2(nfft) + 4) + 12))
print('\n每符号实数乘法次数（近似）：' + ', '.join(f'{k} {v}' for k, v in cplx.items()))

os.makedirs('results', exist_ok=True)
for k in names:
    plt.plot(powers, snr[k], 'o-' if 'LDBP' in k else 's--', label=k)
plt.xlabel('Launch power (dBm)'); plt.ylabel('SNR (dB)'); plt.grid(); plt.legend(fontsize=8)
plt.title('DP-16QAM 32 GBd, 10×80 km SSMF\n(test: independent data + 100 kHz linewidth)', fontsize=10)
plt.savefig('results/nn_coherent_nlc.png', dpi=120)
