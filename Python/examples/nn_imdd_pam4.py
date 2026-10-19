"""神经网络均衡器 vs 传统均衡器：IM-DD PAM4（NN 替代 FFE / Volterra 的典型场景）。

链路：56 GBd PAM4 -> DAC -> MZM（正交偏置、大摆幅 → 调制非线性）-> 4 km SSMF @1550 nm（色散功率衰落）
      -> VOA -> PIN（平方律 → 非线性 ISI）-> ADC (2 sps) -> 均衡器
对比：最小二乘 FFE（线性）、Volterra（2/3 阶）、MLP、1D-CNN、BiLSTM；扫描接收光功率 ROP。
训练与测试使用**独立实现**（不同比特与噪声），避免网络记住序列。
运行（需 torch）：python examples/nn_imdd_pam4.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optsim import tx, devices as dev, channel, dsp, metrics
from optsim.utils import w2dbm, total_power
from optsim.nn_eq import NNEqualizer

rs, sps, L, SWING = 56e9, 4, 4.0, 0.5         # 符号率、仿真过采样、光纤长度 [km]、MZM 驱动摆幅 [Vπ]
fs = rs * sps
lv = tx.pam(4) / 3
NTAPS = 41                                     # 所有均衡器的输入窗长（2 sps 采样点）
EQS = {'FFE (LS)': dict(taps=(NTAPS, 0, 0)),
       'Volterra': dict(taps=(NTAPS, 15, 5)),
       'MLP': dict(arch='mlp', hidden=(128, 64)),
       'CNN': dict(arch='cnn', channels=(16, 16), kernel=7),
       'BiLSTM': dict(arch='bilstm', hidden=32)}


def realization(seed, rop, n=2 ** 16):
    """一次独立的发射-传输-接收，返回 (2 sps 接收信号, 发送符号, 发送比特)。"""
    rng = np.random.default_rng(seed)
    bits = tx.rand_bits(2 * n, rng)
    sym = lv[tx.bits2int(bits, 2)]
    u = dev.dac(tx.pulse_shape(sym, sps, 0.2, 'rc').real, fs, enob=6, bw=0.6 * rs)
    E = dev.mzm(dev.laser(len(u), fs, 10, rin_db=-150, rng=rng), SWING * u, vbias=-0.5)
    E = channel.cd(E, fs, L)
    E = dev.voa(E, w2dbm(total_power(E)) - rop)
    i = dev.photodiode(E, fs, R=0.8, bw=0.7 * rs, T=300, rng=rng)
    y = dev.adc(i, fs, 2 * rs, enob=6)
    return (y - y.mean()) / y.std() * lv.std(), sym, bits


rop = np.arange(-10, -1, 1.0)
ber = {k: [] for k in EQS}
cplx = {}
for p in rop:
    y_tr, s_tr, _ = realization(1, p)
    y_te, s_te, b_te = realization(2, p)
    line = []
    for name, kw in EQS.items():
        t = time.time()
        if 'taps' in kw:                                            # 传统：最小二乘求系数
            _, w = dsp.volterra(y_tr, s_tr, len(s_tr), kw['taps'])
            ye = dsp.volterra_features(y_te, kw['taps'], 2) @ w
            cplx[name] = len(w)
        else:                                                        # 神经网络
            kw = dict(kw)
            eq = NNEqualizer(kw.pop('arch'), ntaps=NTAPS, sps=2, **kw).fit(y_tr, s_tr, verbose=False)
            ye = eq(y_te)
            cplx[name] = eq.mults_per_symbol()
        ber[name].append(max(metrics.ber(b_te, tx.int2bits(tx.decide(ye, lv), 2))[0], 1e-6))
        line.append(f'{name} {ber[name][-1]:.1e}')
    print(f'ROP {p:+.0f} dBm: ' + ' | '.join(line), flush=True)

print('\n每符号实数乘法次数：' + ', '.join(f'{k} {v}' for k, v in cplx.items()))
os.makedirs('results', exist_ok=True)
for k in EQS:
    plt.semilogy(rop, ber[k], 'o-', label=f'{k} ({cplx[k]} mult/sym)')
plt.axhline(metrics.HD_FEC, color='r', ls=':', label='HD-FEC')
plt.xlabel('ROP (dBm)'); plt.ylabel('BER'); plt.grid(True, which='both'); plt.legend(fontsize=8)
plt.title(f'56 GBd PAM4, {L:.0f} km SSMF @1550 nm\n(test on independent data)', fontsize=10)
plt.savefig('results/nn_imdd_pam4.png', dpi=120)
