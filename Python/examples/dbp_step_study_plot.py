"""DBP 步长研究：由 results/dbp_step_study/metrics.csv 统计并作图。

Q 因子两种算法：
  Q_BER : 把 10 组实现的误码数与比特数合并，BER = Σerr/Σbits，Q = 20·log10(√2·erfcinv(2·BER))；
          误差棒为 BER 的 95% Clopper-Pearson 置信区间映射到 Q。
  Q_SNR : 每组用数据辅助有效 SNR 代入 16QAM 高斯信道理论 BER，再换算 Q；给出 10 组均值 ± 标准差。
          高 Q 区（BER < 1e-5，误码过少）Q_BER 统计不可靠，此时以 Q_SNR 为准。
运行：python examples/dbp_step_study_plot.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from optsim import metrics, channel
from optsim.params import fiber_params

RES = os.path.join(os.path.dirname(__file__), '..', 'results', 'dbp_step_study')
METHODS = ['CDC', 'DBP-20km', 'DBP-10km', 'DBP-5km', 'DBP-1km']
LABEL = {'CDC': 'CDC', 'DBP-20km': 'DBP 20 km/step', 'DBP-10km': 'DBP 10 km/step',
         'DBP-5km': 'DBP 5 km/step', 'DBP-1km': 'DBP 1 km/step'}
STYLE = {'CDC': dict(color='#4a4a4a', marker='s', ls='--'),       # 中性灰：基线
         'DBP-20km': dict(color='#2a78d6', marker='o', ls='-'),   # 色盲友好分类色（固定顺序）+ 不同标记
         'DBP-10km': dict(color='#eb6834', marker='^', ls='-'),
         'DBP-5km': dict(color='#1baf7a', marker='D', ls='-'),
         'DBP-1km': dict(color='#eda100', marker='v', ls='-')}
HD_FEC, SD_FEC = 3.8e-3, 2.0e-2                                    # 7% HD-FEC / ~20% SD-FEC BER 门限
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.6,
                     'lines.markersize': 6, 'legend.fontsize': 8.5, 'savefig.dpi': 200, 'figure.dpi': 100})

q_ber = lambda b: metrics.q_factor(np.clip(b, 1e-15, 0.5))
q_snr = lambda s: metrics.q_factor(metrics.ber_theory(16, s))


def summarize(df):
    """按 (距离, 功率, 方法) 合并 10 组实现。"""
    g = df.groupby(['dist_km', 'p_dbm', 'method'])
    s = g.agg(n_err=('n_err', 'sum'), n_bits=('n_bits', 'sum'), n_seed=('seed', 'nunique'),
              snr=('snr_db', 'mean'), snr_std=('snr_db', 'std'), gmi=('gmi', 'mean')).reset_index()
    s['ber'] = s.n_err / s.n_bits
    lo = np.nan_to_num(beta_dist.ppf(0.025, s.n_err, s.n_bits - s.n_err + 1))         # Clopper-Pearson 95%
    hi = beta_dist.ppf(0.975, s.n_err + 1, s.n_bits - s.n_err)
    s['Q_ber'] = np.where(s.n_err > 0, q_ber(s.ber), np.nan)
    s['Q_ber_lo'], s['Q_ber_hi'] = q_ber(hi), np.where(lo > 0, q_ber(lo), np.nan)
    qs = df.assign(Q=q_snr(df.snr_db)).groupby(['dist_km', 'p_dbm', 'method']).Q
    s = s.merge(qs.mean().rename('Q_snr').reset_index()).merge(qs.std().rename('Q_snr_std').reset_index())
    return s


def optimum(s, col):
    """每个 (距离, 方法) 的最优功率：网格最大值附近三点抛物线拟合。"""
    rows = []
    for (d, m), t in s.groupby(['dist_km', 'method']):
        t = t.sort_values('p_dbm'); p, q = t.p_dbm.values, t[col].values
        i = int(np.nanargmax(q)); j = np.clip(i, 1, len(p) - 2)
        c = np.polyfit(p[j - 1:j + 2], q[j - 1:j + 2], 2)
        po = -c[1] / (2 * c[0]) if c[0] < 0 else p[i]
        po = float(np.clip(po, p[0], p[-1]))
        rows.append(dict(dist_km=d, method=m, p_opt_grid=p[i], q_opt_grid=q[i], p_opt_fit=po,
                         q_opt_fit=float(np.polyval(c, po)) if c[0] < 0 else q[i],
                         at_edge=(i in (0, len(p) - 1))))
    return pd.DataFrame(rows)


def gn_q(dist, p):
    """GN 模型（ASE + NLI，无 DSP 实现损伤）对 CDC 的 Q_SNR 预测。"""
    fib = fiber_params('C', 'SSMF', L=100, pmd=0)
    return q_snr(channel.gn_model(p, fib, 40e9, int(dist / 100))['snr'])


def plot_q_vs_power(s, col, fname, ylab, err):
    dists = sorted(s.dist_km.unique())
    fig, axs = plt.subplots(1, len(dists), figsize=(3.1 * len(dists), 3.4), sharey=True, squeeze=False)
    axs = axs.ravel()
    for ax, d in zip(axs, dists):
        for m in METHODS:
            t = s[(s.dist_km == d) & (s.method == m)].sort_values('p_dbm')
            if err == 'ci':
                yerr = np.vstack([t[col] - t.Q_ber_lo, np.nan_to_num(t.Q_ber_hi - t[col], nan=0)])
                few = (t.n_err < 20).values                                   # 误码 < 20：空心标记
                ax.errorbar(t.p_dbm, t[col], yerr=yerr, capsize=2, label=LABEL[m],
                            **STYLE[m])
                ax.plot(t.p_dbm[few], t[col][few], ls='none', marker=STYLE[m]['marker'], mfc='white',
                        mec=STYLE[m]['color'], ms=6, zorder=5)
            else:
                ax.errorbar(t.p_dbm, t[col], yerr=t[col + '_std'], capsize=2, label=LABEL[m], **STYLE[m])
        if err != 'ci':
            pg = np.linspace(0, 7, 71)
            ax.plot(pg, gn_q(d, pg), ':', color='0.45', lw=1.2, label='GN model (CDC)')
        ax.axhline(metrics.q_factor(HD_FEC), color='#e34948', lw=1, ls='-.', label='HD-FEC (BER 3.8e-3)')
        ax.set_title(f'{d} km ({d // 100} spans)')
        ax.set_xlabel('Launch power, dual-pol (dBm)'); ax.set_xticks(range(0, 8))
    axs[0].set_ylabel(ylab)
    axs[-1].legend(loc='lower center')
    fig.tight_layout(); fig.savefig(os.path.join(RES, fname)); plt.close(fig)


def plot_q_vs_distance(opt, s, col, fname, ylab):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for m in METHODS:
        t = opt[opt.method == m].sort_values('dist_km')
        axs[0].plot(t.dist_km, t.q_opt_grid, label=LABEL[m], **STYLE[m])
        axs[1].plot(t.dist_km, t.p_opt_fit, label=LABEL[m], **STYLE[m])
    for q, txt in ((metrics.q_factor(HD_FEC), 'HD-FEC 3.8e-3'), (metrics.q_factor(SD_FEC), 'SD-FEC 2e-2')):
        axs[0].axhline(q, color='#e34948', lw=1, ls='-.'); axs[0].text(610, q + 0.1, txt, color='#e34948', fontsize=7.5)
    axs[0].set_ylabel(ylab + ' at optimum power'); axs[1].set_ylabel('Optimum launch power (dBm)')
    axs[1].set_ylim(-0.2, 7.2)
    for ax in axs:
        ax.set_xlabel('Distance (km)'); ax.set_xticks(sorted(s.dist_km.unique()))
    axs[0].legend(loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(RES, fname)); plt.close(fig)


def plot_q_vs_distance_all_p(s, col, fname, ylab):
    """补充：各固定入纤功率下的 Q vs 距离（每个方法一个子图）。"""
    fig, axs = plt.subplots(1, len(METHODS), figsize=(3.1 * len(METHODS), 3.4), sharey=True, squeeze=False)
    axs = axs.ravel()
    cmap = plt.get_cmap('viridis')
    ps = sorted(s.p_dbm.unique())
    for ax, m in zip(axs, METHODS):
        for k, p in enumerate(ps):
            t = s[(s.method == m) & (s.p_dbm == p)].sort_values('dist_km')
            ax.plot(t.dist_km, t[col], marker='o', ms=4, color=cmap(k / max(len(ps) - 1, 1) * 0.9), label=f'{p} dBm')
        ax.axhline(metrics.q_factor(HD_FEC), color='#e34948', lw=1, ls='-.')
        ax.set_title(LABEL[m]); ax.set_xlabel('Distance (km)'); ax.set_xticks([600, 1200, 1800])
    axs[0].set_ylabel(ylab); axs[-1].legend(ncol=2, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(RES, fname)); plt.close(fig)


def plot_gain(opt, fname):
    """DBP 在最优功率处相对 CDC 的 Q 增益 vs 步长；以及 Q_BER 与 Q_SNR 一致性。"""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    dists = sorted(opt.dist_km.unique())
    cm = plt.get_cmap('viridis')
    for k, d in enumerate(dists):
        t = opt[opt.dist_km == d].set_index('method')
        g = [t.loc[f'DBP-{x}km', 'q_opt_fit'] - t.loc['CDC', 'q_opt_fit'] for x in (20, 10, 5, 1)]
        ax.plot([20, 10, 5, 1], g, marker='o', color=cm(k / max(len(dists) - 1, 1) * 0.9), label=f'{d} km')
    ax.set_xscale('log'); ax.set_xticks([1, 5, 10, 20]); ax.set_xticklabels(['1', '5', '10', '20'])
    ax.invert_xaxis(); ax.set_xlabel('DBP step size (km)'); ax.set_ylabel('ΔQ over CDC at optimum (dB)')
    ax.legend(title='Distance')
    fig.tight_layout(); fig.savefig(os.path.join(RES, fname)); plt.close(fig)


def plot_consistency(s, fname):
    t = s[s.n_err >= 100]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for m in METHODS:
        u = t[t.method == m]
        ax.plot(u.Q_snr, u.Q_ber, ls='none', marker=STYLE[m]['marker'], mfc='none', mec=STYLE[m]['color'],
                label=LABEL[m])
    lim = [t[['Q_snr', 'Q_ber']].min().min() - 0.3, t[['Q_snr', 'Q_ber']].max().max() + 0.3]
    ax.plot(lim, lim, 'k-', lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Q from SNR, Gaussian assumption (dB)'); ax.set_ylabel('Q from counted BER (dB)')
    ax.set_title('Points with ≥100 pooled errors', fontsize=9); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(RES, fname)); plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(RES, 'metrics.csv'))
    df = df.drop_duplicates(['dist_km', 'p_dbm', 'seed', 'method'], keep='last')
    s = summarize(df)
    s.to_csv(os.path.join(RES, 'summary_by_point.csv'), index=False, float_format='%.5g')
    plot_q_vs_power(s, 'Q_ber', 'q_vs_power_ber.png', 'Q factor from BER (dB)', 'ci')
    plot_q_vs_power(s, 'Q_snr', 'q_vs_power_snr.png', 'Q factor from SNR (dB)', 'std')
    ob, os_ = optimum(s, 'Q_ber'), optimum(s, 'Q_snr')
    plot_q_vs_distance(ob, s, 'Q_ber', 'q_vs_distance_ber.png', 'Q from BER (dB)')
    plot_q_vs_distance(os_, s, 'Q_snr', 'q_vs_distance_snr.png', 'Q from SNR (dB)')
    plot_q_vs_distance_all_p(s, 'Q_ber', 'q_vs_distance_all_power_ber.png', 'Q factor from BER (dB)')
    plot_gain(os_, 'dbp_gain_vs_step.png')
    plot_consistency(s, 'q_ber_vs_q_snr.png')
    opt = ob.merge(os_, on=['dist_km', 'method'], suffixes=('_ber', '_snr'))
    opt.to_csv(os.path.join(RES, 'optimum.csv'), index=False, float_format='%.4g')
    pd.set_option('display.width', 200)
    print(f'seeds per point: {s.n_seed.min()}–{s.n_seed.max()}, points: {len(s)}')
    print(opt.round(2).to_string(index=False))
    for d in sorted(s.dist_km.unique()):
        g = channel.gn_model(0, fiber_params('C', 'SSMF', L=100, pmd=0), 40e9, d // 100)
        print(f'GN {d} km: P_opt {g["p_opt_dbm"]:.2f} dBm, SNR_opt {g["snr_opt"]:.2f} dB, Q {q_snr(g["snr_opt"]):.2f} dB')


if __name__ == '__main__':
    main()
