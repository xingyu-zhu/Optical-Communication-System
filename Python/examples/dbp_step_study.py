"""DP-16QAM 单信道长距离传输：DBP 步长研究（数据生成 + 接收 DSP）。

链路：AWG 90 GSa/s -> IQ 调制（100 kHz 线宽）-> 功放 EDFA（双偏振总入纤功率 P）
      -> N × [100 km SSMF + EDFA 完全补偿跨段损耗（NF 5 dB）] -> 相干接收（100 kHz LO）-> 示波器 80 GSa/s
DSP ：CDC 或 DBP（每跨 5/10/20/100 步，即步长 20/10/5/1 km）-> 匹配滤波 -> 定时 -> CMA/RDE -> FOE -> BPS -> 判决

扫描：距离 600/900/1200/1500/1800 km × 功率 0..7 dBm（步进 1 dB）× 10 组独立随机实现（数据、ASE、相位噪声）。

前向光纤用单精度、均匀 100 m 步长的对称 SSFM（比最细的 DBP 步长 1 km 再细 10 倍，作为"真实信道"），
不计 PMD / SRS / 自陡峭，使 DBP 能补偿的确定性 Kerr 效应成为唯一的非线性损伤。

运行（Raymond 环境，项目 Python/ 目录下）：
    python examples/dbp_step_study.py --workers 14 (数据默认写到项目 Data/DP16QAM_40GBd_DBP_step)
    python examples/dbp_step_study.py --quick          # 只跑 1 个点试跑，检查耗时
输出：每个 (距离, 功率, 种子) 一个 rx_*.npz（示波器波形），每个种子一个 tx_*.npz（AWG 波形、帧、比特），
     results/dbp_step_study/metrics.csv（每条 DSP 链的 BER / 误码数 / SNR / EVM / GMI）。
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                           # 每个进程单线程，并行由进程池完成
import sys, time, argparse, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import scipy.fft as sfft
from optsim import devices as dev, metrics
from optsim.channel import _disp_phase
from optsim.system import Config, tx_signal, rx_dsp
from optsim.utils import freq, resample

# ---------------- 研究参数 ----------------
DIST_KM = [600, 900, 1200, 1500, 1800]
POWERS = list(range(0, 8))                  # 双偏振总入纤功率 [dBm]
SEEDS = list(range(1, 11))                  # 10 组独立实现
DBP_STEP_KM = [20, 10, 5, 1]                # DBP 步长 -> 每跨步数 = 100 / 步长
SPAN_KM = 100.0
FS_AWG, FS_SCOPE = 90e9, 80e9
DZ_FWD_KM = 0.1                             # 前向 SSFM 步长
CFG = Config(M=16, rs=40e9, beta=0.1, sps=4, n_blocks=960, train_len=2048, pilot_period=32,
             band='C', fiber='SSMF', L=SPAN_KM, fib_kw={'pmd': 0}, linewidth=100e3, fs_adc=FS_SCOPE)
RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'dbp_step_study')


def ssfm_fast(E, fs, fib, dz_km, nsteps):
    """单跨对称 SSFM（Manakov，均匀步长，单精度，相邻半步色散合并）。物理模型与 channel.ssfm(nsteps=...) 相同。"""
    a = fib['alpha'] * np.log(10) / 10 / 1e3
    w = 2 * np.pi * freq(E.shape[-1], fs)
    dz = dz_km * 1e3
    Hh = np.exp((-a / 2 - 1j * _disp_phase(w, fib['D'], fib['S'], fib['lam'])) * dz / 2).astype(np.complex64)
    Hf = Hh * Hh
    g = np.float32(fib['gamma'] / 1e3 * 8 / 9 * dz)
    X = sfft.fft(E.astype(np.complex64), axis=-1) * Hh
    for i in range(nsteps):
        E = sfft.ifft(X, axis=-1)
        E *= np.exp(-1j * g * (E.real ** 2 + E.imag ** 2).sum(0))
        X = sfft.fft(E, axis=-1)
        X *= Hf if i < nsteps - 1 else Hh
    return sfft.ifft(X, axis=-1)


def transmit(ref, n_spans, p_dbm, rng):
    """AWG -> 调制 -> 功放 -> 多跨链路 -> 相干接收 -> 示波器。返回 (awg 波形, 示波器波形)。"""
    cfg, fib = CFG, CFG.fib
    awg = resample(ref['wave'], cfg.fs, FS_AWG)              # AWG 90 GSa/s 采样点（理想 DAC，无量化/带宽限制）
    x = resample(awg, FS_AWG, cfg.fs)                        # DAC 重建 -> 仿真采样率 160 GSa/s（sps = 4）
    N = x.shape[1]
    E = x * dev.laser(N, cfg.fs, 0, cfg.linewidth, rng=rng)
    E = dev.edfa(E, cfg.fs, p_out_dbm=p_dbm, nf_db=fib['nf'], f0=fib['f0'], rng=rng)
    G, ns = fib['alpha'] * fib['L'], int(round(SPAN_KM / DZ_FWD_KM))
    for _ in range(n_spans):
        E = ssfm_fast(E, cfg.fs, fib, DZ_FWD_KM, ns).astype(complex)
        E = dev.edfa(E, cfg.fs, gain_db=G, nf_db=fib['nf'], f0=fib['f0'], rng=rng)
    lo = dev.laser(N, cfg.fs, 10, cfg.linewidth, rng=rng)
    r = dev.coherent_rx(E, lo, cfg.fs, shot=False)
    scope = dev.adc(r, cfg.fs, FS_SCOPE, enob=None)          # 示波器 80 GSa/s（理想量化，砖墙抗混叠）
    return awg, scope


def methods():
    """DSP 方案列表：(名称, nl_comp, 每跨步数)。"""
    return [('CDC', 'cdc', 0)] + [(f'DBP-{s}km', 'dbp', int(SPAN_KM / s)) for s in DBP_STEP_KM]


def run_job(job):
    dist, p, seed, out = job
    n_spans = int(dist / SPAN_KM)
    t0 = time.time()
    ref = tx_signal(replace(CFG, seed=seed))
    rng = np.random.default_rng([seed, dist, p + 100])
    awg, scope = transmit(ref, n_spans, p, rng)
    if out:
        np.savez(os.path.join(out, f'rx_{dist:04d}km_p{p}dBm_s{seed:02d}.npz'), scope=scope.astype(np.complex64),
                 dist_km=dist, p_launch_dbm=p, seed=seed, fs=FS_SCOPE)
        ftx = os.path.join(out, f'tx_s{seed:02d}.npz')
        if dist == DIST_KM[-1] and p == POWERS[0]:           # 每个种子只写一次（发射数据与距离/功率无关）
            np.savez(ftx, awg=awg.astype(np.complex64), frame=ref['frame'].astype(np.complex64), kind=ref['kind'],
                     bits=np.packbits(ref['bits'].astype(np.uint8), axis=-1), n_bits=ref['bits'].shape[1],
                     const=ref['const'], seed=seed, fs_awg=FS_AWG, rs=CFG.rs, beta=CFG.beta)
    t1 = time.time()
    rows = []
    for name, nl, st in methods():
        cfg = replace(CFG, seed=seed, n_spans=n_spans, p_launch=p, nl_comp=nl, dbp_steps=max(st, 1))
        r = rx_dsp(scope, cfg, ref)
        nerr = int(round(r['ber'] * ref['bits'].size))
        rows.append(dict(dist_km=dist, p_dbm=p, seed=seed, method=name, ber=r['ber'], n_err=nerr,
                         n_bits=ref['bits'].size, snr_db=r['snr'], evm=r['evm'], gmi=r['gmi'],
                         ber_x=r['ber_xy'][0], ber_y=r['ber_xy'][1]))
    t2 = time.time()
    return rows, t1 - t0, t2 - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'DP16QAM_40GBd_DBP_step'))
    ap.add_argument('--quick', action='store_true', help='只跑 1800 km / 4 dBm / seed 1，检查耗时与结果')
    a = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    if a.quick:
        rows, tf, td = run_job((1800, 4, 1, None))
        for r in rows:
            print(f"{r['method']:10s} BER {r['ber']:.2e} ({r['n_err']} err)  SNR {r['snr_db']:.2f} dB  "
                  f"Q {metrics.q_factor(max(r['ber'], 1e-12)):.2f} dB")
        print(f'forward {tf:.1f} s, DSP {td:.1f} s')
        return
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(RES_DIR, 'metrics.csv')
    done = set()
    if os.path.exists(csv_path):                             # 断点续跑：跳过已完成的点
        with open(csv_path, newline='') as f:
            done = {(int(r['dist_km']), int(r['p_dbm']), int(r['seed'])) for r in csv.DictReader(f)}
    jobs = [(d, p, s, a.out) for d in sorted(DIST_KM, reverse=True) for p in POWERS for s in SEEDS
            if (d, p, s) not in done]                        # 长距离优先，均衡负载
    print(f'{len(jobs)} jobs, {a.workers} workers, data -> {a.out}', flush=True)
    keys = ['dist_km', 'p_dbm', 'seed', 'method', 'ber', 'n_err', 'n_bits', 'snr_db', 'evm', 'gmi', 'ber_x', 'ber_y']
    new = not os.path.exists(csv_path)
    t0 = time.time()
    with open(csv_path, 'a', newline='') as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, keys)
        if new:
            w.writeheader()
        for i, (rows, tf, td) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            w.writerows(rows); f.flush()
            r0 = rows[0]
            print(f"[{i}/{len(jobs)}] {r0['dist_km']} km {r0['p_dbm']} dBm s{r0['seed']}: "
                  + ', '.join(f"{r['method']} {r['snr_db']:.2f}" for r in rows)
                  + f" dB | fwd {tf:.0f}s dsp {td:.0f}s | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
