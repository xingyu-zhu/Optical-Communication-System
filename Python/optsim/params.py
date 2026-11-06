"""波段/光纤参数加载：读取共用参数文件 params/fiber_bands.json，换算出指定波段与光纤的传输参数。

示例
----
>>> from optsim.params import fiber_params
>>> fib = fiber_params('C', 'SSMF', L=80)           # C 波段中心 1550 nm 的 SSMF
>>> fib = fiber_params('L', 'ULL', lam=1600)         # 指定波长
>>> fib = fiber_params('C', 'SSMF', pmd=0, raman=True)  # 覆盖任意字段
>>> fib = fiber_params('C', 'HCF', L=100)            # 反谐振空芯光纤（低损耗、低非线性、低时延）
"""
import json
import os
from functools import lru_cache
import numpy as np
from .utils import C

PARAM_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'params', 'fiber_bands.json')


@lru_cache(maxsize=None)
def load(path=PARAM_FILE):
    """读取参数文件，返回 dict（含 bands / fibers / raman / wdm）。"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def dispersion(lam, lambda0, S0):
    """G.652 三项 Sellmeier 色散模型。

    参数: lam, lambda0 [nm]; S0 零色散斜率 [ps/(nm^2 km)]
    返回: (D [ps/(nm km)], S [ps/(nm^2 km)])
    """
    D = S0 / 4 * (lam - lambda0 ** 4 / lam ** 3)
    S = S0 / 4 * (1 + 3 * lambda0 ** 4 / lam ** 4)
    return D, S


def fiber_params(band='C', fiber='SSMF', L=80.0, lam=None, **override):
    """生成某波段、某光纤类型的完整传输参数 dict（可直接传给 channel.ssfm / link / dsp.dbp）。

    参数
    ----
    band   : 'O' | 'E' | 'S' | 'C' | 'L' | 'U'
    fiber  : 'SSMF' | 'ULL' | 'NZDSF' | 'DSF' | 'HCF'（见 JSON，可自行添加）
    L      : 跨段长度 [km]
    lam    : 工作波长 [nm]，默认取波段中心
    override : 覆盖任意输出字段，如 pmd=0, gamma=1.0, raman=True, steep=True

    返回 dict 字段
    ----
    band, fiber, lam [nm], f0 [Hz], L [km], alpha [dB/km], D [ps/(nm km)], S [ps/(nm^2 km)],
    gamma [1/(W km)], aeff [um^2], n2 [m^2/W], pmd [ps/√km], nf [dB], amplifier,
    raman (bool, 是否计入受激拉曼散射), steep (bool, 是否计入自陡峭), fR, tau1, tau2 [s],
    ng 群折射率, latency [µs/km] 单程时延 = ng/c

    光纤条目的可选字段（如 HCF）：alpha_db_km 绝对衰减（不叠加石英波段衰减）；D1550/S1550 线性色散
    D(λ) = D1550 + S1550·(λ−1550)；ng 群折射率（默认 1.468）；medium='air' 时 fR = 0（空气中无石英拉曼响应）
    """
    p = load()
    b, f, r = p['bands'][band], p['fibers'][fiber], p['raman']
    lam = b['center_nm'] if lam is None else lam
    if 'D1550' in f:                                         # 线性色散模型（空芯光纤）
        D, S = f['D1550'] + f['S1550'] * (lam - 1550), f['S1550']
    else:
        D, S = dispersion(lam, f['lambda0_nm'], f['S0'])
    aeff = f['aeff_um2'] * (lam / 1550) ** f['aeff_exp']
    alpha = f['alpha_db_km'] if 'alpha_db_km' in f else b['alpha_db_km'] + f['alpha_offset_db_km']
    ng = f.get('ng', 1.468)
    out = dict(band=band, fiber=fiber, lam=lam, f0=C / (lam * 1e-9), L=L,
               alpha=alpha, D=D, S=S,
               gamma=2 * np.pi * f['n2'] / (lam * 1e-9 * aeff * 1e-12) * 1e3,
               aeff=aeff, n2=f['n2'], pmd=f['pmd'], nf=b['nf_db'], amplifier=b['amplifier'],
               raman=False, steep=False, fR=0.0 if f.get('medium') == 'air' else r['fR'],
               tau1=r['tau1_fs'] * 1e-15, tau2=r['tau2_fs'] * 1e-15, ng=ng, latency=ng / C * 1e9)
    out.update(override)
    return out


def band_table(fiber='SSMF'):
    """打印各波段中心波长处的参数表，便于快速查看。"""
    print(f"{'band':4s} {'λ(nm)':>7s} {'α':>6s} {'D':>7s} {'S':>6s} {'γ':>6s} {'Aeff':>6s} {'NF':>4s}  amplifier")
    for bd in load()['bands']:
        q = fiber_params(bd, fiber)
        print(f"{bd:4s} {q['lam']:7.0f} {q['alpha']:6.3f} {q['D']:7.2f} {q['S']:6.3f} {q['gamma']:6.3f} "
              f"{q['aeff']:6.1f} {q['nf']:4.1f}  {q['amplifier']}")
