"""系统级封装：统一参数 Config、单信道发射 tx_signal、WDM 收发 wdm_tx / wdm_rx、接收 DSP 链 rx_dsp；
IM-DD：ImddConfig、imdd_tx、imdd_link、imdd_rx。

最小用法
--------
>>> cfg = Config(M=16, rs=40e9, band='C', fiber='SSMF', n_spans=10, p_launch=1)
>>> ref = tx_signal(cfg)                          # 发射符号与波形
>>> ...                                           # 器件 + 链路（见 examples/）
>>> res = rx_dsp(adc_output, cfg, ref)            # -> res['ber'], res['snr'], res['gmi'] ...
"""
from dataclasses import dataclass, field, replace
import numpy as np
from . import tx, dsp, metrics, devices as dev
from .params import fiber_params
from .utils import normalize, resample, set_power


@dataclass
class Config:
    # ---- 信号 ----
    M: int = 16                 # 调制阶数
    mod: str = 'qam'            # 'qam' | 'psk'（相干 M-PSK：BPSK/QPSK/8PSK/16PSK）
    rs: float = 40e9            # 符号率 [Bd]
    beta: float = 0.1           # RRC 滚降
    sps: int = 4                # 仿真过采样（fs = sps·rs；WDM 需覆盖全部信道，如 16）
    n_blocks: int = 960         # 导频块数（每块 pilot_period 个符号，首个为导频）
    train_len: int = 2048       # 训练序列长度
    pilot_period: int = 32
    seed: int = 1
    # ---- 链路（光纤参数来自 params/fiber_bands.json）----
    band: str = 'C'             # O / E / S / C / L / U
    fiber: str = 'SSMF'         # SSMF / ULL / NZDSF / DSF
    L: float = 80.0             # 跨段长度 [km]
    n_spans: int = 10
    p_launch: float = 0.0       # 每信道入纤功率 [dBm]
    fib_kw: dict = field(default_factory=dict)   # 覆盖光纤参数，如 {'pmd': 0, 'raman': True}
    phi_max: float = 0.005      # SSFM 自适应步长：每步最大非线性相移 [rad]
    # ---- WDM ----
    n_ch: int = 1               # 信道数（奇数，中心信道为被测信道）
    spacing: float = 50e9       # 信道间隔 [Hz]
    linewidth: float = 100e3    # 激光线宽 [Hz]
    # ---- 接收 DSP ----
    fs_adc: float = 100e9       # ADC 采样率
    nl_comp: object = 'cdc'     # 'cdc' | 'dbp' | 'none' | 可调用对象（如训练好的 LDBP.apply）
    dbp_steps: int = 4          # DBP 每跨步数
    timing: str = 'ff'          # 'ff' | 'gardner' | 'none'
    ntaps: int = 15             # MIMO 抽头数
    mu_cma: float = 1e-3
    mu_rde: float = 3e-4
    cpr: str = 'bps'            # 'bps' | 'vv' | 'pilot'
    dd_lms: bool = False        # CPR 后 1 sps 2x2 DD-LMS 精均衡

    @property
    def fs(self):
        """仿真采样率 [Hz]。"""
        return self.rs * self.sps

    @property
    def fib(self):
        """当前波段/光纤的参数 dict（见 params.fiber_params）。"""
        return fiber_params(self.band, self.fiber, L=self.L, **self.fib_kw)


def tx_signal(cfg):
    """发射端 DSP：比特 -> Gray QAM / PSK（cfg.mod）-> 成帧 [训练|导频+数据] -> RRC 成形。

    返回 dict: const 星座, bits (2, nbits) 数据比特, frame (2, Nf) 帧符号, kind (Nf,) 符号类型
               (0 数据/1 训练/2 导频), wave (2, Nf·sps) 每偏振单位功率波形
    """
    rng = np.random.default_rng(cfg.seed)
    c = tx.constellation(cfg.mod, cfg.M)
    k = int(np.log2(cfg.M))
    nd = cfg.n_blocks * (cfg.pilot_period - 1)
    bits = tx.rand_bits((2, nd * k), rng)
    train = c[rng.integers(0, cfg.M, (2, cfg.train_len))]
    pilot = c[rng.integers(0, cfg.M, (2, cfg.n_blocks))]
    f, kind = tx.frame(tx.modulate(bits, c), train, pilot, cfg.pilot_period)
    return dict(const=c, bits=bits, frame=f, kind=kind, wave=normalize(tx.pulse_shape(f, cfg.sps, cfg.beta)))


def wdm_tx(cfg, rng=None):
    """WDM 发射机：cfg.n_ch 个独立 DP-QAM 信道（各自数据、各自激光相位噪声，理想调制），
    以 cfg.spacing 等间隔排布在中心频率两侧，每信道功率 cfg.p_launch。

    返回: (E (2, N) 合波光场, refs 各信道 tx_signal 结果列表, freqs 各信道频率偏移 [Hz])
    """
    rng = np.random.default_rng(rng)
    freqs = (np.arange(cfg.n_ch) - (cfg.n_ch - 1) / 2) * cfg.spacing
    refs = [tx_signal(replace(cfg, seed=cfg.seed + k)) for k in range(cfg.n_ch)]
    N = refs[0]['wave'].shape[1]
    fields = [set_power(r['wave'] * dev.laser(N, cfg.fs, 0, cfg.linewidth, rng=rng), cfg.p_launch) for r in refs]
    return dev.wdm_mux(fields, freqs, cfg.fs), refs, freqs


def wdm_rx(E, cfg, f_ch, fo=0.0, rng=None):
    """WDM 接收单个信道：WSS 选出 f_ch 信道 -> 本振调谐到 f_ch+fo -> 相干接收 -> ADC（理想量化）。

    返回: (2, N') ADC 输出 @cfg.fs_adc，可直接送入 rx_dsp
    """
    rng = np.random.default_rng(rng)
    Ec = dev.optical_filter(E, cfg.fs, bw=cfg.spacing, order=4, fc=f_ch)
    lo = dev.laser(E.shape[1], cfg.fs, 10, cfg.linewidth, f_offset=f_ch + fo, rng=rng)
    r = dev.coherent_rx(Ec, lo, cfg.fs, shot=False, rng=rng)
    return dev.adc(r, cfg.fs, cfg.fs_adc, enob=None, bw=0.75 * cfg.rs)


def rx_dsp(r, cfg, ref):
    """相干接收 DSP 链。

    参数: r (2, N) ADC 输出 @cfg.fs_adc; cfg Config; ref 该信道的 tx_signal 输出
    返回 dict: ber, ber_xy [X, Y], snr [dB], evm [%], gmi [bit/sym/pol], fo 频偏估计 [Hz],
              lag 帧偏移, tau 定时估计, eq_err 均衡误差, sym (2, nd) 判决前数据符号,
              y (2, Nf) 与 ref['frame'] 逐符号对齐的整帧符号（含训练/导频，供 NN 后均衡使用）
    """
    c, f, kind = ref['const'], ref['frame'], ref['kind']
    rot = tx.rot_order(c)                                    # 旋转对称阶数：QAM/QPSK 4，BPSK 2，8PSK 8
    fib, out = cfg.fib, {}
    fs2 = 2 * cfg.rs
    # 1) 前端：去直流 + GSOP 正交化 + 重采样到 2 sps
    y = resample(dsp.gsop(r), cfg.fs_adc, fs2)
    # 2) 色散 / 非线性补偿
    if cfg.nl_comp == 'cdc':
        y = dsp.cdc(y, fs2, cfg.n_spans * cfg.L, fib['D'], fib['lam'], fib['S'])
    elif cfg.nl_comp == 'dbp':
        y = dsp.dbp(y, fs2, cfg.p_launch, cfg.n_spans, fib, cfg.dbp_steps)
    elif callable(cfg.nl_comp):                              # 自定义补偿器，如训练好的 LDBP.apply
        y = cfg.nl_comp(y)
    # 3) 匹配滤波 + 定时恢复
    y = normalize(tx.matched_filter(y, 2, cfg.beta))
    if cfg.timing == 'ff':
        y, out['tau'] = dsp.timing_ff(y, 2)
    elif cfg.timing == 'gardner':
        y, out['tau'] = dsp.gardner(y, 1e-3, 2)
    # 4) 2x2 MIMO：CMA 预收敛 -> RDE
    n0 = min(10000, y.shape[1] // 2)
    _, w, _ = dsp.mimo_eq(y[:, :2 * n0], cfg.ntaps, cfg.mu_cma, 2, 'cma', c)
    y, w, out['eq_err'] = dsp.mimo_eq(y, cfg.ntaps, cfg.mu_rde, 2, 'rde', c, w=w)
    y = normalize(y)
    # 5) 频偏估计与补偿（1 sps）
    out['fo'] = dsp.foe(y, cfg.rs, m=rot)
    y = dsp.fo_comp(y, out['fo'], cfg.rs)
    # 6) 载波相位恢复 + 同步 + 模糊/周跳消除
    yb, _ = dsp.bps(y, c, rot=rot) if cfg.cpr != 'vv' else dsp.vv(y, 64, rot, c)
    yb, out['lag'], perm = dsp.sync(yb, f)
    if cfg.cpr == 'pilot':
        ya = np.roll(y[perm], -out['lag'], axis=-1)[:, :f.shape[1]]
        y, _ = dsp.pilot_cpr(ya, f, kind == 2, 5)
    else:
        y = dsp.fix_ambiguity(yb, f, kind > 0, n=5, rot=rot)   # 训练+导频辅助的周跳消除
    # 7) 可选 DD-LMS 精均衡
    if cfg.dd_lms:
        y, _, _ = dsp.mimo_eq(y, 7, 1e-4, 1, 'dd', c)
    # 8) 功率对齐、去训练/导频、判决与指标
    y = y / np.sqrt(np.mean(np.abs(y[:, kind > 0]) ** 2, -1, keepdims=True))
    d, s = y[:, kind == 0], f[:, kind == 0]
    rb = tx.demodulate(d, c)
    out.update(y=y, sym=d, ber=metrics.ber(ref['bits'], rb)[0], snr=metrics.snr(d, s),
               evm=metrics.evm(d, s), gmi=metrics.gmi(d, s, c),
               ber_xy=[metrics.ber(ref['bits'][p], rb[p])[0] for p in range(2)])
    return out



# ============================== IM-DD（PAM-M 直接检测） ==============================
@dataclass
class ImddConfig:
    """IM-DD PAM-M 短距系统参数（单跨、无光放大，默认 O 波段 2 km）。"""
    # ---- 信号 ----
    M: int = 4                  # PAM 阶数（2 即 NRZ-OOK）
    rs: float = 56e9            # 符号率 [Bd]
    sps: int = 4                # 仿真过采样
    n_sym: int = 2 ** 15        # 符号数
    n_train: int = 4096         # 均衡器训练符号数（BER 只统计其后的符号）
    beta: float = 0.2           # RC 成形滚降
    seed: int = 1
    # ---- 发射 ----
    tx: str = 'mzm'             # 'mzm'（零啁啾，正交偏置）| 'eml'（EAM，啁啾 α）| 'dml'（直调，瞬态+绝热啁啾）
    p_tx: float = 5.0           # 出纤光功率 [dBm]（MZM/EML 为激光功率；DML 为平均功率）
    swing: float = 0.4          # MZM 驱动摆幅 [Vπ]
    er_db: float = 8.0          # EML / DML 消光比 [dB]
    alpha: float = 0.0          # EML / DML 啁啾 Henry 因子
    k_ad: float = 1e12          # DML 绝热啁啾 [Hz/W]
    linewidth: float = 1e6      # 激光线宽 [Hz]
    rin_db: float = -150.0      # 激光 RIN [dB/Hz]
    dac_enob: float = 6
    dac_bw: float = 0.6         # DAC/驱动带宽（× rs）
    # ---- 链路 ----
    band: str = 'O'
    fiber: str = 'SSMF'
    L: float = 2.0              # 光纤长度 [km]
    # ---- 接收 ----
    rx: str = 'pin'             # 'pin' | 'apd'
    R: float = 0.8              # 响应度 [A/W]
    apd_M: float = 10.0         # APD 倍增因子
    rx_bw: float = 0.7          # PD/TIA 带宽（× rs）
    irnd: float = 15e-12        # TIA 输入等效噪声电流密度 [A/√Hz]
    adc_enob: float = 6
    eq: str = 'ffe'             # 'ffe'(LMS) | 'dfe' | 'volterra' | 'mlse'（FFE + 噪声白化后滤波 + MLSE）
    ntaps: int = 21             # FFE 抽头（2 sps 采样点）
    mu: float = 1e-3            # LMS 步长
    nb: int = 3                 # DFE 反馈抽头

    @property
    def fs(self):
        """仿真采样率 [Hz]。"""
        return self.rs * self.sps

    @property
    def fib(self):
        """当前波段/光纤参数（params.fiber_params）。"""
        return fiber_params(self.band, self.fiber, L=self.L)


def imdd_tx(cfg):
    """IM-DD 发射：比特 -> Gray PAM-M（电平归一化到 [-1, 1]）-> RC 成形 -> DAC -> MZM / EML / DML。

    返回 dict: levels 判决电平, bits, sym (n_sym,), E (N,) 出纤光场
    """
    rng = np.random.default_rng(cfg.seed)
    k = int(np.log2(cfg.M))
    lv = tx.pam(cfg.M) / (cfg.M - 1)
    bits = tx.rand_bits(cfg.n_sym * k, rng)
    sym = lv[tx.bits2int(bits, k)]
    u = dev.dac(tx.pulse_shape(sym, cfg.sps, cfg.beta, 'rc').real, cfg.fs, cfg.dac_enob, cfg.dac_bw * cfg.rs)
    u = u / np.max(np.abs(u))                                   # 归一化驱动到 [-1, 1]
    if cfg.tx == 'dml':
        E = dev.dml(u, cfg.fs, cfg.p_tx, cfg.er_db, cfg.alpha, cfg.k_ad, None, cfg.linewidth, cfg.rin_db, rng)
    else:
        cw = dev.laser(len(u), cfg.fs, cfg.p_tx, cfg.linewidth, rin_db=cfg.rin_db, rng=rng)
        E = dev.mzm(cw, cfg.swing * u, 1.0, -0.5) if cfg.tx == 'mzm' else dev.eml(cw, u, cfg.er_db, cfg.alpha)
    return dict(levels=lv, bits=bits, sym=sym, E=E)


def imdd_link(E, cfg):
    """短距光纤：色散（含斜率，参数随波段）+ 损耗。功率低、距离短，忽略光纤非线性。"""
    from .channel import cd
    fib = cfg.fib
    return cd(E, cfg.fs, cfg.L, fib['D'], fib['lam'], fib['S']) * 10 ** (-fib['alpha'] * cfg.L / 20)


def imdd_rx(E, cfg, ref, rop_dbm, rng=None):
    """IM-DD 接收：VOA 调到 ROP -> PIN 或 APD（+TIA 噪声）-> ADC 2 sps -> 均衡（cfg.eq）-> PAM 判决。

    返回 dict: ber, snr（均衡后电 SNR [dB]）, y（均衡输出 1 sps）, n_bits（统计比特数）
    """
    rng = np.random.default_rng(rng)
    E = dev.voa(E, 10 * np.log10(np.mean(np.abs(E) ** 2) / 1e-3) - rop_dbm)
    kw = dict(bw=cfg.rx_bw * cfg.rs, irnd=cfg.irnd, rng=rng)
    i = dev.apd(E, cfg.fs, cfg.apd_M, cfg.R, **kw) if cfg.rx == 'apd' else dev.photodiode(E, cfg.fs, cfg.R, **kw)
    y = dev.adc(i, cfg.fs, 2 * cfg.rs, cfg.adc_enob)
    lv, sym, nt = ref['levels'], ref['sym'], cfg.n_train
    y = (y - y.mean()) / y.std() * lv.std()
    if cfg.eq == 'dfe':
        ye, _ = dsp.dfe(y, sym, cfg.ntaps, cfg.nb, cfg.mu, 2, nt, lv)
    elif cfg.eq == 'volterra':
        ye, _ = dsp.volterra(y, sym, nt, (cfg.ntaps, 9, 5), 2)
    else:
        ye, _ = dsp.ffe(y, sym, cfg.ntaps, cfg.mu, 2, nt, lv)
    a = dsp.pf_mlse(ye, sym, nt, lv) if cfg.eq == 'mlse' else lv[tx.decide(ye, lv)]
    k = int(np.log2(cfg.M))
    rb = tx.int2bits(tx.decide(a[nt:], lv), k)
    return dict(ber=metrics.ber(ref['bits'][nt * k:], rb)[0], snr=metrics.snr(ye[nt:], sym[nt:]), y=ye,
                n_bits=len(rb))
