"""光纤传感：瑞利（OTDR / φ-OTDR / OFDR）、布里渊（BOTDA）、拉曼（DTS）、光纤布拉格光栅（FBG）。

约定：位置 z [m]；光纤离散为间距 dz 的散射单元，单元 i 的往返时延为 2·ng·z_i/c。
取 dz = c/(2·ng·fs)（cell_len）时，第 i 个单元恰好延迟 i 个采样，背向散射即探测光与 h 的卷积。
传感系数为石英单模光纤 @1550 nm 的典型值（量级可靠，具体光纤需要标定）。
"""
import numpy as np
from scipy.signal import fftconvolve, find_peaks
from scipy.ndimage import uniform_filter1d, median_filter
from .utils import C, H, KB, F0, dbm2w, lpf
from .dsp import _movmean

NG = 1.468                       # 群折射率
ALPHA_S, CAPTURE = 4.6e-5, 1.5e-3   # 瑞利散射系数 [1/m]、背向捕获因子 S


def cell_len(fs, ng=NG):
    """采样率 fs 对应的散射单元长度 dz = c/(2·ng·fs) [m]。"""
    return C / (2 * ng * fs)


# ======================= 瑞利：φ-OTDR / DAS =======================
def rayleigh_fiber(L, dz, alpha_db=0.2, rng=None):
    """随机瑞利散射单元：r_i ~ CN(0, αs·S·dz)，乘往返损耗 e^{-αz}（α 为功率衰减系数）。返回 (z, r)。"""
    rng = np.random.default_rng(rng)
    z = np.arange(int(L / dz)) * dz
    a = alpha_db * np.log(10) / 10 / 1e3
    r = (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z))) * np.sqrt(ALPHA_S * CAPTURE * dz / 2)
    return z, r * np.exp(-a * z)


def backscatter(p, h, n=None):
    """背向散射光场 = 探测光 p 与散射冲激响应 h 的线性卷积（沿最后一维；p、h 可为 (K, ·) 批量，一维 p 对所有行共用）。
    h 一般为 r·exp(jφ)，φ 为扰动引入的往返相位。n 给定时只取前 n 点。"""
    if np.ndim(p) == 1:
        p = np.reshape(p, (1,) * (np.ndim(h) - 1) + (-1,))
    y = fftconvolve(p, h, axes=-1)
    return y[..., :n] if n else y


def lfm(T, B, fs):
    """线性调频（chirp）脉冲复包络：时长 T [s]，频率由 -B/2 线性扫到 B/2。"""
    t = np.arange(int(round(T * fs))) / fs - T / 2
    return np.exp(1j * np.pi * B / T * t ** 2)


def pulse_compress(x, p):
    """脉冲压缩（匹配滤波）：x 与探测脉冲 p 互相关，输出与 x 等长，峰值对齐到散射单元位置。
    LFM 脉冲压缩后空间分辨率 ≈ c/(2·ng·B)，与脉宽无关。"""
    q = np.conj(p[::-1]).reshape((1,) * (np.ndim(x) - 1) + (-1,))
    return fftconvolve(x, q, axes=-1)[..., len(p) - 1:len(p) - 1 + x.shape[-1]] / np.sum(np.abs(p) ** 2)


def ddc(x, fs, f_if, bw):
    """数字下变频：外差探测（如 AOM 移频 f_if）的实电流 → 乘 e^{-j2πf_if·t} → 低通 bw → 复基带。"""
    return lpf(x * np.exp(-2j * np.pi * f_if * np.arange(x.shape[-1]) / fs), fs, bw)


def das_phase(tr, G, n_avg=1):
    """DAS 差分相位解调：标距 G（单元数）两端的相位差 ∠(x[z+G]·x*[z])，沿距离 n_avg 点平均
    （抑制相干衰落），沿慢时间解卷绕并去均值。tr: (K, N) 复轨迹（K 次测量 × N 个单元）。返回 (K, N-G) [rad]。"""
    d = tr[:, G:] * np.conj(tr[:, :-G])
    ph = np.unwrap(np.angle(_movmean(d, n_avg) if n_avg > 1 else d), axis=0)
    return ph - ph.mean(0)


def vib_locate(dphi, n=1):
    """振动定位：各位置差分相位的时间标准差，经 n 点中值滤波（去掉衰落点伪峰）后取最大值。返回 (下标, 能量曲线)。"""
    e = median_filter(dphi.std(0), n) if n > 1 else dphi.std(0)
    return int(np.argmax(e)), e


# ======================= 瑞利：传统 OTDR =======================
def otdr_trace(z, p_dbm, W, alpha_db=0.2, events=(), ng=NG):
    """传统 OTDR 背向散射功率轨迹 [W]（宽谱光源、多次平均，无相干衰落）：
    P(z) = P0·(c·W/2ng)·αs·S·e^{-2αz}；events = [(位置 m, 单程损耗 dB), ...] 之后乘往返损耗；最后按脉宽平滑。"""
    a = alpha_db * np.log(10) / 10 / 1e3
    p = dbm2w(p_dbm) * C * W / (2 * ng) * ALPHA_S * CAPTURE * np.exp(-2 * a * z)
    for ze, loss in events:
        p = p * np.where(z >= ze, 10 ** (-2 * loss / 10), 1.0)
    return uniform_filter1d(p, max(int(C * W / (2 * ng) / (z[1] - z[0])), 1))


def otdr_events(p_db, dz, win, thr=0.1):
    """OTDR 轨迹分析（p_db = 10·lg P，往返量）。
    衰减：平滑后斜率的中位数（事件只占少数点，中位数稳健）；
    事件：去斜后左右各 win 点均值之差为往返台阶，峰值 > 2·thr 判为事件。
    返回 (衰减 [dB/km], 事件下标, 单程损耗 [dB])。"""
    slope = np.median(np.gradient(uniform_filter1d(p_db, win), dz))
    cs = np.r_[0, np.cumsum(p_db - slope * np.arange(len(p_db)) * dz)]
    i = np.arange(win, len(p_db) - win)
    step = np.zeros(len(p_db))
    step[i] = (cs[i] - cs[i - win]) / win - (cs[i + win] - cs[i]) / win
    idx = find_peaks(step, height=2 * thr, distance=win)[0]
    return -slope / 2 * 1e3, idx, step[idx] / 2


# ======================= 布里渊：BOTDA =======================
def bfs(dT=0.0, eps=0.0, vb0=10.85e9, cT=1.0e6, ce=0.05e6):
    """布里渊频移 BFS = vb0 + cT·ΔT + cε·ε [Hz]（ΔT [K]，ε [με]；典型 cT≈1 MHz/K，cε≈0.05 MHz/με）。"""
    return vb0 + cT * np.asarray(dT) + ce * np.asarray(eps)


def brillouin_gain(f, vb, dvb=30e6):
    """归一化洛伦兹布里渊增益谱 g = 1/(1 + (2(f-vb)/Δν_B)²)，Δν_B 为 FWHM（约 30 MHz）。"""
    return 1 / (1 + (2 * (f - vb) / dvb) ** 2)


def botda(vb, f_scan, n_pulse, dvb=30e6):
    """BOTDA 小增益模型：位置 z 处的探测光增益 = 泵浦脉冲覆盖的 n_pulse 个单元局部增益之平均。
    vb: 各单元 BFS [Hz]; f_scan: 泵浦-探测频差扫描 [Hz]。返回 (len(f_scan), len(vb)) 增益谱分布 BGS(f, z)。"""
    return uniform_filter1d(brillouin_gain(f_scan[:, None], vb[None], dvb), n_pulse, axis=1)


def bfs_fit(f, g, frac=0.5):
    """逐位置提取 BFS：取峰值 frac 倍以上的点做二次拟合，返回顶点频率（洛伦兹拟合的快速近似）。"""
    fc = (f - f.mean()) / 1e6               # 中心化并以 MHz 为单位，避免病态
    out = np.empty(g.shape[1])
    for j in range(g.shape[1]):
        m = g[:, j] >= frac * g[:, j].max()
        a, b, _ = np.polyfit(fc[m], g[m, j], 2)
        out[j] = -b / (2 * a) * 1e6
    return out + f.mean()


# ======================= 拉曼：DTS =======================
def raman_ratio(T, dv=13.2e12, lam=1550.0):
    """反斯托克斯/斯托克斯强度比 R(T) = (λs/λas)⁴·exp(-hΔν/kT)，Δν 为拉曼频移（约 13.2 THz）。"""
    f0 = C / (lam * 1e-9)
    return ((f0 + dv) / (f0 - dv)) ** 4 * np.exp(-H * dv / (KB * np.asarray(T)))


def dts_temp(R, R_ref, T_ref, z=0.0, dalpha=0.0, dv=13.2e12):
    """由强度比反演温度（参考段标定，消去未知比例常数）：
    1/T = 1/T_ref − (k/hΔν)·ln(R·e^{Δα·z}/R_ref)，Δα = α_as − α_s 为两波长往返损耗差 [1/m]。"""
    return 1 / (1 / T_ref - KB / (H * dv) * np.log(R * np.exp(dalpha * z) / R_ref))


# ======================= FBG =======================
def fbg_reflect(lam, lam_b, L=10e-3, dn=1e-4, neff=1.447):
    """均匀 FBG 反射谱（耦合模理论解析解，Erdogan 1997）：
    R = sinh²(sL) / (cosh²(sL) − σ²/κ²)，s = √(κ²−σ²)，κ = π·dn/λ，σ = 2π·neff·(1/λ − 1/λ_B)。
    lam、lam_b [nm]，L 栅长 [m]，dn 折射率调制深度。"""
    l = np.asarray(lam) * 1e-9
    k = np.pi * dn / l
    sg = 2 * np.pi * neff * (1 / l - 1 / (lam_b * 1e-9))
    s = np.sqrt(k ** 2 - sg ** 2 + 0j)
    return np.abs(np.sinh(s * L) ** 2 / (np.cosh(s * L) ** 2 - sg ** 2 / k ** 2))


def fbg_shift(lam_b, dT=0.0, eps=0.0, pe=0.22, kT=7e-6):
    """FBG 中心波长漂移 Δλ = λ_B·[(1−pe)·ε + kT·ΔT] [nm]（ε [με]，kT = 热膨胀 + 热光系数）。
    典型 @1550 nm：约 1.2 pm/με、11 pm/K。"""
    return lam_b * ((1 - pe) * np.asarray(eps) * 1e-6 + kT * np.asarray(dT))


def fbg_peak(lam, R, frac=0.5):
    """FBG 峰值解调：反射谱 frac·max 以上部分的质心 [nm]。"""
    m = R >= frac * R.max()
    return np.sum(lam[m] * R[m]) / np.sum(R[m])


# ======================= 瑞利：OFDR =======================
def ofdr_scan(r, dz, eps=0.0, K=0.78, f0=F0, ng=NG):
    """OFDR 一次扫频的复拍频信号（I/Q 相干探测，已线性化到等光频间隔）。
    扫频范围 B = c/(2·ng·dz)，点数 = 单元数；s(ν_k) = Σ_i h_i·e^{-j2πν_kτ_i} 即 h 的 FFT，拍频频率 ∝ 距离。
    应变 ε(z) [με] 使局部光程伸长 K·ε（K≈0.78），累积附加时延 δτ 以 e^{-j2πf0·δτ} 计入（忽略 δτ 随扫频的变化）。"""
    dtau = K * np.cumsum(np.broadcast_to(eps, r.shape)) * 1e-6 * dz * 2 * ng / C
    return np.fft.fft(r * np.exp(-2j * np.pi * f0 * dtau))


def ofdr_shift(s_ref, s_meas, nw, B, pad=8):
    """OFDR 应变/温度解调：两次扫描的距离域剖面按 nw 个单元分段，各段 FFT（补零 pad 倍）得局部瑞利谱，
    互相关求谱移。返回 (各段中心单元下标, 谱移 Δν [Hz])；Δν ≈ −f0·(K·ε + K_T·ΔT)。"""
    p1, p2 = np.fft.ifft(s_ref), np.fft.ifft(s_meas)
    n = len(p1) // nw
    spec = lambda p: np.abs(np.fft.fft(p[:n * nw].reshape(n, nw), nw * pad, axis=1))
    a, b = spec(p1), spec(p2)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    xc = np.abs(np.fft.ifft(np.fft.fft(b, axis=1) * np.conj(np.fft.fft(a, axis=1)), axis=1))
    m = np.argmax(xc, 1)
    m = np.where(m >= nw * pad // 2, m - nw * pad, m)
    return np.arange(n) * nw + nw // 2, m * B / (nw * pad)
