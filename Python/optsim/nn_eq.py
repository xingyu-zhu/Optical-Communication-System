"""神经网络均衡器（PyTorch）：MLP / 1D-CNN / BiLSTM，滑窗输入、输出中心符号。

适用场景（传统线性均衡器失效的非线性失真）
------------------------------------------
1. IM-DD：平方律检测 + 色散功率衰落 + 调制器/驱动非线性 → 非线性 ISI，替代 FFE / Volterra
2. 相干：CDC + MIMO + CPR 之后的残余 Kerr 非线性（SPM/IXPM/IFWM），作为后均衡替代或补充 DBP

设计要点
--------
- 输入：以第 k 个符号为中心、ntaps 个采样的窗（dsp.sym_windows），复信号拆成 [Re, Im]×偏振 通道
- 输出：中心符号；默认残差学习（网络只学 y - x_center 的修正量），收敛更快、小网络也够用
- 训练：MSE 损失 + Adam，按验证集早停并恢复最佳权重
- 评估：必须在**独立实现**（不同比特、不同噪声）上测试，否则网络可能记住序列而高估增益

用法
----
>>> from optsim.nn_eq import NNEqualizer
>>> eq = NNEqualizer('bilstm', ntaps=41, sps=1, hidden=64)
>>> eq.fit(x_train, y_train)          # x: (P, N) 复数 或 (N·sps,) 实数；y: 对应的发送符号
>>> y_hat = eq(x_test)                # 与 y 同格式
>>> eq.mults_per_symbol()             # 每输出符号实数乘法次数（复杂度对比）
"""
import copy
import os
import numpy as np
import torch
from torch import nn
from .dsp import sym_windows


def default_device():
    """计算设备：环境变量 OPTSIM_DEVICE（'cpu' / 'cuda'）优先，否则自动检测 CUDA。
    注意：实测在部分 Windows + CUDA 环境下，若另一进程（如 MATLAB GPU 训练）正在使用 CUDA，torch 初始化 CUDA 可能崩溃（access violation）；
    此时请等待其结束，或设置 OPTSIM_DEVICE=cpu。"""
    return os.environ.get('OPTSIM_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')


# ======================= 网络结构（输入 (B, W, C)，输出 (B, Cout)） =======================
class MLP(nn.Module):
    """全连接网络：展平窗口 → 若干隐藏层 → 输出。"""

    def __init__(self, ntaps, cin, cout, hidden=(128, 64)):
        super().__init__()
        dims, layers = [ntaps * cin, *hidden], []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(0.1)]
        self.net = nn.Sequential(nn.Flatten(), *layers, nn.Linear(dims[-1], cout))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """1D 卷积网络：沿时间卷积提取局部特征 → 展平 → 全连接输出。"""

    def __init__(self, ntaps, cin, cout, channels=(32, 32), kernel=7):
        super().__init__()
        dims, layers = [cin, *channels], []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Conv1d(a, b, kernel, padding=kernel // 2), nn.LeakyReLU(0.1)]
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1] * ntaps, cout)

    def forward(self, x):
        return self.out(self.conv(x.transpose(1, 2)).flatten(1))


class BiLSTM(nn.Module):
    """双向 LSTM：对整个窗口建模前后向记忆，取中心时刻的隐状态输出。"""

    def __init__(self, ntaps, cin, cout, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(cin, hidden, layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, cout)
        self.c = ntaps // 2

    def forward(self, x):
        return self.out(self.lstm(x)[0][:, self.c])


ARCH = dict(mlp=MLP, cnn=CNN, bilstm=BiLSTM)


# ======================= 统一接口 =======================
class NNEqualizer:
    """神经网络均衡器。

    参数
    ----
    arch     : 'mlp' | 'cnn' | 'bilstm'
    ntaps    : 输入窗长（采样点数，按 sps 计；相干后均衡 1 sps 常取 21~61，IM-DD 2 sps 常取 31~61）
    sps      : 输入每符号采样数（输出恒为 1 sps）
    residual : 残差学习（输出 = 中心输入采样 + 网络修正），默认 True
    device   : 'cuda' / 'cpu'，默认 default_device()
    arch_kw  : 透传网络结构，如 MLP hidden=(128, 64)；CNN channels=(32, 32), kernel=7；BiLSTM hidden=64, layers=1
    """

    def __init__(self, arch='mlp', ntaps=31, sps=1, residual=True, device=None, **arch_kw):
        self.arch, self.ntaps, self.sps, self.residual, self.arch_kw = arch, ntaps, sps, residual, arch_kw
        self.device = device or default_device()
        self.model, self.history = None, []

    # ---- 数据格式转换 ----
    def _x(self, x):
        X = sym_windows(x, self.ntaps, self.sps).astype(np.float32)
        return X, X[:, self.ntaps // 2, :]                    # 窗口, 中心采样（残差基准）

    @staticmethod
    def _to_real(y):
        y = np.atleast_2d(y)
        return np.stack([v for p in y for v in (p.real, p.imag)], 1) if np.iscomplexobj(y) else y.T

    def _from_real(self, Y):
        if not self.complex:
            return Y[:, 0]
        return (Y[:, 0::2] + 1j * Y[:, 1::2]).T

    def fit(self, x, y, epochs=60, batch=512, lr=2e-3, val=0.1, patience=8, verbose=True, seed=0):
        """训练：x 为接收信号，y 为对应发送符号（长度 = len(x)/sps）。
        末尾 val 比例（> 0）的样本作验证集，验证 MSE 连续 patience 个 epoch 不降则早停，并恢复最佳权重。"""
        torch.manual_seed(seed)
        self.history = []
        self.complex = np.iscomplexobj(y)
        X, Xc = self._x(x)
        Y = self._to_real(y).astype(np.float32)
        T = Y - Xc if self.residual else Y
        self.model = ARCH[self.arch](self.ntaps, X.shape[2], Y.shape[1], **self.arch_kw).to(self.device)
        n_val = max(int(len(X) * val), 1)                      # 至少 1 个验证样本（避免 X[:-0] 为空）
        to = lambda a: torch.from_numpy(a).to(self.device)
        Xt, Tt, Xv, Tv = to(X[:-n_val]), to(T[:-n_val]), to(X[-n_val:]), to(T[-n_val:])
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 3)
        best, best_state, wait = np.inf, None, 0
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(Xt), batch):
                b = perm[i:i + batch]
                loss = nn.functional.mse_loss(self.model(Xt[b]), Tt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                v = float(sum(nn.functional.mse_loss(self.model(Xv[i:i + 8192]), Tv[i:i + 8192], reduction='sum')
                              for i in range(0, len(Xv), 8192)) / Tv.numel())
            sched.step(v)
            self.history.append(v)
            if v < best:
                best, best_state, wait = v, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
            stop = wait >= patience or ep == epochs - 1
            if verbose and (ep % 10 == 0 or stop):
                print(f'      [{self.arch}] epoch {ep:3d}  val MSE {v:.4e}')
            if stop:
                break
        self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def __call__(self, x):
        """均衡：返回与训练目标同格式的 1 sps 输出。"""
        X, Xc = self._x(x)
        self.model.eval()
        Y = np.concatenate([self.model(torch.from_numpy(X[i:i + 16384]).to(self.device)).cpu().numpy()
                            for i in range(0, len(X), 16384)])
        return self._from_real(Y + Xc if self.residual else Y)

    def n_params(self):
        """可训练参数个数。"""
        return sum(p.numel() for p in self.model.parameters())

    def mults_per_symbol(self):
        """每个输出符号所需实数乘法次数（推理复杂度，忽略激活函数）。"""
        m, W = self.model, self.ntaps
        if self.arch == 'mlp':
            return sum(l.in_features * l.out_features for l in m.modules() if isinstance(l, nn.Linear))
        if self.arch == 'cnn':
            conv = sum(l.in_channels * l.out_channels * l.kernel_size[0] * W for l in m.modules() if isinstance(l, nn.Conv1d))
            return conv + m.out.in_features * m.out.out_features
        h, cin, L = m.lstm.hidden_size, m.lstm.input_size, m.lstm.num_layers
        per = sum(4 * h * ((cin if i == 0 else 2 * h) + h) for i in range(L)) * 2
        return per * W + m.out.in_features * m.out.out_features


# ======================= 学习型数字反向传播 LDBP =======================
class LDBP(nn.Module):
    """学习型数字反向传播（Learned DBP，Häger & Pfister 2018/2021）：物理结构引导的神经网络。

    结构与 DBP 相同，按跨段逆序、每跨 steps 步，每步 = [可学习短复 FIR（色散逆）] -> [非线性相位旋转
    E·exp(jθ_k(|Ex|²+|Ey|²))，θ_k 可学习]。FIR 初始化为该段色散逆的截断冲激响应，θ_k 初始化为
    (8/9)γ·∫e^{-αz}dz（该段有效长度），因此训练前就是一个低步数 DBP，训练后可用更少步数逼近全 DBP。

    参数: fs 采样率（一般 2 sps）; fib 光纤参数; n_spans 跨数; p_launch_dbm 入纤功率（用于功率归一化）;
          steps 每跨步数; ntaps 每步 FIR 抽头数（需覆盖单段色散记忆）
    用法: ldbp = LDBP(2*rs, fib, 10, 4.0, steps=1, ntaps=41).fit(r_train_2sps, frame, beta)
          y = ldbp.apply(r_2sps)             # numpy 输入输出，可作为 Config(nl_comp=ldbp.apply) 嵌入 rx_dsp
    """

    def __init__(self, fs, fib, n_spans, p_launch_dbm, steps=1, ntaps=41, f_pass=None, device=None):
        super().__init__()
        from .channel import beta2
        self.device = device or default_device()
        self.p_launch, self.ntaps, self.n = 1e-3 * 10 ** (p_launch_dbm / 10), ntaps, n_spans * steps
        a = fib['alpha'] * np.log(10) / 10 / 1e3
        dz = fib['L'] * 1e3 / steps
        # FIR 初始化：加权最小二乘设计。目标 = 段色散逆 exp(jβ2ω²dz/2) × 升余弦带限窗（|f| ≤ f_pass 平坦，
        # 到 1.5·f_pass 滚降为 0）；通带权重 1、其余 0.3（全频段都约束，避免过渡带增益在多步级联中发散）。
        f_pass = f_pass or 0.28 * fs                          # ≥ 信号边缘 (1+β)Rs/2 = 0.275·fs（2 sps，滚降 0.1）
        f = np.linspace(-fs / 2, fs / 2, 2048, endpoint=False)
        n = np.arange(ntaps) - ntaps // 2
        A = np.exp(-2j * np.pi * np.outer(f, n) / fs)
        af = np.abs(f)
        win = np.where(af <= f_pass, 1.0, np.where(af >= 1.5 * f_pass, 0.0, 0.5 * (1 + np.cos(np.pi * (af - f_pass) / (0.5 * f_pass)))))
        target = np.exp(1j * beta2(fib['D'], fib['lam']) / 2 * (2 * np.pi * f) ** 2 * dz) * win
        wgt = np.where(af <= f_pass, 1.0, 0.3)
        h = np.linalg.lstsq(A * wgt[:, None], target * wgt, rcond=None)[0]
        taps = torch.tensor(np.tile(h, (self.n, 1)), dtype=torch.complex64)
        # θ 初始化：逆序第 j 段（从跨末端往前）覆盖 z ∈ [L - (j+1)dz, L - j·dz]
        z0 = fib['L'] * 1e3 - (np.arange(steps) + 1) * dz
        th = (8 / 9) * fib['gamma'] / 1e3 * np.exp(-a * z0) * (1 - np.exp(-a * dz)) / a
        self.taps = nn.Parameter(taps)
        self.theta = nn.Parameter(torch.tensor(np.tile(th, n_spans), dtype=torch.float32))
        self.to(self.device)

    def forward(self, E):
        """E: (2, N) complex64 张量，已按入纤功率归一化。"""
        N = E.shape[-1]
        k = (torch.arange(self.ntaps, device=E.device) - self.ntaps // 2) % N
        for i in range(self.n):
            h = torch.zeros(N, dtype=E.dtype, device=E.device).index_add(0, k, self.taps[i])
            E = torch.fft.ifft(torch.fft.fft(E) * torch.fft.fft(h))
            E = E * torch.exp(1j * self.theta[i] * (E.abs() ** 2).sum(0))
        return E

    def _prep(self, r):
        s = np.sqrt(np.sum(np.mean(np.abs(r) ** 2, -1)))
        return torch.from_numpy((r * np.sqrt(self.p_launch) / s).astype(np.complex64)).to(self.device), s

    def fit(self, r, frame, beta, iters=400, lr=3e-3, verbose=True):
        """训练：r 为 (2, N) 2 sps 接收波形（建议用无相位噪声/频偏的训练实现），frame 为 (2, N/2) 发送符号。
        损失：RRC 匹配滤波、抽取 1 sps 后，各偏振做最小二乘复增益，再求与 frame 的 MSE（即最大化有效 SNR）。"""
        from .tx import nyquist_response
        E, _ = self._prep(r)
        H = torch.tensor(nyquist_response(np.fft.fftfreq(E.shape[-1], 0.5), 1.0, beta), dtype=torch.complex64, device=self.device)
        T = torch.from_numpy(np.asarray(frame, np.complex64)).to(self.device)
        assert E.shape[-1] == 2 * T.shape[-1], f'r 应为 2 sps（长度 {2 * T.shape[-1]}），实际 {E.shape[-1]}'
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for it in range(iters):
            y = torch.fft.ifft(torch.fft.fft(self(E)) * H)[:, ::2]
            a = (y.conj() * T).sum(-1, keepdim=True) / (y.abs() ** 2).sum(-1, keepdim=True)
            loss = ((a * y - T).abs() ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and it % 100 == 0:
                print(f'      [ldbp] iter {it:4d}  eff. SNR {-10 * np.log10(loss.item() / (T.abs() ** 2).mean().item()):.2f} dB')
        return self

    @torch.no_grad()
    def apply(self, r):
        """numpy 接口：(2, N) 2 sps 波形 -> 非线性补偿后的波形（保持原功率尺度）。"""
        E, s = self._prep(r)
        return (self(E).cpu().numpy() * s / np.sqrt(self.p_launch)).astype(complex)

    def mults_per_symbol(self, sps=2):
        """每符号实数乘法：每步 FIR 4·ntaps·2 偏振 + 非线性约 12，乘以步数与 sps。"""
        return self.n * sps * (2 * 4 * self.ntaps + 12)
