function cfg = sys_config(varargin)
%SYS_CONFIG 系统默认参数；名值对覆盖任意字段。光纤参数由 band/fiber 从 params/fiber_bands.json 生成。
%   用法: cfg = sys_config();
%         cfg = sys_config('M', 64, 'band', 'L', 'fiber', 'ULL', 'n_spans', 20);
%         cfg = sys_config('fib_kw', {'pmd', 0, 'raman', true});   覆盖光纤参数
%   生成后也可直接改 cfg.fib.xxx。
% ---- 信号 ----
cfg.M = 16;               % 调制阶数
cfg.mod = 'qam';          % 'qam' | 'psk'（相干 M-PSK：BPSK/QPSK/8PSK/16PSK）
cfg.rs = 40e9;            % 符号率 [Bd]
cfg.beta = 0.1;           % RRC 滚降
cfg.sps = 4;              % 仿真过采样（fs = sps·rs；WDM 需覆盖全部信道，如 16）
cfg.n_blocks = 960;       % 导频块数（每块 pilot_period 符号，首个为导频）
cfg.train_len = 2048;     % 训练序列长度
cfg.pilot_period = 32;
cfg.seed = 1;
% ---- 链路 ----
cfg.band = 'C';           % O / E / S / C / L / U
cfg.fiber = 'SSMF';       % SSMF / ULL / NZDSF / DSF
cfg.L = 80;               % 跨段长度 [km]
cfg.n_spans = 10;
cfg.p_launch = 0;         % 每信道入纤功率 [dBm]
cfg.fib_kw = {};          % 光纤参数覆盖（名值对）
cfg.phi_max = 0.005;      % SSFM 自适应步长：每步最大非线性相移 [rad]
% ---- WDM ----
cfg.n_ch = 1;             % 信道数（奇数，中心信道为被测信道）
cfg.spacing = 50e9;       % 信道间隔 [Hz]
cfg.linewidth = 100e3;    % 激光线宽 [Hz]
% ---- 接收 DSP ----
cfg.fs_adc = 100e9;       % ADC 采样率
cfg.nl_comp = 'cdc';      % 'cdc' | 'dbp' | 'none' | 函数句柄（如 @(y) ldbp_apply(p, y)）
cfg.dbp_steps = 4;        % DBP 每跨步数
cfg.timing = 'ff';        % 'ff' | 'gardner' | 'none'
cfg.ntaps = 15;
cfg.mu_cma = 1e-3;
cfg.mu_rde = 3e-4;
cfg.cpr = 'bps';          % 'bps' | 'vv' | 'pilot'
cfg.dd_lms = false;       % CPR 后 1 sps 2×2 DD-LMS 精均衡
for k = 1:2:numel(varargin), cfg.(varargin{k}) = varargin{k+1}; end
cfg.fs = cfg.rs * cfg.sps;
if ~isfield(cfg, 'fib'), cfg.fib = fiber_params(cfg.band, cfg.fiber, 'L', cfg.L, cfg.fib_kw{:}); end
end
