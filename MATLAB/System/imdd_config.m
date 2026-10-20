function cfg = imdd_config(varargin)
%IMDD_CONFIG IM-DD PAM-M 短距系统默认参数（单跨、无光放大，默认 O 波段 2 km）；名值对覆盖任意字段。
%   用法: cfg = imdd_config();
%         cfg = imdd_config('M', 8, 'rs', 112e9/3, 'tx', 'eml', 'alpha', 0.5, 'band', 'C', 'L', 3, 'eq', 'volterra');
% ---- 信号 ----
cfg.M = 4;                % PAM 阶数（2 即 NRZ-OOK）
cfg.rs = 56e9;            % 符号率 [Bd]
cfg.sps = 4;              % 仿真过采样
cfg.n_sym = 2^15;         % 符号数
cfg.n_train = 4096;       % 均衡器训练符号数（BER 只统计其后的符号）
cfg.beta = 0.2;           % RC 成形滚降
cfg.seed = 1;
% ---- 发射 ----
cfg.tx = 'mzm';           % 'mzm'（零啁啾，正交偏置）| 'eml'（EAM，啁啾 α）| 'dml'（直调，瞬态+绝热啁啾）
cfg.p_tx = 5;             % 出纤光功率 [dBm]（MZM/EML 为激光功率；DML 为平均功率）
cfg.swing = 0.4;          % MZM 驱动摆幅 [Vπ]
cfg.er_db = 8;            % EML / DML 消光比 [dB]
cfg.alpha = 0;            % EML / DML 啁啾 Henry 因子
cfg.k_ad = 1e12;          % DML 绝热啁啾 [Hz/W]
cfg.linewidth = 1e6;      % 激光线宽 [Hz]
cfg.rin_db = -150;        % 激光 RIN [dB/Hz]
cfg.dac_enob = 6;
cfg.dac_bw = 0.6;         % DAC/驱动带宽（× rs）
% ---- 链路 ----
cfg.band = 'O'; cfg.fiber = 'SSMF';
cfg.L = 2;                % 光纤长度 [km]
% ---- 接收 ----
cfg.rx = 'pin';           % 'pin' | 'apd'
cfg.R = 0.8;              % 响应度 [A/W]
cfg.apd_M = 10;           % APD 倍增因子
cfg.rx_bw = 0.7;          % PD/TIA 带宽（× rs）
cfg.irnd = 15e-12;        % TIA 输入等效噪声电流密度 [A/√Hz]
cfg.adc_enob = 6;
cfg.eq = 'ffe';           % 'ffe'(LMS) | 'dfe' | 'volterra' | 'mlse'（FFE + 噪声白化后滤波 + MLSE）
cfg.ntaps = 21;           % FFE 抽头（2 sps 采样点）
cfg.mu = 1e-3;            % LMS 步长
cfg.nb = 3;               % DFE 反馈抽头
for k = 1:2:numel(varargin), cfg.(varargin{k}) = varargin{k+1}; end
cfg.fs = cfg.rs * cfg.sps;
cfg.fib = fiber_params(cfg.band, cfg.fiber, 'L', cfg.L);
end
