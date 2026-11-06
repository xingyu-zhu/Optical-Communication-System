function [E, refs] = sdm_tx(cfg, n_modes, swing, enob)
%SDM_TX SDM 发射机：M 个独立 DP-QAM 信号（各自数据/训练序列，seed = cfg.seed + 100·(m-1)），
%   共用一个激光器（分束），各经 DAC + DP-IQ 调制器，每个空间模式入纤功率 cfg.p_launch [dBm]。
%   用法: [E, refs] = sdm_tx(cfg, 3)            E 2M×N，refs 为各模式 tx_signal 输出的元胞数组
%   模式复用器按理想一一映射（任何酉变换都可并入随机模式耦合）。对应 Python multimode.sdm_tx
if nargin < 3, swing = 0.2; end
if nargin < 4, enob = 6; end
refs = cell(n_modes, 1);
for m = 1:n_modes
    c = cfg; c.seed = cfg.seed + 100*(m - 1);
    refs{m} = tx_signal(c);                                % 内部 rng(seed)
end
N = size(refs{1}.wave, 2);
cw = laser_cw(N, cfg.fs, 16, cfg.linewidth);
E = zeros(2*n_modes, N);
for m = 1:n_modes
    u = swing*dac_model(refs{m}.wave, cfg.fs, enob, 0.7*cfg.rs);
    E(2*m-1:2*m, :) = setpow(dp_iq_mod(cw, u, 1, deg2rad(1)), cfg.p_launch);
end
end
