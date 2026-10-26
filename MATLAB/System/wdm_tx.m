function [E, refs, freqs] = wdm_tx(cfg)
%WDM_TX WDM 发射机：cfg.n_ch 个独立 DP-QAM 信道（各自数据 seed+k、各自激光相位噪声，理想调制），
%   以 cfg.spacing 等间隔排布在中心频率两侧，每信道功率 cfg.p_launch。
%   用法: [E, refs, freqs] = wdm_tx(cfg)
%   输出: E 2×N 合波光场 @cfg.fs；refs{k} 为第 k 信道 tx_signal 输出；freqs 各信道频偏 [Hz]（中心信道 = ceil(n_ch/2)）
freqs = ((0:cfg.n_ch-1) - (cfg.n_ch-1)/2) * cfg.spacing;
refs = cell(1, cfg.n_ch); fields = cell(1, cfg.n_ch);
for k = 1:cfg.n_ch
    c = cfg; c.seed = cfg.seed + k - 1;
    refs{k} = tx_signal(c);
    N = size(refs{k}.wave, 2);
    fields{k} = setpow(refs{k}.wave .* laser_cw(N, cfg.fs, 0, cfg.linewidth), cfg.p_launch);
end
E = wdm_mux(fields, freqs, cfg.fs);
end
