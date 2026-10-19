%MAIN_COHERENT_DP_QAM 单信道相干 DP-16QAM 全流程仿真（含全部收发机器件损伤）：
% Tx DSP -> DAC -> 激光器 -> DP-IQ 调制器 -> 功放 -> N×(SSMF + 放大器) -> PDL -> WSS
% -> 本振 + 相干接收机(90°混频器 + BPD) -> ADC -> Rx DSP -> BER/SNR/EVM/GMI
% 注意：单信道只有 SPM 等信道内非线性；WDM 下的 XPM/FWM 见 main_wdm_system.m
clear; close all; setup_paths;
% 波段/光纤参数取自 params/fiber_bands.json；fib_kw 覆盖个别字段
cfg = sys_config('M', 16, 'rs', 40e9, 'band', 'C', 'fiber', 'SSMF', 'n_spans', 10, 'p_launch', 1, ...
                 'fib_kw', {'pmd', 0.1}, 'nl_comp', 'cdc', 'cpr', 'bps');
fib = cfg.fib; fs = cfg.fs;

%% 发射端
ref = tx_signal(cfg);                                % 内部 rng(cfg.seed)
u = 0.2 * dac_model(ref.wave, fs, 6, 0.7*cfg.rs);   % DAC + 驱动带宽，驱动摆幅 0.2Vπ
E = dp_iq_mod(laser_cw(size(u,2), fs, 16, 100e3), u, 1, deg2rad(1));   % 100 kHz 激光 + DP-IQ MZM（1° 正交误差）
E = edfa(E, fs, [], fib.nf, cfg.p_launch, fib.f0);  % 功放（APC）
E_tx = E;

%% 光纤链路（SPM + 色散/斜率 + PMD + ASE；可在 fib_kw 中打开 'raman' / 'steep'）
E = fiber_link(E, fs, fib, cfg.n_spans, true, 'phi_max', cfg.phi_max);
E = pdl_elem(E, 0.5, 0.3);                           % 0.5 dB PDL
E = optical_filter(E, fs, 1.2*cfg.rs*(1+cfg.beta), 3);   % WSS

%% 相干接收机
lo = laser_cw(size(E,2), fs, 13, 100e3, 1.5e9);      % 本振：100 kHz 线宽，1.5 GHz 频偏
skew = [0 1.5e-12 0 -1e-12];                         % Rx IQ skew [s]
r = coherent_rx(E, lo, fs, struct('R', 0.8, 'bw', 0.75*cfg.rs, 'T', 300), skew);
r = adc_model(r, fs, cfg.fs_adc, 6, 0.75*cfg.rs);   % ADC：160 -> 100 GSa/s，ENOB 6

%% 接收端 DSP
r = iq_deskew(r, cfg.fs_adc, skew);                  % 已标定 skew 的补偿
res = rx_dsp(r, cfg, ref);
fprintf('FO est %.3f GHz | BER %.2e (X %.2e, Y %.2e) | SNR %.2f dB | EVM %.1f%% | GMI %.3f b/sym\n', ...
    res.fo/1e9, res.ber, res.ber_xy, res.snr, res.evm, res.gmi);

%% 绘图
figure('Position', [100 100 1400 400]);
subplot(1,3,1); plot_spectrum(E_tx, fs); hold on; plot_spectrum(E, fs); legend('Tx', 'Rx');
subplot(1,3,2); plot_const(res.sym(1,:), 'X pol');
subplot(1,3,3); plot_const(res.sym(2,:), 'Y pol');
exportgraphics(gcf, fullfile('results', 'coherent_dp_qam.png'));
