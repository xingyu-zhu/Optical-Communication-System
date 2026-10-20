%MAIN_LAUNCH_POWER_SWEEP 单信道入纤功率扫描：CDC vs PB-NLC vs DBP，并与 GN 模型预测对比。
% PB-NLC 接在 CDC 链路之后（CPR 后 1 sps），用帧前 n_train 个符号最小二乘拟合微扰系数；三种方法都在其后的数据符号上评估。
clear; close all; setup_paths;
cfg = sys_config('n_blocks', 300, 'n_spans', 10, 'fs_adc', 160e9, 'dbp_steps', 10, 'fib_kw', {'pmd', 0});
fib = cfg.fib;
ref = tx_signal(cfg);
N = size(ref.wave, 2);
P = -2:2:6;
snr = zeros(3, numel(P));                    % 行：CDC / PB-NLC / DBP
n_train = 5000; f = ref.frame;
te = ref.kind == 0 & (1:numel(ref.kind)) > n_train;   % 评估用数据符号
for k = 1:numel(P)
    rng(1);
    E = edfa(ref.wave .* laser_cw(N, cfg.fs, 0), cfg.fs, [], fib.nf, P(k), fib.f0);
    E = fiber_link(E, cfg.fs, fib, cfg.n_spans, true, 'phi_max', cfg.phi_max);
    r = coherent_rx(E, laser_cw(N, cfg.fs, 10), cfg.fs, struct('shot', false));
    cfg.p_launch = P(k);
    cfg.nl_comp = 'cdc'; y = rx_dsp(r, cfg, ref).y;
    snr(1,k) = calc_snr(y(:, te), f(:, te));
    yp = pb_nlc(y, f, n_train, 10); snr(2,k) = calc_snr(yp(:, te), f(:, te));
    cfg.nl_comp = 'dbp'; y = rx_dsp(r, cfg, ref).y; snr(3,k) = calc_snr(y(:, te), f(:, te));
    fprintf('P %+d dBm: SNR CDC %.2f dB, PB-NLC %.2f dB, DBP %.2f dB\n', P(k), snr(:,k));
end
pg = linspace(-4, 8, 100);
gn = gn_model(pg, fib, cfg.rs, cfg.n_spans);
fprintf('GN model: P_opt %.2f dBm, SNR_opt %.2f dB\n', gn.p_opt_dbm, gn.snr_opt);
figure; plot(P, snr, 'o-'); hold on; plot(pg, gn.snr, 'k--', pg, gn.snr_ase, 'k:'); grid on; ylim([10 30]);
xlabel('Launch power (dBm)'); ylabel('SNR (dB)'); legend('SSFM + CDC', 'SSFM + PB-NLC', 'SSFM + DBP', 'GN model (1 ch)', 'ASE only');
exportgraphics(gcf, fullfile('results', 'launch_power_sweep.png'));
