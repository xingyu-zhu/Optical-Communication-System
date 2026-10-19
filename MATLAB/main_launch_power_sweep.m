%MAIN_LAUNCH_POWER_SWEEP 单信道入纤功率扫描：CDC vs DBP，并与 GN 模型预测对比。
clear; close all; setup_paths;
cfg = sys_config('n_blocks', 300, 'n_spans', 10, 'fs_adc', 160e9, 'dbp_steps', 10, 'fib_kw', {'pmd', 0});
fib = cfg.fib;
ref = tx_signal(cfg);
N = size(ref.wave, 2);
P = -2:2:6;
snr = zeros(2, numel(P));
for k = 1:numel(P)
    rng(1);
    E = edfa(ref.wave .* laser_cw(N, cfg.fs, 0), cfg.fs, [], fib.nf, P(k), fib.f0);
    E = fiber_link(E, cfg.fs, fib, cfg.n_spans, true, 'phi_max', cfg.phi_max);
    r = coherent_rx(E, laser_cw(N, cfg.fs, 10), cfg.fs, struct('shot', false));
    cfg.p_launch = P(k);
    cfg.nl_comp = 'cdc'; snr(1,k) = rx_dsp(r, cfg, ref).snr;
    cfg.nl_comp = 'dbp'; snr(2,k) = rx_dsp(r, cfg, ref).snr;
    fprintf('P %+d dBm: SNR CDC %.2f dB, DBP %.2f dB\n', P(k), snr(1,k), snr(2,k));
end
pg = linspace(-4, 8, 100);
gn = gn_model(pg, fib, cfg.rs, cfg.n_spans);
fprintf('GN model: P_opt %.2f dBm, SNR_opt %.2f dB\n', gn.p_opt_dbm, gn.snr_opt);
figure; plot(P, snr, 'o-'); hold on; plot(pg, gn.snr, 'k--', pg, gn.snr_ase, 'k:'); grid on; ylim([10 30]);
xlabel('Launch power (dBm)'); ylabel('SNR (dB)'); legend('SSFM + CDC', 'SSFM + DBP', 'GN model (1 ch)', 'ASE only');
exportgraphics(gcf, fullfile('results', 'launch_power_sweep.png'));
