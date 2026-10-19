%MAIN_BER_VS_OSNR 背靠背 BER-OSNR：仿真（含激光相噪、频偏、完整 Rx DSP）vs AWGN 理论。
clear; close all; setup_paths;
cfg = sys_config('n_blocks', 400, 'n_spans', 0, 'fs_adc', 160e9);
ref = tx_signal(cfg);
N = size(ref.wave, 2);
osnr = 16:24;
ber = zeros(size(osnr));
rng(0);
for k = 1:numel(osnr)
    E = ref.wave .* laser_cw(N, cfg.fs, 0, 100e3);           % 理想调制 + 发射激光相噪
    E = set_osnr(E, osnr(k), cfg.fs);
    r = coherent_rx(E, laser_cw(N, cfg.fs, 10, 100e3, 0.8e9), cfg.fs, struct('shot', false));
    res = rx_dsp(r, cfg, ref);
    ber(k) = res.ber;
    fprintf('OSNR %d dB: BER %.2e, SNR %.2f dB\n', osnr(k), res.ber, res.snr);
end
figure;
semilogy(osnr, ber, 'o-', osnr, ber_theory(cfg.M, osnr2snr(osnr, cfg.rs)), 'k--'); hold on;
yline(3.8e-3, 'r:'); grid on;
xlabel('OSNR (dB/0.1nm)'); ylabel('BER'); legend('simulation', 'theory (AWGN)', 'HD-FEC');
exportgraphics(gcf, fullfile('results', 'ber_vs_osnr.png'));
