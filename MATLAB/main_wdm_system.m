%MAIN_WDM_SYSTEM WDM 相干系统仿真：全场 SSFM 同时包含 SPM / XPM / FWM（可选 SRS、自陡峭），
% 对比 ASE-only、单信道、WDM 三种情形下中心信道的 SNR，并给出 GN 模型预测与各信道 SNR 分布。
%   N 个 DP-16QAM 信道 -> WDM 合波 -> 10×80 km SSMF + 放大器 -> WSS 选信道 -> 本振调谐 -> 相干接收 -> rx_dsp
% 收发机为理想（无 DAC/ADC 量化），差异纯粹来自光纤传输。改 'band' / 'fiber' 即可换波段与光纤类型。（约数分钟）
clear; close all; setup_paths;
N_CH = 7;  RUN_SRS = true;
base = {'M', 16, 'rs', 32e9, 'sps', 16, 'n_blocks', 200, 'band', 'C', 'fiber', 'SSMF', 'n_spans', 10, ...
        'p_launch', 1, 'spacing', 50e9, 'fs_adc', 80e9};
cases = {'ASE only (\gamma=0)', sys_config(base{:}, 'n_ch', 1, 'fib_kw', {'pmd', 0, 'gamma', 0});
         '1 ch: SPM',            sys_config(base{:}, 'n_ch', 1, 'fib_kw', {'pmd', 0});
         sprintf('%d ch: SPM+XPM+FWM', N_CH), sys_config(base{:}, 'n_ch', N_CH, 'fib_kw', {'pmd', 0})};
if RUN_SRS
    cases(end+1, :) = {sprintf('%d ch: +SRS+steepening', N_CH), ...
        sys_config(base{:}, 'n_ch', N_CH, 'fib_kw', {'pmd', 0, 'raman', true, 'steep', true})};
end
snr = cell(size(cases,1), 1);
for m = 1:size(cases,1)
    cfg = cases{m,2};
    fprintf('%s ...\n', cases{m,1});
    [E, refs, freqs] = wdm_tx(cfg);
    if m == 3, E_tx = E; end
    tic; E = fiber_link(E, cfg.fs, cfg.fib, cfg.n_spans, true, 'phi_max', cfg.phi_max); fprintf('    (link %.0f s)\n', toc);
    if m == 3, E_rx = E; end
    snr{m} = zeros(1, cfg.n_ch);
    for k = 1:cfg.n_ch
        res = rx_dsp(wdm_rx(E, cfg, freqs(k)), cfg, refs{k});
        snr{m}(k) = res.snr;
        if k == ceil(cfg.n_ch/2), fprintf('    center channel: SNR %.2f dB, BER %.2e, GMI %.3f\n', res.snr, res.ber, res.gmi); end
        if k == ceil(cfg.n_ch/2) && m == 3, sym_c = res.sym; end
    end
end

fprintf('\nGN model (center channel, ideal transceiver):\n');
fib = cases{2,2}.fib; c0 = cases{2,2};
for n = [1 N_CH]
    g = gn_model(c0.p_launch, fib, c0.rs, c0.n_spans, n, c0.spacing);
    fprintf('    %d ch: SNR %.2f dB (ASE %.2f, NLI %.2f); P_opt %.2f dBm -> SNR_opt %.2f dB\n', ...
        n, g.snr, g.snr_ase, g.snr_nli, g.p_opt_dbm, g.snr_opt);
end
nfull = floor(4.4e12/c0.spacing);
g = gn_model(c0.p_launch, fib, c0.rs, c0.n_spans, nfull, c0.spacing, 'closed');
fprintf('    full C band (%d ch): SNR %.2f dB; P_opt %.2f dBm -> SNR_opt %.2f dB\n', nfull, g.snr, g.p_opt_dbm, g.snr_opt);

figure('Position', [100 100 1500 420]);
subplot(1,3,1); plot_spectrum(E_tx, cases{3,2}.fs); hold on; plot_spectrum(E_rx, cases{3,2}.fs);
legend('Tx', 'Rx'); title(sprintf('%d-ch WDM spectrum (X pol)', N_CH));
subplot(1,3,2); idx = (1:N_CH) - ceil(N_CH/2); hold on;
for m = 3:size(cases,1), plot(idx, snr{m}, 'o-', 'DisplayName', cases{m,1}); end
for m = 1:2, yline(snr{m}, '--', cases{m,1}, 'HandleVisibility', 'off'); end
xlabel('Channel index'); ylabel('SNR (dB)'); legend('Location', 'east'); grid on; ylim([min([snr{:}]) - 0.5, max([snr{:}]) + 0.5]);
subplot(1,3,3); plot_const(sym_c(1,:), 'Center channel, X pol');
exportgraphics(gcf, fullfile('results', 'wdm_system.png'));
