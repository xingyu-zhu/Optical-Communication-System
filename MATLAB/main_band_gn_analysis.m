%MAIN_BAND_GN_ANALYSIS 多波段 (O/E/S/C/L/U) × 多光纤类型的满载 WDM 性能估计（GN 闭式，快速）。
% 展示参数文件 params/fiber_bands.json 的用法：每个波段自动取对应的 α、D、S、γ、放大器 NF。
% 注意：GN 假设高色散（近零色散时 FWM 相位匹配，结果仅供参考），未计入波段间 SRS（ISRS-GN）与收发机噪声。
clear; close all; setup_paths;
rs = 32e9; spacing = 50e9; n_spans = 10; L = 80;
p = load_params(); c0 = phys_const();
bands = fieldnames(p.bands)'; fibers = {'SSMF', 'ULL', 'NZDSF'};
snr_opt = zeros(numel(fibers), numel(bands));
fprintf('%-6s %-4s %4s %6s %5s %5s %4s %10s %11s\n', 'fiber', 'band', 'Nch', 'D', 'alpha', 'gamma', 'NF', 'P_opt(dBm)', 'SNR_opt(dB)');
for i = 1:numel(fibers)
    for j = 1:numel(bands)
        fib = fiber_params(bands{j}, fibers{i}, 'L', L);
        rg = p.bands.(bands{j}).range_nm;
        n_ch = floor((c0.c/(rg(1)*1e-9) - c0.c/(rg(2)*1e-9)) / spacing);
        g = gn_model(0, fib, rs, n_spans, n_ch, spacing, 'closed');
        snr_opt(i,j) = g.snr_opt;
        fprintf('%-6s %-4s %4d %6.2f %5.2f %5.2f %4.1f %10.2f %11.2f\n', fibers{i}, bands{j}, n_ch, fib.D, fib.alpha, fib.gamma, fib.nf, g.p_opt_dbm, g.snr_opt);
    end
end
fib = fiber_params('C', 'SSMF', 'L', L); P = linspace(-6, 8, 200);
figure('Position', [100 100 1200 450]);
subplot(1,2,1); hold on;
for n = [1 7 21 87], g = gn_model(P, fib, rs, n_spans, n, spacing, 'closed'); plot(P, g.snr, 'DisplayName', sprintf('%d ch', n)); end
xlabel('Launch power per channel (dBm)'); ylabel('SNR (dB)'); title('C-band SSMF, 10×80 km (GN)'); legend; grid on; ylim([5 30]);
subplot(1,2,2); bar(categorical(bands, bands), snr_opt.'); legend(fibers); ylabel('Optimum SNR (dB)');
title('Fully-loaded band, 10×80 km (GN)'); grid on;
exportgraphics(gcf, fullfile('results', 'band_gn_analysis.png'));
