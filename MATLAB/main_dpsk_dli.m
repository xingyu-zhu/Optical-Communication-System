%MAIN_DPSK_DLI 差分相位调制 DBPSK / DQPSK / D8PSK + 延迟线干涉仪（DLI）直接检测（无本振、无载波恢复）。
% 发射：差分编码 diff_psk -> RRC 成形 -> 激光（相位噪声）；信道：OSNR 加载（ASE 在两个偏振上）；
% 接收：光匹配滤波（RRC 光带通）-> dli_rx（两个 1 符号延迟 DLI + BPD，偏振不敏感）-> 符号中心采样 -> 差分判决。
% (a) BER–OSNR 与差分检测理论（ber_theory(..., 'dpsk')）及同阶相干 PSK 理论对比。
%     注：理论只计信号同偏振 ASE；偏振不敏感 DLI 还会收到正交偏振的 ASE-ASE 拍频，低 OSNR 下（DBPSK 约 7 dB）
%     带来约 0.9 dB 额外代价（DLI 前加偏振片即可消除）；DQPSK/D8PSK 工作 OSNR 高，影响 ≤ 0.2 dB。
% (b) 相位噪声容限：固定 OSNR，扫描激光线宽 Δν，BER 随 Δν·T 上升
clear; close all; setup_paths;
HD = 3.8e-3; rs = 28e9; sps = 8; beta = 0.2; nsym = 2^15; fs = rs*sps;
ranges = {2, 5:13; 4, 9:17; 8, 14:23}; names = containers.Map({2, 4, 8}, {'DBPSK', 'DQPSK', 'D8PSK'});
figure('Position', [80 80 1300 480]);
subplot(1,2,1); hold on; col = lines(3);
og = linspace(0, 30, 3001); sg = osnr2snr(og, rs, 12.5e9, 1);
for m = 1:size(ranges,1)
    M = ranges{m,1}; osnr = ranges{m,2}; ber = zeros(size(osnr));
    for k = 1:numel(osnr), ber(k) = run_dpsk(M, osnr(k), 100e3, osnr(k), rs, sps, beta, nsym); end
    o2 = linspace(osnr(1), osnr(end), 100); s2 = osnr2snr(o2, rs, 12.5e9, 1);
    semilogy(osnr, max(ber, 1e-6), 'o', 'Color', col(m,:), 'DisplayName', [names(M) ' + DLI (sim)']);
    semilogy(o2, ber_theory(M, s2, 'dpsk'), '-', 'Color', col(m,:), 'HandleVisibility', 'off');
    semilogy(o2, ber_theory(M, s2, 'psk'), ':', 'Color', col(m,:), 'HandleVisibility', 'off');
    sd = sensitivity(og, ber_theory(M, sg, 'dpsk'), HD); sc = sensitivity(og, ber_theory(M, sg, 'psk'), HD);
    fprintf('%s: OSNR @ HD-FEC sim %.2f dB, DPSK theory %.2f dB, coherent PSK theory %.2f dB (differential penalty %.2f dB)\n', ...
        names(M), sensitivity(osnr, ber, HD), sd, sc, sd - sc);
end
yline(HD, 'k--', 'DisplayName', 'HD-FEC'); set(gca, 'YScale', 'log'); ylim([1e-6 0.3]); grid on; legend('FontSize', 8);
xlabel('OSNR (dB / 0.1 nm)'); ylabel('BER'); title('Single-pol M-DPSK, 28 GBd (solid: DPSK theory, dotted: coherent PSK)');

lws = logspace(5, 9, 9);
subplot(1,2,2); hold on;
cases = {2, 10; 4, 14; 8, 20};
for m = 1:size(cases,1)
    b = arrayfun(@(lw) max(run_dpsk(cases{m,1}, cases{m,2}, lw, 3, rs, sps, beta, nsym), 1e-6), lws);
    loglog(lws/rs, b, 'o-', 'Color', col(m,:), 'DisplayName', sprintf('%s @ OSNR %d dB', names(cases{m,1}), cases{m,2}));
end
yline(HD, 'k--', 'HandleVisibility', 'off'); set(gca, 'XScale', 'log', 'YScale', 'log'); grid on; legend('FontSize', 8, 'Location', 'northwest');
xlabel('Linewidth \times symbol period  \Delta\nu T'); ylabel('BER'); title('(b) Laser phase-noise tolerance of DLI detection');
exportgraphics(gcf, fullfile('results', 'dpsk_dli.png'));

function b = run_dpsk(M, osnr, linewidth, seed, rs, sps, beta, nsym)
% 单偏振 M-DPSK 一次传输，返回 BER（跳过首个符号）
rng(seed); fs = rs*sps; k = log2(M);
bits = uint8(randi([0 1], 1, nsym*k));
E = pulse_shape(diff_psk(bits, M), sps, beta) .* laser_cw(nsym*sps, fs, 0, linewidth);
E = set_osnr([E; zeros(size(E))], osnr, fs);              % 信号在 X 偏振，ASE 落在两个偏振
E = matched_filter(E, sps, beta);                         % 光匹配滤波（光带通）
z = dli_rx(E, fs, rs, struct('shot', false));
z = z(1:sps:end);
b = calc_ber(bits(k+1:end), diff_psk_demod(z(2:end), M));
end
