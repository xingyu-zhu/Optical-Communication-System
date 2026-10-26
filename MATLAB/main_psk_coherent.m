%MAIN_PSK_COHERENT 相干 DP-M-PSK（BPSK / QPSK / 8PSK / 16PSK）背靠背 BER–OSNR，与 AWGN 理论对比。
% 复用 rx_dsp（sys_config('mod','psk')）：FOE 的 M 次方、BPS 搜索范围、模糊消除步长均由星座旋转对称阶数
% rot_order 自动确定（BPSK 2，QPSK 4，8PSK 8，16PSK 16）。含 100 kHz 收发激光线宽与 0.5 GHz 频偏。
% 另加 DP-16QAM 作对比：同为 4 bit/符号，16QAM 最小欧氏距离更大，HD-FEC 门限处比 16PSK 约省 3.7 dB OSNR。
clear; close all; setup_paths;
HD = 3.8e-3;
fmts = {'psk', 2, 4:12; 'psk', 4, 8:16; 'psk', 8, 13:21; 'psk', 16, 18:26; 'qam', 16, 14:22};
figure('Position', [80 80 1400 480]);
subplot(1,3,[1 2]); hold on; col = lines(size(fmts,1));
for m = 1:size(fmts,1)
    cfg = sys_config('M', fmts{m,2}, 'mod', fmts{m,1}, 'rs', 32e9, 'n_blocks', 300, 'n_spans', 0, 'fs_adc', 128e9);
    osnr = fmts{m,3}; ber = zeros(size(osnr));
    for k = 1:numel(osnr), [res, ~] = b2b(cfg, osnr(k)); ber(k) = res.ber; end
    if m == 3, sym8 = res.sym(1,:); end
    nm = sprintf('DP-%d%s', cfg.M, upper(cfg.mod)); if cfg.M == 2, nm = 'DP-BPSK'; elseif cfg.M == 4, nm = 'DP-QPSK'; end
    o2 = linspace(osnr(1), osnr(end), 100); th = ber_theory(cfg.M, osnr2snr(o2, cfg.rs), cfg.mod);
    semilogy(osnr, max(ber, 1e-6), 'o', 'Color', col(m,:), 'DisplayName', [nm ' (sim)']);
    semilogy(o2, th, '-', 'Color', col(m,:), 'HandleVisibility', 'off');
    fprintf('%-9s: OSNR @ HD-FEC %5.2f dB (theory %5.2f dB)\n', nm, sensitivity(osnr, ber, HD), sensitivity(o2, th, HD));
end
yline(HD, 'k:', 'DisplayName', 'HD-FEC'); set(gca, 'YScale', 'log'); ylim([1e-6 0.3]); grid on; legend('NumColumns', 2, 'FontSize', 8);
xlabel('OSNR (dB / 0.1 nm)'); ylabel('BER'); title('DP-M-PSK, 32 GBd B2B (lines: AWGN theory)');
subplot(1,3,3); plot_const(sym8, 'DP-8PSK @ 21 dB OSNR, X pol');
exportgraphics(gcf, fullfile('results', 'psk_coherent.png'));

function [res, ref] = b2b(cfg, osnr)
% 理想调制 + 发射/本振激光相噪 + 频偏 + OSNR 加载 -> 相干接收 -> rx_dsp
ref = tx_signal(cfg); N = size(ref.wave, 2);
rng(round(osnr*10));
E = set_osnr(ref.wave .* laser_cw(N, cfg.fs, 0, 100e3), osnr, cfg.fs);
r = coherent_rx(E, laser_cw(N, cfg.fs, 10, 100e3, 0.5e9), cfg.fs, struct('shot', false));
res = rx_dsp(r, cfg, ref);
end
