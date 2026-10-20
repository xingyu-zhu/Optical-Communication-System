%MAIN_IMDD_SYSTEM IM-DD PAM-M 系统仿真（imdd_config / imdd_tx / imdd_link / imdd_rx）：
% (a) 112 Gb/s 下 PAM2 / PAM4 / PAM8 的 BER–ROP 与 KP4 灵敏度，O 波段 2 km，MZM + PIN；另加 PAM4 + APD
%     （DAC/ADC 取 ENOB 7：ENOB 6 时量化+削波使 PAM8 有效 SNR 封顶约 21 dB，低于 KP4 所需约 23.7 dB）
% (b) C 波段 10 km 色散功率衰落：MZM（α=0）、EML（α=±1）、DML（瞬态+绝热啁啾）的小信号响应 vs 理论
% (c) C 波段 PAM4 56 GBd 色散受限传输距离：FFE / DFE / FFE+后滤波+MLSE / Volterra 的 BER–光纤长度
%     （第一个功率衰落陷波 f1 = sqrt(c/(2λ²DL)) 进入信号带宽后，线性均衡无法恢复 → BER 陡升）
% (d) O 波段 PAM4 接收眼图（均衡前）
clear; close all; setup_paths;
KP4 = 2.4e-4;
figure('Position', [80 80 1250 900]);

%% (a) 112 Gb/s：PAM2 / PAM4 / PAM8
rop = -18:-2;
q = {'dac_enob', 7, 'adc_enob', 7};
cases = {'PAM2 112 GBd',      imdd_config('M', 2, 'rs', 112e9, 'n_sym', 2^16, q{:});
         'PAM4 56 GBd',       imdd_config('M', 4, 'rs', 56e9, 'n_sym', 2^15, q{:});
         'PAM8 37.3 GBd',     imdd_config('M', 8, 'rs', 112e9/3, 'n_sym', 21846, q{:});
         'PAM4 56 GBd + APD', imdd_config('M', 4, 'rs', 56e9, 'n_sym', 2^15, 'rx', 'apd', q{:})};
subplot(2,2,1); hold on;
fprintf('(a) O-band 2 km, 112 Gb/s, KP4 sensitivity:\n');
for m = 1:size(cases,1)
    b = ber_curve(cases{m,2}, rop);
    semilogy(rop, max(b, 1e-6), 'o-', 'MarkerSize', 4, 'DisplayName', cases{m,1});
    fprintf('    %-18s: %6.2f dBm\n', cases{m,1}, sensitivity(rop, b, KP4));
end
yline(KP4, 'k:', 'DisplayName', 'KP4 FEC'); set(gca, 'YScale', 'log'); ylim([1e-6 0.5]); grid on; legend('FontSize', 8);
xlabel('ROP (dBm)'); ylabel('BER'); title('(a) 112 Gb/s, O-band 2 km, FFE');

%% (b) C 波段 10 km 色散功率衰落
n = 2^16; fs = 400e9; L = 10; rng(0);
u = elpf(randn(1, n), fs, 60e9); u = 0.1*u/std(u);          % 小信号驱动
cw = sqrt(1e-3)*ones(1, n); f = freqaxis(n, fs); fp = linspace(0.5e9, 40e9, 400);
txs = {'MZM, \alpha=0', mzm(cw, 0.3*u, 1, -0.5), 0;
       'EML, \alpha=+1', eml(cw, 0.5 + u, 8, 1), 1;
       'EML, \alpha=-1', eml(cw, 0.5 + u, 8, -1), -1;
       'DML, \alpha=3, \kappa=1 GHz/mW', dml(u, fs, 5, 4, 3, 1e12, [], 0, []), NaN};
subplot(2,2,2); hold on; col = lines(4); k = f > 0 & f < 40e9;
for m = 1:size(txs,1)
    E0 = txs{m,2}; P0 = abs(E0).^2; P1 = abs(fiber_cd(E0, fs, L, 17, 1550)).^2;
    Hs = abs(fft(P1 - mean(P1)) ./ fft(P0 - mean(P0)));
    plot(f(k)/1e9, 20*log10(movmean(Hs(k), 15)), 'Color', col(m,:), 'DisplayName', [txs{m,1} ' (sim)']);
    if ~isnan(txs{m,3})
        plot(fp/1e9, 20*log10(power_fading(fp, L, 17, 1550, txs{m,3}) + 1e-6), '--', 'Color', col(m,:), 'HandleVisibility', 'off');
    end
end
ylim([-30 10]); grid on; legend('FontSize', 8, 'Location', 'southwest');
xlabel('Frequency (GHz)'); ylabel('|H(f)| (dB)'); title('(b) C-band 10 km power fading (dashed: theory)');
fprintf('(b) first fading notch (alpha=0) at %.1f GHz\n', sqrt(3e8/(2*(1550e-9)^2*17e-6*L*1e3))/1e9);

%% (c) C 波段 PAM4：色散受限距离与均衡器对比
Ls = 0:0.5:6; ROP_C = -4;
eqs = {'ffe', 'FFE (LMS)'; 'dfe', 'FFE+DFE'; 'mlse', 'FFE+PF+MLSE'; 'volterra', 'Volterra'};
ber = zeros(size(eqs,1), numel(Ls));
for j = 1:numel(Ls)
    c = imdd_config('M', 4, 'rs', 56e9, 'band', 'C', 'L', Ls(j), 'n_sym', 2^14, 'ntaps', 31);
    ref = imdd_tx(c); E = imdd_link(ref.E, c);
    for m = 1:size(eqs,1)
        c.eq = eqs{m,1}; rng(1); r = imdd_rx(E, c, ref, ROP_C); ber(m,j) = r.ber;
    end
end
subplot(2,2,3); hold on;
fprintf('(c) C-band PAM4 56 GBd, ROP %d dBm: max reach at KP4 [km]\n', ROP_C);
for m = 1:size(eqs,1)
    semilogy(Ls, max(ber(m,:), 1e-6), 'o-', 'MarkerSize', 4, 'DisplayName', eqs{m,2});
    ok = Ls(ber(m,:) <= KP4); fprintf('    %-12s: %.1f km\n', eqs{m,2}, max([ok 0]));
end
yline(KP4, 'k:', 'DisplayName', 'KP4 FEC'); set(gca, 'YScale', 'log'); ylim([1e-6 0.5]); grid on; legend('FontSize', 8, 'Location', 'northwest');
xlabel('Fiber length (km)'); ylabel('BER'); title(sprintf('(c) PAM4 56 GBd, C-band SSMF, ROP %d dBm', ROP_C));

%% (d) 眼图
c = cases{2,2}; c.n_sym = 4096; ref = imdd_tx(c);
i = photodiode(voa(imdd_link(ref.E, c), 10), c.fs, struct('R', 0.8, 'bw', 0.7*c.rs, 'irnd', 15e-12));
subplot(2,2,4); plot_eye(circshift(i, c.sps/2)*1e3, c.sps, 600);    % 移半个符号使眼图居中
ylabel('Photocurrent (mA)'); title('(d) PAM4 eye, O-band 2 km (before EQ)');
exportgraphics(gcf, fullfile('results', 'imdd_system.png'));

function b = ber_curve(cfg, rops)
% 一次发射与传输，扫描 ROP 返回 BER
ref = imdd_tx(cfg); E = imdd_link(ref.E, cfg);
b = zeros(size(rops));
for k = 1:numel(rops), rng(k); r = imdd_rx(E, cfg, ref, rops(k)); b(k) = r.ber; end
end
