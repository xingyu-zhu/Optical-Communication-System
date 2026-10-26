%MAIN_NN_COHERENT_NLC 神经网络非线性补偿 vs DBP：相干 DP-16QAM 长距单信道（入纤功率扫描）。
% 对比（同一套 rx_dsp，只替换色散/非线性补偿级）：
%   CDC；DBP 1 / 10 步每跨；LDBP 1 / 4 步每跨（学习型 DBP，物理引导神经网络）；
%   CNN 后均衡（黑盒网络作用于 CDC+CPR 后符号，对照用）
% 训练实现无激光相位噪声；测试实现为独立比特/噪声 + 100 kHz 线宽。需要 Deep Learning Toolbox。（约 15 分钟）
clear; close all; setup_paths;
base = sys_config('M', 16, 'rs', 32e9, 'n_blocks', 1000, 'n_spans', 10, 'fs_adc', 128e9, 'fib_kw', {'pmd', 0});
fib = base.fib;
powers = [0 2 4 6];
ldbp_cfg = {'LDBP 1 st/span', 1, 41; 'LDBP 4 st/span', 4, 15};
names = [{'CDC', 'DBP 1 st/span', 'DBP 10 st/span'}, ldbp_cfg(:,1).', {'CNN post-eq'}];
snr = zeros(numel(names), numel(powers)); cplx = zeros(1, numel(names));
for k = 1:numel(powers)
    tic; P = powers(k);
    cfg = base; cfg.p_launch = P;
    [c1, ref1, r1] = realization(cfg, 11, 0);           % 训练实现
    [c2, ref2, r2] = realization(cfg, 22, 100e3);       % 测试实现
    res = rx_dsp(r2, c2, ref2);
    snr(1,k) = res.snr;
    c = c2; c.nl_comp = 'dbp'; c.dbp_steps = 1;  snr(2,k) = rx_dsp(r2, c, ref2).snr;
    c.dbp_steps = 10;                            snr(3,k) = rx_dsp(r2, c, ref2).snr;
    x1 = resamp(gsop(r1), cfg.fs_adc, 2*cfg.rs);
    for j = 1:size(ldbp_cfg, 1)
        st = ldbp_cfg{j,2}; nt = ldbp_cfg{j,3};
        p = ldbp_train(ldbp_init(2*cfg.rs, fib, cfg.n_spans, P, st, nt), x1, ref1.frame, cfg.beta);
        c = c2; c.nl_comp = @(y) ldbp_apply(p, y);
        snr(3+j, k) = rx_dsp(r2, c, ref2).snr;
        cplx(3+j) = cfg.n_spans*st*2*(2*4*nt + 12);
    end
    eq = nn_eq_train(rx_dsp(r1, c1, ref1).y, ref1.frame, 'cnn', 41, 1, 'channels', [16 16], 'kernel', 9);
    ye = nn_eq_apply(eq, res.y); m = ref2.kind == 0;
    snr(end,k) = calc_snr(ye(:, m), ref2.frame(:, m)); cplx(end) = nn_eq_mults(eq);
    fprintf('P %+d dBm: %s  (%.0f s)\n', P, fmt_list(names, snr(:,k), '%.2f', ' | '), toc);
end
nfft = 2^12;                                              % DBP 复杂度（FFT 实现，与 LDBP 同口径）
cplx(2) = base.n_spans*1*2*(2*(4*log2(nfft) + 4) + 12); cplx(3) = 10*cplx(2);
fprintf('\n每符号实数乘法次数(近似): %s\n', fmt_list(names(2:end), round(cplx(2:end)), '%d', ', '));
figure; plot(powers, snr, 'o-'); grid on; legend(names, 'Location', 'northwest');
xlabel('Launch power (dBm)'); ylabel('SNR (dB)');
title({'DP-16QAM 32 GBd, 10×80 km SSMF', '(test: independent data + 100 kHz linewidth)'});
exportgraphics(gcf, fullfile('results', 'nn_coherent_nlc.png'));

function [c, ref, r] = realization(cfg, seed, lw)
% 一次独立的传输，返回 (cfg, ref, ADC 输出)
c = cfg; c.seed = seed; fib = cfg.fib;
ref = tx_signal(c);                                       % 内部 rng(seed)
N = size(ref.wave, 2);
E = edfa(ref.wave .* laser_cw(N, c.fs, 0, lw), c.fs, [], fib.nf, c.p_launch, fib.f0);
E = fiber_link(E, c.fs, fib, c.n_spans, true, 'phi_max', c.phi_max);
r = coherent_rx(E, laser_cw(N, c.fs, 10, lw), c.fs, struct('shot', false));
end

function s = fmt_list(names, v, f, sep)
% 'name1 v1 | name2 v2 ...' 格式化输出
s = strjoin(arrayfun(@(i) sprintf(['%s ' f], names{i}, v(i)), 1:numel(names), 'UniformOutput', false), sep);
end
