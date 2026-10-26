%MAIN_NN_IMDD_PAM4 神经网络均衡器 vs 传统均衡器：IM-DD PAM4（NN 替代 FFE / Volterra 的典型场景）。
% 链路：56 GBd PAM4 -> DAC -> MZM（正交偏置、大摆幅 → 调制非线性）-> 4 km SSMF @1550 nm（色散功率衰落）
%       -> VOA -> PIN（平方律 → 非线性 ISI）-> ADC (2 sps) -> 均衡器
% 对比：最小二乘 FFE、Volterra（2/3 阶）、MLP、CNN、BiLSTM；扫描接收光功率。训练与测试使用独立实现。
% 需要 Deep Learning Toolbox（有 GPU 时自动使用）。
clear; close all; setup_paths;
prm = struct('rs', 56e9, 'sps', 4, 'L', 4, 'swing', 0.5, 'lv', pam_const(4)/3);   % 符号率、过采样、光纤 [km]、MZM 摆幅 [Vπ]、电平
LV = prm.lv;
NTAPS = 41;
names = {'FFE (LS)', 'Volterra', 'MLP', 'CNN', 'BiLSTM'};
rop = -10:-2;
ber = zeros(numel(names), numel(rop)); cplx = zeros(1, numel(names));
for k = 1:numel(rop)
    [ytr, str] = realization(1, rop(k), prm);
    [yte, ~, bte] = realization(2, rop(k), prm);
    for m = 1:numel(names)
        switch names{m}
            case 'FFE (LS)', [~, w] = volterra_eq(ytr, str, numel(str), [NTAPS 0 0], 2); ye = (volterra_features(yte, [NTAPS 0 0], 2)*w).'; cplx(m) = numel(w);
            case 'Volterra', [~, w] = volterra_eq(ytr, str, numel(str), [NTAPS 15 5], 2); ye = (volterra_features(yte, [NTAPS 15 5], 2)*w).'; cplx(m) = numel(w);
            case 'MLP',    eq = nn_eq_train(ytr, str, 'mlp', NTAPS, 2, 'hidden', [128 64]);
            case 'CNN',    eq = nn_eq_train(ytr, str, 'cnn', NTAPS, 2, 'channels', [16 16], 'kernel', 7);
            case 'BiLSTM', eq = nn_eq_train(ytr, str, 'bilstm', NTAPS, 2, 'hidden', 32);
        end
        if m > 2, ye = nn_eq_apply(eq, yte); cplx(m) = nn_eq_mults(eq); end
        ber(m, k) = max(calc_ber(bte, sym2bits(ye, LV)), 1e-6);
    end
    fprintf('ROP %+d dBm: %s\n', rop(k), fmt_list(names, ber(:,k), '%.1e', ' | '));
end
fprintf('\n每符号实数乘法次数: %s\n', fmt_list(names, cplx, '%d', ', '));
figure; semilogy(rop, ber, 'o-'); hold on; yline(3.8e-3, 'r:'); grid on;
legend([arrayfun(@(i) sprintf('%s (%d mult/sym)', names{i}, cplx(i)), 1:numel(names), 'UniformOutput', false), {'HD-FEC'}], 'Location', 'southwest');
xlabel('ROP (dBm)'); ylabel('BER'); title(sprintf('56 GBd PAM4, %d km SSMF @1550 nm (test on independent data)', prm.L));
exportgraphics(gcf, fullfile('results', 'nn_imdd_pam4.png'));

function [y, sym, bits] = realization(seed, rop, prm)
% 一次独立的发射-传输-接收，返回 2 sps 接收信号、发送符号、发送比特
RS = prm.rs; SPS = prm.sps; LV = prm.lv;
rng(seed); fs = RS*SPS; n = 2^16;
bits = uint8(randi([0 1], 1, 2*n));
sym = bits2sym(bits, LV);
u = dac_model(pulse_shape(sym, SPS, 0.2, 'rc'), fs, 6, 0.6*RS);
E = mzm(laser_cw(numel(u), fs, 10, 0, 0, -150), prm.swing*u, 1, -0.5);
E = fiber_cd(E, fs, prm.L);
E = voa(E, w2dbm(sum(sigpow(E))) - rop);
i = photodiode(E, fs, struct('R', 0.8, 'bw', 0.7*RS, 'T', 300));
y = adc_model(i, fs, 2*RS, 6, []);
y = (y - mean(y)) / std(y) * std(LV);
end

function s = fmt_list(names, v, f, sep)
% 'name1 v1 | name2 v2 ...' 格式化输出
s = strjoin(arrayfun(@(i) sprintf(['%s ' f], names{i}, v(i)), 1:numel(names), 'UniformOutput', false), sep);
end
