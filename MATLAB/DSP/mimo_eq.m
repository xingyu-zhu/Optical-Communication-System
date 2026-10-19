function [y, W, err] = mimo_eq(x, ntaps, mu, sps, mode, c, ref, n_train, W)
%MIMO_EQ 2×2 蝶形 FIR 自适应均衡（逐符号随机梯度 W ← W + μ·e·x^H），完成偏振解复用与残余 CD/PMD/ISI 补偿。
%   用法: [y, W] = mimo_eq(x, 15, 1e-3, 2, 'cma', c)                    盲 CMA
%         [y, W] = mimo_eq(x, 15, 3e-4, 2, 'rde', c, [], 0, W)           以 W 为初值的 RDE
%         y = mimo_eq(x, 15, 1e-3, 2, 'lms', c, ref, 4096)               前 4096 符号训练，之后判决引导
%         y = mimo_eq(y, 7, 1e-4, 1, 'dd', c)                            CPR 后 1 sps DD-LMS
%   mode: 'cma'  e = y(R2 - |y|²)
%         'rde'  e = y(R_d² - |y|²)，R_d 为最近星座半径（多模 QAM）
%         'mma'  实虚分离多模
%         'lms'  前 n_train 符号用 ref 训练（数据辅助），之后判决引导
%         'dd'   判决引导 LMS（需先完成载波恢复）
%   输入 x P×N（sps 倍过采样）；W 为 P×(P·ntaps) 初始抽头（[] 则中心单抽头）
%   输出 y P×nsym（1 sps）；err 为 |e|
if nargin < 7, ref = []; end
if nargin < 8, n_train = 0; end
[P, N] = size(x);
nsym = floor(N/sps); h = floor(ntaps/2);
xp = [x(:, N-h+1:N), x, x(:, 1:h)];
if nargin < 9 || isempty(W)
    W = zeros(P, P*ntaps);
    for i = 1:P, W(i, (i-1)*ntaps + h + 1) = 1; end
end
R2 = mean(abs(c).^4)/mean(abs(c).^2);
Rr = mean(real(c).^4)/mean(real(c).^2);
radii = unique(round(abs(c(:)), 6));                 % 列向量
y = zeros(P, nsym); err = zeros(P, nsym);
for k = 1:nsym
    X = xp(:, (k-1)*sps + (1:ntaps));
    X = reshape(X.', [], 1);                         % [pol1 taps; pol2 taps]
    yk = W * X;
    switch mode
        case 'cma', e = yk .* (R2 - abs(yk).^2);
        case 'rde'
            [~, i] = min(abs(abs(yk) - radii.'), [], 2);
            e = yk .* (radii(i).^2 - abs(yk).^2);
        case 'mma', e = real(yk).*(Rr - real(yk).^2) + 1j*imag(yk).*(Rr - imag(yk).^2);
        otherwise
            if strcmp(mode, 'lms') && k <= n_train, d = ref(:, k); else, [~, d] = sym_decide(yk, c); end
            e = d - yk;
    end
    W = W + mu * e * X';
    y(:, k) = yk; err(:, k) = abs(e);
end
end
