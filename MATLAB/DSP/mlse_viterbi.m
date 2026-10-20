function a = mlse_viterbi(x, h, levels)
%MLSE_VITERBI MLSE（Viterbi）序列检测：x_k = Σ_{i=0}^{L} h_i·a_{k-i} + 噪声，a ∈ levels，状态为最近 L 个符号（M^L 个）。
%   通常接在短 FFE 之后，此时剩余 ISI 只有 2~3 个抽头（h 由 ch_est 得到，numel(h) ≥ 2）。
%   用法: a = mlse_viterbi(x, h, levels)    返回判决电平序列（1×N）
M = numel(levels); L = numel(h) - 1; S = M^L; D = M^(L-1);
levels = levels(:).'; h = h(:).';
sp = (0:S-1).';
pred = mod(sp, D)*M + (0:M-1);                       % 状态 s' 的 M 个前驱 s（0-based），S×M
out = h(1)*levels(floor(sp/D) + 1).' + zeros(1, M);  % 分支期望输出
for i = 1:L
    out = out + h(i+1)*levels(mod(floor(pred/M^(L-i)), M) + 1);   % 前驱中的 a_{k-i}
end
N = numel(x); metric = zeros(S, 1); tb = zeros(N, S, 'uint8');
for k = 1:N
    [metric, j] = min(metric(pred + 1) + (x(k) - out).^2, [], 2);
    tb(k, :) = j;
    metric = metric - min(metric);
end
[~, s] = min(metric); s = s - 1;
a = zeros(1, N);
for k = N:-1:1
    a(k) = levels(floor(s/D) + 1);
    s = pred(s + 1, tb(k, s + 1));
end
end
