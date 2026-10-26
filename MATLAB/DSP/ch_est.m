function h = ch_est(x, ref, ntaps)
%CH_EST 最小二乘估计因果 FIR 信道 x_k ≈ Σ_{i<ntaps} h_i·ref_{k-i}（1 sps，循环）。用于 MLSE 的信道模型。
%   用法: h = ch_est(x, ref, 3)    h 为 1×ntaps
A = zeros(numel(ref), ntaps);
for i = 1:ntaps, A(:, i) = circshift(ref(:), i-1); end
h = (A \ reshape(x(1:numel(ref)), [], 1)).';
end
