function [y, w] = volterra_eq(x, ref, n_train, taps, sps, lam)
%VOLTERRA_EQ 实数 Volterra 均衡器（IM-DD 传统非线性均衡）：前 n_train 个符号岭回归最小二乘求核，再作用于全序列。
%   用法: [y, w] = volterra_eq(x, ref, n_train)                    默认 taps = [31 9 5]、sps = 2
%         [y, w] = volterra_eq(x, ref, n_train, [41 15 5], 2)
%         [y, w] = volterra_eq(x, ref, n_train, [41 0 0], 2)       二/三阶为 0 即最小二乘 FFE
%   输出 y 为 1 sps 均衡结果（行向量），w 为核系数（乘法复杂度 ≈ numel(w) 次/符号）
if nargin < 4, taps = [31 9 5]; end
if nargin < 5, sps = 2; end
if nargin < 6, lam = 1e-6; end
F = volterra_features(x, taps, sps);
A = F(1:n_train, :);
w = (A.'*A + lam*n_train*eye(size(A,2))) \ (A.' * ref(1:n_train).');
y = (F * w).';
end
