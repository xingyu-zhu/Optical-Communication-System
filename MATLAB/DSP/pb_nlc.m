function [yc, C] = pb_nlc(y, ref, n_train, L, lam)
%PB_NLC 一阶微扰非线性补偿（PB-NLC），微扰系数由训练符号最小二乘拟合：
%   Δy_p[k] = Σ_{m,n} C_mn·(y_p[k+m]·y_p*[k+m+n] + y_q[k+m]·y_q*[k+m+n])·y_p[k+n]，
%   取 |m|,|n| ≤ L 且 |m·n| ≤ L 的三元组（双曲剪枝），两偏振共享 C；输出 y − Δy。
%   用法: [yc, C] = pb_nlc(y, ref, n_train)          默认 L = 10，岭回归 lam = 1e-3
%         [yc, C] = pb_nlc(y, ref, n_train, L, lam)  y、ref 为 P×N：CDC + CPR 后、已同步的 1 sps 符号
if nargin < 4, L = 10; end
if nargin < 5, lam = 1e-3; end
[m, n] = meshgrid(-L:L); keep = abs(m.*n) <= L; m = m(keep); n = n(keep);
[P, N] = size(y); F = numel(m);
sh = @(v, d) circshift(v, -d, 2);                    % sh(v, d)(k) = v(k+d)
T = zeros(P, N, F);
for f = 1:F
    T(:,:,f) = sum(sh(y, m(f)) .* conj(sh(y, m(f) + n(f))), 1) .* sh(y, n(f));
end
A = reshape(permute(T(:, 1:n_train, :), [2 1 3]), [], F);
tgt = reshape((y(:, 1:n_train) - ref(:, 1:n_train)).', [], 1);
C = (A'*A + lam*size(A,1)*eye(F)) \ (A'*tgt);
yc = y - reshape(reshape(T, [], F)*C, P, N);
end
