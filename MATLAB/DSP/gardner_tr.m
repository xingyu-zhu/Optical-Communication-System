function [y, taus] = gardner_tr(x, mu, sps)
%GARDNER_TR Gardner 反馈定时恢复：e = Re{(y[k-1] - y[k])·y*[k-1/2]}，τ ← τ + μe，三次 Lagrange 插值。
%   用法: [y, taus] = gardner_tr(x, 1e-3)      2 sps 输入/输出；taus 为每符号 τ 轨迹；可跟踪时钟频偏
if nargin < 3, sps = 2; end
[P, N] = size(x);
nsym = floor(N/sps);
y = zeros(P, 2*nsym);
taus = zeros(1, nsym);
tau = 0;
for k = 1:nsym-1
    t = k*sps + tau;
    yk = lagr3(x, t); ym = lagr3(x, t - sps/2); yp = lagr3(x, t - sps);
    tau = tau + mu*sum(real((yp - yk) .* conj(ym)));
    y(:, 2*k+1) = yk; y(:, 2*k) = ym; taus(k+1) = tau;
end
end

function v = lagr3(x, t)
% 三次 Lagrange 插值（Farrow），t 为 0-based 分数位置，循环索引
i = floor(t); m = t - i;
c = [-m*(m-1)*(m-2)/6; (m+1)*(m-1)*(m-2)/2; -(m+1)*m*(m-2)/2; (m+1)*m*(m-1)/6];
v = x(:, mod(i + (-1:2), size(x,2)) + 1) * c;
end
