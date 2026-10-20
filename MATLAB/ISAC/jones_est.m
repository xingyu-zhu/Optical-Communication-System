function J = jones_est(y, x, n)
%JONES_EST 通感一体（前向）：数据辅助块 Jones 矩阵估计，每 n 个符号最小二乘 J = Y·X^H·(X·X^H)^{-1}。
%   用法: J = jones_est(y, x, n)    y、x 为 2×N 接收 / 已知（或判决）发送符号（1 sps，已同步）
%   输出 2×2×nb（nb = floor(N/n)，每页一个块）
nb = floor(size(y,2)/n);
J = zeros(2, 2, nb);
for b = 1:nb
    k = (b-1)*n + (1:n);
    J(:,:,b) = (y(:,k)*x(:,k)') / (x(:,k)*x(:,k)');
end
end
