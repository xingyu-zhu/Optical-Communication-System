function [y, tau] = timing_ff(x, sps)
%TIMING_FF 前馈 Godard 时钟音定时估计：c = Σ_{f>0} X(f)·X*(f-Rs)，τ = -∠c/(2π)·sps。
%   用法: [y, tau] = timing_ff(x, 2)    返回校正后信号与估计时延 tau [采样]；块处理、无时钟频偏，2 sps 可用
X = fft(x, [], 2);
N = size(X, 2);
k = 0:floor(N/2)-1;
c = sum(sum(X(:, k+1) .* conj(X(:, mod(k - N/sps, N) + 1))));
tau = -angle(c)/(2*pi) * sps;
y = fracdelay(x, -tau, 1);
end
