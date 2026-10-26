function E = ldbp_forward(taps, theta, E)
%LDBP_FORWARD LDBP 前向传播（ldbp_apply / ldbp_train 内部使用；支持 dlarray 自动微分与 gpuArray）。
%   E 为 2×N 光场（已按入纤功率归一化）；taps n×ntaps 复；theta 1×n。每步：循环 FIR -> 非线性相位旋转。
[n, ntaps] = size(taps); N = size(E, 2); c = floor(ntaps/2) + 1;
for i = 1:n
    h = [taps(i, c:end), zeros(1, N - ntaps, 'like', taps), taps(i, 1:c-1)];   % 中心抽头置于 0 时刻
    E = ifft(fft(E, [], 2) .* fft(h, [], 2), [], 2);
    E = E .* exp(1j*theta(i)*sum(abs(E).^2, 1));
end
end
