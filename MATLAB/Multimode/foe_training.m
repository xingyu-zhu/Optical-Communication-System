function f = foe_training(y, train, max_lag, sps, fs, nfft)
%FOE_TRAINING 训练序列辅助的粗频偏估计（MIMO 之前、模式混合与群时延扩展未补偿时使用）。
%   用法: fo = foe_training(y, frame(:, 1:train_len), max_lag, 2, rs)
%   各通道共用同一对收发激光器，频偏 Δf 相同：y_i(t) = e^{j2πΔf t}·Σ_j h_ij ⊗ s_j。对每个时延 k ∈ [-max_lag, max_lag]
%   （符号）计算 z_ij,k(t) = y_i(t)·s_j*(t-k)，其频谱在 Δf 处有峰；对全部 (i, j, k) 非相干累加 |FFT|² 后取峰值。
%   y P×N（sps 倍过采样，帧起始在 0 附近）；train P×T 各通道训练序列（1 sps）；fs 符号率（返回值单位随之）。
%   返回频偏估计（范围 ±fs/2，分辨率 fs/nfft）。对应 Python multimode.foe_training
[P, T] = size(train);
if nargin < 6 || isempty(nfft), nfft = 4*2^nextpow2(T); end
x = y(:, 1:sps:end);
acc = zeros(1, nfft);
for k = -max_lag:max_lag
    xs = circshift(x, -k, 2);
    z = reshape(xs(:, 1:T), P, 1, T) .* reshape(conj(train), 1, P, T);     % P×P×T
    acc = acc + reshape(sum(abs(fft(z, nfft, 3)).^2, [1 2]), 1, nfft);
end
fa = freqaxis(nfft, fs);
[~, i] = max(acc);
f = fa(i);
end
