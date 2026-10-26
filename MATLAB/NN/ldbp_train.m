function p = ldbp_train(p, r, frame, beta, iters, lr)
%LDBP_TRAIN 端到端训练 LDBP（Adam，自动微分，有 GPU 时自动使用）。
%   用法: p = ldbp_train(p, r, frame, beta)            默认 400 次迭代、学习率 3e-3
%   r 为 2×N 的 2 sps 接收波形（建议用无相位噪声/频偏的训练实现），frame 为 2×N/2 发送符号。
%   损失：RRC 匹配滤波、抽取 1 sps 后各偏振做最小二乘复增益，再求与 frame 的 MSE（等价于最大化有效 SNR）。
if nargin < 5, iters = 400; end
if nargin < 6, lr = 3e-3; end
assert(size(r,2) == 2*size(frame,2), 'r 应为 2 sps（长度 %d），实际 %d', 2*size(frame,2), size(r,2));
s = sqrt(sum(sigpow(r)));
E = r * sqrt(p.p_launch)/s;
H = nyquist_resp(freqaxis(size(r,2), 2), 1, beta, true);
T = frame;
if canUseGPU, E = gpuArray(E); H = gpuArray(H); T = gpuArray(T); end
taps = dlarray(cast_like(p.taps, E)); theta = dlarray(cast_like(p.theta, real(E)));
avgG = []; avgSq = [];
for it = 1:iters
    [loss, g] = dlfeval(@lossfun, taps, theta, E, H, T);
    [prm, avgG, avgSq] = adamupdate({taps, theta}, g, avgG, avgSq, it, lr);
    taps = prm{1}; theta = prm{2};
    if mod(it-1, 100) == 0
        fprintf('      [ldbp] iter %4d  eff. SNR %.2f dB\n', it-1, -10*log10(gather(extractdata(loss))/mean(abs(T(:)).^2)));
    end
end
p.taps = double(gather(extractdata(taps))); p.theta = double(gather(extractdata(theta)));
end

function [loss, g] = lossfun(taps, theta, E, H, T)
y = ifft(fft(ldbp_forward(taps, theta, E), [], 2) .* H, [], 2);
y = y(:, 1:2:end);
a = sum(conj(y).*T, 2) ./ sum(abs(y).^2, 2);
loss = mean(abs(a.*y - T).^2, 'all');
g = cell(1, 2);
[g{1}, g{2}] = dlgradient(loss, taps, theta);
end

function x = cast_like(x, ref)
if isa(ref, 'gpuArray'), x = gpuArray(x); end
end
