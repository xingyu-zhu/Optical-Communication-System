function [iw, dnu] = ofdr_shift(s_ref, s_meas, nw, B, pad)
%OFDR_SHIFT OFDR 应变/温度解调：两次扫描的距离域剖面按 nw 个单元分段，各段 FFT（补零 pad 倍）得局部瑞利谱，
%   互相关求谱移。Δν ≈ −f0·(K·ε + K_T·ΔT)。
%   用法: [iw, dnu] = ofdr_shift(s_ref, s_meas, 256, B)    iw 各段中心单元下标，dnu 谱移 [Hz]
if nargin < 5, pad = 8; end
p1 = ifft(s_ref); p2 = ifft(s_meas);
n = floor(numel(p1)/nw);
spec = @(p) abs(fft(reshape(p(1:n*nw), nw, n), nw*pad, 1));     % 每列一段的局部谱
a = spec(p1); b = spec(p2);
a = a - mean(a, 1); b = b - mean(b, 1);
xc = abs(ifft(fft(b, [], 1) .* conj(fft(a, [], 1)), [], 1));
[~, m] = max(xc, [], 1); m = m - 1;
m(m >= nw*pad/2) = m(m >= nw*pad/2) - nw*pad;
iw = (0:n-1)*nw + floor(nw/2) + 1;
dnu = m * B/(nw*pad);
end
