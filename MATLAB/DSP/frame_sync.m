function [ya, lag, perm] = frame_sync(y, ref, use_abs)
%FRAME_SYNC 帧同步 + 偏振交换检测（FFT 循环互相关，取 |相关| 峰）。
%   用法: [ya, lag, perm] = frame_sync(y, frame)            相位已恢复（相干相关）
%         [ya, lag, perm] = frame_sync(y, frame, true)      用 |y| 相关（相位未恢复时，同参考代码 *_abs）
%   输出 ya = y(perm, lag+1 : lag+Nf)（循环），lag 为 0-based 偏移，perm 为偏振顺序
if nargin < 3, use_abs = false; end
N = size(y,2);
f = @(v) v;
if use_abs, f = @(v) abs(v) - mean(abs(v), 2); end
Y = fft(f(y), [], 2); R = fft(f(ref), N, 2);
c = zeros(2, 2, N);
for i = 1:2, for j = 1:2, c(i,j,:) = abs(ifft(Y(i,:) .* conj(R(j,:)))); end, end
if max(c(1,1,:)) + max(c(2,2,:)) >= max(c(1,2,:)) + max(c(2,1,:)), perm = [1 2]; else, perm = [2 1]; end
[~, lag] = max(squeeze(c(perm(1),1,:) + c(perm(2),2,:)));
lag = lag - 1;
ya = circshift(y(perm,:), -lag, 2);
ya = ya(:, 1:size(ref,2));
end
