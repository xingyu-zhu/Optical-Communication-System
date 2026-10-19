function y = fdfilter(x, H)
%FDFILTER 频域滤波（循环卷积）：ifft(fft(x)·H)，按行。
%   用法: y = fdfilter(x, H)   H 为 FFT 顺序响应（1×N 行向量对所有行广播，或与 x 同尺寸）
y = ifft(fft(x, [], 2) .* H, [], 2);
end
