function y = pulse_compress(x, p)
%PULSE_COMPRESS 脉冲压缩（匹配滤波）：x 与探测脉冲 p 互相关，输出与 x 等长，峰值对齐到散射单元位置。
%   用法: y = pulse_compress(x, p)    x 按行（K×N），p 为行向量；LFM 压缩后分辨率 ≈ c/(2·ng·B)，与脉宽无关
L = size(x,2) + numel(p) - 1;
y = ifft(fft(x, L, 2) .* fft(conj(fliplr(p)), L, 2), [], 2);
y = y(:, numel(p):numel(p)+size(x,2)-1) / sum(abs(p).^2);
end
