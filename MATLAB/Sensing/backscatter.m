function y = backscatter(p, h, n)
%BACKSCATTER 背向散射光场 = 探测光 p 与散射冲激响应 h 的线性卷积（按行；p 为单行时对 h 的所有行共用）。
%   用法: y = backscatter(p, h)       h 一般为 r.*exp(1j*phi)（phi 为扰动引入的往返相位），可为 K×N
%         y = backscatter(p, h, n)    只取前 n 点
L = size(p,2) + size(h,2) - 1;
y = ifft(fft(p, L, 2) .* fft(h, L, 2), [], 2);
if nargin > 2, y = y(:, 1:n); end
end
