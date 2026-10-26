function y = quantize_enob(x, enob, clip)
%QUANTIZE_ENOB 按有效位数量化（复数时 I/Q 分别量化）。
%   用法: y = quantize_enob(x, 6)          满量程 = 3 × 每行 RMS
%         y = quantize_enob(x, 6, clip)    clip 决定削波程度
if nargin < 3, clip = 3; end
q = @(v) quant(v, enob, clip);
if isreal(x), y = q(x); else, y = q(real(x)) + 1j*q(imag(x)); end
end

function y = quant(v, enob, clip)
fsr = clip * sqrt(mean(v.^2, 2));
step = 2*fsr / 2^enob;
lim = 2^enob/2 - 0.5;
y = min(max(round(v./step - 0.5) + 0.5, -lim), lim) .* step;
end
