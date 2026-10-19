function y = dac_model(x, fs, enob, bw, kind)
%DAC_MODEL DAC：ENOB 量化 + 模拟带宽（含驱动放大器响应）。
%   用法: y = dac_model(x, fs, 6, 28e9)          6 bit、28 GHz 高斯响应
%         y = dac_model(x, fs, [], bw, 'bessel')  enob/bw 置 [] 则跳过该项
if nargin < 5, kind = 'gaussian'; end
y = x;
if ~isempty(enob), y = quantize_enob(y, enob); end
if ~isempty(bw),   y = elpf(y, fs, bw, kind); end
end
