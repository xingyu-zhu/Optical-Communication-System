function y = adc_model(x, fs_in, fs_out, enob, bw, kind)
%ADC_MODEL ADC：前端带宽 -> 重采样到 fs_out -> ENOB 量化。
%   用法: y = adc_model(x, 160e9, 100e9, 6, 30e9)    enob/bw 置 [] 则跳过该项
if nargin < 6, kind = 'gaussian'; end
y = x;
if ~isempty(bw),   y = elpf(y, fs_in, bw, kind); end
y = resamp(y, fs_in, fs_out);
if ~isempty(enob), y = quantize_enob(y, enob); end
end
