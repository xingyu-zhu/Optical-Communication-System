function E = optical_filter(E, fs, bw, order, fc)
%OPTICAL_FILTER 超高斯光带通（WSS / 解复用器）：|H|^2 = exp(-ln2·(2(f-fc)/bw)^(2·order))。
%   用法: E = optical_filter(E, fs, 45e9)              3 dB 带宽 45 GHz，2 阶
%         E = optical_filter(E, fs, bw, order, fc)     order=1 高斯，越大越矩形；fc 中心频偏 [Hz]（WDM 选信道）
if nargin < 4, order = 2; end
if nargin < 5, fc = 0; end
f = freqaxis(size(E,2), fs) - fc;
E = fdfilter(E, exp(-log(2)/2 * (2*f/bw).^(2*order)));
end
