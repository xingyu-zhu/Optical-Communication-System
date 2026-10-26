function s = sensitivity(x, ber, thr)
%SENSITIVITY 接收灵敏度：在 log10(BER)–x 曲线上线性插值，求 BER 恰为 thr 时的 x（如 ROP [dBm] 或 OSNR [dB]）。
%   用法: s = sensitivity(rop, ber)              默认 thr = 2.4e-4（KP4 FEC）
%         s = sensitivity(osnr, ber, 3.8e-3)     HD-FEC
%   BER 须随 x 单调下降；未跨过门限时返回 NaN。
if nargin < 3, thr = 2.4e-4; end
lb = log10(max(ber, 1e-12)); lt = log10(thr);
i = find(lb(1:end-1) >= lt & lb(2:end) < lt, 1);
if isempty(i), s = NaN; return; end
s = x(i) + (lt - lb(i)) / (lb(i+1) - lb(i)) * (x(i+1) - x(i));
end
