function snr = osnr2snr(osnr_db, rs, bref, npol)
%OSNR2SNR OSNR（参考带宽 bref）-> 每偏振符号 SNR [dB]：SNR = OSNR + 10·log10(2·bref/(npol·rs))。
%   用法: snr = osnr2snr(20, 32e9)     默认 bref = 12.5 GHz、npol = 2
if nargin < 3, bref = 12.5e9; end
if nargin < 4, npol = 2; end
snr = osnr_db + 10*log10(2*bref/(npol*rs));
end
