function osnr = snr2osnr(snr_db, rs, bref, npol)
%SNR2OSNR 每偏振符号 SNR -> OSNR(bref) [dB]（osnr2snr 的逆）。  用法: osnr = snr2osnr(15, 32e9)
if nargin < 3, bref = 12.5e9; end
if nargin < 4, npol = 2; end
osnr = snr_db - 10*log10(2*bref/(npol*rs));
end
