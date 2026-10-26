function y = addawgn(x, snr_db, sps)
%ADDAWGN 按每符号 SNR [dB] 加复高斯白噪声（每行独立按自身功率计算）。
%   用法: y = addawgn(x, snr_db)         1 sps 符号
%         y = addawgn(x, snr_db, sps)    sps 倍过采样波形：噪声铺满仿真带宽，匹配滤波后 SNR = snr_db
if nargin < 3, sps = 1; end
s2 = sigpow(x) * sps / 10^(snr_db/10);
y = x + sqrt(s2/2) .* (randn(size(x)) + 1j*randn(size(x)));
end
