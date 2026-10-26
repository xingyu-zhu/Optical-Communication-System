function Y = ofdm_demod(y, nfft, ncp)
%OFDM_DEMOD OFDM 解调：串并转换 -> 去循环前缀 -> FFT（与 ofdm_mod 互逆）。
%   用法: Y = ofdm_demod(y, nfft, ncp)    y 行向量，Y 为 nsym×nfft
n = nfft + ncp; ns = floor(numel(y)/n);
Y = reshape(y(1:ns*n), n, ns).';
Y = fft(Y(:, ncp+1:end), [], 2) / sqrt(nfft);
end
