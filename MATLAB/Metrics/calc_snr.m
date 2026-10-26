function s = calc_snr(rx, tx)
%CALC_SNR 数据辅助有效 SNR [dB]：最小二乘复增益 a = <tx,rx>/<tx,tx>，SNR = |a|²P_tx / E|rx - a·tx|²。
%   用法: s = calc_snr(rx, tx)
rx = rx(:); tx = tx(:);
a = (tx' * rx) / (tx' * tx);
s = 10*log10(abs(a)^2 * mean(abs(tx).^2) / mean(abs(rx - a*tx).^2));
end
