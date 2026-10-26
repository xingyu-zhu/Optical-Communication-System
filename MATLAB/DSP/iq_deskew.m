function y = iq_deskew(x, fs, skew)
%IQ_DESKEW 补偿 IQ skew：对各支路施加相反时延。
%   用法: y = iq_deskew(x, fs, [XI XQ YI YQ])    skew 为标定/估计的时延 [s]
y = x;
for p = 1:size(x,1)
    y(p,:) = iqdelay(x(p,:), -skew(2*p-1), -skew(2*p), fs);
end
end
