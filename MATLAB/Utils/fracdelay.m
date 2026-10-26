function y = fracdelay(x, tau, fs)
%FRACDELAY 分数时延（频域线性相位，循环）。
%   用法: y = fracdelay(x, tau, fs)   tau [s] 为标量，或每行一个值的列向量
y = fdfilter(x, exp(-2j*pi*freqaxis(size(x,2), fs) .* tau(:)));
if isreal(x), y = real(y); end
end
