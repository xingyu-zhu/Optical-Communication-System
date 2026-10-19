function p = sigpow(x)
%SIGPOW 每行（每偏振）平均功率 E|x|^2，输出列向量。总功率: sum(sigpow(x))。
%   用法: p = sigpow(x)      x: P×N
p = mean(abs(x).^2, 2);
end
