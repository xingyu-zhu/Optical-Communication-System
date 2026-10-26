function y = normpow(x)
%NORMPOW 每行归一化到单位平均功率。  用法: y = normpow(x)
y = x ./ sqrt(sigpow(x));
end
