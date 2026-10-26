function y = setpow(x, p_dbm)
%SETPOW 缩放信号使总功率（所有行之和）为 p_dbm。  用法: E = setpow(E, 0)
y = x * sqrt(dbm2w(p_dbm) / sum(sigpow(x)));
end
