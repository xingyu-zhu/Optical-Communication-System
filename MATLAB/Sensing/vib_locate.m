function [i, e] = vib_locate(dphi, n)
%VIB_LOCATE 振动定位：各位置差分相位的时间标准差，经 n 点中值滤波（去掉相干衰落点伪峰）后取最大值。
%   用法: [i, e] = vib_locate(dphi, n)    i 为下标，e 为能量曲线
e = std(dphi, 0, 1);
if nargin > 1 && n > 1, e = medfilt1(e, n); end
[~, i] = max(e);
end
