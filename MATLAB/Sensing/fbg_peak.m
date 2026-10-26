function lc = fbg_peak(lam, R, frac)
%FBG_PEAK FBG 峰值解调：反射谱 frac·max 以上部分的质心 [nm]。
%   用法: lc = fbg_peak(lam, R)      默认 frac = 0.5
if nargin < 3, frac = 0.5; end
m = R >= frac*max(R);
lc = sum(lam(m).*R(m)) / sum(R(m));
end
