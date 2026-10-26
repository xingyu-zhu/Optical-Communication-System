function R = raman_ratio(T, dv, lam)
%RAMAN_RATIO 反斯托克斯/斯托克斯强度比 R(T) = (λs/λas)⁴·exp(-hΔν/kT)，Δν 为拉曼频移（约 13.2 THz）。
%   用法: R = raman_ratio(T)     T [K]；lam 泵浦波长 [nm]（默认 1550）
if nargin < 2, dv = 13.2e12; end
if nargin < 3, lam = 1550; end
c0 = phys_const();
f0 = c0.c/(lam*1e-9);
R = ((f0 + dv)/(f0 - dv))^4 * exp(-c0.h*dv ./ (c0.kB*T));
end
