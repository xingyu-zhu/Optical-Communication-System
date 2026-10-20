function T = dts_temp(R, R_ref, T_ref, z, dalpha, dv)
%DTS_TEMP 拉曼 DTS 温度反演（参考段标定，消去未知比例常数）：
%   1/T = 1/T_ref − (k/hΔν)·ln(R·e^{Δα·z}/R_ref)，Δα = α_as − α_s 为两波长往返损耗差 [1/m]。
%   用法: T = dts_temp(R, R_ref, T_ref)
%         T = dts_temp(R, R_ref, T_ref, z, dalpha)
if nargin < 4, z = 0; end
if nargin < 5, dalpha = 0; end
if nargin < 6, dv = 13.2e12; end
c0 = phys_const();
T = 1 ./ (1/T_ref - c0.kB/(c0.h*dv) * log(R.*exp(dalpha*z)/R_ref));
end
