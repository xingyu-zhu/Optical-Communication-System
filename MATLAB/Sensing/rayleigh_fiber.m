function [z, r] = rayleigh_fiber(L, dz, alpha_db)
%RAYLEIGH_FIBER 随机瑞利散射单元：r_i ~ CN(0, αs·S·dz)，乘往返损耗 e^{-αz}（α 为功率衰减系数）。
%   用法: [z, r] = rayleigh_fiber(2000, 0.41)          默认 0.2 dB/km
%         [z, r] = rayleigh_fiber(L, dz, alpha_db)     z、r 为 1×N
if nargin < 3, alpha_db = 0.2; end
s = sens_const();
z = (0:floor(L/dz)-1)*dz;
a = alpha_db*log(10)/10/1e3;
r = (randn(size(z)) + 1j*randn(size(z))) * sqrt(s.alpha_s*s.capture*dz/2) .* exp(-a*z);
end
