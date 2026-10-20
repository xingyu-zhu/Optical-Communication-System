function [Ec, Ed] = dli(E, fs, tau, phi)
%DLI 延迟线干涉仪（非对称 MZI）：两臂时延差 tau [s]、相位偏置 phi。
%   用法: [Ec, Ed] = dli(E, fs, 1/rs, 0)
%   输出 相长口 Ec = (E(t) + e^{jφ}E(t-τ))/2，相消口 Ed = (E(t) - e^{jφ}E(t-τ))/2
if nargin < 4, phi = 0; end
Edel = exp(1j*phi) * fracdelay(E, tau, fs);
Ec = (E + Edel)/2; Ed = (E - Edel)/2;
end
