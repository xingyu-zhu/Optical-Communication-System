function J = jones_rotator(theta, phi)
%JONES_ROTATOR 偏振旋转 Jones 矩阵（方位角 theta，相位差 phi）。  用法: E = jones_rotator(pi/6)*E
if nargin < 2, phi = 0; end
J = [cos(theta), -sin(theta)*exp(-1j*phi); sin(theta)*exp(1j*phi), cos(theta)];
end
