function E = pdl_elem(E, pdl_db, theta)
%PDL_ELEM 偏振相关损耗：方位角 theta 方向的偏振衰减 pdl_db。  用法: E = pdl_elem(E, 0.5, 0.3)
if nargin < 3, theta = 0; end
R = jones_rotator(theta);
E = R * diag([1, 10^(-pdl_db/20)]) * R' * E;
end
