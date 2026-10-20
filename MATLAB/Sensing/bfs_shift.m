function vb = bfs_shift(dT, eps, vb0, cT, ce)
%BFS_SHIFT 布里渊频移 BFS = vb0 + cT·ΔT + cε·ε [Hz]（ΔT [K]，ε [με]）。
%   用法: vb = bfs_shift(dT, eps)      典型 vb0 = 10.85 GHz，cT ≈ 1 MHz/K，cε ≈ 0.05 MHz/με
if nargin < 3, vb0 = 10.85e9; end
if nargin < 4, cT = 1.0e6; end
if nargin < 5, ce = 0.05e6; end
vb = vb0 + cT*dT + ce*eps;
end
