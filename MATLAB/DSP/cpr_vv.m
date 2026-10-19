function [y, phi] = cpr_vv(x, n, m, c)
%CPR_VV Viterbi-Viterbi M 次方相位估计：φ = (unwrap∠<x^M>_n - ∠E[c^M]) / M。含 2π/M 模糊。
%   用法: [y, phi] = cpr_vv(x, 64)            滑动窗 64，M=4
%         [y, phi] = cpr_vv(x, 64, 4, c)      给出星座 c 时扣除 E[c^M] 的固有相位
if nargin < 3, m = 4; end
ref = pi; if nargin > 3, ref = angle(sum(c.^m)); end
phi = (unwrap(angle(movmean(x.^m, n, 2)), [], 2) - ref) / m;
y = x .* exp(-1j*phi);
end
