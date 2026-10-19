function [y, phi] = cpr_pilot(x, tx, mask, n)
%CPR_PILOT 导频辅助相位恢复（推广自参考代码 PilotBased_CPE）：导频处 ∠(x·tx*)，n 个导频滑动平均，
%   线性插值到全部符号。无相位模糊。
%   用法: [y, phi] = cpr_pilot(x, frame, kind == 2, 5)    x 与 tx 须已帧同步；mask 为导频位置逻辑向量
if nargin < 4, n = 3; end
idx = find(mask);
r = movmean(x(:, idx) .* conj(tx(:, idx)), n, 2);
ph = unwrap(angle(r), [], 2);
phi = zeros(size(x));
for p = 1:size(x,1)
    phi(p,:) = interp1(idx, ph(p,:), 1:size(x,2), 'linear', 'extrap');
end
y = x .* exp(-1j*phi);
end
