function [L_est, m] = cd_estimate(x, fs, rs, L_grid, D, lam)
%CD_ESTIMATE 盲色散估计：扫描补偿长度，代价函数为平方律时钟音 |Σ|x|²·e^{-j2πRs t}|（正确补偿时最大）。
%   用法: [L_est, m] = cd_estimate(x, fs, rs, 300:20:700)        返回最佳长度 [km] 与代价函数
%         [L_est, m] = cd_estimate(x, fs, rs, L_grid, D, lam)
if nargin < 5, D = 17; end
if nargin < 6, lam = 1550; end
tone = exp(-2j*pi*rs*(0:size(x,2)-1)/fs);
m = zeros(size(L_grid));
for k = 1:numel(L_grid)
    m(k) = abs(sum(sum(abs(cdc(x, fs, L_grid(k), D, lam)).^2, 1) .* tone));
end
[~, i] = max(m);
L_est = L_grid(i);
end
