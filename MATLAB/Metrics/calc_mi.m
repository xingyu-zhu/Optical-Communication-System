function I = calc_mi(rx, tx, c, prior)
%CALC_MI 符号级互信息 MI [bit/sym/pol]（高斯辅助信道）：E[log2 q(y|x) / Σ_s p(s)q(y|s)]。
%   用法: I = calc_mi(rx, tx, c)   /   I = calc_mi(rx, tx, c, prior)
rx = rx(:); tx = tx(:);
if nargin < 4 || isempty(prior), prior = ones(size(c))/numel(c); end
s2 = mean(abs(rx - tx).^2);
a = -abs(rx - c(:).').^2/s2 + log(prior(:).');
mx = max(a, [], 2);
den = mx + log(sum(exp(a - mx), 2));
I = mean(-abs(rx - tx).^2/s2 - den) / log(2);
end
