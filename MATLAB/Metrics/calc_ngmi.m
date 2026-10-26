function g = calc_ngmi(rx, tx, c, prior)
%CALC_NGMI 归一化 GMI：NGMI = 1 − (H(X) − GMI)/log2(M)（SD-FEC 门限常用 NGMI 判定，如 0.8~0.9）。
%   用法: g = calc_ngmi(rx, tx, c)  /  g = calc_ngmi(rx, tx, c, prior)
if nargin < 4, prior = []; end
k = log2(numel(c));
H = k; if ~isempty(prior), H = -sum(prior .* log2(prior)); end
g = 1 - (H - calc_gmi(rx, tx, c, prior))/k;
end
