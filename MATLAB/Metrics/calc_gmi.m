function g = calc_gmi(rx, tx, c, prior)
%CALC_GMI 广义互信息 GMI [bit/sym/pol]（BICM 可达速率，高斯辅助信道）：
%   GMI = H(X) - (1/N)·Σ_k Σ_i log2(1 + exp(-(1-2b_ki)·LLR_ki))
%   用法: g = calc_gmi(rx, tx, c)            均匀输入
%         g = calc_gmi(rx, tx, c, prior)     PCS（prior 为符号先验）
if nargin < 4, prior = []; end
rx = rx(:); tx = tx(:); k = log2(numel(c));
s2 = mean(abs(rx - tx).^2);
L = calc_llr(rx, c, s2, prior);
b = reshape((dec2bin(sym_decide(tx, c), k) - '0').', 1, []);
H = k; if ~isempty(prior), H = -sum(prior .* log2(prior)); end
z = -(1 - 2*b) .* L;
g = H - mean(max(z, 0) + log1p(exp(-abs(z)))) / log(2) * k;     % log(1+e^z) 数值稳定形式
end
