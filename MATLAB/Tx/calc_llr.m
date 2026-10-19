function L = calc_llr(x, c, sigma2, prior)
%CALC_LLR 逐比特对数似然比 LLR = log P(b=0|y)/P(b=1|y)（高斯信道，精确 log-sum-exp）。
%   用法: L = calc_llr(y, c, sigma2)          sigma2 为复噪声方差
%         L = calc_llr(y, c, sigma2, prior)   prior 为符号先验（PCS）
%   输出 1×(N·k)，顺序 [符号1的k比特, 符号2的k比特, ...]
k = log2(numel(c));
m = -abs(x(:) - c(:).').^2 / sigma2;                 % N×M
if nargin > 3 && ~isempty(prior), m = m + log(prior(:).'); end
lab = dec2bin(0:numel(c)-1, k) - '0';                % M×k
L = zeros(numel(x), k);
for i = 1:k
    L(:,i) = lse(m(:, lab(:,i)==0)) - lse(m(:, lab(:,i)==1));
end
L = reshape(L.', 1, []);
end

function y = lse(a)
mx = max(a, [], 2);
y = mx + log(sum(exp(a - mx), 2));
end
