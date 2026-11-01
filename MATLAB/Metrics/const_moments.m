function [mu4, mu6] = const_moments(c, p)
%CONST_MOMENTS 星座归一化高阶矩 mu4 = E|x|^4/(E|x|^2)^2，mu6 = E|x|^6/(E|x|^2)^3。
%   决定非线性干扰对调制格式的依赖（EGN 模型修正项含 mu4-2 与 mu6）：
%   QPSK (1, 1)，16QAM (1.32, 1.96)，64QAM (1.381, 2.226)，复高斯 (2, 6)。
%   用法: [mu4, mu6] = const_moments(qam_const(64))
%         [mu4, mu6] = const_moments(c, p)      p 为 PCS 先验概率
c = c(:);
if nargin < 2, p = ones(size(c)) / numel(c); else, p = p(:); end
m2 = sum(p .* abs(c).^2);
mu4 = sum(p .* abs(c).^4) / m2^2;
mu6 = sum(p .* abs(c).^6) / m2^3;
end
