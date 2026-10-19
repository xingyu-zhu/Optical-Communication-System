function [idx, d] = sym_decide(x, c)
%SYM_DECIDE 最小欧氏距离硬判决。
%   用法: [idx, d] = sym_decide(x, c)   idx 为 0-based 比特标签（与 x 同尺寸），d 为判决后的符号
[~, i] = min(abs(x(:) - c(:).'), [], 2);
idx = reshape(i - 1, size(x));
d = reshape(c(i), size(x));
end
