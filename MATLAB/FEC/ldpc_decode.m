function [b, ok] = ldpc_decode(llr, code, iters, alpha)
%LDPC_DECODE 归一化最小和（NMS）译码（flooding 调度，满足全部校验即提前停止）。
%   用法: [b, ok] = ldpc_decode(llr, code)     llr 为 nb×n 信道 LLR = log P(b=0)/P(b=1)（与 calc_llr 一致）
%         [b, ok] = ldpc_decode(llr, code, iters, alpha)    默认 30 次迭代，alpha = 0.75
%   输出 b 为 nb×n 硬判决码字（uint8），ok 为 nb×1 是否满足全部校验；信息位为 b(:, code.free)
if nargin < 3, iters = 30; end
if nargin < 4, alpha = 0.75; end
cn = code.cn; ve = code.ve;
[m, dc] = size(cn); [n, dv] = size(ve); nb = size(llr, 1);
L = llr.';                                           % n×nb
c2v = zeros(m*dc, nb);                               % 边 × 码字
for it = 1:iters
    tot = L + reshape(sum(reshape(c2v(ve(:), :), n, dv, nb), 2), n, nb);
    hard = tot < 0;
    ok = ~any(mod(reshape(sum(reshape(hard(cn(:), :), m, dc, nb), 2), m, nb), 2), 1);
    if all(ok), break; end
    v2c = reshape(tot(cn(:), :), m, dc, nb) - reshape(c2v, m, dc, nb);
    a = abs(v2c);
    [as, i1] = sort(a, 2);
    mag = as(:,1,:) + (as(:,2,:) - as(:,1,:)) .* ((1:dc) == i1(:,1,:));   % 自身是最小值时用次小值
    sg = 1 - 2*(v2c < 0);
    c2v = reshape(alpha * prod(sg, 2) .* sg .* mag, m*dc, nb);
end
b = uint8(hard.');
ok = ok(:);
end
