function y = fix_ambiguity(y, ref, mask, n)
%FIX_AMBIGUITY 用已知符号（mask 标记的训练/导频）消除每偏振 π/2 整数倍相位模糊。
%   用法: y = fix_ambiguity(y, frame)                  全部符号已知、全局一个旋转量
%         y = fix_ambiguity(y, frame, kind > 0)        仅用训练+导频，全局一个旋转量
%         y = fix_ambiguity(y, frame, kind > 0, 5)     已知符号 5 点滑动平均逐段判定（导频辅助周跳消除）
N = size(y,2);
if nargin < 3 || isempty(mask), mask = true(1, N); end
idx = find(mask);
r = y(:, idx) .* conj(ref(:, idx));
if nargin < 4
    k = round(angle(sum(r, 2)) / (pi/2));
else
    k = round(angle(movmean(r, n, 2)) / (pi/2));
    pos = max(cumsum(mask), 1);                      % 每个符号对应的前一个已知符号
    k = k(:, pos);
end
y = y .* exp(-1j*k*pi/2);
end
