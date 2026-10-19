function F = volterra_features(x, taps, sps)
%VOLTERRA_FEATURES 实数 Volterra 特征矩阵 Nsym×F：[1, 一阶 L1 项, 二阶 L2(L2+1)/2 项, 三阶 C(L3+2,3) 项]。
%   用法: F = volterra_features(x, [41 15 5], 2)      各阶为居中子窗；二/三阶记忆为 0 时省略该阶
%   训练得到核 w 后，新数据的均衡输出 = (volterra_features(x_new, taps, sps) * w).'
W = sym_windows(x, taps(1), sps);                         % Nsym×L1
c = floor(taps(1)/2);
F = [ones(size(W,1), 1), W];
for order = 2:3
    L = taps(order);
    if L == 0, continue; end
    sub = W(:, c - floor(L/2) + (1:L));
    comb = nmultichoosek(1:L, order);                     % 有放回组合 i ≤ j (≤ k)
    P = ones(size(W,1), size(comb,1));
    for m = 1:order, P = P .* sub(:, comb(:, m)); end
    F = [F, P]; %#ok<AGROW>
end
end

function C = nmultichoosek(v, k)
% 有放回组合（非降序），每行一组
if k == 2
    [a, b] = ndgrid(v, v); C = [a(:), b(:)]; C = C(C(:,1) <= C(:,2), :);
else
    [a, b, c] = ndgrid(v, v, v); C = [a(:), b(:), c(:)]; C = C(C(:,1) <= C(:,2) & C(:,2) <= C(:,3), :);
end
end
