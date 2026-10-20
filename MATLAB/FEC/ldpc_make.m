function code = ldpc_make(n, dv, dc)
%LDPC_MAKE 随机规则 (dv, dc) LDPC（Gallager 构造，n 须为 dc 的整数倍；码率 ≥ 1 − dv/dc），
%   并做 GF(2) 高斯消元得到系统编码所需的校验位生成矩阵。
%   用法: code = ldpc_make(1944, 3, 6)
%   code 字段: H (m×n 逻辑矩阵); cn (m×dc 各校验节点相连的变量下标); ve (n×dv 各变量节点的边下标，边按 cn(:) 编号);
%              piv / free（校验位 / 信息位列）; P（校验位 = P·u mod 2）; k 信息位数
if nargin < 2, dv = 3; end
if nargin < 3, dc = 6; end
base = reshape(1:n, dc, []).';
cn = base;
for i = 2:dv, pm = randperm(n); cn = [cn; pm(base)]; end %#ok<AGROW>
m = size(cn, 1);
H = false(m, n);
H(sub2ind([m n], repmat((1:m).', 1, dc), cn)) = true;
R = H; piv = []; r = 1;                              % GF(2) 消元 -> 行最简形
for c = 1:n
    i = find(R(r:end, c), 1) + r - 1;
    if isempty(i), continue; end
    R([r i], :) = R([i r], :);
    o = find(R(:, c)); o(o == r) = [];
    R(o, :) = xor(R(o, :), R(r, :));
    piv(end+1) = c; r = r + 1; %#ok<AGROW>
    if r > m, break; end
end
free = setdiff(1:n, piv);
[~, ord] = sort(cn(:));
code = struct('H', H, 'cn', cn, 've', reshape(ord, dv, n).', 'piv', piv, 'free', free, ...
              'P', double(R(1:r-1, free)), 'k', numel(free));
end
