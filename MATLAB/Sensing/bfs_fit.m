function vb = bfs_fit(f, g, frac)
%BFS_FIT 逐位置提取 BFS：取峰值 frac 倍以上的点做二次拟合，返回顶点频率（洛伦兹拟合的快速近似）。
%   用法: vb = bfs_fit(f_scan, g)     g 为 numel(f)×Nz（botda_bgs 的输出），vb 为 1×Nz [Hz]
if nargin < 3, frac = 0.5; end
f = f(:); fm = mean(f); fc = (f - fm)/1e6;          % 中心化并以 MHz 为单位，避免病态
vb = zeros(1, size(g, 2));
for j = 1:size(g, 2)
    m = g(:, j) >= frac*max(g(:, j));
    c = polyfit(fc(m), g(m, j), 2);
    vb(j) = -c(2)/(2*c(1))*1e6 + fm;
end
end
