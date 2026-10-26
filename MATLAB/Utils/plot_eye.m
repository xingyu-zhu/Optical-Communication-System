function plot_eye(x, sps, n)
%PLOT_EYE 眼图（取实部，2 UI 一段，最多 n 段，默认 500）。  用法: plot_eye(y, sps)
if nargin < 3, n = 500; end
L = 2*sps; x = real(x(:));
n = min(n, floor(numel(x)/L));
plot((0:L-1)/sps, reshape(x(1:n*L), L, n), 'b'); xlabel('Time (UI)');
end
