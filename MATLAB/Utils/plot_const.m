function plot_const(x, ttl)
%PLOT_CONST 星座密度图（2D 直方图）。  用法: plot_const(res.sym(1,:), 'X pol')
if nargin < 2, ttl = ''; end
x = x(:); lim = 1.3*max(abs([real(x); imag(x)]));
histogram2(real(x), imag(x), linspace(-lim, lim, 200), linspace(-lim, lim, 200), ...
    'DisplayStyle', 'tile', 'ShowEmptyBins', 'off', 'EdgeColor', 'none');
axis equal tight; colormap(turbo); title(ttl);
end
