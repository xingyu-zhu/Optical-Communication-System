function f = mm_overlap(hg, w, npts)
%MM_OVERLAP 模场交叠积分 f_ab = ∬ψa²ψb² dA [1/m²]，ψ_(m,n)(x,y) = u_m(x)·u_n(y)（可分离，逐维积分）。
%   用法: f = mm_overlap([0 0; 1 0; 0 1], 6e-6)
%   hg M×2 各空间模式的 HG 阶数；w 模场半径 [m]；f_aa = 1/Aeff,a（基模 Aeff = π·w²）。对应 Python multimode.overlap
if nargin < 3, npts = 4001; end
nmax = max(hg(:));
x = linspace(-1, 1, npts)*w*(6 + nmax);
u2 = hg1d(nmax, x, w).^2;
I = u2*u2.'*(x(2) - x(1));                                 % I(i,j) = ∫u_i² u_j² dx
f = I(hg(:,1)+1, hg(:,1)+1) .* I(hg(:,2)+1, hg(:,2)+1);
end
