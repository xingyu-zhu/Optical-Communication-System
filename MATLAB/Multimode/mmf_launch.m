function P = mmf_launch(mmf, kind, w_um, offset_um, npts)
%MMF_LAUNCH 激励条件 -> 各模式群功率 P_g（1×G）。对应 Python multimode.mmf_launch
%   用法: P = mmf_launch(mmf, 'ofl')                      满注入：各模式等功率，P_g ∝ g，ΣP_g = 1
%         P = mmf_launch(mmf, 'gauss', 4)                  居中高斯光斑（模场半径 4 µm）
%         P = mmf_launch(mmf, 'gauss', 4, 8)               横向偏移 8 µm
%   'gauss'：对 HG 模做交叠积分 |∫ψ_launch·ψ_mn dA|²，按 N = m+n 归入第 N+1 群；ΣP_g ≤ 1（其余耦合到非导模）
if nargin < 2, kind = 'ofl'; end
G = mmf.n_groups;
if strcmp(kind, 'ofl')
    P = (1:G)/sum(1:G);
    return
end
if nargin < 3 || isempty(w_um), w_um = mmf.w0_um; end
if nargin < 4, offset_um = 0; end
if nargin < 5, npts = 4001; end
w0 = mmf.w0_um*1e-6; ws = w_um*1e-6; d = offset_um*1e-6;
x = linspace(-1, 1, npts)*w0*(6 + G) + d/2;
dx = x(2) - x(1);
u = hg1d(G - 1, x, w0);
gx = exp(-(x - d).^2/ws^2); gy = exp(-x.^2/ws^2);
gx = gx/sqrt(sum(gx.^2)*dx); gy = gy/sqrt(sum(gy.^2)*dx);
cx = (u*gx.'*dx).^2; cy = (u*gy.'*dx).^2;
P = zeros(1, G);
for N = 0:G-1
    n = 0:N;
    P(N+1) = sum(cx(n+1) .* cy(N-n+1));
end
end
