function u = hg1d(nmax, x, w)
%HG1D 归一化一维 Hermite-Gauss 模 u_n(x)，n = 0..nmax，输出 (nmax+1)×numel(x)，∫u_n² dx = 1。
%   用法: u = hg1d(5, x, 6e-6)
%   u_n(x) = (2/w²)^{1/4}·h_n(√2·x/w)，h_n 为 Hermite 函数（三项递推，数值稳定）。对应 Python multimode.hg1d
s = sqrt(2)*x(:).'/w;
h = zeros(nmax+1, numel(s));
h(1,:) = pi^-0.25*exp(-s.^2/2);
if nmax > 0, h(2,:) = sqrt(2)*s.*h(1,:); end
for k = 1:nmax-1
    h(k+2,:) = sqrt(2/(k+1))*s.*h(k+1,:) - sqrt(k/(k+1))*h(k,:);
end
u = h*(2/w^2)^0.25;
end
