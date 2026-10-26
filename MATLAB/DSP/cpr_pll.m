function [y, phi] = cpr_pll(x, c, mu, rho)
%CPR_PLL 二阶判决引导锁相环（DD-PLL）：e = Im{y·d*}，f ← f + ρe，θ ← θ + μe + f。
%   用法: [y, phi] = cpr_pll(x, c)              默认 mu = 0.05, rho = 1e-3
%         [y, phi] = cpr_pll(x, c, mu, rho)
if nargin < 3, mu = 0.05; end
if nargin < 4, rho = 1e-3; end
y = zeros(size(x)); phi = zeros(size(x));
for p = 1:size(x,1)
    th = 0; fr = 0;
    for k = 1:size(x,2)
        y(p,k) = x(p,k) * exp(-1j*th);
        [~, d] = sym_decide(y(p,k), c);
        e = imag(y(p,k) * conj(d));
        fr = fr + rho*e;
        th = th + mu*e + fr;
        phi(p,k) = th;
    end
end
end
