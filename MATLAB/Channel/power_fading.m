function H = power_fading(f, L, D, lam, alpha)
%POWER_FADING IM-DD 色散功率衰落的小信号频率响应 |H(f)| = |cos θ - α·sin θ|，θ = π·λ²·D·L·f²/c。
%   用法: H = power_fading(f, 10)                     C 波段 SSMF 10 km，无啁啾
%         H = power_fading(f, L, D, lam, alpha)       f [Hz]，L [km]，D [ps/(nm km)]，lam [nm]，α 为啁啾 Henry 因子
%   α=0 时第 k 个陷波位于 f_k = sqrt((2k-1)·c / (2·λ²·D·L))。
if nargin < 3, D = 17; end
if nargin < 4, lam = 1550; end
if nargin < 5, alpha = 0; end
c = phys_const();
th = pi*(lam*1e-9)^2*D*1e-6*L*1e3*f.^2/c.c;
H = abs(cos(th) - alpha*sin(th));
end
