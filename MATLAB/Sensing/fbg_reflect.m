function R = fbg_reflect(lam, lam_b, L, dn, neff)
%FBG_REFLECT 均匀 FBG 反射谱（耦合模理论解析解，Erdogan 1997）：
%   R = sinh²(sL) / (cosh²(sL) − σ²/κ²)，s = √(κ²−σ²)，κ = π·dn/λ，σ = 2π·neff·(1/λ − 1/λ_B)
%   用法: R = fbg_reflect(lam, 1550)                 lam、lam_b [nm]
%         R = fbg_reflect(lam, lam_b, L, dn, neff)   L 栅长 [m]（默认 10 mm），dn 调制深度（默认 1e-4）
if nargin < 3, L = 10e-3; end
if nargin < 4, dn = 1e-4; end
if nargin < 5, neff = 1.447; end
l = lam*1e-9;
k = pi*dn./l;
sg = 2*pi*neff*(1./l - 1/(lam_b*1e-9));
s = sqrt(complex(k.^2 - sg.^2));
R = abs(sinh(s*L).^2 ./ (cosh(s*L).^2 - sg.^2./k.^2));
end
