function b3 = beta3_from_DS(D, S, lam)
%BETA3_FROM_DS 色散斜率换算：S [ps/(nm^2 km)] -> β3 [s^3/m]，β3 = (λ/2πc)^2·(λ^2·S + 2λ·D)。
%   用法: b3 = beta3_from_DS(17, 0.057, 1550)
c = phys_const(); l = lam*1e-9;
b3 = (l/(2*pi*c.c))^2 * (l^2*S*1e3 + 2*l*D*1e-6);
end
