function b2 = beta2_from_D(D, lam)
%BETA2_FROM_D 色散换算：D [ps/(nm km)], lam [nm] -> β2 [s^2/m]。  用法: b2 = beta2_from_D(17, 1550)
c = phys_const();
b2 = -D*1e-6 * (lam*1e-9)^2 / (2*pi*c.c);
end
