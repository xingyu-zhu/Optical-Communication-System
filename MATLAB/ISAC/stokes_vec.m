function S = stokes_vec(E)
%STOKES_VEC Jones 向量 -> 归一化 Stokes 向量：[|Ex|²−|Ey|²; 2Re(Ex·Ey*); −2Im(Ex·Ey*)]/S0。
%   用法: S = stokes_vec(E)    E 为 2×N，S 为 3×N
s0 = abs(E(1,:)).^2 + abs(E(2,:)).^2;
S = [abs(E(1,:)).^2 - abs(E(2,:)).^2; 2*real(E(1,:).*conj(E(2,:))); -2*imag(E(1,:).*conj(E(2,:)))] ./ s0;
end
