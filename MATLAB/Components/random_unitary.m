function U = random_unitary()
%RANDOM_UNITARY 随机 SU(2) Jones 矩阵（Poincaré 球上均匀分布的 SOP 旋转）。  用法: E = random_unitary()*E
a = randn(4,1); a = a / norm(a);
U = [a(1)+1j*a(2), a(3)+1j*a(4); -a(3)+1j*a(4), a(1)-1j*a(2)];
end
