function m = rot_order(c)
%ROT_ORDER 星座旋转对称阶数 m：绕原点旋转 2π/m 后与自身重合的最大 m（QAM/QPSK 4，BPSK 2，8PSK 8）。
%   用法: m = rot_order(psk_const(8))
%   决定 FOE 的 M 次方、BPS 搜索范围 ±π/m 与相位模糊步长 2π/m。
c = c(:);
for m = numel(c):-1:2
    if all(min(abs(c*exp(2j*pi/m) - c.'), [], 2) < 1e-9), return; end
end
m = 1;
end
