function c = psk_const(M)
%PSK_CONST Gray 映射 M-PSK（QPSK 旋转 π/4），单位幅度。  用法: c = psk_const(8)
g = bitxor(0:M-1, bitshift(0:M-1, -1));
c = zeros(1, M);
c(g+1) = exp(1j*(2*pi*(0:M-1)/M + pi/4*(M==4)));
end
