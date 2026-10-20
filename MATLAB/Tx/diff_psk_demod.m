function bits = diff_psk_demod(z, M)
%DIFF_PSK_DEMOD 差分 M-PSK 判决：z = y_k·conj(y_{k-1})，就近判 Δφ = 2π·i/M，i -> Gray 标签 -> 比特。
%   用法: bits = diff_psk_demod(z, 4)    z 1×n，bits 1×(n·k) uint8
k = log2(M);
i = mod(round(angle(z) / (2*pi/M)), M);
g = bitxor(0:M-1, bitshift(0:M-1, -1));
b = dec2bin(g(i + 1), k) - '0';
bits = uint8(reshape(b.', 1, []));
end
