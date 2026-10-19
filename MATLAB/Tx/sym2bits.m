function bits = sym2bits(x, c)
%SYM2BITS 硬判决解映射：符号 -> 比特。
%   用法: bits = sym2bits(x, c)    x: P×n，bits: P×(n·k) uint8
k = log2(numel(c));
bits = zeros(size(x,1), size(x,2)*k, 'uint8');
for p = 1:size(x,1)
    b = dec2bin(sym_decide(x(p,:), c), k) - '0';      % n×k
    bits(p,:) = reshape(b.', 1, []);
end
end
