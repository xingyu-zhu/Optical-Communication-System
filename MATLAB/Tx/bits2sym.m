function s = bits2sym(bits, c)
%BITS2SYM 比特 -> 符号（每 log2(M) 比特一组，MSB 在前）。
%   用法: s = bits2sym(bits, qam_const(16))    bits: P×(n·k)，s: P×n
k = log2(numel(c));
s = zeros(size(bits,1), size(bits,2)/k);
for p = 1:size(bits,1)
    idx = reshape(double(bits(p,:)), k, []).' * 2.^(k-1:-1:0).';
    s(p,:) = c(idx + 1);
end
end
