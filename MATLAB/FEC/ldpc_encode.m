function c = ldpc_encode(u, code)
%LDPC_ENCODE 系统编码：码字信息位 = u，校验位 = P·u (mod 2)。
%   用法: c = ldpc_encode(u, code)    u 为 nb×k，c 为 nb×n uint8
c = zeros(size(u,1), size(code.H,2), 'uint8');
c(:, code.free) = u;
c(:, code.piv) = mod(double(u)*code.P.', 2);
end
