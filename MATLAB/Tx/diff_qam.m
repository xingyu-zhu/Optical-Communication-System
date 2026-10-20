function s = diff_qam(bits, M)
%DIFF_QAM 差分象限编码 QAM（对 π/2 整数倍相位模糊和周跳免疫）：每符号前 2 比特（Gray）差分编码象限
%   q_k = q_{k-1} + u_k (mod 4)，其余比特映射到第一象限的点，再乘 j^{q_k}。M=4 即 DQPSK。
%   用法: s = diff_qam(bits, 16)    bits: P×(n·k)，s: P×n；解调用 diff_qam_demod
k = log2(M); n = size(bits,2)/k;
cq = qam_quadrant(M); g = [0 1 3 2];
s = zeros(size(bits,1), n);
for p = 1:size(bits,1)
    b = reshape(double(bits(p,:)), k, n).';           % n×k
    u = g(2*b(:,1) + b(:,2) + 1);
    j = zeros(n, 1);
    if k > 2, j = b(:,3:end) * 2.^(k-3:-1:0).'; end
    s(p,:) = cq(j(:).' + 1) .* 1j.^mod(cumsum(u(:).'), 4);
end
end
