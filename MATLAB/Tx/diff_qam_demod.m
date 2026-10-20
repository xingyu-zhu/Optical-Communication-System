function bits = diff_qam_demod(y, M)
%DIFF_QAM_DEMOD diff_qam 的硬判决解调：象限 q = ⌊∠y/(π/2)⌋，旋回第一象限判决内层比特，u = q_k − q_{k-1}。
%   用法: bits = diff_qam_demod(y, 16)    y: P×n，bits: P×(n·k) uint8（首符号以 q_{-1}=0 为参考）
k = log2(M); cq = qam_quadrant(M); g = [0 1 3 2];
bits = zeros(size(y,1), size(y,2)*k, 'uint8');
for p = 1:size(y,1)
    q = mod(floor(mod(angle(y(p,:)), 2*pi)/(pi/2)), 4);
    u = g(mod(diff([0, q]), 4) + 1);
    b = [floor(u(:)/2), mod(u(:), 2)];
    if k > 2, b = [b, dec2bin(sym_decide(y(p,:) .* (-1j).^q, cq), k-2) - '0']; end
    bits(p,:) = reshape(b.', 1, []);
end
end
