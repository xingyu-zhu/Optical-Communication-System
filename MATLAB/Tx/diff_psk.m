function s = diff_psk(bits, M)
%DIFF_PSK 差分 M-PSK 编码（DBPSK / DQPSK / D8PSK）：每 log2(M) 比特（Gray）决定相位增量 Δφ = 2π·i/M，
%   s_k = s_{k-1}·e^{jΔφ_k}，s_0 之前取 1。
%   用法: s = diff_psk(bits, 4)    bits 1×(n·k)，s 1×n 单位幅度符号
%   接收端只需差分积 z_k = y_k·conj(y_{k-1})（DLI 直接检测或数字差分），用 diff_psk_demod 判决。
k = log2(M);
lab = reshape(double(bits), k, []).' * 2.^(k-1:-1:0).';
g = bitxor(0:M-1, bitshift(0:M-1, -1));
inv = zeros(1, M); inv(g + 1) = 0:M-1;                    % Gray 标签 -> 增量序号
s = exp(2j*pi/M * cumsum(inv(lab + 1)));
end
