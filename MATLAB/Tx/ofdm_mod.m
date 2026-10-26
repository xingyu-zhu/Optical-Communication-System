function x = ofdm_mod(X, ncp)
%OFDM_MOD OFDM 调制：X（nsym×nfft 子载波符号，FFT 顺序，0 为空载波）-> IFFT（×√nfft 保持功率）
%   -> 加循环前缀 ncp -> 串行输出 1×(nsym·(nfft+ncp))。解调见 ofdm_demod。
%   用法: x = ofdm_mod(X, ncp)
x = ifft(X, [], 2) * sqrt(size(X,2));
x = [x(:, end-ncp+1:end), x];
x = reshape(x.', 1, []);
end
