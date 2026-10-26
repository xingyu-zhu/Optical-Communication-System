function y = resamp(x, fs_in, fs_out)
%RESAMP 周期（FFT）重采样，按行。
%   y = resamp(x, fs_in, fs_out)
%   等效理想砖墙抗混叠，与本仿真的循环信号假设一致（无边缘瞬态）。输出点数 round(N·fs_out/fs_in)。
N = size(x, 2); M = round(N * fs_out / fs_in);
X = fft(x, [], 2);
K = min(N, M); np_ = ceil(K/2); nn = floor(K/2);
Y = zeros(size(x,1), M);
Y(:, 1:np_) = X(:, 1:np_);                               % 0 与正频率
Y(:, end-nn+1:end) = X(:, end-nn+1:end);                 % 负频率
if mod(K, 2) == 0, Y(:, end-nn+1) = 0; end               % 丢弃 Nyquist 分量（信号带限，可忽略）
y = ifft(Y, [], 2) * M / N;
if isreal(x), y = real(y); end
end
