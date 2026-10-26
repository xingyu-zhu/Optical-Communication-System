function f = foe(x, fs, method, m)
%FOE 频偏估计 [Hz]（x 一般为 1 sps 符号，fs = 符号率）。
%   用法: f = foe(x, rs)                 'fft'：M 次方谱峰/M（补零 4 倍），范围 ±fs/(2M)
%         f = foe(x, rs, 'diff')         差分 M 次方（Leven）：∠Σ(x[k]x*[k-1])^M / (2πM)·fs
%         f = foe(x, rs, method, m)      m 默认 4（QPSK/QAM）
if nargin < 3, method = 'fft'; end
if nargin < 4, m = 4; end
if strcmp(method, 'fft')
    nfft = 4*2^nextpow2(size(x,2));
    s = sum(abs(fft(x.^m, nfft, 2)).^2, 1);
    fa = freqaxis(nfft, fs);
    [~, i] = max(s);
    f = fa(i)/m;
else
    d = x(:, 2:end) .* conj(x(:, 1:end-1));
    f = angle(sum(d(:).^m)) / (2*pi*m) * fs;
end
end
