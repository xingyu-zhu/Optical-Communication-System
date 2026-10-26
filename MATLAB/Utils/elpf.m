function y = elpf(x, fs, bw, kind, order)
%ELPF 电低通滤波（DAC / ADC / 驱动 / PD 带宽）。
%   y = elpf(x, fs, bw)                  高斯（零相位）
%   y = elpf(x, fs, bw, kind, order)     kind: 'gaussian' | 'bessel'(线性相位, 有群时延) | 'butter' | 'brickwall'
%   bw 为 3 dB 带宽 [Hz]；order 为贝塞尔/巴特沃斯阶数（默认 4）。实输入返回实输出。
if nargin < 4, kind = 'gaussian'; end
if nargin < 5, order = 4; end
f = freqaxis(size(x,2), fs);
switch kind
    case 'gaussian',  H = exp(-log(2)/2 * (f/bw).^2);
    case 'brickwall', H = double(abs(f) <= bw);
    case 'bessel',    [b,a] = besself(order, 2*pi*bw); H = freqs(b, a, 2*pi*f);
    case 'butter',    [b,a] = butter(order, 2*pi*bw, 's'); H = freqs(b, a, 2*pi*f);
end
y = fdfilter(x, H);
if isreal(x), y = real(y); end
end
