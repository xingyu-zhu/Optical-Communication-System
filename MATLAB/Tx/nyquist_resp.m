function H = nyquist_resp(f, rs, beta, root)
%NYQUIST_RESP 升余弦 (root=false) / 根升余弦 (root=true) 频率响应。
%   用法: H = nyquist_resp(freqaxis(N, fs), rs, 0.1, true)    f 与 rs 单位一致
f = abs(f); f1 = (1-beta)*rs/2; f2 = (1+beta)*rs/2;
H = double(f <= f1);
tr = f > f1 & f <= f2;
H(tr) = 0.5*(1 + cos(pi/(beta*rs)*(f(tr) - f1)));
if root, H = sqrt(H); end
end
