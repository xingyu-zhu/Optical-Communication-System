function y = cdc(x, fs, L, D, lam, S)
%CDC 频域色散补偿（fiber_cd 的逆）：H = exp(+j(β2ω²/2 + β3ω³/6)·L)。
%   用法: y = cdc(x, fs, L)                  L 总长度 [km]，默认 D=17、λ=1550
%         y = cdc(x, fs, L, D, lam, S)       S≠0 时同时补偿色散斜率
if nargin < 4, D = 17; end
if nargin < 5, lam = 1550; end
if nargin < 6, S = 0; end
w = 2*pi*freqaxis(size(x,2), fs);
ph = beta2_from_D(D, lam)/2*w.^2;
if S, ph = ph + beta3_from_DS(D, S, lam)/6*w.^3; end
y = fdfilter(x, exp(1j*ph*L*1e3));
end
