function E = fiber_cd(E, fs, L, D, lam, S)
%FIBER_CD 线性色散传输（无损耗、无非线性），工程约定 H = exp(-j(β2ω^2/2 + β3ω^3/6)L)。
%   用法: E = fiber_cd(E, fs, L)                 L [km]，默认 D=17、λ=1550
%         E = fiber_cd(E, fs, L, D, lam, S)      S 色散斜率 [ps/(nm^2 km)]（0 则忽略 β3）
if nargin < 4, D = 17; end
if nargin < 5, lam = 1550; end
if nargin < 6, S = 0; end
w = 2*pi*freqaxis(size(E,2), fs);
ph = beta2_from_D(D, lam)/2*w.^2;
if S, ph = ph + beta3_from_DS(D, S, lam)/6*w.^3; end
E = fdfilter(E, exp(-1j*ph*L*1e3));
end
