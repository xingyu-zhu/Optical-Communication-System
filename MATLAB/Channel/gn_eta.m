function eta = gn_eta(fib, rs, n_ch, spacing, method, ppc)
%GN_ETA 单跨、中心信道非线性干扰系数 η [1/W^2]：P_NLI = η·P_ch^3（非相干 GN 模型，矩形谱，双偏振）。
%   用法: eta = gn_eta(fib, 32e9)                              单信道
%         eta = gn_eta(fib, 32e9, 7, 50e9)                     7 信道、50 GHz 间隔，数值积分
%         eta = gn_eta(fib, 32e9, 87, 50e9, 'closed')          闭式近似（大信道数时快）
%   'numeric'：G_NLI(0) = 16/27·γ²·∬G(f1)G(f2)G(f1+f2)·|(1-e^{-αL}e^{j4π²β2L f1f2})/(α-j4π²β2 f1f2)|² df1df2
%   'closed' ：η = 8/27·γ²·Leff²·asinh(π²/2·|β2|·Leff,a·Rs²·Nch^{2Rs/Δf}) / (π·|β2|·Leff,a·Rs²)
%   ppc 为每信道积分点数（默认 40）
if nargin < 3, n_ch = 1; end
if nargin < 4 || isempty(spacing), spacing = rs; end
if nargin < 5, method = 'numeric'; end
if nargin < 6, ppc = 40; end
a = fib.alpha*log(10)/10/1e3; Lm = fib.L*1e3; g = fib.gamma/1e3; b2 = abs(beta2_from_D(fib.D, fib.lam));
Leff = (1 - exp(-a*Lm))/a; Lea = 1/a;
if strcmp(method, 'closed')
    eta = 8/27*g^2*Leff^2*asinh(pi^2/2*b2*Lea*rs^2*n_ch^(2*rs/spacing)) / (pi*b2*Lea*rs^2);
    return
end
fc = ((0:n_ch-1) - (n_ch-1)/2) * spacing;
B = fc(end) - fc(1) + rs; df = rs/ppc;
f = (-B/2 + df/2):df:(B/2);
inband = @(x) any(abs(x(:) - fc) <= rs/2, 2).';
f = f(inband(f));
tot = 0;
for i = 1:numel(f)                                        % 逐行积分，节省内存
    x = 4*pi^2*b2*Lm*f(i)*f;
    mu = (1 - exp(-a*Lm)*exp(1j*x)) ./ (a - 1j*x/Lm);
    tot = tot + sum(inband(f(i) + f) .* abs(mu).^2);
end
eta = 16/27*g^2*tot*df^2/rs^2;
end
