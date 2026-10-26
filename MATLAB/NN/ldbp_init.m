function p = ldbp_init(fs, fib, n_spans, p_launch_dbm, steps, ntaps, f_pass)
%LDBP_INIT 学习型数字反向传播（Learned DBP，Häger & Pfister）参数初始化。
%   用法: p = ldbp_init(2*rs, fib, 10, 4)                每跨 1 步、41 抽头
%         p = ldbp_init(2*rs, fib, 10, 4, 4, 15)         每跨 4 步、15 抽头
%   结构同 DBP：按跨段逆序、每跨 steps 步，每步 = [可学习短复 FIR（色散逆）] -> [E·exp(jθ_k(|Ex|²+|Ey|²))]。
%   FIR：加权最小二乘设计，目标 = 段色散逆 exp(jβ2ω²dz/2) × 升余弦带限窗（|f| ≤ f_pass 平坦，1.5·f_pass 处为 0），
%        通带权重 1、其余 0.3（全频段约束，避免过渡带增益在多步级联中发散）。f_pass 默认 0.28·fs（2 sps）。
%   θ：初始化为 (8/9)γ·∫e^{-αz}dz（该段有效长度）。
%   注意：ntaps 必须覆盖单段色散记忆（80 km SSMF、2 sps 约 22 个采样），否则无法收敛。
%   输出 p：taps (n×ntaps 复), theta (1×n), p_launch [W], ntaps
if nargin < 5, steps = 1; end
if nargin < 6, ntaps = 41; end
if nargin < 7, f_pass = 0.28*fs; end
a = fib.alpha*log(10)/10/1e3; dz = fib.L*1e3/steps; n = n_spans*steps;
f = linspace(-fs/2, fs/2, 2049).'; f = f(1:end-1);
k = (0:ntaps-1) - floor(ntaps/2);
A = exp(-2j*pi*f*k/fs);
af = abs(f);
win = (af <= f_pass) + (af > f_pass & af < 1.5*f_pass) .* 0.5.*(1 + cos(pi*(af - f_pass)/(0.5*f_pass)));
target = exp(1j*beta2_from_D(fib.D, fib.lam)/2*(2*pi*f).^2*dz) .* win;
wgt = 0.3 + 0.7*(af <= f_pass);
h = (A .* wgt) \ (target .* wgt);
z0 = fib.L*1e3 - (1:steps)*dz;                           % 逆序第 j 段覆盖 [L-j·dz, L-(j-1)·dz]
th = (8/9)*fib.gamma/1e3 * exp(-a*z0) * (1 - exp(-a*dz))/a;
p = struct('taps', repmat(h.', n, 1), 'theta', repmat(th, 1, n_spans), ...
           'p_launch', 1e-3*10^(p_launch_dbm/10), 'ntaps', ntaps);
end
