function y = dbp(x, fs, p_launch_dbm, n_spans, fib, steps, xi)
%DBP 数字反向传播（改写自参考代码 DBP_Raymond）：逐跨撤销放大器增益，再以 -α,-D,-S,-γ 做 Manakov SSFM。
%   用法: y = dbp(x, fs, p_launch_dbm, n_spans, fib)             每跨 10 步，全补偿
%         y = dbp(x, fs, p_launch_dbm, n_spans, fib, steps, xi)
%   p_launch_dbm 每跨入纤总功率（把接收信号还原为物理功率）；fib 光纤参数；
%   steps 每跨步数；xi 非线性补偿系数（0 退化为 CDC，常取 0.5~1 优化）
if nargin < 6, steps = 10; end
if nargin < 7, xi = 1; end
s = sqrt(sum(sigpow(x)));
E = setpow(x, p_launch_dbm);
b = fib; b.alpha = -fib.alpha; b.D = -fib.D; b.S = -fib.S; b.gamma = -fib.gamma*xi;
b.pmd = 0; b.raman = false; b.steep = false;
for k = 1:n_spans
    E = E * 10^(-fib.alpha*fib.L/20);
    E = ssfm_manakov(E, fs, b, 'nsteps', steps);
end
y = E / sqrt(sum(sigpow(E))) * s;
end
