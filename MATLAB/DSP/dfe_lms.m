function [y, wf, wb] = dfe_lms(x, ref, nf, nb, mu, sps, n_train, levels)
%DFE_LMS 实数 LMS 判决反馈均衡（IM-DD / PAM）：y_k = wf·x_k − wb·[d_{k-1}, ..., d_{k-nb}]，
%   训练段 d 取 ref，之后取判决。FFE 抑制前导 ISI，DFE 无噪声增强地消除后续 ISI。
%   用法: [y, wf, wb] = dfe_lms(x, ref, 21, 3, 1e-3, 2, 4096, levels)   输出 1 sps；n_train 为空表示全程训练
N = floor(numel(x)/sps); h = floor(nf/2);
if nargin < 7 || isempty(n_train), n_train = N; end
xp = [x(end-h+1:end), x, x(1:h)];
wf = zeros(1, nf); wf(h+1) = 1;
wb = zeros(1, nb); d = zeros(1, nb);
y = zeros(1, N);
for k = 1:N
    X = xp((k-1)*sps + (1:nf));
    y(k) = wf*X.' - wb*d.';
    if k <= n_train, dk = ref(k); else, [~, i] = min(abs(levels - y(k))); dk = levels(i); end
    e = dk - y(k);
    wf = wf + mu*e*X;
    wb = wb - mu*e*d;
    d = [dk, d(1:end-1)];
end
end
