function [y, w] = ffe_lms(x, ref, ntaps, mu, sps, n_train, levels)
%FFE_LMS 实数 LMS 前馈均衡器（IM-DD / PAM）。
%   用法: [y, w] = ffe_lms(x, ref, 21, 1e-3, 2, 4096, levels)
%   前 n_train 符号用 ref 训练，之后对 levels 判决引导；输出 1 sps
N = floor(numel(x)/sps); h = floor(ntaps/2);
xp = [x(end-h+1:end), x, x(1:h)];
w = zeros(1, ntaps); w(h+1) = 1;
y = zeros(1, N);
for k = 1:N
    X = xp((k-1)*sps + (1:ntaps));
    y(k) = w * X.';
    if k <= n_train, d = ref(k); else, [~, i] = min(abs(levels - y(k))); d = levels(i); end
    w = w + mu*(d - y(k))*X;
end
end
