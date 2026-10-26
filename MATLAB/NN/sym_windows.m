function X = sym_windows(x, ntaps, sps)
%SYM_WINDOWS 滑窗取样：以第 k 个符号（采样点 (k-1)·sps+1）为中心取 ntaps 个采样（循环边界）。
%   用法: X = sym_windows(x, ntaps, sps)
%   x 为 1×N 实信号或 P×N 复信号；X 为 Nsym×ntaps×C 实数组，C = 1（实）或 2P（[Re1 Im1 Re2 Im2 ...]）
%   NN 均衡器与 Volterra 均衡器共用。
if ~isreal(x)
    xr = zeros(2*size(x,1), size(x,2));
    xr(1:2:end,:) = real(x); xr(2:2:end,:) = imag(x); x = xr;
end
N = size(x, 2); nsym = floor(N/sps);
idx = mod((0:nsym-1).'*sps + (0:ntaps-1) - floor(ntaps/2), N) + 1;       % nsym×ntaps
X = zeros(nsym, ntaps, size(x,1));
for c = 1:size(x,1)
    v = x(c,:); X(:,:,c) = v(idx);
end
end
