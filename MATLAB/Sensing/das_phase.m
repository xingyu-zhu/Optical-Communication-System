function dphi = das_phase(tr, G, n_avg)
%DAS_PHASE DAS 差分相位解调：标距 G（单元数）两端的相位差 ∠(x[z+G]·x*[z])，沿距离 n_avg 点平均
%   （抑制相干衰落），沿慢时间解卷绕并去均值。
%   用法: dphi = das_phase(tr, G, n_avg)    tr: K×N 复轨迹（K 次测量 × N 个单元），dphi: K×(N-G) [rad]
if nargin < 3, n_avg = 1; end
d = tr(:, G+1:end) .* conj(tr(:, 1:end-G));
if n_avg > 1, d = movmean(d, n_avg, 2); end
dphi = unwrap(angle(d), [], 1);
dphi = dphi - mean(dphi, 1);
end
