function [a, idx, loss] = otdr_events(p_db, dz, win, thr)
%OTDR_EVENTS OTDR 轨迹分析（p_db = 10·lg P，往返量）。
%   衰减：平滑后斜率的中位数（事件只占少数点，中位数稳健）；
%   事件：去斜后左右各 win 点均值之差为往返台阶，峰值 > 2·thr 判为事件。
%   用法: [a, idx, loss] = otdr_events(p_db, dz, win, thr)   a [dB/km]，idx 事件下标，loss 单程损耗 [dB]
if nargin < 4, thr = 0.1; end
N = numel(p_db);
slope = median(gradient(movmean(p_db, win), dz));
cs = [0, cumsum(p_db - slope*(0:N-1)*dz)];
i = win+1:N-win;
step = zeros(1, N);
step(i) = (cs(i) - cs(i-win))/win - (cs(i+win) - cs(i))/win;
[~, idx] = findpeaks(step, 'MinPeakHeight', 2*thr, 'MinPeakDistance', win);
a = -slope/2*1e3;
loss = step(idx)/2;
end
