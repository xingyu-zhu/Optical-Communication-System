function [f, kind] = framing(data, train, pilot, period)
%FRAMING 帧结构：[训练序列 | 数据块...]，每块 period 个符号、首个为导频。
%   用法: [f, kind] = framing(data, train, pilot, 32)
%   data P×n_d（n_d 须为 period-1 的整数倍）、train P×n_t、pilot P×(≥n_d/(period-1))
%   kind 1×N_f：0 数据 / 1 训练 / 2 导频（用于导频 CPR、去导频、模糊消除）
nblk = size(data,2) / (period-1);
kb = zeros(1, nblk*period);
kb(1:period:end) = 2;
body = zeros(size(data,1), nblk*period);
body(:, kb==2) = pilot(:, 1:nblk);
body(:, kb==0) = data;
f = [train, body];
kind = [ones(1, size(train,2)), kb];
end
