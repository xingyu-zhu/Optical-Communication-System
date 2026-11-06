function g = mm_amp_gain(mmf)
%MM_AMP_GAIN 少模 EDFA 各通道增益 [dB]（2M×1）：补偿平均跨段损耗 + 模式相关增益 MDG（去均值）。
%   用法: g = mm_amp_gain(mmf)
gi = repelem(mmf.groups, 2);
loss = mean(mmf.alpha_db_km + mmf.mdl_db_km(gi))*mmf.L;
mdg = mmf.mdg_db(gi);
g = (loss + mdg - mean(mdg)).';
end
