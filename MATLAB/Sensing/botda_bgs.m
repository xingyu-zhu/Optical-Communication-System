function g = botda_bgs(vb, f_scan, n_pulse, dvb)
%BOTDA_BGS BOTDA 小增益模型：位置 z 处的探测光增益 = 泵浦脉冲覆盖的 n_pulse 个单元局部增益之平均。
%   用法: g = botda_bgs(vb, f_scan, n_pulse)    vb 各单元 BFS [Hz]；f_scan 泵浦-探测频差 [Hz]
%   输出 numel(f_scan)×numel(vb) 的布里渊增益谱分布 BGS(f, z)
if nargin < 4, dvb = 30e6; end
g = movmean(brillouin_gain(f_scan(:), vb(:).', dvb), n_pulse, 2);
end
