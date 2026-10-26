function g = brillouin_gain(f, vb, dvb)
%BRILLOUIN_GAIN 归一化洛伦兹布里渊增益谱 g = 1/(1 + (2(f-vb)/Δν_B)²)，Δν_B 为 FWHM（约 30 MHz）。
%   用法: g = brillouin_gain(f, vb)    f、vb 可隐式扩展（如列 × 行）
if nargin < 3, dvb = 30e6; end
g = 1 ./ (1 + (2*(f - vb)/dvb).^2);
end
