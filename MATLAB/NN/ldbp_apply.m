function y = ldbp_apply(p, r)
%LDBP_APPLY 用（训练好的）LDBP 补偿 2 sps 接收波形，保持原功率尺度。
%   用法: y = ldbp_apply(p, r)
%   嵌入 rx_dsp：cfg.nl_comp = @(y) ldbp_apply(p, y)
s = sqrt(sum(sigpow(r)));
y = ldbp_forward(p.taps, p.theta, r * sqrt(p.p_launch)/s) * s/sqrt(p.p_launch);
end
