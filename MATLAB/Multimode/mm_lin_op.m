function lin = mm_lin_op(mmf, w)
%MM_LIN_OP FMF 单位长度线性算子 [1/m]（2M×N，每个通道一行）：-α/2 - j(τω + β2ω²/2 + β3ω³/6)。
%   用法: lin = mm_lin_op(mmf, 2*pi*freqaxis(N, fs))
%   τ 为各通道 DMGD 去均值后的值（以通道平均群时延为参考帧），α/D/S 按所属模式群取值
gi = repelem(mmf.groups, 2);
a = (mmf.alpha_db_km + mmf.mdl_db_km(gi))*log(10)/10/1e3;
tau = mmf.dmgd_ps_km(gi)*1e-15;
tau = tau - mean(tau);
D = mmf.D(gi).'; S = mmf.S(gi).';
ph = tau(:).*w + beta2_from_D(D, mmf.lam)/2.*w.^2 + beta3_from_DS(D, S, mmf.lam)/6.*w.^3;
lin = -a(:)/2 - 1j*ph;
end
