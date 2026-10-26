function i = apd(E, fs, pd)
%APD 雪崩光电二极管：i = M·R·P；倍增散粒噪声方差 2q·M²·F(M)·(R·P̄ + i_dark)·B，
%   过剩噪声因子 F = ka·M + (1-ka)(2 - 1/M)（McIntyre；InGaAs/InP 的 ka ≈ 0.4~0.5，Si ≈ 0.02）；再加 TIA 噪声。
%   用法: i = apd(E, fs, struct('M', 10, 'R', 0.8, 'bw', 40e9, 'irnd', 15e-12))
%   pd 字段（均可省略）：M=10；R=0.8；ka=0.5；bw=[]；i_dark=10e-9；irnd TIA 输入等效噪声电流密度 [A/√Hz]=[]；
%                        T 温度=[]；RL=50。增益压过热噪声，但 M 过大时倍增噪声主导 → 存在最优 M。
d = struct('M',10,'R',0.8,'ka',0.5,'bw',[],'i_dark',10e-9,'irnd',[],'T',[],'RL',50);
if nargin > 2, for f = fieldnames(pd)', d.(f{1}) = pd.(f{1}); end, end
c = phys_const();
i = d.M * d.R * sum(abs(E).^2, 1);
B = fs/2;
F = d.ka*d.M + (1 - d.ka)*(2 - 1/d.M);
v = 2*c.q*d.M^2*F*(mean(i)/d.M + d.i_dark)*B;
if ~isempty(d.T), v = v + 4*c.kB*d.T*B/d.RL; end
if ~isempty(d.irnd), v = v + d.irnd^2*B; end
i = i + sqrt(v)*randn(size(i));
if ~isempty(d.bw), i = elpf(i, fs, d.bw); end
end
