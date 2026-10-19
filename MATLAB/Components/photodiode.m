function i = photodiode(E, fs, pd)
%PHOTODIODE PIN 光电二极管：i = R·Σ|E|^2 + i_dark + 散粒噪声 + 热噪声，再经带宽 bw 低通。
%   用法: i = photodiode(E, fs)
%         i = photodiode(E, fs, struct('R',0.8,'bw',30e9,'T',300))
%   pd 字段（均可省略）：R 响应度 [A/W]=1；bw 带宽 [Hz]=[]（不限带）；i_dark 暗电流 [A]=0；
%                        T 温度 [K]=[]（不计热噪声）；RL 负载电阻 [Ω]=50；shot 散粒噪声=true
%   多行输入（多偏振）时各行功率相加。
d = struct('R',1,'bw',[],'i_dark',0,'T',[],'RL',50,'shot',true);
if nargin > 2, for f = fieldnames(pd)', d.(f{1}) = pd.(f{1}); end, end
c = phys_const();
i = d.R * sum(abs(E).^2, 1) + d.i_dark;
B = fs/2;
v = 0;
if d.shot, v = v + 2*c.q*mean(i)*B; end
if ~isempty(d.T), v = v + 4*c.kB*d.T*B/d.RL; end
i = i + sqrt(v)*randn(size(i));
if ~isempty(d.bw), i = elpf(i, fs, d.bw); end
end
