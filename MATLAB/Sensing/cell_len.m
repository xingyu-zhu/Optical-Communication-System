function dz = cell_len(fs)
%CELL_LEN 采样率 fs 对应的散射单元长度 dz = c/(2·ng·fs) [m]：此时第 i 个单元恰好延迟 i 个采样。
%   用法: dz = cell_len(250e6)     % ≈ 0.41 m
c0 = phys_const(); s = sens_const();
dz = c0.c/(2*s.ng*fs);
end
