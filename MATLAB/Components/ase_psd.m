function S = ase_psd(G, nf_db, f0)
%ASE_PSD 单偏振 ASE 功率谱密度 [W/Hz] = n_sp·hν·(G-1)，n_sp = NF/2。
%   用法: S = ase_psd(G, nf_db)        G 线性增益；f0 默认 193.414 THz
%         S = ase_psd(G, nf_db, f0)    f0 放大器中心频率 [Hz]（波段相关，fib.f0）
c = phys_const();
if nargin < 3, f0 = c.f0; end
S = 10^(nf_db/10)/2 * c.h * f0 * max(G - 1, 0);
end
