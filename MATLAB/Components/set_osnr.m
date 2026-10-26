function E = set_osnr(E, osnr_db, fs, bref)
%SET_OSNR 直接按 OSNR 加 ASE（双偏振噪声）。
%   用法: E = set_osnr(E, 20, fs)            参考带宽默认 12.5 GHz (0.1 nm @1550)
%         E = set_osnr(E, 20, fs, bref)
if nargin < 4, bref = 12.5e9; end
psd = sum(sigpow(E)) / (2 * 10^(osnr_db/10) * bref);
E = add_noise_psd(E, psd, fs);
end
