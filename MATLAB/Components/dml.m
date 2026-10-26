function E = dml(u, fs, p_dbm, er_db, alpha, k_ad, bw, linewidth, rin_db)
%DML 直调激光器（大信号啁啾模型，Koch & Bowers）。
%   用法: E = dml(u, fs, 5, 4, 3, 1e12)                          默认无带宽限制、1 MHz 线宽、RIN -145 dB/Hz
%         E = dml(u, fs, p_dbm, er_db, alpha, k_ad, bw, linewidth, rin_db)
%   P(t) = P̄·(1 + m·u)，m = (ER-1)/(ER+1)，u 为归一化驱动 [-1, 1]（bw 为驱动/弛豫振荡带宽限制，[] 不限）；
%   频率啁啾 Δν = α/(4π)·[d ln P/dt + κP]，绝热啁啾系数 k_ad = ακ/(4π) [Hz/W]（典型 0.5~2 GHz/mW = 0.5~2e12 Hz/W）。
%   绝热啁啾的平均频移只是载波偏移，已去除以保持循环连续。输出 1×N 复光场。
if nargin < 7, bw = []; end
if nargin < 8, linewidth = 1e6; end
if nargin < 9, rin_db = -145; end
if ~isempty(bw), u = elpf(u, fs, bw); end
m = (10^(er_db/10) - 1)/(10^(er_db/10) + 1);
P = dbm2w(p_dbm) * max(1 + m*u, 1e-3);
phi = alpha/2*log(P) + 2*pi*k_ad*cumsum(P - mean(P))/fs;
E = sqrt(P) .* exp(1j*phi) .* laser_cw(numel(P), fs, 0, linewidth, 0, rin_db) / sqrt(1e-3);
end
