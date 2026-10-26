function E = laser_cw(n, fs, p_dbm, linewidth, f_offset, rin_db)
%LASER_CW 连续光激光器。
%   用法: E = laser_cw(n, fs, p_dbm)                                  理想 CW
%         E = laser_cw(n, fs, p_dbm, linewidth, f_offset, rin_db)
%   输入: linewidth 线宽 Δν [Hz]（Wiener 相位噪声，增量方差 2πΔν/fs）；f_offset 频偏 [Hz]（WDM 信道频率也用它）；
%         rin_db 相对强度噪声 [dB/Hz]（[] 不计）
%   输出: 1×n 复光场 [sqrt(W)]
%   为与循环（FFT）仿真相容：相位噪声取 Brownian bridge（首尾相接），f_offset 取整到频率网格 fs/n，
%   否则首尾相位跳变会经 CD/滤波扩散成误码。
if nargin < 4, linewidth = 0; end
if nargin < 5, f_offset = 0; end
if nargin < 6, rin_db = []; end
t = (0:n-1)/fs;
f_offset = round(f_offset*n/fs) * fs/n;
phi = cumsum(sqrt(2*pi*linewidth/fs) * randn(1, n));
phi = phi - (1:n)/n * phi(end);
P = dbm2w(p_dbm) * ones(1, n);
if ~isempty(rin_db), P = P .* (1 + sqrt(10^(rin_db/10)*fs/2) * randn(1, n)); end
E = sqrt(abs(P)) .* exp(1j*(phi + 2*pi*f_offset*t));
end
