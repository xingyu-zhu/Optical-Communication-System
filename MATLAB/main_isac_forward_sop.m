%MAIN_ISAC_FORWARD_SOP 通感一体示例 2：相干收发机前向传感（积分式，类 Zhan 2021 / Mecozzi 2021 / Mazur 2022）。
% 链路：DP-16QAM -> 光纤（某处扰动使双折射轴以 60 Hz 摆动 -> SOP 振荡；另一处振动产生 120 Hz 相位调制）
%       -> 相干接收 -> 2×2 LMS（训练后判决引导，同时跟踪 SOP 与相位）-> 16QAM 判决（通信）
% 传感：用判决符号做数据辅助块 Jones 矩阵估计（每 1000 符号一次 = 10 kHz 采样）
%       -> 输出 SOP（Stokes）与公共相位的时间序列 -> 频谱识别扰动频率、带通恢复振动波形。
% 缩小规模：10 MBd、50 ms。激光相噪按 Δν·Ts 起作用，故发/收合计线宽取 10 Hz（Δν/Rs = 1e-6，相当于 32 GBd 下 32 kHz；
% 相位传感本身也需要超稳激光，见 Marra 2018）。
clear; close all; setup_paths; rng(0);
rs = 10e6; N = 500000; nb = 1000;                 % 符号率、符号数（50 ms）、Jones 估计块长
t = (0:N-1)/rs;
c = qam_const(16);
bits = uint8(randi([0 1], 2, 4*N));
s = bits2sym(bits, c);

% ---- 信道：R2·R(θ(t))·R1 的 SOP 扰动 + 振动相位 + 激光相噪（发/收合计 10 Hz）+ AWGN ----
f_sop = 60; f_ph = 120;                           % 取在频率网格（1/50 ms = 20 Hz）上
theta = 0.4*sin(2*pi*f_sop*t);
phi = 1.0*sin(2*pi*f_ph*t);
Rt = reshape([cos(theta); sin(theta); -sin(theta); cos(theta)], 2, 2, N);
J = pagemtimes(random_unitary(), pagemtimes(Rt, random_unitary()));
y = reshape(pagemtimes(J, reshape(s, 2, 1, N)), 2, N) .* exp(1j*(phi + angle(laser_cw(N, rs, 0, 10))));
y = addawgn(y, 18);

% ---- 通信：LMS 训练 5000 符号后判决引导（ntaps = 3，1 sps）----
ye = mimo_eq(y, 3, 3e-3, 1, 'lms', c, s, 5000);
[~, d] = sym_decide(ye, c);
d(:, 1:5000) = s(:, 1:5000);                      % 训练段用已知训练符号
k0 = 10001:N;
ber = calc_ber(bits(:, 4*10000+1:end), sym2bits(ye(:, k0), c));
fprintf('Comm: BER %.2e, SNR %.2f dB (AWGN 18 dB)\n', ber, calc_snr(ye(:, k0), s(:, k0)));

% ---- 传感：判决符号辅助的块 Jones 估计 -> SOP 与相位 ----
Jh = jones_est(y, d, nb);
S = jones_sop(Jh);                                % 3×nblk
ph = jones_phase(Jh);
fsj = rs/nb; tb = ((0:numel(ph)-1) + 0.5)/fsj;
f = freqaxis(numel(ph), fsj);
P_sop = sum(abs(fft(detrend(S.'), [], 1)).^2, 2).';
P_ph = abs(fft(detrend(ph.'))).^2.';
band = find(f > 5 & f < 1000);
[~, i] = max(P_sop(band)); fs_hat = f(band(i));
[~, i] = max(P_ph(band)); fp_hat = f(band(i));
ph_rec = real(ifft(fft(ph) .* (abs(abs(f) - fp_hat) < 20)));    % ±20 Hz 带通
ph_true = sin(2*pi*f_ph*tb);
cc = corrcoef(ph_rec, ph_true);
fprintf('Sensing: SOP peak %.0f Hz (true %d), phase peak %.0f Hz (true %d); phase waveform corr %.3f, amplitude %.2f rad (true 1.00)\n', ...
        fs_hat, f_sop, fp_hat, f_ph, cc(1,2), std(ph_rec)*sqrt(2));

figure('Position', [50 100 1900 400]);
subplot(1,4,1); plot(tb*1e3, S.'); legend('S1', 'S2', 'S3'); xlabel('Time (ms)'); ylabel('Normalized Stokes'); title('SOP from receiver DSP');
m = f > 0 & f < 1000;
subplot(1,4,2); semilogy(f(m), P_sop(m)/max(P_sop(m)), f(m), P_ph(m)/max(P_ph(m))); legend('SOP', 'phase');
xlabel('Frequency (Hz)'); ylabel('Normalized PSD'); title('Spectra');
subplot(1,4,3); plot(tb*1e3, ph_rec, tb*1e3, ph_true, '--'); legend('recovered (band-pass)', 'true');
xlabel('Time (ms)'); ylabel('Phase (rad)'); title(sprintf('%d Hz vibration', f_ph));
subplot(1,4,4); plot(real(ye(1, end-19999:end)), imag(ye(1, end-19999:end)), '.', 'MarkerSize', 1); axis equal;
title('Comm: X-pol after LMS');
exportgraphics(gcf, fullfile('results', 'isac_forward_sop.png'));
