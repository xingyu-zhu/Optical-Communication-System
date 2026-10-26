%MAIN_ISAC_OFDM_DAS 通感一体示例 1：CP-OFDM 数据信号同时作 DAS 探测光（后向，可定位）。
% 发射机：16QAM-OFDM（nfft 16384，CP 5000 ≥ 1 km 光纤往返时延 4900 采样）-> 窄线宽激光 -> 环形器 -> 1 km 光纤
%   通信：远端相干接收 -> 去 CP/FFT -> LS 信道估计 + 导频 CPE -> 16QAM 判决
%   传感：近端用同一激光作本振接收瑞利背向散射 -> 每个 OFDM 符号逐子载波除以已知发送符号 -> IFFT
%         -> 散射冲激响应 h(z)（一个 OFDM 符号 = 一次 "脉冲"）-> 差分相位解调（同 φ-OTDR）
% 仿真为缩小规模（500 MS/s），原理与高速系统相同：距离分辨率 ≈ c/(2·ng·B_OFDM)，慢时间采样率 = 1/OFDM 符号时长。
clear; close all; setup_paths; rng(0);
fs = 500e6; nfft = 16384; ncp = 5000; K = 300;   % 采样率 = OFDM 带宽；K 个 OFDM 符号
dz = cell_len(fs);                                % 距离单元 0.20 m
nz = floor(1000/dz);                              % 1 km 光纤（散射记忆 4900 采样 ≤ CP）
f_vib = 800; A_vib = 1.0; z1 = 600; z2 = 605;     % 振动：800 Hz，单程 1 rad，600~605 m
G = round(10/dz); n_avg = 5;                      % 标距 10 m，距离平均 1 m
n_s = nfft + ncp; Ts = n_s/fs;                    % OFDM 符号时长 = 慢时间采样间隔

% ---- 发射：16QAM-OFDM，DC 与两侧 10% 子载波置空，每 64 个子载波 1 个导频 ----
c = qam_const(16);
used = abs(freqaxis(nfft, 1)) < 0.45; used(1) = false;
X = c(randi(16, K, nfft)) .* used;
pil = find(used); pil = pil(1:64:end);
x = ofdm_mod(X, ncp) / sqrt(mean(used));          % 单位平均功率
lo = laser_cw(K*n_s, fs, 0, 100)/sqrt(1e-3);      % 100 Hz 超窄线宽，发射与本振同源（单位功率）
Etx = sqrt(dbm2w(10)) * x .* lo;                  % 入纤 10 dBm

% ---- 光纤：瑞利散射 + 振动（往返相位 2φ）。每个 OFDM 符号内扰动视为不变 ----
[z, r] = rayleigh_fiber(nz*dz, dz, 0.2);
t_slow = (0:K-1).'*Ts;
vib = A_vib*sin(2*pi*f_vib*t_slow);
h = r .* exp(2j*vib*min(max((z - z1)/(z2 - z1), 0), 1));
Ebs = reshape(backscatter(reshape(Etx, n_s, K).', h, n_s).', 1, []);   % 各符号自身 CP 吸收散射记忆

% ---- 传感接收：同源本振 0 dBm，90° 混频 + BPD ----
o = hybrid90(Ebs, sqrt(dbm2w(0))*lo);
pd = struct('R', 0.8, 'T', 300);
i_bs = bpd(o(1,:), o(2,:), fs, pd) + 1j*bpd(o(3,:), o(4,:), fs, pd);
prof = ofdm_profile(i_bs, X, nfft, ncp); prof = prof(:, 1:nz);
dphi = das_phase(prof, G, n_avg);
zc = z(1:end-G) + G*dz/2;
[iz, e] = vib_locate(dphi, 25);
true_sig = 2*vib;
cc = corrcoef(dphi(:, iz), true_sig);
fprintf('Sensing: located %.1f m (true %d-%d m), waveform corr %.3f, range resolution ~%.2f m, slow-time rate %.1f kHz\n', ...
        zc(iz), z1, z2, abs(cc(1,2)), cell_len(0.9*fs), 1/Ts/1e3);

% ---- 通信接收：远端，前向单程振动相位 + 独立本振（100 Hz）+ AWGN 20 dB ----
fwd = repelem(exp(1j*vib.'), n_s);                % 前向扰动相位（单程，不可定位）
Erx = Etx .* fwd .* conj(laser_cw(K*n_s, fs, 0, 100))/sqrt(1e-3);
y = addawgn(Erx/sqrt(dbm2w(10)), 20);
nt = 8;                                           % 训练符号数（LS 信道估计）
Z = ofdm_eq(ofdm_demod(y, nfft, ncp), X, nt, pil);
dat = used; dat(pil) = false;
nerr = 0;
for k = nt+1:K
    nerr = nerr + sum(sym2bits(Z(k, dat), c) ~= sym2bits(X(k, dat), c));
end
ber = nerr/((K - nt)*sum(dat)*4);
fprintf('Comm: BER %.2e, SNR %.2f dB (AWGN 20 dB), net rate %.2f Gb/s, CP overhead %.0f %%\n', ...
        ber, calc_snr(Z(nt+1:end, dat), X(nt+1:end, dat)), sum(dat)*4/Ts/1e9, ncp/n_s*100);

figure('Position', [50 100 1900 400]);
subplot(1,4,1); imagesc(t_slow*1e3, zc, dphi.'); axis xy; clim([-2.5 2.5]);
xlabel('Time (ms)'); ylabel('Distance (m)'); title('ISAC: differential phase');
subplot(1,4,2); plot(zc, e); xlabel('Distance (m)'); ylabel('RMS (rad)'); title('Vibration energy');
subplot(1,4,3); plot(t_slow*1e3, dphi(:, iz), t_slow*1e3, true_sig, '--'); legend('recovered', 'true');
xlabel('Time (ms)'); ylabel('Phase (rad)'); title('Waveform');
zz = Z(nt+1:end, dat); zz = zz(1:20000);
subplot(1,4,4); plot(real(zz), imag(zz), '.', 'MarkerSize', 1); axis equal; title('Comm: 16QAM-OFDM');
exportgraphics(gcf, fullfile('results', 'isac_ofdm_das.png'));
