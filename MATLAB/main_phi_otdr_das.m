%MAIN_PHI_OTDR_DAS 光纤传感：φ-OTDR 分布式声波传感 (DAS)，相干 IQ 解调。
% 窄线宽激光 -> AOM 脉冲 -> 光纤（随机瑞利散射 + 局部振动调相）-> 背向散射与本振 90° 混频 -> BPD
% DSP：IQ -> 标距 G 差分相位 -> 慢时间解卷绕 -> 振动定位与波形恢复
clear; close all; setup_paths; rng(0);
c0 = phys_const();
fs = 250e6; ng = 1.468;
dz = c0.c/ng/(2*fs);                       % 每采样对应长度 (~0.41 m)
nz = floor(2000/dz); z = (0:nz-1)*dz;      % 2 km 光纤
W = 100e-9; prr = 10e3; K = 1000;          % 脉宽 (~10 m 分辨率)、重复频率、脉冲数
alpha = 0.2e-3*log(10)/10;                 % 功率衰减 [1/m]
f_vib = 500; A_vib = 1.5; z1 = 1200; z2 = 1210;    % 振动 500 Hz、1.5 rad、1200~1210 m
G = round(20/dz);                          % 差分相位标距 20 m

% 瑞利散射体（单程散射系数 αs·S·dz）与往返损耗
r = (randn(1,nz) + 1j*randn(1,nz))/sqrt(2) * sqrt(4.6e-5*1.5e-3*dz) .* exp(-alpha*z);
prof = min(max((z - z1)/(z2 - z1), 0), 1);  % 振动区内相位线性累积，其后保持
np = round(W*fs);
t_slow = (0:K-1)/prr;
pd = struct('R', 0.8, 'T', 300);
traces = zeros(K, nz);
for k = 1:K
    bs = conv(r .* exp(2j*A_vib*sin(2*pi*f_vib*t_slow(k))*prof), ones(1, np));
    Es = sqrt(dbm2w(20)) * bs(1:nz) .* laser_cw(nz, fs, 0, 1e3)/sqrt(1e-3);  % 1 kHz 激光相噪
    o = hybrid90(Es, sqrt(dbm2w(0))*ones(1, nz));
    traces(k,:) = bpd(o(1,:), o(2,:), fs, pd) + 1j*bpd(o(3,:), o(4,:), fs, pd);
end

% 差分相位解调
d = movmean(traces(:, G+1:end) .* conj(traces(:, 1:end-G)), np, 2);
dphi = unwrap(angle(d), [], 1);
dphi = dphi - mean(dphi, 1);
zc = z(1:end-G) + G*dz/2;
rms_z = medfilt1(std(dphi, 0, 1), 2*np);   % 中值滤波抑制相干衰落伪峰
[~, iz] = max(rms_z);
true_sig = 2*A_vib*sin(2*pi*f_vib*t_slow).';
cc = corrcoef(dphi(:, iz), true_sig);
fprintf('Vibration located at %.1f m (true %d-%d m), waveform corr %.3f\n', zc(iz), z1, z2, abs(cc(1,2)));

figure('Position', [100 100 1500 400]);
subplot(1,3,1); imagesc(t_slow*1e3, zc, dphi.'); axis xy; colormap(gca, 'parula'); clim([-1 1]*2.5*A_vib);
xlabel('Time (ms)'); ylabel('Distance (m)'); title('Differential phase');
subplot(1,3,2); plot(zc, rms_z); xlabel('Distance (m)'); ylabel('RMS (rad)');
subplot(1,3,3); plot(t_slow*1e3, dphi(:, iz), t_slow*1e3, true_sig, '--'); legend('recovered', 'true');
xlabel('Time (ms)');
exportgraphics(gcf, fullfile('results', 'phi_otdr_das.png'));
