%MAIN_PHI_OTDR_DAS 光纤传感：φ-OTDR 分布式声波传感 (DAS)，相干 IQ 解调。
% 窄线宽激光 -> AOM 脉冲 -> 光纤（随机瑞利散射 + 局部振动调相）-> 背向散射与本振 90° 混频 -> BPD
% DSP：IQ -> 标距 G 差分相位 -> 慢时间解卷绕 -> 振动定位与波形恢复
clear; close all; setup_paths; rng(0);
fs = 250e6;
dz = cell_len(fs);                         % 每采样对应长度 (~0.41 m)
W = 100e-9; prr = 10e3; K = 1000;          % 脉宽 (~10 m 分辨率)、重复频率、脉冲数
f_vib = 500; A_vib = 1.5; z1 = 1200; z2 = 1210;    % 振动 500 Hz、单程 1.5 rad、1200~1210 m
G = round(20/dz);                          % 差分相位标距 20 m

% 光纤与扰动：振动区内相位线性累积，其后保持（往返相位 = 2×单程）
[z, r] = rayleigh_fiber(2000, dz, 0.2); nz = numel(z);
t_slow = (0:K-1).'/prr;
prof = min(max((z - z1)/(z2 - z1), 0), 1);
h = r .* exp(2j*A_vib*sin(2*pi*f_vib*t_slow)*prof);         % K×nz
np = round(W*fs);

% 背向散射 + 1 kHz 激光相噪（相对本振）+ 相干接收
Es = sqrt(dbm2w(20)) * backscatter(ones(1, np), h, nz);
pd = struct('R', 0.8, 'T', 300);
traces = zeros(K, nz);
for k = 1:K
    o = hybrid90(Es(k,:) .* laser_cw(nz, fs, 0, 1e3)/sqrt(1e-3), sqrt(dbm2w(0))*ones(1, nz));
    traces(k,:) = bpd(o(1,:), o(2,:), fs, pd) + 1j*bpd(o(3,:), o(4,:), fs, pd);
end

% 差分相位解调 + 定位（中值滤波抑制相干衰落伪峰）
dphi = das_phase(traces, G, np);
zc = z(1:end-G) + G*dz/2;
[iz, rms_z] = vib_locate(dphi, 2*np);
true_sig = 2*A_vib*sin(2*pi*f_vib*t_slow);
cc = corrcoef(dphi(:, iz), true_sig);
fprintf('Vibration located at %.1f m (true %d-%d m), waveform corr %.3f\n', zc(iz), z1, z2, abs(cc(1,2)));

figure('Position', [100 100 1500 400]);
subplot(1,3,1); imagesc(t_slow*1e3, zc, dphi.'); axis xy; colormap(gca, 'parula'); clim([-1 1]*2.5*A_vib);
xlabel('Time (ms)'); ylabel('Distance (m)'); title('Differential phase');
subplot(1,3,2); plot(zc, rms_z); xlabel('Distance (m)'); ylabel('RMS (rad)');
subplot(1,3,3); plot(t_slow*1e3, dphi(:, iz), t_slow*1e3, true_sig, '--'); legend('recovered', 'true');
xlabel('Time (ms)'); ylabel('Phase (rad)');
exportgraphics(gcf, fullfile('results', 'phi_otdr_das.png'));
