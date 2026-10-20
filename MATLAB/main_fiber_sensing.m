%MAIN_FIBER_SENSING 光纤传感综合示例（6 种常用技术，各自给出解调误差）：
% (a) OTDR：衰减与损耗事件识别          (b) φ-OTDR：矩形脉冲 vs LFM 脉冲压缩的空间分辨率
% (c) BOTDA：布里渊增益谱 -> BFS -> 温度/应变   (d) 拉曼 DTS：反斯托克斯/斯托克斯比 -> 温度
% (e) FBG：反射谱峰值解调应变                (f) OFDR：局部瑞利谱互相关 -> 毫米级应变分布
clear; close all; setup_paths; rng(0);
c0 = phys_const();
figure('Position', [50 50 1600 850]);

% ---- (a) OTDR：20 km，100 ns 脉冲，5 km 处 0.5 dB、12 km 处 1.0 dB 损耗 ----
z = 0:1:20e3-1;
p = otdr_trace(z, 20, 100e-9, 0.2, [5e3 0.5; 12e3 1.0]);
p_db = 10*log10(abs(p + 2e-3*p(1)*10^-0.8*randn(size(z))));    % 平均后的残余噪声
[a_est, idx, loss] = otdr_events(p_db, 1, 200, 0.1);
fprintf('(a) OTDR: attenuation %.3f dB/km (true 0.200); events', a_est);
fprintf(' %.2f km/%.2f dB', [z(idx)/1e3; loss]); fprintf(' (true 5.00/0.50, 12.00/1.00)\n');
subplot(2,3,1); plot(z/1e3, p_db - p_db(1)); xline(z(idx)/1e3, 'r:');
xlabel('Distance (km)'); ylabel('Backscatter (dB, round trip)'); title('(a) OTDR events');

% ---- (b) φ-OTDR 空间分辨率：矩形 100 ns vs LFM 1 µs / 100 MHz（同峰值功率）----
fs = 250e6; dz = cell_len(fs);
[zb, r] = rayleigh_fiber(300, dz, 0.2);
i1 = floor(150/dz) + 1; i2 = floor(153/dz) + 1;
rect = ones(1, round(100e-9*fs)); chirp = lfm_pulse(1e-6, 100e6, fs);
width = @(y) sum(abs(y).^2 >= 0.5*max(abs(y).^2))*dz;           % 单反射点响应 -3 dB 宽度
one = zeros(size(zb)); one(i1) = 1;
w_r = width(backscatter(rect, one, numel(zb)));
w_c = width(pulse_compress(backscatter(chirp, one, numel(zb)), chirp));
h = r; h([i1 i2]) = h([i1 i2]) + 20*std(r);                        % 两个相距 3 m 的强反射点
yr = backscatter(rect, h, numel(zb));
yc = pulse_compress(backscatter(chirp, h, numel(zb)), chirp);
fprintf('(b) phi-OTDR resolution: rect %.1f m, LFM compressed %.2f m (theory c/2nB = %.2f m)\n', w_r, w_c, cell_len(100e6));
m = zb > 140 & zb < 165;
subplot(2,3,2); plot(zb(m), 20*log10(abs(yr(m))/max(abs(yr(m)))), zb(m), 20*log10(abs(yc(m))/max(abs(yc(m)))));
ylim([-40 2]); legend('rect 100 ns', 'LFM 1 µs, 100 MHz'); xlabel('Distance (m)'); ylabel('|trace| (dB)'); title('(b) Pulse compression');

% ---- (c) BOTDA：2 km，20 ns 泵浦（2 m），600 m 处 +30 K、1200 m 处 300 με ----
zc = 0:0.5:1999.5;
dT = 30*(zc >= 600 & zc < 620); eps = 300*(zc >= 1200 & zc < 1230);
f_scan = 10.75e9:2e6:10.998e9;
g = botda_bgs(bfs_shift(dT, eps), f_scan, round(2/0.5));
vb = bfs_fit(f_scan, g + 0.02*randn(size(g)));
hot = zc >= 605 & zc < 615; st = zc >= 1205 & zc < 1225; bg = dT == 0 & eps == 0;
fprintf('(c) BOTDA: BFS rms error %.2f MHz; hot spot %.1f K (true 30); strain %.0f µε (true 300)\n', ...
        std(vb(bg) - 10.85e9)/1e6, mean(vb(hot) - 10.85e9)/1e6, mean(vb(st) - 10.85e9)/0.05e6);
subplot(2,3,3); imagesc(zc, f_scan/1e9, g); axis xy; hold on; plot(zc, vb/1e9, 'w', 'LineWidth', 0.6);
xlabel('Distance (m)'); ylabel('Pump-probe offset (GHz)'); title('(c) BOTDA gain spectrum + BFS');

% ---- (d) 拉曼 DTS：4 km，基线 20 °C，2000~2050 m 处 70 °C，前 100 m 为已知温度的参考段 ----
zd = 0:3999;
T = 293.15 + 50*(zd >= 2000 & zd < 2050);
dalpha = 0.05*log(10)/10/1e3*2;                   % 两波长往返损耗差 0.05 dB/km
R = movmean(raman_ratio(T), 3) .* exp(-dalpha*zd) .* (1 + 5e-3*randn(size(zd)));
ref = zd < 100;
T_est = dts_temp(R, mean(R(ref).*exp(dalpha*zd(ref))), 293.15, zd, dalpha);
fprintf('(d) Raman DTS: background rms error %.2f K; hot section %.1f °C (true 70.0)\n', ...
        std(T_est(zd > 2100) - 293.15), mean(T_est(zd >= 2005 & zd < 2045)) - 273.15);
subplot(2,3,4); plot(zd, T_est - 273.15, zd, T - 273.15, 'k--'); legend('DTS', 'true');
xlabel('Distance (m)'); ylabel('Temperature (°C)'); title('(d) Raman DTS');

% ---- (e) FBG：λB = 1550 nm，10 mm，Δn = 1e-4；应变 0 / 500 / 1000 με ----
lam = linspace(1549, 1552, 3001);
subplot(2,3,5); hold on;
for e = [0 500 1000]
    Rf = fbg_reflect(lam, 1550 + fbg_shift(1550, 0, e)) .* 10.^(0.002*randn(size(lam)));
    d_pk = fbg_peak(lam, Rf) - 1550;
    fprintf('(e) FBG: strain %4d µε -> peak shift %7.1f pm -> %.0f µε\n', e, d_pk*1e3, d_pk/fbg_shift(1550, 0, 1));
    plot(lam, Rf, 'DisplayName', sprintf('%d µε', e));
end
legend; xlabel('Wavelength (nm)'); ylabel('Reflectivity'); title('(e) FBG spectrum');

% ---- (f) OFDR：1 THz 扫频（分辨率 0.1 mm），3.3 m 光纤，1.0~1.2 m 处 200 με、2.0~2.05 m 处 100 με ----
B = 1e12; nk = 2^15; dzf = cell_len(B);
[zf, rf] = rayleigh_fiber(nk*dzf, dzf, 0.2);
ef = 200*(zf >= 1.0 & zf < 1.2) + 100*(zf >= 2.0 & zf < 2.05);
noise = @(s) s + 0.03*std(s)*(randn(size(s)) + 1j*randn(size(s)));
[iw, dnu] = ofdr_shift(noise(ofdr_scan(rf, dzf)), noise(ofdr_scan(rf, dzf, ef)), 256, B);
e_est = -dnu/(0.78*c0.f0)*1e6;
ins = zf(iw) > 1.03 & zf(iw) < 1.17;
quiet = movmax(ef, 256); quiet = quiet(iw) == 0;          % 整窗都不受应变的位置
fprintf('(f) OFDR: window %.1f mm; strain in 1.0-1.2 m %.0f µε (true 200), unstrained rms %.1f µε (windows fully outside strain)\n', ...
        256*dzf*1e3, mean(e_est(ins)), std(e_est(quiet)));
subplot(2,3,6); plot(zf(iw), e_est, '.-', zf, ef, 'k--'); legend('OFDR', 'true');
xlabel('Distance (m)'); ylabel('Strain (µε)'); title('(f) OFDR strain');
exportgraphics(gcf, fullfile('results', 'fiber_sensing.png'));
