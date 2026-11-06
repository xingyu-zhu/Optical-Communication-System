%MAIN_HCF_LINK 空芯光纤（HCF，反谐振 DNANF 型）与 SSMF 对比：参数与时延、入纤功率扫描（SSFM vs GN 模型）、模间干扰（IMI）容限。
% 1) 参数表：衰减、色散、非线性系数、群折射率与单程时延（params/fiber_bands.json 中的 'HCF' 典型值）
% 2) 单信道 DP-16QAM 40 GBd、10×80 km：SSFM + CDC 的接收 SNR 随入纤功率变化，与 GN 模型对比
%    （HCF 的 γ 约为 SSMF 的 1/5000，最优入纤功率高出 20 dB 以上，实际受放大器输出功率限制；
%    HCF 高功率时 SSFM 结果约 36 dB 封顶，来自接收 DSP 自身的 SNR 上限，不是光纤效应）
% 3) HCF 链路每跨加一次 IMI（fiber_imi），时延远大于均衡器窗口时 IMI 表现为加性干扰：1/SNR ≈ 1/SNR_0 + N·η
% 与 Python examples/hcf_link.py 对应
clear; close all; setup_paths;
names = {'SSMF', 'HCF'};
for i = 1:2
    cfgs{i} = sys_config('n_blocks', 300, 'n_spans', 10, 'fs_adc', 160e9, 'fiber', names{i}, 'fib_kw', {'pmd', 0}); %#ok<SAGROW>
end
ref = tx_signal(cfgs{1});
N = size(ref.wave, 2); f = ref.frame; data = ref.kind == 0;
pg = linspace(-6, 36, 400);

%% 1) 参数与时延
fprintf('%-5s %8s %6s %9s %7s %6s %9s %12s %10s\n', 'fiber', 'a dB/km', 'D', 'g 1/W/km', 'ng', 'us/km', '800 km ms', 'GN P_opt dBm', 'SNR_opt dB');
for i = 1:2
    fib = cfgs{i}.fib;
    gn{i} = gn_model(pg, fib, cfgs{i}.rs, cfgs{i}.n_spans); %#ok<SAGROW>
    fprintf('%-5s %8.3f %6.2f %9.2e %7.4f %6.3f %9.3f %12.2f %10.2f\n', names{i}, fib.alpha, fib.D, fib.gamma, fib.ng, ...
        fib.latency, fib.latency*cfgs{i}.n_spans*fib.L/1e3, gn{i}.p_opt_dbm, gn{i}.snr_opt);
end

%% 2) 入纤功率扫描
powers = {-2:2:6, [0 10 20 26 32]};
snr = cell(1, 2);
for i = 1:2
    snr{i} = arrayfun(@(p) run_link(cfgs{i}, ref, p, [], N, f, data), powers{i});
    fprintf('%s SSFM SNR [dB]: %s\n', names{i}, mat2str(round(snr{i}, 2)));
end

%% 3) IMI 容限（HCF，+10 dBm）
imis = -50:5:-20; p_imi = 10;
s0 = run_link(cfgs{2}, ref, p_imi, [], N, f, data);
s_imi = arrayfun(@(q) run_link(cfgs{2}, ref, p_imi, q, N, f, data), imis);
th = -10*log10(10^(-s0/10) + cfgs{2}.n_spans*10.^(imis/10));
fprintf('HCF IMI @ %+d dBm: 无 IMI SNR %.2f dB\n', p_imi, s0);
for k = 1:numel(imis), fprintf('  IMI %+d dB/跨: SNR %.2f dB（理论 %.2f dB）\n', imis(k), s_imi(k), th(k)); end

%% 绘图
figure('Position', [100 100 1500 420]);
col = {[0 0.45 0.74], [0.85 0.33 0.1]};
subplot(1,3,1); hold on;
for i = 1:2
    plot(powers{i}, snr{i}, 'o', 'Color', col{i}); plot(pg, gn{i}.snr, '-', 'Color', col{i});
end
ylim([10 52]); grid on; xlabel('Launch power per channel (dBm)'); ylabel('SNR (dB)'); title('10×80 km, DP-16QAM 40 GBd, 1 ch');
legend('SSMF SSFM + CDC', 'SSMF GN model', 'HCF SSFM + CDC', 'HCF GN model', 'Location', 'south');
subplot(1,3,2); hold on; Lk = linspace(0, 2000, 50);
for i = 1:2, plot(Lk, cfgs{i}.fib.latency*Lk/1e3, 'Color', col{i}); end
grid on; xlabel('Distance (km)'); ylabel('One-way latency (ms)'); title('Propagation latency');
legend(sprintf('SSMF (ng = %.4f)', cfgs{1}.fib.ng), sprintf('HCF (ng = %.4f)', cfgs{2}.fib.ng), 'Location', 'northwest');
subplot(1,3,3); plot(imis, s_imi, 'o', 'Color', col{2}); hold on; plot(imis, th, 'k--');
grid on; xlabel('IMI per span (dB)'); ylabel('SNR (dB)'); title(sprintf('HCF IMI tolerance, %+d dBm', p_imi));
legend('SSFM + IMI per span', '1/SNR_0 + N\eta');
exportgraphics(gcf, fullfile('results', 'hcf_link.png'));

function s = run_link(cfg, ref, p, imi_db, N, f, data)
%RUN_LINK 理想收发机：激光器 + 功放 -> 链路（可选每跨 IMI）-> 相干接收 -> rx_dsp，返回数据符号 SNR [dB]。
rng(1); fib = cfg.fib;
E = edfa(ref.wave .* laser_cw(N, cfg.fs, 0), cfg.fs, [], fib.nf, p, fib.f0);
for k = 1:cfg.n_spans
    E = fiber_link(E, cfg.fs, fib, 1, true, 'phi_max', cfg.phi_max);
    if ~isempty(imi_db), E = fiber_imi(E, cfg.fs, imi_db, 10e-12*fib.L*1e3); end   % LP11 DMGD 取 10 ps/m（典型量级，待核实）
end
r = coherent_rx(E, laser_cw(N, cfg.fs, 10), cfg.fs, struct('shot', false));
cfg.p_launch = p;
y = rx_dsp(r, cfg, ref).y;
s = calc_snr(y(:, data), f(:, data));
end
