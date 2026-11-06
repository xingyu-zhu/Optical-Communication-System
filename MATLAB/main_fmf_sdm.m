%MAIN_FMF_SDM 少模光纤（FMF）相干空分复用（SDM）：模式耦合、群时延扩展、MDL、多模非线性与 2M×2M MIMO 接收。
% 模型（Multimode/）：耦合段模型（每段 lc 内各模式群独立的 DMGD/MDL/色散，段间随机酉耦合）+ 模式群平均的
% 多模 Manakov 非线性 + 少模 EDFA（MDG + ASE）；M 个 DP-16QAM 发射机共用一个激光器，接收端 M 个相干接收机共用本振，
% DSP 为训练序列辅助粗频偏估计 + 2M×2M DA-LMS→RDE MIMO + BPS（sdm_rx_dsp）。
% 1) FMF3（LP01 + LP11a/b）群时延扩展随长度：强耦合 ∝ √L，弱耦合 ∝ L（不耦合上限 DMGD·L·√(2/9)）；MDL 随长度累积
% 2) FMF3 强耦合 4×50 km、6×6 MIMO：入纤功率扫描（每模式功率）（低功率端 SNR 约 21 dB 封顶，来自收发机损伤）
% 3) FMF6（3 个模式群）弱耦合 1×10 km、12×12 MIMO：各模式 BER / SNR
% 与 Python examples/fmf_sdm.py 对应（随机数实现不同，统计结果一致）
clear; close all; setup_paths;
cfg = sys_config('M', 16, 'rs', 32e9, 'n_blocks', 480, 'fs_adc', 80e9);

%% 1) 群时延扩展与 MDL
lengths = [5 10 20 50 100 200];
regs = {'strong', 'weak'};
gd_std = zeros(2, numel(lengths)); mdl = gd_std;
for r = 1:2
    for i = 1:numel(lengths)
        q = mm_params('FMF3', 'L', lengths(i), 'regime', regs{r});
        for sd = 0:3
            H = mm_transfer(64e9, 2^14, q, 1, sd);
            gd_std(r,i) = gd_std(r,i) + mean(std(mm_group_delays(H, 64e9), 1, 2))/4;
            mdl(r,i) = mdl(r,i) + mean(mm_mdl(H))/4;
        end
    end
    fprintf('FMF3 %-6s：群时延标准差 [ps] %s；MDL [dB] %s\n', regs{r}, mat2str(round(gd_std(r,:)*1e12, 1)), mat2str(round(mdl(r,:), 2)));
end
q = mm_params('FMF3');
uncoupled = q.dmgd_ps_km(2)*lengths*sqrt(2/9);
for nm = {'FMF3', 'FMF6'}
    for r = 1:2
        [~, Gm] = mm_nl_matrix(mm_params(nm{1}, 'regime', regs{r}));
        fprintf('%s %-6s 非线性系数 Γ [1/(W·km)]：%s\n', nm{1}, regs{r}, mat2str(round(Gm*1e3, 3)));
    end
end

%% 2) FMF3 强耦合 4×50 km 功率扫描
fmf3 = mm_params('FMF3', 'L', 50, 'regime', 'strong');
powers = [-4 0 4 8 12];
snr_p = zeros(numel(powers), fmf3.n_modes);
for k = 1:numel(powers)
    c = cfg; c.p_launch = powers(k);
    [res, ntaps] = sdm_run(c, fmf3, 4, 7);
    snr_p(k,:) = res.snr_mode;
    fprintf('FMF3 强耦合 4×50 km，%+d dBm/模式，%d 抽头：BER %.2e，各模式 SNR %s dB\n', powers(k), ntaps, res.ber, mat2str(round(res.snr_mode, 2)));
end

%% 3) FMF6 弱耦合 1×10 km，12×12 MIMO
fmf6 = mm_params('FMF6', 'L', 10);
c = sys_config('M', 16, 'rs', 32e9, 'n_blocks', 960, 'train_len', 8192, 'fs_adc', 80e9, 'p_launch', 0);
[res6, nt6] = sdm_run(c, fmf6, 1, 7);
fprintf('FMF6 弱耦合 1×10 km，12×12 MIMO，%d 抽头：BER %.2e\n', nt6, res6.ber);
for m = 1:fmf6.n_modes, fprintf('  %-5s BER %.2e  SNR %.2f dB\n', fmf6.modes{m}, res6.ber_mode(m), res6.snr_mode(m)); end

%% 绘图
figure('Position', [100 100 1900 420]);
subplot(1,4,1); loglog(lengths, gd_std(1,:)*1e12, 'o-', lengths, gd_std(2,:)*1e12, 's-'); hold on;
loglog(lengths, gd_std(1,4)*1e12*sqrt(lengths/lengths(4)), 'k:', lengths, uncoupled, 'k--'); grid on;
xlabel('Length (km)'); ylabel('Group-delay std (ps)'); title('FMF3 group-delay spread');
legend('strong coupling', 'weak coupling', '\propto\surdL', 'uncoupled DMGD·L·\surd(2/9)', 'Location', 'northwest');
subplot(1,4,2); plot(lengths, mdl(1,:), 'o-', lengths, mdl(2,:), 's-'); grid on;
xlabel('Length (km)'); ylabel('Mean MDL (dB)'); title('FMF3 mode-dependent loss'); legend('strong coupling', 'weak coupling');
subplot(1,4,3); plot(powers, snr_p, 'o-'); grid on; legend(fmf3.modes);
xlabel('Launch power per mode (dBm)'); ylabel('SNR (dB)'); title('FMF3 strong, 4×50 km, 6×6 MIMO');
subplot(1,4,4); plot_const(res6.sym(11,:), sprintf('FMF6 weak 10 km, %s X', fmf6.modes{6}));
exportgraphics(gcf, fullfile('results', 'fmf_sdm.png'));

function [res, ntaps] = sdm_run(cfg, mmf, n_spans, seed)
%SDM_RUN SDM 全链路一次：发射 -> 链路 -> 相干接收 -> MIMO DSP。抽头数按群时延扩展自动选取。
gd = mm_group_delays(mm_transfer(cfg.fs, 2^14, mmf, n_spans, seed), cfg.fs);
ntaps = 4*ceil(mean(gd(:,end) - gd(:,1))*cfg.rs) + 1;
[E, refs] = sdm_tx(cfg, mmf.n_modes);
E = mm_link(E, cfg.fs, mmf, n_spans, true, seed);
r = sdm_rx(E, cfg, 1.5e9);
res = sdm_rx_dsp(r, cfg, refs, mmf, n_spans, ntaps);
end
