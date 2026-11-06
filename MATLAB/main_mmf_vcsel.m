%MAIN_MMF_VCSEL 多模光纤（OM3 / OM4，850 nm）短距 IM-DD：模式群时延、有效模式带宽（EMB）、激励条件、PAM4 BER 与传输距离。
% 模型（Multimode/）：α 剖面渐变折射率光纤，G = floor(V/2) 个主模式群，群时延按 Olshansky-Keck（ε=0）计算；
% 激励功率分布由满注入（OFL）或高斯光斑对 HG 模的交叠积分给出；探测器对各模式群功率求和。
% 发射机用 dml 近似直调 VCSEL（啁啾 + RIN），接收复用 imdd_rx（PIN + TIA + FFE）。
% 注意：离散模式群冲激响应 Σ P_g δ(t - τ_g) 在 -3 dB 带宽之外有深陷波，比实际光纤更悲观；
%       传输距离结果只用于说明 EMB 对距离的限制趋势，不对应具体标准的链路预算。
% 与 Python examples/mmf_vcsel.py 对应
clear; close all; setup_paths;
names = {'OM3', 'OM4'};
fibers = {mm_params('OM3'), mm_params('OM4')};
m4 = fibers{2}; G = m4.n_groups;
fprintf('OM3/OM4 @ 850 nm: V = %.2f，模式群数 G = %d，基模模场半径 %.2f um，Δ = %.4f，α_opt = %.4f，D = %.1f ps/(nm·km)\n', ...
    m4.V, G, m4.w0_um, m4.delta, m4.alpha_opt, m4.D);

%% 1) 模式群时延
cases = {mm_params('OM4', 'alpha_profile_offset', 0), fibers{1}, fibers{2}};
labels = {'α_opt', 'OM3', 'OM4'};
dmd = zeros(3, G);
for k = 1:3
    t = mmf_groups(cases{k}).tau*1e12;
    dmd(k,:) = t - t(1);
    fprintf('%-5s α = %.4f：群时延相对 LP01 群最大 %.1f ps/km，满注入 EMB %.0f MHz·km\n', labels{k}, cases{k}.alpha_profile, ...
        max(dmd(k,:)) - min(dmd(k,:)), mmf_emb(cases{k}, mmf_launch(cases{k}, 'ofl')));
end

%% 2) EMB 随剖面偏差
offs = linspace(-0.04, 0.05, 91);
emb = arrayfun(@(o) mmf_emb(mm_params('OM4', 'alpha_profile_offset', o), mmf_launch(m4, 'ofl')), offs);

%% 3) 激励条件
launch = {mmf_launch(m4, 'ofl'), mmf_launch(m4, 'gauss', 4.0), mmf_launch(m4, 'gauss', 4.0, 8.0)};
lnames = {'OFL', 'Gaussian w = 4 µm', 'Gaussian w = 4 µm, offset 8 µm'};
for k = 1:3
    fprintf('OM4 %-32s: 耦合效率 %.3f，EMB %.0f MHz·km\n', lnames{k}, sum(launch{k}), mmf_emb(m4, launch{k}));
end

%% 4) PAM4 BER vs 长度
cfg = imdd_config('M', 4, 'rs', 53.125e9, 'n_sym', 2^15, 'tx', 'dml', 'p_tx', 0, 'er_db', 6, 'alpha', 3, 'k_ad', 1e12, ...
                  'linewidth', 50e6, 'rin_db', -145, 'dac_bw', 0.6, 'rx_bw', 0.7, 'ntaps', 31);
ref = imdd_tx(cfg);
rop = -4;
lengths = [20 40 60 80 100 150 200 300];
ber = zeros(3, numel(lengths));
for i = 1:2
    P = mmf_launch(fibers{i}, 'ofl');
    for k = 1:numel(lengths)
        rng(1); ber(i,k) = imdd_rx(mmf_link(ref.E, cfg.fs, fibers{i}, P, lengths(k)/1e3), cfg, ref, rop).ber;
    end
    fprintf('%s PAM4 53.125 GBd，ROP %d dBm，BER：%s\n', names{i}, rop, mat2str(ber(i,:), 3));
end
for k = 1:numel(lengths)
    rng(1); ber(3,k) = imdd_rx(mmf_link(ref.E, cfg.fs, m4, [1 zeros(1, G-1)], lengths(k)/1e3), cfg, ref, rop).ber;
end
fprintf('仅色度色散（单个模式群）参考：%s\n', mat2str(ber(3,:), 3));

%% 绘图
figure('Position', [100 100 1900 420]);
subplot(1,4,1); plot(1:G, dmd, 'o-', 'MarkerSize', 4); grid on; legend(labels, 'Location', 'northwest');
xlabel('Principal mode group'); ylabel('Relative group delay (ps/km)'); title('Mode-group delay (Olshansky-Keck, ε = 0)');
subplot(1,4,2); semilogy(offs, emb, 'k-'); hold on;
for i = 1:2, semilogy(fibers{i}.alpha_profile_offset, fibers{i}.emb_nominal_mhz_km, 'o'); end
grid on; xlabel('\alpha - \alpha_{opt}'); ylabel('EMB, OFL (MHz·km)'); title('EMB vs index profile');
legend('model', 'OM3 (2000 MHz·km)', 'OM4 (4700 MHz·km)');
subplot(1,4,3); bar(1:G, cell2mat(cellfun(@(p) p(:)/sum(p), launch, 'UniformOutput', false))); grid on;
xlabel('Principal mode group'); ylabel('Normalized power'); title('Launch conditions (OM4)'); legend(lnames);
subplot(1,4,4); semilogy(lengths, max(ber(1:2,:), 1e-6), 'o-'); hold on; semilogy(lengths, max(ber(3,:), 1e-6), 'k--');
yline(2.4e-4, ':', 'KP4'); grid on; xlabel('Fiber length (m)'); ylabel('BER'); title(sprintf('PAM4 53.125 GBd, ROP %d dBm', rop));
legend('OM3, OFL', 'OM4, OFL', 'CD only', 'Location', 'northwest');
exportgraphics(gcf, fullfile('results', 'mmf_vcsel.png'));
