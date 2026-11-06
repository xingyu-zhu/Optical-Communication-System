function out = sdm_rx_dsp(r, cfg, refs, mmf, n_spans, ntaps, mu_lms, mu_rde, train_epochs)
%SDM_RX_DSP SDM 接收 DSP。对应 Python multimode.sdm_rx_dsp
%   用法: res = sdm_rx_dsp(r, cfg, refs, mmf, n_spans, ntaps)
%   流程: GSOP → 2 sps → CDC（通道平均色散）→ 匹配滤波 → 定时 → 训练序列辅助粗频偏估计（foe_training）
%         → 2M×2M MIMO（训练序列 DA-LMS 预收敛 train_epochs 遍，再以其抽头为初值全帧 RDE）
%         → 残余频偏（4 次方谱）→ BPS → 帧同步 → 导频辅助模糊/周跳消除 → 判决
%   ntaps（2 sps）须覆盖模式群时延扩展（mm_group_delays；经验上 ntaps ≈ 4×扩展符号数 + 1）。
%   步长默认按 LMS 稳定条件 μ < 2/tr(R) ∝ 1/(P·ntaps) 归一化：mu_lms = 0.5/(P·ntaps)，mu_rde = 0.15/(P·ntaps)（P = 2M）。
%   仿真中帧起始在 0 附近（循环仿真、CDC 已去除平均时延）；实际系统需先用前导码做粗帧同步。
%   输出字段: ber, ber_mode 1×M, snr_mode [dB], snr（平均）, fo [Hz], lag, sym 2M×nd, W
if nargin < 6 || isempty(ntaps), ntaps = 31; end
P = size(r, 1);
if nargin < 7 || isempty(mu_lms), mu_lms = 0.5/(P*ntaps); end
if nargin < 8 || isempty(mu_rde), mu_rde = 0.15/(P*ntaps); end
if nargin < 9 || isempty(train_epochs), train_epochs = 4; end
c = refs{1}.const; kind = refs{1}.kind;
f = cell2mat(cellfun(@(q) q.frame, refs(:), 'UniformOutput', false));
bits = cell2mat(cellfun(@(q) q.bits, refs(:), 'UniformOutput', false));
rot = rot_order(c);
fs2 = 2*cfg.rs;
gi = repelem(mmf.groups, 2);
y = resamp(gsop(r), cfg.fs_adc, fs2);
y = cdc(y, fs2, n_spans*mmf.L, mean(mmf.D(gi)), mmf.lam, mean(mmf.S(gi)));
y = normpow(matched_filter(y, 2, cfg.beta));
y = timing_ff(y, 2);
nt = cfg.train_len;
fo = foe_training(y, f(:, 1:nt), floor(ntaps/4) + 1, 2, cfg.rs);
y = fo_comp(y, fo, fs2);
W = [];
for e = 1:train_epochs
    [~, W] = mimo_eq(y(:, 1:2*nt), ntaps, mu_lms, 2, 'lms', c, f, nt, W);
end
[y, W] = mimo_eq(y, ntaps, mu_rde, 2, 'rde', c, [], 0, W);
y = normpow(y);
fo2 = foe(y, cfg.rs, 'fft', rot);
y = fo_comp(y, fo2, cfg.rs);
out.fo = fo + fo2;
y = cpr_bps(y, c, 32, 64, rot);
[~, out.lag] = frame_sync(y(1:2, :), refs{1}.frame);
y = circshift(y, -out.lag, 2);
y = y(:, 1:size(f,2));
y = fix_ambiguity(y, f, kind > 0, 5, rot);
y = y ./ sqrt(mean(abs(y(:, kind > 0)).^2, 2));
d = y(:, kind == 0); s = f(:, kind == 0);
rb = sym2bits(d, c);
M = numel(refs);
out.ber_mode = zeros(1, M); out.snr_mode = zeros(1, M);
for m = 1:M
    k = 2*m-1:2*m;
    out.ber_mode(m) = calc_ber(bits(k, :), rb(k, :));
    out.snr_mode(m) = calc_snr(d(k, :), s(k, :));
end
out.ber = mean(out.ber_mode);
out.snr = 10*log10(mean(10.^(out.snr_mode/10)));
out.sym = d; out.W = W;
end
