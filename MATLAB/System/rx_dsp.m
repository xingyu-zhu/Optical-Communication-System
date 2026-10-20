function out = rx_dsp(r, cfg, ref)
%RX_DSP 相干接收 DSP 链。
%   用法: res = rx_dsp(r, cfg, ref)
%   输入: r 2×N ADC 输出 @cfg.fs_adc；cfg 见 sys_config；ref 为该信道 tx_signal 输出
%   输出字段: ber, ber_xy [X Y], snr [dB], evm [%], gmi [bit/sym/pol], fo 频偏估计 [Hz], lag 帧偏移,
%             tau 定时估计, eq_err 均衡误差, sym 2×nd 判决前数据符号,
%             y 2×Nf 与 ref.frame 逐符号对齐的整帧符号（含训练/导频，供 NN 后均衡使用）
%   流程: GSOP -> 重采样 2 sps -> CDC/DBP/自定义 -> 匹配滤波 -> 定时 -> MIMO(CMA→RDE) -> FOE -> CPR
%         -> 帧同步 -> 模糊/周跳消除 -> (DD-LMS) -> 判决与指标
%   cfg.nl_comp 可为函数句柄（输入/输出 2 sps 波形），如训练好的 LDBP：@(y) ldbp_apply(p, y)
c = ref.const; f = ref.frame; kind = ref.kind; fib = cfg.fib;
rot = rot_order(c);                                % 旋转对称阶数：QAM/QPSK 4，BPSK 2，8PSK 8
fs2 = 2*cfg.rs;
% 1) 前端：去直流 + GSOP 正交化 + 重采样到 2 sps
y = resamp(gsop(r), cfg.fs_adc, fs2);
% 2) 色散 / 非线性补偿
if isa(cfg.nl_comp, 'function_handle')
    y = cfg.nl_comp(y);
else
    switch cfg.nl_comp
        case 'cdc', y = cdc(y, fs2, cfg.n_spans*fib.L, fib.D, fib.lam, fib.S);
        case 'dbp', y = dbp(y, fs2, cfg.p_launch, cfg.n_spans, fib, cfg.dbp_steps);
    end
end
% 3) 匹配滤波 + 定时恢复
y = normpow(matched_filter(y, 2, cfg.beta));
switch cfg.timing
    case 'ff',      [y, out.tau] = timing_ff(y, 2);
    case 'gardner', [y, out.tau] = gardner_tr(y, 1e-3, 2);
end
% 4) 2×2 MIMO：CMA 预收敛 -> RDE
n0 = min(10000, floor(size(y,2)/4));
[~, W] = mimo_eq(y(:, 1:2*n0), cfg.ntaps, cfg.mu_cma, 2, 'cma', c);
[y, ~, out.eq_err] = mimo_eq(y, cfg.ntaps, cfg.mu_rde, 2, 'rde', c, [], 0, W);
y = normpow(y);
% 5) 频偏估计与补偿（1 sps）
out.fo = foe(y, cfg.rs, 'fft', rot);
y = fo_comp(y, out.fo, cfg.rs);
% 6) 载波相位恢复 + 同步 + 模糊消除（训练+导频辅助周跳消除）
if strcmp(cfg.cpr, 'vv'), yb = cpr_vv(y, 64, rot, c); else, yb = cpr_bps(y, c, 32, 64, rot); end
[yb, out.lag, perm] = frame_sync(yb, f);
if strcmp(cfg.cpr, 'pilot')
    ya = circshift(y(perm,:), -out.lag, 2);
    y = cpr_pilot(ya(:, 1:size(f,2)), f, kind == 2, 5);
else
    y = fix_ambiguity(yb, f, kind > 0, 5, rot);
end
% 7) 可选 DD-LMS 精均衡
if cfg.dd_lms, y = mimo_eq(y, 7, 1e-4, 1, 'dd', c); end
% 8) 功率对齐、去训练/导频、判决与指标
y = y ./ sqrt(mean(abs(y(:, kind > 0)).^2, 2));
d = y(:, kind == 0); s = f(:, kind == 0);
rb = sym2bits(d, c);
out.y = y; out.sym = d;
out.ber = calc_ber(ref.bits, rb);
out.ber_xy = [calc_ber(ref.bits(1,:), rb(1,:)), calc_ber(ref.bits(2,:), rb(2,:))];
out.snr = calc_snr(d, s);
out.evm = calc_evm(d, s);
out.gmi = calc_gmi(d, s, c);
end
