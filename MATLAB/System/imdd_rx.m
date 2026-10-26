function out = imdd_rx(E, cfg, ref, rop_dbm)
%IMDD_RX IM-DD 接收：VOA 调到 ROP -> PIN 或 APD（+TIA 噪声）-> ADC 2 sps -> 均衡（cfg.eq）-> PAM 判决。
%   用法: res = imdd_rx(imdd_link(ref.E, cfg), cfg, ref, -10)
%   cfg.eq: 'ffe'（LMS）| 'dfe'（dfe_lms）| 'volterra'（volterra_eq，二/三阶记忆 9/5）| 'mlse'（FFE + pf_mlse）
%   输出字段: ber, snr（均衡后电 SNR [dB]）, y（均衡输出 1 sps）, n_bits 统计比特数
E = voa(E, w2dbm(mean(abs(E).^2)) - rop_dbm);
pd = struct('R', cfg.R, 'bw', cfg.rx_bw*cfg.rs, 'irnd', cfg.irnd);
if strcmp(cfg.rx, 'apd'), pd.M = cfg.apd_M; i = apd(E, cfg.fs, pd); else, i = photodiode(E, cfg.fs, pd); end
y = adc_model(i, cfg.fs, 2*cfg.rs, cfg.adc_enob, []);
lv = ref.levels; sym = ref.sym; nt = cfg.n_train;
y = (y - mean(y)) / std(y) * std(lv);
switch cfg.eq
    case 'dfe',      ye = dfe_lms(y, sym, cfg.ntaps, cfg.nb, cfg.mu, 2, nt, lv);
    case 'volterra', ye = volterra_eq(y, sym, nt, [cfg.ntaps 9 5], 2);
    otherwise,       ye = ffe_lms(y, sym, cfg.ntaps, cfg.mu, 2, nt, lv);
end
if strcmp(cfg.eq, 'mlse'), a = pf_mlse(ye, sym, nt, lv); else, [~, a] = sym_decide(ye, lv); end
k = log2(cfg.M);
rb = sym2bits(a(nt+1:end), lv);
out.ber = calc_ber(ref.bits(nt*k+1:end), rb);
out.snr = calc_snr(ye(nt+1:end), sym(nt+1:end));
out.y = ye; out.n_bits = numel(rb);
end
