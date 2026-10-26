function r = gn_model(p_ch_dbm, fib, rs, n_spans, n_ch, spacing, method)
%GN_MODEL GN 模型链路 SNR 预测（理想收发机，非相干累积 ε=0）。
%   用法: r = gn_model(-4:0.1:8, fib, 32e9, 10)                  单信道
%         r = gn_model(P, fib, 32e9, 10, 7, 50e9)                 7 信道 WDM
%         r = gn_model(P, fib, 32e9, 10, 87, 50e9, 'closed')     满载 C 波段
%   输出字段: snr, snr_ase, snr_nli [dB]（与 p_ch_dbm 同尺寸）; eta 全链路 [1/W^2]; p_opt_dbm; snr_opt [dB]
%   P_ASE = N_span·NF·hν·(G-1)·Rs（双偏振、信号带宽内），P_NLI = η·P^3，SNR = P/(P_ASE + P_NLI)
if nargin < 5, n_ch = 1; end
if nargin < 6, spacing = []; end
if nargin < 7, method = 'numeric'; end
c = phys_const();
P = dbm2w(p_ch_dbm);
G = 10^(fib.alpha*fib.L/10);
p_ase = n_spans * 10^(fib.nf/10) * c.h * fib.f0 * (G - 1) * rs;
r.eta = n_spans * gn_eta(fib, rs, n_ch, spacing, method);
p_nli = r.eta * P.^3;
p_opt = (p_ase/(2*r.eta))^(1/3);
r.snr = 10*log10(P ./ (p_ase + p_nli));
r.snr_ase = 10*log10(P / p_ase);
r.snr_nli = 10*log10(P ./ p_nli);
r.p_opt_dbm = w2dbm(p_opt);
r.snr_opt = 10*log10(p_opt / (1.5*p_ase));
end
