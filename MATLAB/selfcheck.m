%SELFCHECK 自检：DSP 算法恢复能力 + 光纤物理模型（SPM / 孤子 / SRS / FWM / GN）正确性。
clear; setup_paths; rng(1);
M = 16; rs = 32e9; sps = 2; fs = rs*sps;
c = qam_const(M);
bits = uint8(randi([0 1], 2, 4*2^14));
s = bits2sym(bits, c);
x = normpow(pulse_shape(s, sps, 0.1));

fprintf('---- DSP ----\n');
% 1) 映射/解映射 + AWGN 理论 BER
r = addawgn(s, 14);
b = calc_ber(bits, sym2bits(r, c)); bt = ber_theory(M, 14);
rep('QAM map / AWGN BER', abs(b/bt - 1) < 0.15, sprintf('sim %.2e theory %.2e GMI %.2f', b, bt, calc_gmi(r, s, c)));

% 2) CD(含斜率) + CDC + 盲 CD 估计
y = fiber_cd(x, fs, 500, 17, 1550, 0.057);
Lh = cd_estimate(y, fs, rs, 300:20:700);
z = matched_filter(cdc(y, fs, 500, 17, 1550, 0.057), sps, 0.1); z = z(:, 1:sps:end);
rep('CD / CDC / CD estimate', calc_snr(z, s) > 25 && Lh == 500, sprintf('SNR %.1f dB, L_est %d km', calc_snr(z, s), Lh));

% 3) FOE
xf = fo_comp(s, -1.2e9, rs);
fa = foe(xf, rs); fd = foe(xf, rs, 'diff');
rep('FOE fft / diff', abs(fa-1.2e9) < 5e6 && abs(fd-1.2e9) < 5e7, sprintf('%.4f / %.4f GHz (true 1.2)', fa/1e9, fd/1e9));

% 4) CPR：合计 200 kHz 线宽
r = addawgn(s .* laser_cw(size(s,2), rs, 0, 200e3)/sqrt(1e-3), 18);
sb = calc_snr(fix_ambiguity(cpr_bps(r, c), s), s);
sv = calc_snr(fix_ambiguity(cpr_vv(r, 64, 4, c), s), s);
rep('CPR BPS / VV', min(sb, sv) > 16, sprintf('SNR BPS %.1f VV %.1f dB (AWGN 18)', sb, sv));
mask = false(1, size(s,2)); mask(1:32:end) = true;
sp = calc_snr(cpr_pilot(r, s, mask, 5), s);
sl = calc_snr(fix_ambiguity(cpr_pll(r*exp(0.3j), c), s), s);
rep('CPR pilot / DD-PLL', sp > 16 && sl > 16, sprintf('SNR pilot %.1f PLL %.1f dB', sp, sl));

% 5) 定时恢复：0.37 采样时延
xd = fracdelay(x, 0.37/fs, fs);
[~, tau] = timing_ff(xd, sps);
[~, taus] = gardner_tr(xd, 2e-3, sps);
rep('Timing FF / Gardner', abs(tau-0.37) < 0.05 && abs(taus(end)-0.37) < 0.1, sprintf('FF %.3f, Gardner %.3f sa (true 0.37)', tau, taus(end)));

% 6) MIMO：随机 SOP + DGD 10 ps
xm = matched_filter(addawgn(dgd_elem(random_unitary()*x, fs, 10e-12), 20, sps), sps, 0.1);
[~, W] = mimo_eq(xm, 15, 1e-3, sps, 'cma', c);
ym = mimo_eq(xm, 15, 5e-4, sps, 'rde', c, [], 0, W);
[ys, lag] = frame_sync(cpr_bps(ym, c), s);
ys = fix_ambiguity(ys, s);
sm = calc_snr(ys(:, 5001:end), s(:, 5001:end));
rep('MIMO CMA->RDE + sync', sm > 17, sprintf('SNR %.1f dB (AWGN 20), lag %d', sm, lag));

% 7) Kramers-Kronig：载波 + 单边带信号（偏移 0.6 Rs），CSPR 10 dB
xs = normpow(pulse_shape(s(1,:), 4, 0.1)); t = 0:numel(xs)-1; fsh = 0.6/4;
Ek = kk_rx(abs(sqrt(10) + xs.*exp(2j*pi*fsh*t)).^2);
rec = matched_filter((Ek - mean(Ek)).*exp(-2j*pi*fsh*t), 4, 0.1); rec = rec(1:4:end);
rep('Kramers-Kronig', calc_snr(rec, s(1,:)) > 20, sprintf('SNR %.1f dB', calc_snr(rec, s(1,:))));

fprintf('---- Fiber physics ----\n');
fib = fiber_params('C', 'SSMF', 'pmd', 0);
% 8) SPM：CW 非线性相移 = γ·P·Leff
P0 = 0.1;
Eo = ssfm_manakov(sqrt(P0)*ones(1, 1024), 100e9, fib, 'nsteps', 200);
a = fib.alpha*log(10)/10; phi_th = fib.gamma*P0*(1 - exp(-a*fib.L))/a;
rep('SPM phase (CW)', abs(-angle(Eo(1))/phi_th - 1) < 1e-3, sprintf('%.4f rad vs gPLeff %.4f', -angle(Eo(1)), phi_th));

% 9) 基阶孤子：P0 = |β2|/(γT0²)，无损传输 5 个色散长度后形状不变
T0 = 5e-12; fsol = 2e12; tt = ((0:4095) - 2048)/fsol;
b2 = abs(beta2_from_D(fib.D, fib.lam));
Ps = b2/(fib.gamma/1e3*T0^2); LD = T0^2/b2/1e3;
u0 = sqrt(Ps)./cosh(tt/T0);
f0 = fib; f0.alpha = 0; f0.S = 0; f0.L = 5*LD;
u1 = ssfm_manakov(u0, fsol, f0, 'nsteps', 500);
f0.gamma = 0; u2 = ssfm_manakov(u0, fsol, f0, 'nsteps', 10);
err = max(abs(abs(u1) - abs(u0)))/sqrt(Ps);
rep('Fundamental soliton', err < 0.01, sprintf('shape error %.1e (linear-only peak drops to %.2f)', err, max(abs(u2))^2/Ps));

% 10) SRS：相距 13.2 THz 的两 CW（各 0.5 W），与双波解析解 P_low = P/(1+e^{-gPz}) 比较
fsr = 40e12; nr = 4000; fr = 6.6e12; tr = (0:nr-1)/fsr;
E2 = sqrt(0.5)*(exp(2j*pi*fr*tr) + exp(-2j*pi*fr*tr));
fr_ = fib; fr_.alpha = 0; fr_.L = 1; fr_.raman = true;
Sp = abs(fft(ssfm_manakov(E2, fsr, fr_, 'nsteps', 50))/nr).^2;
[~, il] = min(abs(freqaxis(nr, fsr) + fr)); lo = Sp(il);
gR = 2*fib.gamma/1e3*fib.fR*abs(imag(raman_response(2*pi*2*fr, fib.tau1, fib.tau2)));
lo_th = 1/(1 + exp(-gR*1e3));
rep('SRS (2-tone, 1 km)', abs(lo/lo_th - 1) < 0.01, sprintf('P_low %.3f W vs analytic %.3f W (gain %.2f /W/km)', lo, lo_th, gR*1e3));

% 11) FWM：两 CW（±25 GHz）在 DSF 与 SSMF 中产生的 ±75 GHz 分量
fw = 400e9; nw = 4096; tw = (0:nw-1)/fw; fa_ = freqaxis(nw, fw);
Ew = sqrt(0.01)*(exp(2j*pi*25e9*tw) + exp(-2j*pi*25e9*tw));
names = {'DSF', 'SSMF'}; fwm = zeros(1, 2);
for k = 1:2
    Sw = abs(fft(ssfm_manakov(Ew, fw, fiber_params('C', names{k}, 'pmd', 0), 'nsteps', 400))).^2;
    [~, i3] = min(abs(fa_ - 75e9)); [~, i1] = min(abs(fa_ - 25e9));
    fwm(k) = 10*log10(Sw(i3)/Sw(i1));
end
rep('FWM (DSF vs SSMF)', fwm(1) > fwm(2) + 15, sprintf('product/pump: DSF %.1f dB, SSMF %.1f dB', fwm(1), fwm(2)));

% 12) GN 模型：数值积分 vs 闭式（5 信道 Nyquist WDM）
en = gn_eta(fib, 32e9, 5, 32e9, 'numeric'); ec = gn_eta(fib, 32e9, 5, 32e9, 'closed');
rep('GN numeric vs closed', abs(10*log10(en/ec)) < 1, sprintf('eta %.1f vs %.1f 1/W^2 per span (%+.2f dB)', en, ec, 10*log10(en/ec)));

function rep(name, ok, info)
tag = {'BAD', 'OK '};
fprintf('[%s] %-26s %s\n', tag{ok+1}, name, info);
end
