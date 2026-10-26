%MAIN_IMDD_PAM4 IM-DD PAM4 短距：DAC -> MZM(正交偏置) -> SSMF(CD 功率衰落) -> VOA -> PIN -> ADC -> LMS-FFE
clear; close all; setup_paths; rng(0);
rs = 56e9; sps = 4; L = 2; fs = rs*sps;              % 56 GBd PAM4，2 km SSMF @1550 nm
lv = pam_const(4)/3;                                 % Gray PAM4 电平，归一化到 [-1, 1]
bits = uint8(randi([0 1], 1, 2*2^15));
sym = bits2sym(bits, lv);
u = dac_model(pulse_shape(sym, sps, 0.2, 'rc'), fs, 6, 0.6*rs);
E = mzm(laser_cw(numel(u), fs, 10, 0, 0, -150), 0.4*u, 1, -0.5);   % 正交点偏置，±0.4Vπ 摆幅
E = fiber_cd(E, fs, L) * 10^(-0.2*L/20);

n_train = 4096;
rop = -14:-4;
ber = zeros(size(rop));
for k = 1:numel(rop)
    Er = voa(E, w2dbm(sum(sigpow(E))) - rop(k));    % VOA 调到目标 ROP
    i = photodiode(Er, fs, struct('R', 0.8, 'bw', 0.7*rs, 'T', 300));
    y = adc_model(i, fs, 2*rs, 6, []);
    y = (y - mean(y)) / std(y) * std(lv);
    ye = ffe_lms(y, sym, 21, 1e-3, 2, n_train, lv);
    rb = sym2bits(ye(n_train+1:end), lv);
    ber(k) = calc_ber(bits(2*n_train+1:end), rb);
    fprintf('ROP %+d dBm: BER %.2e\n', rop(k), ber(k));
end
figure('Position', [100 100 1000 400]);
subplot(1,2,1); semilogy(rop, max(ber, 1e-6), 'o-'); yline(3.8e-3, 'r:'); grid on;
xlabel('ROP (dBm)'); ylabel('BER');
subplot(1,2,2); histogram(ye(n_train+1:end), 200); title('FFE output');
exportgraphics(gcf, fullfile('results', 'imdd_pam4.png'));
