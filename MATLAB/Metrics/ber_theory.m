function b = ber_theory(M, snr_db, kind)
%BER_THEORY AWGN 下 Gray 映射理论 BER，snr_db 为每符号 SNR (Es/N0)。
%   用法: b = ber_theory(16, 10:20)           M-QAM（最近邻近似）
%         b = ber_theory(8, snr, 'psk')       kind: 'qam'(默认) | 'psk'（相干，BPSK/QPSK 精确）| 'pam' |
%                                             'dpsk'（差分检测：DBPSK 精确 ½e^{-SNR}，DQPSK 精确 Marcum-Q 式，M≥8 近似）
if nargin < 3, kind = 'qam'; end
s = 10.^(snr_db/10); k = log2(M);
Qf = @(v) 0.5*erfc(v/sqrt(2));
switch kind
    case 'qam', b = 4/k*(1 - 1/sqrt(M)) * Qf(sqrt(3*s/(M-1)));
    case 'psk'
        if M == 2, b = Qf(sqrt(2*s));
        elseif M == 4, b = Qf(sqrt(s));
        else, b = 2/k*Qf(sqrt(2*s)*sin(pi/M)); end
    case 'dpsk'
        if M == 2, b = 0.5*exp(-s);
        elseif M == 4                                          % Proakis 式 (8.6-62)，γb = SNR/2
            a2 = s*(1 - 1/sqrt(2)); b2 = s*(1 + 1/sqrt(2));
            b = marcumq(sqrt(a2), sqrt(b2)) - 0.5*besseli(0, sqrt(a2.*b2), 1).*exp(sqrt(a2.*b2) - (a2 + b2)/2);
        else, b = 2/k*Qf(sqrt(2*s)*sin(pi/(sqrt(2)*M)));
        end
    case 'pam', b = 2*(M-1)/(M*k) * Qf(sqrt(3*s/(M^2-1)));
end
end
