function b = ber_theory(M, snr_db, kind)
%BER_THEORY AWGN 下 Gray 映射理论 BER（最近邻近似），snr_db 为每符号 SNR。
%   用法: b = ber_theory(16, 10:20)           M-QAM
%         b = ber_theory(8, snr, 'psk')       kind: 'qam'(默认) | 'psk' | 'pam'
if nargin < 3, kind = 'qam'; end
s = 10.^(snr_db/10); k = log2(M);
Qf = @(v) 0.5*erfc(v/sqrt(2));
switch kind
    case 'qam', b = 4/k*(1 - 1/sqrt(M)) * Qf(sqrt(3*s/(M-1)));
    case 'psk'
        if M <= 4, b = Qf(sqrt(s)); else, b = 2/k*Qf(sqrt(2*s)*sin(pi/M)); end
    case 'pam', b = 2*(M-1)/(M*k) * Qf(sqrt(3*s/(M^2-1)));
end
end
