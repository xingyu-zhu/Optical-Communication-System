function q = q_factor(ber)
%Q_FACTOR 由 BER 换算 Q 因子 [dB]：20·log10(sqrt(2)·erfcinv(2·BER))。  用法: q = q_factor(1e-3)
q = 20*log10(sqrt(2)*erfcinv(2*ber));
end
