function c = qam_const(M)
%QAM_CONST Gray 映射方形 M-QAM（M = 4, 16, 64, 256, ...），单位平均功率。
%   用法: c = qam_const(16)    c(label+1) 为比特标签 label（MSB 在前，前一半比特→I）对应符号
k = log2(M)/2; L = 2^k;
p = pam_const(L);
lab = 0:M-1;
c = p(bitshift(lab, -k) + 1) + 1j*p(bitand(lab, L-1) + 1);
c = c / sqrt(mean(abs(c).^2));
end
