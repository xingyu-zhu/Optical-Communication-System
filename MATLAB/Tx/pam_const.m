function lv = pam_const(M)
%PAM_CONST Gray 映射 PAM-M 电平（±1, ±3, ...，未归一化）。
%   用法: lv = pam_const(4)    lv(label+1) 为比特标签 label 对应电平
g = bitxor(0:M-1, bitshift(0:M-1, -1));
lv = zeros(1, M);
lv(g+1) = 2*(0:M-1) - M + 1;
end
