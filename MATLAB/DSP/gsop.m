function y = gsop(x)
%GSOP Gram-Schmidt 正交化：去直流并补偿每行 IQ 幅度/相位不平衡（同参考代码 IQimbalance 思想）。
%   用法: y = gsop(x)    输出每行 I、Q 单位功率且正交
I = real(x) - mean(real(x), 2);
Q = imag(x) - mean(imag(x), 2);
I = I ./ sqrt(mean(I.^2, 2));
Q = Q - mean(I.*Q, 2) .* I;
Q = Q ./ sqrt(mean(Q.^2, 2));
y = I + 1j*Q;
end
