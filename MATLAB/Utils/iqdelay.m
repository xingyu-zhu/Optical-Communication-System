function y = iqdelay(x, tau_i, tau_q, fs)
%IQDELAY I、Q 分别施加时延（模拟或补偿 IQ skew）。  用法: y = iqdelay(x, tau_i, tau_q, fs)
y = fracdelay(real(x), tau_i, fs) + 1j*fracdelay(imag(x), tau_q, fs);
end
