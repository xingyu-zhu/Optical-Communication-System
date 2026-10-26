function H = raman_response(w, tau1, tau2)
%RAMAN_RESPONSE 石英 Raman 响应频域形式 H_R(ω)（h_R(t) = (τ1²+τ2²)/(τ1τ2²)·e^{-t/τ2}·sin(t/τ1) 的 FFT，H_R(0)=1）。
%   用法: H = raman_response(w, 12.2e-15, 32e-15)    w 角频率 [rad/s]；|Im H| 峰值约在 13.2 THz
H = (tau1^2 + tau2^2)/(tau1*tau2^2)/tau1 ./ ((1/tau2 + 1j*w).^2 + 1/tau1^2);
end
