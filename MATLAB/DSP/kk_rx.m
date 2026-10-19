function E = kk_rx(i_pd)
%KK_RX Kramers-Kronig 接收：由单 PD 光电流（强载波 + 单边带信号，满足最小相位条件）重建复光场（含载波）。
%   用法: E = kk_rx(i_pd)     E = sqrt(I)·exp(j·H{ln sqrt(I)})，H 为 Hilbert 变换
a = log(sqrt(max(i_pd, 1e-12*max(i_pd))));
E = exp(a + 1j*imag(hilbert(a)));
end
