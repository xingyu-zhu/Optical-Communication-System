function Eo = iq_mod(E, u, vpi, phase_err, gain_imb_db)
%IQ_MOD IQ 调制器（两个零点偏置子 MZM + 90° 移相）。
%   用法: Eo = iq_mod(E, u, vpi)
%         Eo = iq_mod(E, u, vpi, phase_err, gain_imb_db)
%   u 复驱动电压（实部→I，虚部→Q）；phase_err 正交误差 [rad]；gain_imb_db I/Q 幅度不平衡 [dB]
%   Eo = E/2·[MZM(uI) + e^{j(π/2+φe)}·MZM(uQ)]
if nargin < 4, phase_err = 0; end
if nargin < 5, gain_imb_db = 0; end
g = 10^(gain_imb_db/40);
Eo = 0.5*(g*mzm(E, real(u), vpi, -vpi) + exp(1j*(pi/2 + phase_err))/g*mzm(E, imag(u), vpi, -vpi));
end
