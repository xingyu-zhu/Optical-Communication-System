function Eo = dp_iq_mod(E, u, vpi, varargin)
%DP_IQ_MOD 双偏振 IQ 调制器：PBS 均分 -> 2×iq_mod -> PBC 合束。
%   用法: Eo = dp_iq_mod(E, u, vpi)                    u 为 2×N 复驱动
%         Eo = dp_iq_mod(E, u, vpi, phase_err, gain_imb_db)   其余参数透传 iq_mod
Eo = [iq_mod(E/sqrt(2), u(1,:), vpi, varargin{:}); iq_mod(E/sqrt(2), u(2,:), vpi, varargin{:})];
end
