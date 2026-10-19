function Eo = phase_mod(E, v, vpi)
%PHASE_MOD 相位调制器：E·exp(jπv/Vπ)。  用法: Eo = phase_mod(E, v, vpi)
Eo = E .* exp(1j*pi*v/vpi);
end
