function Eo = mzm(E, v, vpi, vbias)
%MZM 推挽 MZM 场传输：E·cos(π(v+vbias)/(2Vπ))。
%   用法: Eo = mzm(E, v, vpi, vbias)
%   vbias = -Vπ：零点偏置（相干 IQ 子调制器，场 ∝ sin，近线性）；vbias = -Vπ/2：正交点（IM-DD）
Eo = E .* cos(pi*(v + vbias) / (2*vpi));
end
