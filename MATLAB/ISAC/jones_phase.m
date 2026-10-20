function ph = jones_phase(J)
%JONES_PHASE Jones 矩阵（2×2×nb）的公共相位 ∠det(J)/2（沿时间解卷绕）：前向传输光相位（含激光相噪）。
%   用法: ph = jones_phase(J)    输出 1×nb [rad]
d = squeeze(J(1,1,:).*J(2,2,:) - J(1,2,:).*J(2,1,:)).';
ph = unwrap(angle(d)) / 2;
end
