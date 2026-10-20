function c = constellation(fmt, M)
%CONSTELLATION 按格式名取相干星座（单位平均功率）。  用法: c = constellation('psk', 8)   fmt: 'qam' | 'psk'
switch fmt
    case 'qam', c = qam_const(M);
    case 'psk', c = psk_const(M);
end
end
