function r = coherent_rx(E, Elo, fs, pd, skew)
%COHERENT_RX 偏振分集相干接收机：PBS + 2×90°混频器 + 4×BPD。
%   用法: r = coherent_rx(E, Elo, fs)
%         r = coherent_rx(E, Elo, fs, pd, skew)
%   E 2×N 信号；Elo 1×N 本振（45° 线偏，均分到两偏振）；pd 见 photodiode；
%   skew = [XI XQ YI YQ] 各支路时延 [s]。输出 2×N 复电流 I+jQ ∝ R·Es·conj(Elo)。
if nargin < 4, pd = struct(); end
if nargin < 5, skew = [0 0 0 0]; end
r = zeros(size(E));
for p = 1:2
    o = hybrid90(E(p,:), Elo/sqrt(2));
    r(p,:) = bpd(o(1,:), o(2,:), fs, pd) + 1j*bpd(o(3,:), o(4,:), fs, pd);
    if any(skew), r(p,:) = iqdelay(r(p,:), skew(2*p-1), skew(2*p), fs); end
end
end
