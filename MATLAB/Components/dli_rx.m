function z = dli_rx(E, fs, rs, pd)
%DLI_RX 差分相位直接检测接收机（DPSK/DQPSK/D8PSK）：3 dB 分光 -> 两个 1 符号延迟 DLI（φ = 0、π/2）-> 两个 BPD。
%   用法: z = dli_rx(E, fs, rs)              pd 同 photodiode（默认含散粒噪声）
%         z = dli_rx(E, fs, rs, struct('shot', false))
%   I = Re{E(t)E*(t-T)}/2，Q = Im{E(t)E*(t-T)}/2，返回 z = I + jQ ∝ y_k·conj(y_{k-1})。
%   无需本振，对激光相位噪声（Δν·T << 1）与小频偏不敏感；多行输入（双偏振）时功率相加，偏振不敏感。
if nargin < 4, pd = struct(); end
[a, b] = dli(E/sqrt(2), fs, 1/rs, 0);
[c, d] = dli(E/sqrt(2), fs, 1/rs, pi/2);
z = bpd(a, b, fs, pd) + 1j*bpd(c, d, fs, pd);
end
