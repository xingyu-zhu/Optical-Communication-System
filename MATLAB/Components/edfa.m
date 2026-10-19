function E = edfa(E, fs, gain_db, nf_db, p_out_dbm, f0)
%EDFA 光放大器（EDFA/TDFA/... 通用集总模型）：增益 + ASE。
%   用法: E = edfa(E, fs, 16, 5)                  固定增益 16 dB，NF 5 dB
%         E = edfa(E, fs, [], 5, 0)               自动功率控制：总输出 0 dBm
%         E = edfa(E, fs, G, nf, [], fib.f0)      指定中心频率（波段相关）
if nargin < 6, f0 = phys_const().f0; end
if nargin > 4 && ~isempty(p_out_dbm)
    G = dbm2w(p_out_dbm) / sum(sigpow(E));
else
    G = 10^(gain_db/10);
end
E = add_noise_psd(sqrt(G)*E, ase_psd(G, nf_db, f0), fs);
end
