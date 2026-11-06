function Eg = mmf_link(E, fs, mmf, P, L)
%MMF_LINK MMF 链路：单模输入光场 E (1×N) -> 各模式群光场 (G×N)。对应 Python multimode.mmf_link
%   用法: Eg = mmf_link(ref.E, cfg.fs, mmf, mmf_launch(mmf, 'ofl'), 0.1)
%   第 g 群 = sqrt(P_g)·衰减·色散(D, S @ lam)·时延 (τ_g - τ̄)·L（τ̄ 为按 P_g 加权的平均群时延）。
%   探测时 photodiode 对各行 |E|² 求和（全面积探测器，模式正交），可直接送入 imdd_rx。
%   P 长度须为模式群数 G（单个模式群用 [1, zeros(1, G-1)]）。
if nargin < 5 || isempty(L), L = mmf.L; end
q = mmf_groups(mmf);
P = P(:).';
if numel(P) ~= q.n_groups, error('P 长度应为模式群数 %d', q.n_groups); end
tau = (q.tau - sum(P.*q.tau)/sum(P))*L;
Ec = fiber_cd(E, fs, L, mmf.D, mmf.lam, mmf.S) * 10^(-mmf.alpha_db_km*L/20);
Eg = zeros(numel(P), size(E,2));
for g = 1:numel(P)
    Eg(g,:) = sqrt(P(g))*fracdelay(Ec, tau(g), fs);
end
end
