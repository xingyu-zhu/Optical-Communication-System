function q = mmf_groups(mmf)
%MMF_GROUPS MMF 主模式群（α 剖面渐变折射率光纤）。对应 Python multimode.mmf_groups
%   用法: q = mmf_groups(mm_params('OM4'))
%   输出字段: n_groups G, n_modes 1×G 每群空间模式数 g, tau 1×G 各群单位长度群时延 [s/km]（绝对值），
%             x 1×G 归一化模式序号 m/M ≈ g²/(G(G+1))
%   群时延（Gloge-Marcatili / Olshansky-Keck，忽略剖面色散 ε）：
%     τ_g = ng/c·[1 + Δ(α-2)/(α+2)·x^{2α/(α+2)} + Δ²(3α-2)/(2(α+2))·x^{4α/(α+2)}]
G = mmf.n_groups; a = mmf.alpha_profile; D = mmf.delta;
c = phys_const();
g = 1:G;
q.n_groups = G; q.n_modes = g;
q.x = g.^2/(G*(G + 1));
q.tau = mmf.ng/c.c*1e3 * (1 + D*(a - 2)/(a + 2)*q.x.^(2*a/(a + 2)) ...
                          + D^2*(3*a - 2)/(2*(a + 2))*q.x.^(4*a/(a + 2)));
end
