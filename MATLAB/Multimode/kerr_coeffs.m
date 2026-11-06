function [K, g] = kerr_coeffs(f, labels)
%KERR_COEFFS 模式群平均的多模 Manakov 非线性系数 κ_gh [1/m²]。对应 Python multimode.kerr_coeffs
%   用法: [K, g] = kerr_coeffs(mm_overlap(hg, w), labels)
%   labels: 每个空间-偏振通道（顺序 [模式1 X, 模式1 Y, 模式2 X, ...]，共 2M 个）所属的群
%   推导：通道 i=(a,p) 的 Kerr 项 NL_i = Σ T_ijkl E_j* E_k E_l，各向同性石英 P_NL ∝ (1/3)[2(E*·E)E + (E·E)E*]。
%   对各群内独立的 Haar 随机酉变换求平均后只剩 Σ_h κ_gh |E_h|² E_g；用复高斯矩
%   <x_i*x_j*x_k x_l> = δ_ik δ_jl + δ_il δ_jk 缩并两边得
%     κ_gg = Σ_{i,j∈g} S_ij / (n_g(n_g+1))，κ_gh = Σ_{i∈g, j∈h} S_ij / (n_g n_h)，S_ij = f_ab·(2/3 + 4/3·δ_pq)
%   单模（一个群、2 个偏振通道）：κ = 8/9·f，即 Manakov 系数。输出 K (G×G)、群标签 g (1×G)
labels = labels(:).';
M = size(f, 1);
a = repelem(1:M, 2); p = repmat([0 1], 1, M);
S = f(a, a) .* (2/3 + 4/3*(p.' == p));
g = unique(labels);
K = zeros(numel(g));
for i = 1:numel(g)
    for j = 1:numel(g)
        mi = labels == g(i); mj = labels == g(j);
        ni = sum(mi); nj = sum(mj);
        if i == j, nrm = ni*(ni + 1); else, nrm = ni*nj; end
        K(i,j) = sum(S(mi, mj), 'all') / nrm;
    end
end
end
