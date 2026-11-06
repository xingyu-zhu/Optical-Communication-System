function [lab, Gam] = mm_nl_matrix(mmf)
%MM_NL_MATRIX FMF 非线性系数矩阵 Γ = (n2·ω0/c)·κ [1/(W m)]。对应 Python multimode.nl_matrix
%   用法: [lab, Gam] = mm_nl_matrix(mm_params('FMF3'))
%   lab 1×2M 每个通道所属非线性群的下标（1-based）；regime='strong' 时全部通道为一个群，'weak' 时为各模式群
comp = repelem(mmf.groups, 2);
if strcmp(mmf.regime, 'strong'), labels = zeros(size(comp)); else, labels = comp; end
[K, g] = kerr_coeffs(mm_overlap(mmf.hg, mmf.w0_um*1e-6), labels);
[~, lab] = ismember(labels, g);
c = phys_const();
Gam = mmf.n2*2*pi*mmf.f0/c.c*K;
end
