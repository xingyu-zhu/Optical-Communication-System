function S = jones_sop(J, e_in)
%JONES_SOP 输入偏振 e_in 经 Jones 矩阵 J（2×2×nb）后的输出 Stokes 向量（3×nb）。
%   用法: S = jones_sop(J)            默认 e_in = [1; 0]（X 偏振）
%         S = jones_sop(J, e_in)
if nargin < 2, e_in = [1; 0]; end
S = stokes_vec(reshape(pagemtimes(J, e_in), 2, []));
end
