function cq = qam_quadrant(M)
%QAM_QUADRANT 方形 M-QAM 第一象限的 M/4 个点（I、Q 符号位均为 1 的标签，象限内保持 Gray）。diff_qam 用。
%   用法: cq = qam_quadrant(16)    cq(j+1) 对应象限内 k-2 比特标签 j
k = log2(M); h = k/2; j = 0:M/4-1;
lab = bitor(bitor(2^(k-1), bitshift(bitshift(j, -(h-1)), h)), bitor(2^(h-1), bitand(j, 2^(h-1)-1)));
c = qam_const(M);
cq = c(lab + 1);
end
