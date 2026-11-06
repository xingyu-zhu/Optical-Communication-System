function m = mm_mdl(H)
%MM_MDL 模式相关损耗 MDL [dB] = 20lg(σmax/σmin)（H(f) 的奇异值，每个频点一个值，输出 1×n）。
%   用法: m = mm_mdl(mm_transfer(fs, n, mmf, n_spans, seed))
sv = pagesvd(H);
m = 20*log10(squeeze(sv(1,1,:) ./ sv(end,1,:))).';
end
