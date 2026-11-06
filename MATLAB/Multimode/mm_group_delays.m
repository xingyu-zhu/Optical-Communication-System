function gd = mm_group_delays(H, fs)
%MM_GROUP_DELAYS 群时延算符 j·H⁻¹·dH/dω 的本征值 [s]（输出 n×2M，每行按实部升序；H = e^{-jωτ} 时得 τ）。
%   用法: gd = mm_group_delays(mm_transfer(fs, n, mmf, n_spans, seed), fs)
%   dH/dω 用中心差分，要求频点间隔 fs/n 远小于 1/(2·最大群时延差)，否则相位卷绕使结果失真。
n = size(H, 3);
dw = 2*pi*fs/n;
dH = (circshift(H, -1, 3) - circshift(H, 1, 3))/(2*dw);
ev = pageeig(1j*pagemldivide(H, dH));
gd = sort(real(reshape(ev, size(H,1), n)), 1).';
end
