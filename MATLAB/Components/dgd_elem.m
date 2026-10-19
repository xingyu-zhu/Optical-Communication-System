function E = dgd_elem(E, fs, tau)
%DGD_ELEM 差分群时延（一阶 PMD）：两本征偏振相对时延 tau [s]。  用法: E = dgd_elem(E, fs, 10e-12)
w = 2*pi*freqaxis(size(E,2), fs);
E = fdfilter(E, [exp(1j*w*tau/2); exp(-1j*w*tau/2)]);
end
