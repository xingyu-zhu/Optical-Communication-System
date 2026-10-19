function E = add_noise_psd(E, psd, fs)
%ADD_NOISE_PSD 按单偏振 PSD [W/Hz] 在仿真带宽 fs 内给每行加复高斯白噪声。  用法: E = add_noise_psd(E, psd, fs)
E = E + sqrt(psd*fs/2) * (randn(size(E)) + 1j*randn(size(E)));
end
