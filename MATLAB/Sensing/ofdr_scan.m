function s = ofdr_scan(r, dz, eps, K)
%OFDR_SCAN OFDR 一次扫频的复拍频信号（I/Q 相干探测，已线性化到等光频间隔）。
%   扫频范围 B = c/(2·ng·dz)，点数 = 单元数；s(ν_k) = Σ_i h_i·e^{-j2πν_kτ_i} 即 h 的 FFT，拍频频率 ∝ 距离。
%   应变 ε(z) [με] 使局部光程伸长 K·ε（K≈0.78），累积附加时延 δτ 以 e^{-j2πf0·δτ} 计入（忽略 δτ 随扫频的变化）。
%   用法: s = ofdr_scan(r, dz)            参考扫描
%         s = ofdr_scan(r, dz, eps)       eps 为 1×N 应变分布 [με]
if nargin < 3, eps = 0; end
if nargin < 4, K = 0.78; end
c0 = phys_const(); sc = sens_const();
dtau = K*cumsum(eps.*ones(size(r)))*1e-6*dz*2*sc.ng/c0.c;
s = fft(r .* exp(-2j*pi*c0.f0*dtau));
end
