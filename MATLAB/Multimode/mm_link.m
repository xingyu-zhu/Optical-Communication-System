function E = mm_link(E, fs, mmf, n_spans, ase, seed, varargin)
%MM_LINK FMF 多跨链路：每跨 ssfm_mm + 少模 EDFA（各通道增益 mm_amp_gain，ASE 按各自增益与 nf_db）。
%   用法: E = mm_link(E, fs, mmf, 4, true, 7)                       模式耦合种子 7，ASE 用全局随机数
%         E = mm_link(E, fs, mmf, 4, false, 7, 'nonlinear', false)  线性、无 ASE（与 mm_transfer(.., 7) 逐点一致）
%   seed 只决定模式耦合（独立 RandStream），其余参数透传 ssfm_mm。对应 Python multimode.mm_link
if nargin < 5 || isempty(ase), ase = true; end
if nargin < 6 || isempty(seed), seed = 0; end
s = RandStream('mt19937ar', 'Seed', seed);
g = mm_amp_gain(mmf);
psd = arrayfun(@(gk) ase_psd(10^(gk/10), mmf.nf_db, mmf.f0), g);
for k = 1:n_spans
    E = ssfm_mm(E, fs, mmf, s, varargin{:}) .* 10.^(g/20);
    if ase, E = E + sqrt(psd*fs/2) .* (randn(size(E)) + 1j*randn(size(E))); end
end
end
