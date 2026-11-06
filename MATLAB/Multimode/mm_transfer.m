function H = mm_transfer(fs, n, mmf, n_spans, seed)
%MM_TRANSFER FMF 线性传递矩阵 H(f)（2M×2M×n，第 3 维为 FFT 频率顺序）。对应 Python multimode.mm_transfer
%   用法: H = mm_transfer(fs, 2^14, mmf, 4, 7)
%   与 mm_link(.., n_spans, false, seed, 'nonlinear', false) 逐点一致（含跨段放大器增益与 MDG，不含非线性与 ASE）。
if nargin < 4 || isempty(n_spans), n_spans = 1; end
if nargin < 5 || isempty(seed), seed = 0; end
s = RandStream('mt19937ar', 'Seed', seed);
lin = mm_lin_op(mmf, 2*pi*freqaxis(n, fs));                 % 2M×n
P = size(lin, 1);
lin = reshape(lin, P, 1, n);
g = 10.^(mm_amp_gain(mmf)/20);
H = repmat(eye(P), 1, 1, n);
Lm = mmf.L*1e3; lc = mmf.lc_km*1e3;
for k = 1:n_spans
    z = 0; z_next = lc;
    while z < Lm - 1e-9
        dz = min(z_next - z, Lm - z);
        H = exp(lin*dz) .* H;
        z = z + dz;
        if z >= z_next - 1e-9
            H = pagemtimes(mm_coupling(mmf, mmf.lc_km, s), H);
            z_next = z_next + lc;
        end
    end
    H = g .* H;
end
end
