function U = mm_coupling(mmf, dz_km, s)
%MM_COUPLING 一个耦合段末端的随机模式耦合矩阵 (2M×2M)。对应 Python multimode.coupling
%   用法: U = mm_coupling(mmf, mmf.lc_km, stream)
%   strong：Haar(2M)；weak：各模式群内 Haar（块对角）后乘 expm(j·sqrt(xt)·H)，H 为只含群间元素的随机厄米矩阵
%   （按行归一化，使每段从一个通道耦合到其他群的功率 ≈ xt = 10^(xt_db_km/10)·dz_km）
if nargin < 3 || isempty(s), s = RandStream.getGlobalStream; end
comp = repelem(mmf.groups, 2);
n = numel(comp);
if strcmp(mmf.regime, 'strong'), U = haar_unitary(n, s); return; end
U = zeros(n);
for g = unique(comp)
    idx = find(comp == g);
    U(idx, idx) = haar_unitary(numel(idx), s);
end
xt = 10^(mmf.xt_db_km/10)*dz_km;
if xt > 0 && numel(unique(comp)) > 1
    Hm = randn(s, n) + 1j*randn(s, n);
    Hm(comp.' == comp) = 0;
    Hm = (Hm + Hm')/2;
    Hm = Hm / sqrt(mean(sum(abs(Hm).^2, 2)));
    U = expm(1j*sqrt(xt)*Hm)*U;
end
end
