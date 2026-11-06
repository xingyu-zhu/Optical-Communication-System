function E = ssfm_mm(E, fs, mmf, s, varargin)
%SSFM_MM FMF 单跨传输：耦合段模型 + 对称分步傅里叶（线性半步 - 群平均 Kerr 相移 - 线性半步）。
%   用法: E = ssfm_mm(E, fs, mmf, stream)                          默认 'phi_max' 0.005
%         E = ssfm_mm(E, fs, mmf, stream, 'nonlinear', false)      纯线性
%   E 2M×N 光场 [sqrt(W)]（行顺序 [模式1 X, 模式1 Y, 模式2 X, ...]）；mmf 由 mm_params 生成（L [km]）；
%   stream 只用于随机耦合矩阵的 RandStream（与 mm_transfer 用同一种子即得到同一条光纤实现）
%   每段长 lc 内各通道独立传播（MDL、DMGD、色散，见 mm_lin_op），段末乘 mm_coupling；
%   非线性：∂E_g/∂z = -j·Σ_h Γ_gh·|E_h|²·E_g（mm_nl_matrix），精确相位旋转。对应 Python multimode.ssfm_mm
o = struct('phi_max', 0.005, 'nonlinear', true);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
if nargin < 4 || isempty(s), s = RandStream.getGlobalStream; end
lin = mm_lin_op(mmf, 2*pi*freqaxis(size(E,2), fs));
[lab, Gam] = mm_nl_matrix(mmf);
G = size(Gam, 1);
Lm = mmf.L*1e3; lc = mmf.lc_km*1e3;
z = 0; z_next = lc;
while z < Lm - 1e-9
    dz = min(z_next - z, Lm - z);
    if o.nonlinear
        dz = min(dz, o.phi_max/max(max(Gam(:))*max(sum(abs(E).^2, 1)), 1e-30));
    end
    Hh = exp(lin*dz/2);
    E = fdfilter(E, Hh);
    if o.nonlinear                                           % 精确相位旋转（各群功率不变）
        P = zeros(G, size(E,2));
        for g = 1:G, P(g,:) = sum(abs(E(lab == g, :)).^2, 1); end
        ph = Gam*P;
        E = E .* exp(-1j*ph(lab, :)*dz);
    end
    E = fdfilter(E, Hh);
    z = z + dz;
    if z >= z_next - 1e-9
        E = mm_coupling(mmf, mmf.lc_km, s)*E;
        z_next = z_next + lc;
    end
end
end
