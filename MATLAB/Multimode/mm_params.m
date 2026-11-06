function p = mm_params(name, varargin)
%MM_PARAMS 少模/多模光纤参数（params/fiber_bands.json 的 "multimode" 部分）。对应 Python multimode.mm_params
%   用法: p = mm_params('FMF3')                               3 模 FMF（默认 L = 50 km）
%         p = mm_params('FMF6', 'L', 10, 'regime', 'strong')  覆盖任意字段
%         p = mm_params('OM4', 'L', 0.1)                       多模光纤（默认 L = 0.1 km）
%   name: 'FMF3' | 'FMF6' | 'OM3' | 'OM4'（可在 JSON 中添加）；'L' 长度 [km]，'lam' 波长 [nm]；
%   其余名值对先覆盖 JSON 原始字段，再计算派生量（不在 JSON 中的字段最后直接写入）。
%   输出（除 JSON 字段外）: name, lam [nm], f0 [Hz], L [km]；
%     FMF: n_modes 空间模式数 M；groups/mdl_db_km/D/S/dmgd_ps_km/mdg_db 为行向量，hg 为 M×2
%     MMF: a [µm] 纤芯半径, delta 相对折射率差, V 归一化频率, n_groups 模式群数 G, w0_um 基模模场半径,
%          alpha_opt 最优剖面指数 2 - 12Δ/5, alpha_profile 实际剖面指数, D [ps/(nm km)], S [ps/(nm^2 km)]
P = load_params();
raw = P.multimode.(name);
L = []; lam = []; extra = {};
for k = 1:2:numel(varargin)
    key = varargin{k}; val = varargin{k+1};
    switch key
        case 'L', L = val;
        case 'lam', lam = val;
        otherwise
            if isfield(raw, key), raw.(key) = val; else, extra(end+1:end+2) = {key, val}; end %#ok<AGROW>
    end
end
p = raw; p.name = name;
if isempty(lam), p.lam = raw.lam_nm; else, p.lam = lam; end
c = phys_const();
p.f0 = c.c/(p.lam*1e-9);
if strcmp(raw.type, 'FMF')
    if isempty(L), p.L = 50; else, p.L = L; end
    for k = {'groups', 'mdl_db_km', 'D', 'S', 'dmgd_ps_km', 'mdg_db'}
        p.(k{1}) = double(p.(k{1})(:)).';
    end
    p.hg = double(raw.hg);
    p.n_modes = numel(p.groups);
else
    if isempty(L), p.L = 0.1; else, p.L = L; end
    a = raw.core_um/2; na = raw.na; k0 = 2*pi/(p.lam*1e-9);
    p.a = a; p.delta = na^2/(2*raw.n1^2); p.V = k0*a*1e-6*na;
    p.n_groups = floor(p.V/2);
    p.w0_um = sqrt(2*a*1e-6/(k0*na))*1e6;                   % 抛物线剖面基模模场半径
    p.alpha_opt = 2 - 12*p.delta/5;
    p.alpha_profile = p.alpha_opt + raw.alpha_profile_offset;
    p.D = raw.S0/4*(p.lam - raw.lambda0_nm^4/p.lam^3);
    p.S = raw.S0/4*(1 + 3*raw.lambda0_nm^4/p.lam^4);
end
for k = 1:2:numel(extra), p.(extra{k}) = extra{k+1}; end
end
