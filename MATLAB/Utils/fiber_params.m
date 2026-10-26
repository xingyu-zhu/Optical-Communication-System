function fib = fiber_params(band, fiber, varargin)
%FIBER_PARAMS 从共用参数文件 params/fiber_bands.json 生成指定波段、光纤类型的传输参数。
%
%   fib = fiber_params(band, fiber)
%   fib = fiber_params(band, fiber, 'L', 100, 'lam', 1600, 'pmd', 0, 'raman', true, ...)
%
%   输入
%     band   'O' | 'E' | 'S' | 'C' | 'L' | 'U'
%     fiber  'SSMF' | 'ULL' | 'NZDSF' | 'DSF'（可在 JSON 中自行添加）
%     名值对 'lam' 工作波长 [nm]（默认波段中心）、'L' 跨段长度 [km]（默认 80），
%            以及任意输出字段的覆盖值
%   输出 fib 结构体
%     band, fiber, lam [nm], f0 [Hz], L [km], alpha [dB/km], D [ps/(nm km)], S [ps/(nm^2 km)],
%     gamma [1/(W km)], aeff [um^2], n2 [m^2/W], pmd [ps/sqrt(km)], nf [dB], amplifier,
%     raman / steep（是否计入 SRS / 自陡峭，默认 false）, fR, tau1, tau2 [s]
%   模型
%     D(λ) = S0/4·(λ - λ0^4/λ^3)，S(λ) = S0/4·(1 + 3λ0^4/λ^4)
%     Aeff(λ) = Aeff_1550·(λ/1550)^aeff_exp，γ = 2π·n2/(λ·Aeff)
%   示例
%     fib = fiber_params('C', 'SSMF');  fib = fiber_params('L', 'ULL', 'pmd', 0);
p = load_params();
b = p.bands.(band); f = p.fibers.(fiber); r = p.raman;
opt = struct(varargin{:});
lam = b.center_nm; if isfield(opt, 'lam'), lam = opt.lam; end
c = phys_const();
aeff = f.aeff_um2 * (lam/1550)^f.aeff_exp;
fib = struct('band', band, 'fiber', fiber, 'lam', lam, 'f0', c.c/(lam*1e-9), 'L', 80, ...
    'alpha', b.alpha_db_km + f.alpha_offset_db_km, ...
    'D', f.S0/4*(lam - f.lambda0_nm^4/lam^3), 'S', f.S0/4*(1 + 3*f.lambda0_nm^4/lam^4), ...
    'gamma', 2*pi*f.n2/(lam*1e-9*aeff*1e-12)*1e3, 'aeff', aeff, 'n2', f.n2, 'pmd', f.pmd, ...
    'nf', b.nf_db, 'amplifier', b.amplifier, 'raman', false, 'steep', false, ...
    'fR', r.fR, 'tau1', r.tau1_fs*1e-15, 'tau2', r.tau2_fs*1e-15);
for k = fieldnames(opt)', fib.(k{1}) = opt.(k{1}); end
end
