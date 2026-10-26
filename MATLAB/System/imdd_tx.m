function ref = imdd_tx(cfg)
%IMDD_TX IM-DD 发射：比特 -> Gray PAM-M（电平归一化到 [-1, 1]）-> RC 成形 -> DAC -> MZM / EML / DML（内部 rng(cfg.seed)）。
%   用法: ref = imdd_tx(imdd_config('M', 4))
%   输出字段: levels 判决电平; bits 1×(n_sym·k); sym 1×n_sym; E 1×N 出纤光场 @cfg.fs
rng(cfg.seed);
k = log2(cfg.M);
ref.levels = pam_const(cfg.M) / (cfg.M - 1);
ref.bits = uint8(randi([0 1], 1, cfg.n_sym*k));
ref.sym = bits2sym(ref.bits, ref.levels);
u = dac_model(pulse_shape(ref.sym, cfg.sps, cfg.beta, 'rc'), cfg.fs, cfg.dac_enob, cfg.dac_bw*cfg.rs);
u = u / max(abs(u));                                      % 归一化驱动到 [-1, 1]
switch cfg.tx
    case 'dml'
        ref.E = dml(u, cfg.fs, cfg.p_tx, cfg.er_db, cfg.alpha, cfg.k_ad, [], cfg.linewidth, cfg.rin_db);
    case 'mzm'
        ref.E = mzm(laser_cw(numel(u), cfg.fs, cfg.p_tx, cfg.linewidth, 0, cfg.rin_db), cfg.swing*u, 1, -0.5);
    case 'eml'
        ref.E = eml(laser_cw(numel(u), cfg.fs, cfg.p_tx, cfg.linewidth, 0, cfg.rin_db), u, cfg.er_db, cfg.alpha);
end
end
