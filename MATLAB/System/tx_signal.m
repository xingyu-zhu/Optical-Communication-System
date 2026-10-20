function ref = tx_signal(cfg)
%TX_SIGNAL 发射端 DSP：比特 -> Gray QAM / PSK（cfg.mod）-> 成帧 [训练 | 导频+数据] -> RRC 成形（内部 rng(cfg.seed)）。
%   用法: ref = tx_signal(cfg)
%   输出字段: const 星座; bits 2×nbits 数据比特; frame 2×Nf 帧符号; kind 1×Nf (0 数据/1 训练/2 导频);
%             wave 2×(Nf·sps) 每偏振单位功率波形 @cfg.fs
rng(cfg.seed);
c = constellation(cfg.mod, cfg.M); k = log2(cfg.M);
nd = cfg.n_blocks * (cfg.pilot_period - 1);
ref.const = c;
ref.bits  = uint8(randi([0 1], 2, nd*k));
train = c(randi(cfg.M, 2, cfg.train_len));
pilot = c(randi(cfg.M, 2, cfg.n_blocks));
[ref.frame, ref.kind] = framing(bits2sym(ref.bits, c), train, pilot, cfg.pilot_period);
ref.wave = normpow(pulse_shape(ref.frame, cfg.sps, cfg.beta));
end
