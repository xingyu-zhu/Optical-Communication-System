function E = imdd_link(E, cfg)
%IMDD_LINK 短距光纤：色散（含斜率，参数随波段）+ 损耗。功率低、距离短，忽略光纤非线性。  用法: E = imdd_link(ref.E, cfg)
f = cfg.fib;
E = fiber_cd(E, cfg.fs, cfg.L, f.D, f.lam, f.S) * 10^(-f.alpha*cfg.L/20);
end
