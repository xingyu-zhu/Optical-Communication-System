function dl = fbg_shift(lam_b, dT, eps, pe, kT)
%FBG_SHIFT FBG 中心波长漂移 Δλ = λ_B·[(1−pe)·ε + kT·ΔT] [nm]（ε [με]，kT = 热膨胀 + 热光系数）。
%   用法: dl = fbg_shift(1550, dT, eps)     典型 @1550 nm：约 1.2 pm/με、11 pm/K
if nargin < 4, pe = 0.22; end
if nargin < 5, kT = 7e-6; end
dl = lam_b*((1 - pe)*eps*1e-6 + kT*dT);
end
