function E = ssfm_manakov(E, fs, fib, varargin)
%SSFM_MANAKOV 对称分步傅里叶法单跨传输（Manakov 方程；单行输入则为标量 NLSE）。
%   用法: E = ssfm_manakov(E, fs, fib, 'nsteps', 40)          均匀步长
%         E = ssfm_manakov(E, fs, fib, 'phi_max', 0.005)      自适应步长（每步最大非线性相移 [rad]，WDM 推荐）
%         可选 'dz_max' 自适应步长上限 [km]（默认 2）
%   fib 由 fiber_params 生成：L, alpha, D, S, gamma, lam, f0, pmd, raman, steep, fR, tau1, tau2
%
%   方程（工程约定 E·exp(+jω0t)）：
%     ∂E/∂z = -α/2·E - j(β2ω²/2 + β3ω³/6)E
%             - j·γ·(8/9)·(1 + ω/ω0)·F{ E·[(1-fR)|E|² + fR·(h_R ⊗ |E|²)] }
%   - SPM / XPM / FWM：对全场（单信道或 WDM 合波）求解时自动包含（含信道内 IXPM/IFWM）
%   - SRS：fib.raman=true 时计入 fR·h_R 延迟响应（WDM 中高频信道向低频转移功率）
%   - 自陡峭：fib.steep=true 时计入 (1+ω/ω0)
%   - 以上两项开启时非线性子步用 RK4，否则用精确相位旋转
%   - PMD：fib.pmd>0 时每步随机 SOP + DGD（粗步长，平均 DGD = pmd·sqrt(L)）
o = struct('nsteps', [], 'phi_max', [], 'dz_max', 2);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
if ~isfield(fib, 'S'), fib.S = 0; end
Lm = fib.L*1e3;
a  = fib.alpha*log(10)/10/1e3;                           % 功率衰减 [1/m]
w  = 2*pi*freqaxis(size(E,2), fs);
ph = beta2_from_D(fib.D, fib.lam)/2*w.^2;
if fib.S, ph = ph + beta3_from_DS(fib.D, fib.S, fib.lam)/6*w.^3; end
lin = -a/2 - 1j*ph;                                      % 线性算子 [1/m]
g  = fib.gamma/1e3 * (1 - (size(E,1)==2)/9);            % 双偏振取 8/9
full = fib.raman || fib.steep;
fR = 0; HR = 1; sh = 1;
if fib.raman, fR = fib.fR; HR = raman_response(w, fib.tau1, fib.tau2); end
if fib.steep, sh = 1 + w/(2*pi*fib.f0); end
nl = @(E) -1j*g*ifft(fft(E .* ((1-fR)*sum(abs(E).^2,1) + fR*real(ifft(fft(sum(abs(E).^2,1)).*HR))), [], 2) .* sh, [], 2);
z = 0;
while z < Lm - 1e-9
    if ~isempty(o.phi_max)
        dz = min([o.dz_max*1e3, o.phi_max/max(abs(g)*max(sum(abs(E).^2,1)), 1e-30), Lm - z]);
    else
        dz = Lm/o.nsteps;
    end
    Hh = exp(lin*dz/2);
    E = fdfilter(E, Hh);
    if full                                              % RK4 非线性子步
        k1 = dz*nl(E); k2 = dz*nl(E + k1/2); k3 = dz*nl(E + k2/2); k4 = dz*nl(E + k3);
        E = E + (k1 + 2*k2 + 2*k3 + k4)/6;
    else                                                 % 纯 Kerr：精确相位旋转
        E = E .* exp(-1j*g*sum(abs(E).^2,1)*dz);
    end
    E = fdfilter(E, Hh);
    if fib.pmd > 0 && size(E,1) == 2
        E = dgd_elem(random_unitary()*E, fs, fib.pmd*1e-12*sqrt(dz/1e3)*sqrt(3*pi/8));
    end
    z = z + dz;
end
end
