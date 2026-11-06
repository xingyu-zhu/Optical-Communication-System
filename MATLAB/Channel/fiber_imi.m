function E = fiber_imi(E, fs, imi_db, tau_max, n_paths)
%FIBER_IMI 空芯光纤模间干扰（IMI, intermodal interference）：基模沿光纤分布式耦合到高阶模（如 LP11），
%   以不同群速度传播后再耦合回基模，在接收端表现为一串弱的、随机相位的延迟回波。
%   用法: E = fiber_imi(E, fs, -50, 10e-12*80e3)          每跨 -50 dB，最大时延 = DMGD [s/m] × 长度 [m]
%         E = fiber_imi(E, fs, imi_db, tau_max, n_paths)  n_paths 离散回波数（默认 64）
%   模型: E_out = sqrt(1-η)·E + Σ_k sqrt(η/K)·e^{jφ_k}·E(t - τ_k)，τ_k ~ U(0, tau_max)，φ_k ~ U(0, 2π)
%   imi_db = 10lg η 为总 IMI 功率比 [dB]（随长度和高阶模抑制比变化）。对应 Python channel.imi
if nargin < 5, n_paths = 64; end
eta = 10^(imi_db/10);
w = 2*pi*freqaxis(size(E,2), fs);
tau = tau_max*rand(n_paths, 1);
ph = 2*pi*rand(n_paths, 1);
h = sqrt(1 - eta) + sqrt(eta/n_paths)*sum(exp(1j*ph - 1j*tau.*w), 1);
E = fdfilter(E, h);
end
