function E = fiber_link(E, fs, fib, n_spans, ase, varargin)
%FIBER_LINK 多跨段链路：每跨 ssfm_manakov + 放大器（增益 = 跨段损耗，NF = fib.nf，ASE 中心频率 fib.f0）。
%   用法: E = fiber_link(E, fs, fib, 10)                           默认 'phi_max' 0.005，含 ASE
%         E = fiber_link(E, fs, fib, 10, false)                    理想无噪放大
%         E = fiber_link(E, fs, fib, 10, true, 'nsteps', 40)       其余参数透传 ssfm_manakov
if nargin < 5, ase = true; end
if isempty(varargin), varargin = {'phi_max', 0.005}; end
G = fib.alpha*fib.L;
for k = 1:n_spans
    E = ssfm_manakov(E, fs, fib, varargin{:});
    if ase, E = edfa(E, fs, G, fib.nf, [], fib.f0); else, E = E * 10^(G/20); end
end
end
