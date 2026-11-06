function [emb, f, H] = mmf_emb(mmf, P, f_max, n)
%MMF_EMB 有效模式带宽 EMB [MHz·km]：模式群冲激响应 h(t) = Σ P_g δ(t - τ_g·1 km) 的 -3 dB 光带宽 × 1 km。
%   用法: emb = mmf_emb(mmf, mmf_launch(mmf, 'ofl'))
%         [emb, f, H] = mmf_emb(mmf, P, f_max, n)       默认 f_max = 1 THz、n = 100001 点
%   |H(f)| 首次降到 0.5（H(0) = 1）；只含模式色散，不含色度色散；f_max 内未降到 0.5 时 emb = Inf。
%   对应 Python multimode.mmf_emb
if nargin < 3 || isempty(f_max), f_max = 1e12; end
if nargin < 4 || isempty(n), n = 100001; end
q = mmf_groups(mmf);
P = P(:)/sum(P);
f = linspace(0, f_max, n).';
H = abs(exp(-2j*pi*f*q.tau)*P);
i = find(H < 0.5, 1);
if isempty(i), emb = Inf; else, emb = f(i)/1e6; end
end
