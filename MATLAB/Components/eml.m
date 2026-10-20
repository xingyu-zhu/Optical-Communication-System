function Eo = eml(E, u, er_db, alpha)
%EML 电吸收调制器 EAM（EML 中与激光集成），带瞬态啁啾。
%   用法: Eo = eml(E, u, 8, 0.5)
%   功率透过率 T(u) = 10^(-ER·(1-u)/20)，u ∈ [-1, 1]（u=1 最亮，u=-1 最暗，消光比 ER [dB]）；
%   Eo = E·T^((1+jα)/2)，即 Δφ = (α/2)·ln T。α>0 在 C 波段 SSMF 中加剧色散功率衰落，α<0 可部分抵消。
if nargin < 3, er_db = 8; end
if nargin < 4, alpha = 0.5; end
T = 10.^(-er_db*(1 - min(max(u, -1), 1))/20);
Eo = E .* T.^((1 + 1j*alpha)/2);
end
