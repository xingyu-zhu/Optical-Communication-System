function o = hybrid90(Es, Elo)
%HYBRID90 2×4 90° 光混频器。  用法: o = hybrid90(Es, Elo)
%   输出 4×N：[I+; I-; Q+; Q-] = [Es+Elo; Es-Elo; Es+jElo; Es-jElo]/2
o = [Es + Elo; Es - Elo; Es + 1j*Elo; Es - 1j*Elo] / 2;
end
