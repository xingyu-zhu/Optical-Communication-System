function U = haar_unitary(n, s)
%HAAR_UNITARY n×n Haar 随机酉矩阵（复高斯矩阵 QR 分解并校正 R 对角相位）。
%   用法: U = haar_unitary(6)              用全局随机数流
%         U = haar_unitary(6, stream)      用指定 RandStream（模式耦合用，便于复现同一光纤实现）
if nargin < 2 || isempty(s), s = RandStream.getGlobalStream; end
Z = (randn(s, n) + 1j*randn(s, n))/sqrt(2);
[Q, R] = qr(Z);
d = diag(R);
U = Q .* (d./abs(d)).';
end
