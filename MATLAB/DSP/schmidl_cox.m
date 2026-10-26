function [d, fo, Mt] = schmidl_cox(y, L, fs)
%SCHMIDL_COX Schmidl-Cox 定时与频偏估计：前导为两个相同的半符号（各 L 点）。
%   M(d) = |P(d)|²/R(d)²，P(d) = Σ_m y*[d+m]·y[d+m+L]，R(d) = Σ_m |y[d+m+L]|²
%   用法: [d, fo, Mt] = schmidl_cox(y, L, fs)   d 为前导起点（1-based 下标）；fo = ∠P(d)·fs/(2πL)，范围 ±fs/(2L)
if nargin < 3, fs = 1; end
y = y(:).';
P = conv(conj(y(1:end-L)) .* y(L+1:end), ones(1, L), 'valid');
R = conv(abs(y(L+1:end)).^2, ones(1, L), 'valid');
Mt = abs(P).^2 ./ max(R, 1e-30).^2;
[~, d] = max(Mt);
fo = angle(P(d))*fs/(2*pi*L);
end
