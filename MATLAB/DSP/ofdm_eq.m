function [Z, H] = ofdm_eq(Y, X, n_train, pilots)
%OFDM_EQ OFDM 单抽头频域均衡：前 n_train 个训练符号做 LS 信道估计 H = <Y./X>（各训练符号先对齐公共相位再平均，
%   避免激光相噪使平均失效），Y./H；给出 pilots（导频子载波下标，1-based）时再逐符号去除公共相位误差
%   CPE = ∠Σ_p Y·X*。X 为已知发送符号（只用到训练行与导频列）。
%   用法: [Z, H] = ofdm_eq(Y, X, n_train, pilots)
if nargin < 3, n_train = 1; end
Xt = X(1:n_train, :); Xt(Xt == 0) = 1;
Hk = Y(1:n_train, :) ./ Xt;
H = mean(Hk .* exp(-1j*angle(sum(Hk .* conj(Hk(1,:)), 2))), 1);
Hd = H; Hd(Hd == 0) = 1;
Z = Y ./ Hd;
if nargin > 3 && ~isempty(pilots)
    Z = Z .* exp(-1j*angle(sum(Z(:, pilots) .* conj(X(1:size(Z,1), pilots)), 2)));
end
end
