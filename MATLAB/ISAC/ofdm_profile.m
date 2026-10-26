function h = ofdm_profile(y, X, nfft, ncp)
%OFDM_PROFILE 通感一体（后向）：CP-OFDM 背向散射 -> 距离剖面。
%   H = ofdm_demod(y)./X（空载波置 0，OFDM 雷达式处理，Sturm & Wiesbeck 2011），h = IFFT(H)。
%   要求 CP ≥ 散射记忆（被测光纤往返时延对应的采样数），否则出现空间 ISI。
%   用法: h = ofdm_profile(y, X, nfft, ncp)   y 与发送对齐的背向散射基带（行向量）；X 为 nsym×nfft 已知发送符号
%   输出 nsym×nfft，第 i 列对应距离单元 i（单元长 cell_len(fs)）
Y = ofdm_demod(y, nfft, ncp);
Y = Y(1:size(X,1), :);
H = zeros(size(X));
m = X ~= 0;
H(m) = Y(m) ./ X(m);
h = ifft(H, [], 2);
end
