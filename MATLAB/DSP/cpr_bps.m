function [y, phi] = cpr_bps(x, c, n_test, n, rot)
%CPR_BPS 盲相位搜索 BPS（Pfau 2009）：在 [-π/4, π/4) 测试 n_test 个相位，取 n 点窗内判决距离最小者，
%   再做 π/2 周期解卷绕。适用任意 QAM，结果含 π/2 模糊（用 fix_ambiguity 消除）。
%   用法: [y, phi] = cpr_bps(x, c)              默认 32 个测试相位、窗长 64
%         [y, phi] = cpr_bps(x, c, n_test, n)
if nargin < 3, n_test = 32; end
if nargin < 4, n = 64; end
if nargin < 5, rot = 4; end                       % 星座旋转对称阶数（rot_order）：QAM/QPSK 4，BPSK 2，8PSK 8
phis = ((0:n_test-1)/n_test - 0.5) * 2*pi/rot;
phi = zeros(size(x));
for p = 1:size(x,1)
    d = zeros(n_test, size(x,2));
    for k = 1:n_test
        d(k,:) = min(abs(x(p,:).'*exp(-1j*phis(k)) - c(:).').^2, [], 2).';
    end
    [~, i] = min(movmean(d, n, 2), [], 1);
    phi(p,:) = unwrap(phis(i)*rot) / rot;
end
y = x .* exp(-1j*phi);
end
