function p = otdr_trace(z, p_dbm, W, alpha_db, events)
%OTDR_TRACE 传统 OTDR 背向散射功率轨迹 [W]（宽谱光源、多次平均，无相干衰落）：
%   P(z) = P0·(c·W/2ng)·αs·S·e^{-2αz}；events 每行 [位置 m, 单程损耗 dB]，其后乘往返损耗；最后按脉宽平滑。
%   用法: p = otdr_trace(z, 20, 100e-9, 0.2, [5e3 0.5; 12e3 1.0])
if nargin < 4, alpha_db = 0.2; end
if nargin < 5, events = zeros(0, 2); end
c0 = phys_const(); s = sens_const();
a = alpha_db*log(10)/10/1e3;
p = dbm2w(p_dbm)*c0.c*W/(2*s.ng)*s.alpha_s*s.capture*exp(-2*a*z);
for k = 1:size(events, 1)
    p(z >= events(k,1)) = p(z >= events(k,1)) * 10^(-2*events(k,2)/10);
end
p = movmean(p, max(round(c0.c*W/(2*s.ng)/(z(2)-z(1))), 1));
end
