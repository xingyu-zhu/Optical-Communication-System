function i = bpd(E1, E2, fs, pd)
%BPD 平衡探测器：photodiode(E1) - photodiode(E2)（抑制 LO 直流与信号-信号拍频）。
%   用法: i = bpd(E1, E2, fs, pd)    pd 同 photodiode
if nargin < 4, pd = struct(); end
i = photodiode(E1, fs, pd) - photodiode(E2, fs, pd);
end
