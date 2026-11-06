function r = sdm_rx(E, cfg, f_offset, lo_dbm, enob)
%SDM_RX 相干接收机阵列：模式解复用（理想）后 M 个偏振分集相干接收机，共用一个本振（分束），再经 ADC。
%   用法: r = sdm_rx(E, cfg)                 本振频偏默认 1.5 GHz、13 dBm，ADC ENOB 6
%         r = sdm_rx(E, cfg, f_offset, lo_dbm, enob)
%   输出 2M×N' ADC 输出 @cfg.fs_adc。对应 Python multimode.sdm_rx
if nargin < 3, f_offset = 1.5e9; end
if nargin < 4, lo_dbm = 13; end
if nargin < 5, enob = 6; end
lo = laser_cw(size(E,2), cfg.fs, lo_dbm, cfg.linewidth, f_offset);
pd = struct('R', 0.8, 'bw', 0.75*cfg.rs, 'T', 300);
r = zeros(size(E));
for m = 1:size(E,1)/2
    r(2*m-1:2*m, :) = coherent_rx(E(2*m-1:2*m, :), lo, cfg.fs, pd);
end
r = adc_model(r, cfg.fs, cfg.fs_adc, enob, 0.75*cfg.rs);
end
