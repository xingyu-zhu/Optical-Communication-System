function r = wdm_rx(E, cfg, f_ch, fo)
%WDM_RX WDM 接收单个信道：WSS 选出 f_ch 信道 -> 本振调谐到 f_ch+fo -> 相干接收 -> ADC（理想量化）。
%   用法: r = wdm_rx(E, cfg, freqs(k))          输出 2×N' @cfg.fs_adc，可直接送入 rx_dsp
%         r = wdm_rx(E, cfg, freqs(k), 1e9)     本振额外频偏 1 GHz
if nargin < 4, fo = 0; end
Ec = optical_filter(E, cfg.fs, cfg.spacing, 4, f_ch);
lo = laser_cw(size(E,2), cfg.fs, 10, cfg.linewidth, f_ch + fo);
r = coherent_rx(Ec, lo, cfg.fs, struct('shot', false));
r = adc_model(r, cfg.fs, cfg.fs_adc, [], 0.75*cfg.rs);
end
