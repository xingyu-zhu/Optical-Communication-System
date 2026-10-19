function E = wdm_mux(fields, freqs, fs)
%WDM_MUX WDM 合波器（理想耦合）：Σ_k freq_shift(fields{k}, freqs(k))。
%   用法: E = wdm_mux({E1, E2, E3}, [-50e9 0 50e9], fs)    各 fields{k} 为同尺寸光场
E = zeros(size(fields{1}));
for k = 1:numel(fields), E = E + freq_shift(fields{k}, freqs(k), fs); end
end
