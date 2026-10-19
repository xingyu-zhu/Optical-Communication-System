function E = freq_shift(E, f, fs)
%FREQ_SHIFT 频移 f [Hz]（乘 exp(j2πft)），把基带信道搬到 WDM 栅格或搬回基带。  用法: E = freq_shift(E, 50e9, fs)
E = E .* exp(2j*pi*f*(0:size(E,2)-1)/fs);
end
