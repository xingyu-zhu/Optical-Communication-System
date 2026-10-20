function p = lfm_pulse(T, B, fs)
%LFM_PULSE 线性调频（chirp）脉冲复包络：时长 T [s]，频率由 -B/2 线性扫到 B/2。
%   用法: p = lfm_pulse(1e-6, 100e6, 250e6)
t = (0:round(T*fs)-1)/fs - T/2;
p = exp(1j*pi*B/T*t.^2);
end
