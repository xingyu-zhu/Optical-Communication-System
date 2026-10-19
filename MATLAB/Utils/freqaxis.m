function f = freqaxis(N, fs)
%FREQAXIS FFT 顺序的频率轴（行向量）[Hz]，与 fft 输出一一对应。  用法: f = freqaxis(N, fs)
f = [0:ceil(N/2)-1, -floor(N/2):-1] * fs / N;
end
