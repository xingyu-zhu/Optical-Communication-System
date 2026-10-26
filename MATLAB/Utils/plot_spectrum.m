function plot_spectrum(x, fs, varargin)
%PLOT_SPECTRUM Welch 功率谱（只画第一行），双边、中心化。
%   用法: plot_spectrum(E, fs); hold on; plot_spectrum(E2, fs, 'r')   varargin 透传 plot
[p, f] = pwelch(x(1,:), 2048, [], 2048, fs, 'centered');
plot(f/1e9, 10*log10(p + 1e-30), varargin{:});
xlabel('Frequency (GHz)'); ylabel('PSD (dB/Hz)'); grid on;
end
