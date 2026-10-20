function y = ddc(x, fs, f_if, bw)
%DDC 数字下变频：外差探测（如 AOM 移频 f_if）的实电流 -> 乘 e^{-j2πf_if·t} -> 低通 bw -> 复基带。
%   用法: y = ddc(x, fs, 80e6, 20e6)
y = elpf(x .* exp(-2j*pi*f_if*(0:size(x,2)-1)/fs), fs, bw);
end
