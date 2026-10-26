function y = fo_comp(x, f, fs)
%FO_COMP 频偏补偿：x·exp(-j2πft)。  用法: y = fo_comp(x, foe(x, rs), rs)
y = x .* exp(-2j*pi*f*(0:size(x,2)-1)/fs);
end
