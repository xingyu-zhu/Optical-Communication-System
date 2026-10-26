function s = sens_const()
%SENS_CONST 光纤传感常数（石英单模光纤 @1550 nm 典型值）。
%   用法: s = sens_const();  s.ng 群折射率；s.alpha_s 瑞利散射系数 [1/m]；s.capture 背向捕获因子 S
s.ng = 1.468;
s.alpha_s = 4.6e-5;
s.capture = 1.5e-3;
end
