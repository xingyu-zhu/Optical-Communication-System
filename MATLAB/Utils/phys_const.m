function c = phys_const()
%PHYS_CONST 物理常数（SI）。  用法: c = phys_const(); c.c, c.h, c.q, c.kB, c.f0
c.c  = 299792458;          % 光速 [m/s]
c.h  = 6.62607015e-34;     % 普朗克常数 [J s]
c.q  = 1.602176634e-19;    % 电子电荷 [C]
c.kB = 1.380649e-23;       % 玻尔兹曼常数 [J/K]
c.f0 = 193.414e12;         % 默认载波频率 [Hz] (1550 nm)
end
