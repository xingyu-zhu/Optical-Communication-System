function E = voa(E, att_db)
%VOA 可调光衰减器。  用法: E = voa(E, 10)   衰减 10 dB
E = E * 10^(-att_db/20);
end
