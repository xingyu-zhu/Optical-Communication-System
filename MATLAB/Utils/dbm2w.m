function w = dbm2w(p)
%DBM2W dBm -> W
w = 1e-3 * 10.^(p/10);
end
