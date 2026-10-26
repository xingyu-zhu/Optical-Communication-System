function e = calc_evm(rx, tx)
%CALC_EVM rms EVM [%] = sqrt(E|rx-tx|² / E|tx|²)。  用法: e = calc_evm(rx, tx)
e = 100*sqrt(mean(abs(rx(:) - tx(:)).^2) / mean(abs(tx(:)).^2));
end
