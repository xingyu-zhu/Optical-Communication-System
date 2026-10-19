function [ber, nerr] = calc_ber(tx_bits, rx_bits)
%CALC_BER 误码率与误比特数。  用法: [ber, nerr] = calc_ber(tx_bits, rx_bits)
nerr = sum(tx_bits(:) ~= rx_bits(:));
ber = nerr / numel(tx_bits);
end
