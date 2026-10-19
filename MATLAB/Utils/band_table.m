function band_table(fiber)
%BAND_TABLE 打印某光纤在各波段中心波长处的参数表。  用法: band_table('SSMF')
if nargin < 1, fiber = 'SSMF'; end
fprintf('%-4s %7s %6s %7s %6s %6s %6s %4s  %s\n', 'band', 'lam', 'alpha', 'D', 'S', 'gamma', 'Aeff', 'NF', 'amplifier');
for b = fieldnames(load_params().bands)'
    q = fiber_params(b{1}, fiber);
    fprintf('%-4s %7.0f %6.3f %7.2f %6.3f %6.3f %6.1f %4.1f  %s\n', b{1}, q.lam, q.alpha, q.D, q.S, q.gamma, q.aeff, q.nf, q.amplifier);
end
end
