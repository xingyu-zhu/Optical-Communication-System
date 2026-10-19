function [s, lab, p, H] = pcs_mb(c, nu, n)
%PCS_MB Maxwell-Boltzmann 概率整形：p ∝ exp(-ν|c|^2)，按分布抽样 n 个符号（理想分布匹配器近似，未实现 CCDM）。
%   用法: [s, lab, p, H] = pcs_mb(qam_const(64), 0.05, 1e4)
%   输出: s 符号（重新归一化到单位功率）, lab 0-based 标签, p 概率, H 熵 [bit/sym]；ν 越大整形越强
p = exp(-nu*abs(c).^2); p = p / sum(p);
H = -sum(p .* log2(p));
lab = randsample(numel(c), n, true, p) - 1;
s = c(lab + 1) / sqrt(sum(p .* abs(c).^2));
lab = lab(:).';
s = s(:).';
end
