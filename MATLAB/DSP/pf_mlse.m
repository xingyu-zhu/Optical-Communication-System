function a = pf_mlse(ye, ref, n_train, levels, mem)
%PF_MLSE FFE + 噪声白化后滤波 + MLSE（IM-DD 带宽受限 / 色散功率衰落的常用方案）。
%   用法: a = pf_mlse(ye, sym, 4096, levels)       ye 为 FFE 输出（1 sps 行向量），返回判决电平序列
%         a = pf_mlse(ye, sym, n_train, levels, 2)  白化滤波器阶数 mem（默认 1）
%   FFE 输出噪声因高频增强而有色。训练段最小二乘拟合 AR(mem) 预测器 e_k ≈ -Σ p_i·e_{k-i}，
%   后滤波 z = (1 + Σ p_i·D^i)·ye 使噪声变白，同时引入已知 ISI h = [1, p_1..p_mem]，再用 mlse_viterbi 序列检测。
if nargin < 5, mem = 1; end
e = ye(1:n_train) - ref(1:n_train);
A = zeros(n_train, mem);
for i = 1:mem, A(:, i) = circshift(e, i).'; end
p = -(A(mem+1:end, :) \ e(mem+1:end).');
h = [1, p.'];
z = ye;
for i = 1:mem, z = z + h(i+1)*circshift(ye, i); end
a = mlse_viterbi(z, h, levels);
end
