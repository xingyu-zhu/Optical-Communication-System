function [Xin, Xc, fmt] = nn_eq_input(eq, x)
%NN_EQ_INPUT 把接收信号整理成对应网络结构的输入张量（nn_eq_train / nn_eq_apply 内部使用）。
%   输出: Xin 网络输入，Xc Nsym×C 中心采样（残差基准），fmt 数据格式（'BC' / 'SSCB' / 'CBT'）
X = sym_windows(x, eq.ntaps, eq.sps);                    % Nsym×W×C
Xc = squeeze(X(:, floor(eq.ntaps/2) + 1, :));
if isrow(Xc), Xc = Xc.'; end
switch eq.arch
    case 'mlp',    Xin = reshape(X, size(X,1), []);       fmt = 'BC';
    case 'cnn',    Xin = permute(X, [2 4 3 1]);           fmt = 'SSCB';   % W×1×C×Nsym
    case 'bilstm', Xin = permute(X, [3 1 2]);             fmt = 'CBT';    % C×Nsym×W
end
end
