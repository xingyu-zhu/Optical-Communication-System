function m = nn_eq_mults(eq)
%NN_EQ_MULTS 每个输出符号所需实数乘法次数（推理复杂度，忽略激活函数）。  用法: m = nn_eq_mults(eq)
m = 0; W = eq.ntaps;
for L = eq.net.Layers.'
    if isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
        m = m + numel(L.Weights);
    elseif isa(L, 'nnet.cnn.layer.Convolution2DLayer')
        m = m + numel(L.Weights) * W;                           % 'same' 填充，每个时间位置都计算
    elseif isa(L, 'nnet.cnn.layer.BiLSTMLayer')
        m = m + (numel(L.InputWeights) + numel(L.RecurrentWeights)) * W;   % 两个方向、W 个时间步
    end
end
end
