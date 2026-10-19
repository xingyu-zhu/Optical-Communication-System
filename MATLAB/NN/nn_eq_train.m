function eq = nn_eq_train(x, y, arch, ntaps, sps, varargin)
%NN_EQ_TRAIN 训练神经网络均衡器（Deep Learning Toolbox，trainnet）：滑窗输入、输出中心符号。
%   用法: eq = nn_eq_train(x, y, 'mlp', 41, 2)
%         eq = nn_eq_train(x, y, 'cnn', 41, 2, 'channels', [16 16], 'kernel', 7)
%         eq = nn_eq_train(x, y, 'bilstm', 41, 2, 'hidden', 32, 'epochs', 60)
%   输入: x 接收信号（1×N 实，或 P×N 复）; y 对应发送符号（长度 N/sps）
%         arch 'mlp' | 'cnn' | 'bilstm'; ntaps 窗长（采样点）; sps 输入每符号采样数
%   名值对: hidden（MLP 默认 [128 64]，BiLSTM 默认 32）、channels [16 16]、kernel 7、
%           epochs 60、batch 512、lr 2e-3、val 0.1（验证集比例，早停 patience 8）、residual true
%   输出 eq 结构体：net, arch, ntaps, sps, residual, cplx（是否复数）；用 nn_eq_apply 均衡，nn_eq_mults 估计复杂度
%   残差学习：网络只学 y - 中心输入 的修正量。训练与测试请使用独立实现（不同比特、噪声）。
o = struct('hidden', [], 'channels', [16 16], 'kernel', 7, 'epochs', 60, 'batch', 512, 'lr', 2e-3, ...
           'val', 0.1, 'residual', true);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
eq = struct('arch', arch, 'ntaps', ntaps, 'sps', sps, 'residual', o.residual, 'cplx', ~isreal(y));
[Xin, Xc, fmt] = nn_eq_input(eq, x);
T = to_real(y);
if o.residual, T = T - Xc; end
C = size(Xc, 2); Cout = size(T, 2);
switch arch
    case 'mlp'
        h = o.hidden; if isempty(h), h = [128 64]; end
        layers = featureInputLayer(ntaps*C);
        for k = 1:numel(h), layers = [layers, fullyConnectedLayer(h(k)), leakyReluLayer(0.1)]; end %#ok<AGROW>
        layers = [layers, fullyConnectedLayer(Cout)];
    case 'cnn'
        layers = imageInputLayer([ntaps 1 C], 'Normalization', 'none');
        for k = 1:numel(o.channels)
            layers = [layers, convolution2dLayer([o.kernel 1], o.channels(k), 'Padding', 'same'), leakyReluLayer(0.1)]; %#ok<AGROW>
        end
        layers = [layers, fullyConnectedLayer(Cout)];
    case 'bilstm'
        h = o.hidden; if isempty(h), h = 32; end
        c = floor(ntaps/2) + 1;
        layers = [sequenceInputLayer(C, 'MinLength', ntaps), bilstmLayer(h, 'OutputMode', 'sequence'), ...
                  functionLayer(@(X) dlarray(reshape(stripdims(X(:,:,c)), size(X,1), []), 'CB'), 'Formattable', true), ...
                  fullyConnectedLayer(Cout)];
end
nv = max(round(o.val * size(T,1)), 1); it = 1:size(T,1)-nv; iv = size(T,1)-nv+1:size(T,1);
pick = @(A, i) select_obs(A, i, fmt);
opts = trainingOptions('adam', 'MaxEpochs', o.epochs, 'MiniBatchSize', o.batch, 'InitialLearnRate', o.lr, ...
    'Shuffle', 'every-epoch', 'ValidationData', {pick(Xin, iv), T(iv,:)}, 'ValidationPatience', 8, ...
    'ValidationFrequency', ceil(numel(it)/o.batch), 'OutputNetwork', 'best-validation', ...
    'InputDataFormats', fmt, 'TargetDataFormats', 'BC', 'Verbose', false);
eq.net = trainnet(pick(Xin, it), T(it,:), dlnetwork(layers), 'mse', opts);
end

function A = select_obs(A, i, fmt)
switch fmt
    case 'BC',   A = A(i,:);
    case 'SSCB', A = A(:,:,:,i);
    case 'CBT',  A = A(:,i,:);
end
end

function T = to_real(y)
if isreal(y), T = y(:); return; end
T = zeros(size(y,2), 2*size(y,1));
T(:,1:2:end) = real(y).'; T(:,2:2:end) = imag(y).';
end
