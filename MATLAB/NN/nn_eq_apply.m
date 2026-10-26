function y = nn_eq_apply(eq, x)
%NN_EQ_APPLY 用训练好的 NN 均衡器均衡：输出 1 sps，格式与训练目标相同（实 1×Nsym 或复 P×Nsym）。
%   用法: y = nn_eq_apply(eq, x)
[Xin, Xc, fmt] = nn_eq_input(eq, x);
Y = minibatchpredict(eq.net, Xin, 'InputDataFormats', fmt, 'OutputDataFormats', 'BC', 'MiniBatchSize', 16384);
Y = double(gather(Y));
if eq.residual, Y = Y + Xc; end
if eq.cplx, y = (Y(:,1:2:end) + 1j*Y(:,2:2:end)).'; else, y = Y.'; end
end
